"""Dynamic chunk-aligned image view with lazy per-level loading."""

import traceback

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui


class DynamicImageView(pg.ImageView):
    """pyqtgraph ImageView that lazy-loads only the visible region at the
    best-fit pyramid level and keeps the world transform correct across
    level transitions."""

    levelChanged = QtCore.pyqtSignal(int, tuple)
    cursorMoved = QtCore.pyqtSignal(float, float)  # world coords

    _RELOAD_DEBOUNCE_MS = 100

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.multires_image = None
        self.current_level = 0
        self.current_slice = 0
        self._last_load_params = None
        self._initial_load_done = False
        self._current_display_data = None

        self._reload_timer = QtCore.QTimer(self)
        self._reload_timer.setSingleShot(True)
        self._reload_timer.timeout.connect(self._reload_visible_region)
        self.view.sigRangeChanged.connect(self._on_view_changed)

        if getattr(self, "imageItem", None) is not None:
            self.imageItem.setAutoDownsample(True)

        # Report cursor position (world coords) for the status bar.
        self.scene.sigMouseMoved.connect(self._emit_cursor_moved)

    def _emit_cursor_moved(self, scene_pos):
        if self.image is None:
            return
        p = self.view.mapSceneToView(scene_pos)
        self.cursorMoved.emit(float(p.x()), float(p.y()))

    def set_multires_image(self, multires_image):
        """Attach a MultiResolutionImage and display the coarsest level."""
        self.multires_image = multires_image
        self._last_load_params = None
        self._initial_load_done = False

        level0_shape = self.multires_image.get_level_shape(0)
        if level0_shape is None or len(level0_shape) < 3:
            self.current_slice = 0
        else:
            self.current_slice = int(level0_shape[0] // 2)

        initial_level = len(self.multires_image.levels) - 1
        self.current_level = initial_level
        self._show_initial_frame(initial_level)
        self._initial_load_done = True

    def set_slice(self, slice_idx_level0: int):
        """Set slice index in level-0 coordinates and reload."""
        self.current_slice = max(0, int(slice_idx_level0))
        self._last_load_params = None
        self._reload_visible_region()

    def _on_view_changed(self, *_):
        if self._initial_load_done:
            self._reload_timer.start(self._RELOAD_DEBOUNCE_MS)

    @staticmethod
    def _prepare_for_display(image):
        if image is None or image.size == 0:
            return image
        if image.dtype != np.float32:
            image = image.astype(np.float32, copy=False)
        return image

    def _show_initial_frame(self, level):
        """Load and display the full coarsest-level plane as the opening view."""
        lvl_shape = self.multires_image.get_level_shape(level)
        if lvl_shape is None or len(lvl_shape) < 3:
            return

        level0_shape = self.multires_image.get_level_shape(0)
        scale_z = level0_shape[0] / lvl_shape[0] if lvl_shape[0] > 0 else 1.0
        z_idx_L = int(np.clip(self.current_slice / scale_z, 0, lvl_shape[0] - 1))

        img = self.multires_image.get_chunk_data_aligned(
            level, z_idx_L, (0, lvl_shape[1]), (0, lvl_shape[2])
        )
        if img is None:
            print(f"[sview] failed to load initial frame at level {level}")
            return

        display_img = self._prepare_for_display(img)
        self._current_display_data = display_img

        downsample_y = level0_shape[-2] / lvl_shape[-2]
        downsample_x = level0_shape[-1] / lvl_shape[-1]
        _, y_off, x_off = self.multires_image.get_world_offset(level)

        # Translate-then-scale: level pixel (i, j) -> (x_off + i*ds_x,
        # y_off + j*ds_y). pyqtgraph's setImage(pos=, scale=) applies these
        # in the opposite order and misplaces the image, so we build the
        # transform ourselves.
        tr = QtGui.QTransform()
        tr.translate(float(x_off), float(y_off))
        tr.scale(float(downsample_x), float(downsample_y))
        self.setImage(display_img, autoLevels=True, autoRange=True, transform=tr)

    def _compute_view_scale(self):
        view_box = self.view.viewRect()
        widget_size = self.view.size()
        if widget_size.width() <= 0 or view_box.width() <= 0 or view_box.height() <= 0:
            return 1.0, view_box
        ppx = widget_size.width() / view_box.width()
        ppy = widget_size.height() / view_box.height()
        return min(ppx, ppy), view_box

    def _reload_visible_region(self):
        """Fetch only the chunks that cover the visible region at the best
        level, place them in world coordinates, and swap the image in."""
        if self.multires_image is None:
            return

        try:
            view_scale, view_box = self._compute_view_scale()
            level0_shape = self.multires_image.get_level_shape(0)

            optimal_level = self.multires_image.get_optimal_level(
                view_scale, axes=(-2, -1), current_level=self.current_level
            )
            lvl_shape = self.multires_image.get_level_shape(optimal_level)
            if lvl_shape is None or len(lvl_shape) < 3:
                return

            z_size, y_size, x_size = lvl_shape[-3:]
            scale_z = level0_shape[0] / z_size if z_size > 0 else 1.0
            z_idx_L = int(np.clip(self.current_slice / scale_z, 0, z_size - 1))

            downsample_y = level0_shape[-2] / lvl_shape[-2]
            downsample_x = level0_shape[-1] / lvl_shape[-1]
            _, y_off, x_off = self.multires_image.get_world_offset(optimal_level)

            # View bounds (world) -> level pixel indices, minus translation.
            vx0 = int(np.floor((view_box.left() - x_off) / downsample_x))
            vy0 = int(np.floor((view_box.top() - y_off) / downsample_y))
            vx1 = int(np.ceil((view_box.right() - x_off) / downsample_x))
            vy1 = int(np.ceil((view_box.bottom() - y_off) / downsample_y))
            vx0 = max(0, min(vx0, x_size))
            vy0 = max(0, min(vy0, y_size))
            vx1 = max(0, min(vx1, x_size))
            vy1 = max(0, min(vy1, y_size))
            if vx1 <= vx0 or vy1 <= vy0:
                vx0, vy0, vx1, vy1 = 0, 0, x_size, y_size

            load_params = (optimal_level, z_idx_L, vy0, vy1, vx0, vx1)
            if load_params == self._last_load_params:
                return

            level_changed = (optimal_level != self.current_level)
            if level_changed:
                print(f"[sview] level transition: {self.current_level} -> {optimal_level}")

            self._last_load_params = load_params
            self.current_level = optimal_level

            img = self.multires_image.get_chunk_data_aligned(
                optimal_level, z_idx_L, (vy0, vy1), (vx0, vx1)
            )
            if img is None or img.size == 0:
                return

            display = self._prepare_for_display(img)
            self._current_display_data = display

            x0_world = vx0 * downsample_x + x_off
            y0_world = vy0 * downsample_y + y_off

            tr = QtGui.QTransform()
            tr.translate(float(x0_world), float(y0_world))
            tr.scale(float(downsample_x), float(downsample_y))
            self.setImage(display, autoLevels=False, autoRange=False, transform=tr)

            if level_changed:
                self.levelChanged.emit(optimal_level, tuple(lvl_shape))

        except Exception as e:
            print(f"[sview] reload error: {e}")
            traceback.print_exc()

    def get_current_image_data(self):
        """Return the pixel data currently on screen (for contrast/stats)."""
        return self._current_display_data
