"""Dynamic chunk-aligned image view with lazy per-level loading."""

# Bind PyOpenGL to the same GL libraries as Qt before pyqtgraph imports it.
from zview.sysinfo import apply_gl_compat as _apply_gl_compat
_apply_gl_compat()

import math
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui

from zview import _log

# dtypes pyqtgraph renders natively (with levels + LUT); anything else is
# converted to float32 once per frame.
_NATIVE_DTYPES = {np.dtype(t) for t in (np.uint8, np.uint16, np.int8, np.int16,
                                        np.float32, np.float64)}


class DynamicImageView(pg.ImageView):
    """pyqtgraph ImageView that lazy-loads only the visible region at the
    best-fit pyramid level and keeps the world transform correct across
    level transitions."""

    levelChanged = QtCore.pyqtSignal(int, tuple)
    # World coords (h, v) under the cursor and the displayed pixel value
    # there (NaN when outside the image).
    cursorMoved = QtCore.pyqtSignal(float, float, float)
    # Ctrl+wheel over the pane: request a slice step of +/-1.
    sliceStepRequested = QtCore.pyqtSignal(int)

    # Internal: worker thread -> GUI thread hand-off of fetched pixels.
    _fetched = QtCore.pyqtSignal(int, object)

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

        self.scene.sigMouseMoved.connect(self._emit_cursor_moved)
        self.ui.graphicsView.viewport().installEventFilter(self)

        # Region fetches run off the GUI thread so panning/slicing never
        # blocks on I/O. One worker per pane keeps requests ordered; stale
        # requests are skipped by generation number.
        self._loader = ThreadPoolExecutor(max_workers=1, thread_name_prefix="zview-pane")
        self._req_gen = 0
        self.has_refined = False   # a region fetch has been shown since attach
        self._fetched.connect(self._on_fetched)

    # ------------------------------------------------------------ async fetch

    def _submit_fetch(self, fetch, apply):
        """Run `fetch()` on the pane's worker; call `apply(result)` on the GUI
        thread if no newer request has been made in the meantime."""
        self._req_gen += 1
        gen = self._req_gen

        def work():
            if gen != self._req_gen:
                return  # superseded before it started
            try:
                res = fetch()
            except Exception as e:  # surfaced on the GUI thread
                res = e
            self._fetched.emit(gen, (res, apply))

        self._loader.submit(work)

    def _cancel_pending(self):
        self._req_gen += 1
        self.has_refined = False

    def _on_fetched(self, gen, payload):
        if gen != self._req_gen:
            return
        res, apply = payload
        if isinstance(res, Exception):
            _log.error(f"[{type(self).__name__}] fetch error: {res}")
            self._last_load_params = None
            return
        try:
            apply(res)
            self.has_refined = True
        except Exception as e:
            _log.error(f"[{type(self).__name__}] apply error: {e}")
            traceback.print_exc()

    def schedule_refine(self, delay_ms=None):
        """Reload the visible region at the right level after `delay_ms`."""
        if self.multires_image is not None:
            self._reload_timer.start(self._RELOAD_DEBOUNCE_MS if delay_ms is None else delay_ms)

    # ------------------------------------------------------------ events

    def eventFilter(self, obj, event):
        if (event.type() == QtCore.QEvent.Wheel
                and event.modifiers() & QtCore.Qt.ControlModifier):
            delta = event.angleDelta().y()
            if delta:
                self.sliceStepRequested.emit(1 if delta > 0 else -1)
            return True
        return super().eventFilter(obj, event)

    def pixel_value_at(self, scene_pos):
        """Displayed value under `scene_pos`, or NaN."""
        data = self._current_display_data
        if data is None or data.ndim != 2 or self.imageItem is None:
            return math.nan
        p = self.imageItem.mapFromScene(scene_pos)
        col, row = int(math.floor(p.x())), int(math.floor(p.y()))
        if 0 <= row < data.shape[0] and 0 <= col < data.shape[1]:
            try:
                return float(data[row, col])
            except (TypeError, ValueError):
                return math.nan
        return math.nan

    def _emit_cursor_moved(self, scene_pos):
        if self.image is None:
            return
        p = self.view.mapSceneToView(scene_pos)
        self.cursorMoved.emit(float(p.x()), float(p.y()), self.pixel_value_at(scene_pos))

    # ------------------------------------------------------------ public

    def set_multires_image(self, multires_image):
        """Attach a MultiResolutionImage and display the coarsest level."""
        self._cancel_pending()
        self.multires_image = multires_image
        self._last_load_params = None
        self._initial_load_done = False

        level0_shape = self.multires_image.get_level_shape(0)
        self.current_slice = int(level0_shape[0] // 2) if level0_shape else 0

        initial_level = len(self.multires_image.levels) - 1
        self.current_level = initial_level
        self._show_initial_frame(initial_level)
        self._initial_load_done = True
        lvl_shape = self.multires_image.get_level_shape(initial_level)
        if lvl_shape is not None:
            self.levelChanged.emit(initial_level, tuple(lvl_shape))
        self.schedule_refine()

    def detach(self):
        """Drop the dataset and blank the pane (e.g. on File > Close)."""
        self._reload_timer.stop()
        self._cancel_pending()
        self.multires_image = None
        self._last_load_params = None
        self._initial_load_done = False
        self._current_display_data = None
        self.clear()

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
        if image.dtype in _NATIVE_DTYPES:
            return image
        return image.astype(np.float32, copy=False)

    def _set_image_with_transform(self, img, pos_x, pos_y, ds_x, ds_y,
                                  auto_levels=False, auto_range=False):
        """Show `img` with level pixel (i, j) -> world (pos_x + i*ds_x, pos_y + j*ds_y).

        Translate-then-scale: pyqtgraph's setImage(pos=, scale=) applies these
        in the opposite order and misplaces the image, so we build the
        transform ourselves.
        """
        display = self._prepare_for_display(img)
        self._current_display_data = display
        tr = QtGui.QTransform()
        tr.translate(float(pos_x), float(pos_y))
        tr.scale(float(ds_x), float(ds_y))
        self.setImage(display, autoLevels=auto_levels, autoRange=auto_range, transform=tr)

    def _show_initial_frame(self, level):
        """Load and display the full coarsest-level plane as the opening view."""
        lvl_shape = self.multires_image.get_level_shape(level)
        if lvl_shape is None:
            return
        level0_shape = self.multires_image.get_level_shape(0)
        scale_z = level0_shape[0] / lvl_shape[0] if lvl_shape[0] > 0 else 1.0
        z_idx_L = int(np.clip(self.current_slice / scale_z, 0, lvl_shape[0] - 1))

        img = self.multires_image.get_chunk_data_aligned(
            level, z_idx_L, (0, lvl_shape[1]), (0, lvl_shape[2])
        )
        if img is None:
            _log.error(f"[sview] failed to load initial frame at level {level}")
            return

        _, y_off, x_off = self.multires_image.get_world_offset(level)
        self._set_image_with_transform(
            img, x_off, y_off,
            level0_shape[2] / lvl_shape[2], level0_shape[1] / lvl_shape[1],
            auto_levels=True, auto_range=True,
        )

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
            if lvl_shape is None:
                return

            z_size, y_size, x_size = lvl_shape
            scale_z = level0_shape[0] / z_size if z_size > 0 else 1.0
            z_idx_L = int(np.clip(self.current_slice / scale_z, 0, z_size - 1))

            downsample_y = level0_shape[1] / y_size
            downsample_x = level0_shape[2] / x_size
            _, y_off, x_off = self.multires_image.get_world_offset(optimal_level)

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
            self._last_load_params = load_params

            mi = self.multires_image

            def fetch():
                return mi.get_chunk_data_aligned(optimal_level, z_idx_L, (vy0, vy1), (vx0, vx1))

            def apply(img):
                if img is None or img.size == 0:
                    return
                level_changed = optimal_level != self.current_level
                self.current_level = optimal_level
                self._set_image_with_transform(
                    img, vx0 * downsample_x + x_off, vy0 * downsample_y + y_off,
                    downsample_x, downsample_y,
                )
                if level_changed:
                    self.levelChanged.emit(optimal_level, tuple(lvl_shape))

            self._submit_fetch(fetch, apply)

        except Exception as e:
            _log.error(f"[sview] reload error: {e}")
            traceback.print_exc()

    def get_current_image_data(self):
        """Return the pixel data currently on screen (for contrast/stats)."""
        return self._current_display_data
