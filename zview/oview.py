"""Axis-oriented ortho view (XY / XZ / YZ) with crosshair + click-to-sync."""

import traceback

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui

from zview.sview import DynamicImageView


# Per-axis geometry: (vertical world axis index, horizontal world axis index,
# fixed-slice world axis index). Indexing is on (z, y, x).
_AXIS_GEOM = {
    'XY': (1, 2, 0),  # v=y, h=x, fixed=z
    'XZ': (0, 2, 1),  # v=z, h=x, fixed=y
    'YZ': (0, 1, 2),  # v=z, h=y, fixed=x
}

# Downsample-selection axis pair for each pane. XY uses Y/X; XZ/YZ need Z so
# anisotropic volumes don't pick an over-coarse level in Z.
_SEL_AXES = {
    'XY': (-2, -1),
    'XZ': (-3, -1),
    'YZ': (-3, -2),
}


class OrthoImageView(DynamicImageView):
    """A single ortho pane: XY, XZ, or YZ, with crosshair, click-to-move-slice,
    and the same chunk-aligned lazy loading as sview."""

    crosshairMoved = QtCore.pyqtSignal(int, int)

    def __init__(self, axis_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if axis_name not in _AXIS_GEOM:
            raise ValueError(f"axis_name must be XY/XZ/YZ, got {axis_name!r}")
        self.axis_name = axis_name
        self.slice_indices = [0, 0, 0]  # [z, y, x] in level-0 coords

        self.vLine = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('y', width=1))
        self.hLine = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('y', width=1))
        self.addItem(self.vLine, ignoreBounds=True)
        self.addItem(self.hLine, ignoreBounds=True)

        self.scene.sigMouseMoved.connect(self._on_mouse_moved)
        self.scene.sigMouseClicked.connect(self._on_mouse_clicked)

        self.view.setMouseEnabled(x=True, y=True)
        self.view.setMenuEnabled(False)
        self.view.setMouseMode(pg.ViewBox.PanMode)

    # ---------- geometry helpers ----------

    def _axes(self):
        return _AXIS_GEOM[self.axis_name]

    def _fixed_slice_from_indices(self):
        return self.slice_indices[self._axes()[2]]

    def _downsample(self, lvl_shape, level0_shape):
        """(downsample_x_world, downsample_y_world) for this pane and level."""
        v, h, _ = self._axes()
        return level0_shape[h] / lvl_shape[h], level0_shape[v] / lvl_shape[v]

    def _world_offset_xy(self, level):
        """(x_offset_world, y_offset_world) for this pane at `level`."""
        off = self.multires_image.get_world_offset(level)  # (z, y, x)
        v, h, _ = self._axes()
        return off[h], off[v]

    def _world_bounds_xy(self, level0_shape):
        """(x_max_world, y_max_world) = level-0 extent along pane axes."""
        v, h, _ = self._axes()
        return level0_shape[h], level0_shape[v]

    # ---------- public ----------

    def set_multires_image(self, multires_image):
        self.multires_image = multires_image
        self._last_load_params = None

        initial_level = len(self.multires_image.levels) - 1
        self.current_level = initial_level

        level0_shape = self.multires_image.get_level_shape(0)
        if level0_shape and len(level0_shape) >= 3:
            self.slice_indices = [int(level0_shape[i] // 2) for i in range(3)]
            self.current_slice = self._fixed_slice_from_indices()

        # Only load the coarsest frame; let user interaction trigger finer
        # reloads. Immediate full-level reload here is the dominant cause of
        # startup stalls on large volumes.
        self._show_initial_frame(initial_level)
        self._initial_load_done = True

    def set_slice_indices(self, z, y, x):
        if self.slice_indices == [z, y, x]:
            return
        self.slice_indices = [z, y, x]
        self.current_slice = self._fixed_slice_from_indices()
        self._last_load_params = None
        self._reload_visible_region()

    def update_crosshair(self, pos_x, pos_y):
        self.vLine.setPos(pos_x)
        self.hLine.setPos(pos_y)

    # ---------- mouse ----------

    def _on_mouse_moved(self, scene_pos):
        if self.image is None:
            return
        p = self.view.mapSceneToView(scene_pos)
        self.vLine.setPos(p.x())
        self.hLine.setPos(p.y())

    def _on_mouse_clicked(self, event):
        if self.image is None or self.multires_image is None:
            return
        level0_shape = self.multires_image.get_level_shape(0)
        if level0_shape is None or len(level0_shape) < 3:
            return
        p = self.view.mapSceneToView(event.scenePos())
        # Clamp to world (level-0) extent, not current-level pixel extent —
        # at coarse levels that collapses the clamp range.
        x_max, y_max = self._world_bounds_xy(level0_shape)
        x = max(0, min(int(p.x()), x_max - 1))
        y = max(0, min(int(p.y()), y_max - 1))
        self.crosshairMoved.emit(x, y)

    # ---------- rendering ----------

    def _set_image_with_transform(self, img, pos_x, pos_y, ds_x, ds_y, auto):
        tr = QtGui.QTransform()
        tr.translate(float(pos_x), float(pos_y))
        tr.scale(float(ds_x), float(ds_y))
        self.setImage(
            self._prepare_for_display(img),
            autoLevels=auto,
            autoRange=auto,
            transform=tr,
        )

    def _show_initial_frame(self, level):
        lvl_shape = self.multires_image.get_level_shape(level)
        if lvl_shape is None or len(lvl_shape) < 3:
            return
        level0_shape = self.multires_image.get_level_shape(0)

        _, _, fixed = self._axes()
        axis_size_0 = level0_shape[fixed]
        axis_size_L = lvl_shape[fixed]
        slice_scale = axis_size_0 / axis_size_L if axis_size_L > 0 else 1.0
        slice_idx_L = int(np.clip(
            self._fixed_slice_from_indices() / slice_scale,
            0, max(0, axis_size_L - 1),
        ))

        img = self.multires_image.get_slice_2d(level, self.axis_name.lower(), slice_idx_L)
        if img is None:
            return

        ds_x, ds_y = self._downsample(lvl_shape, level0_shape)
        pos_x, pos_y = self._world_offset_xy(level)
        self._set_image_with_transform(img, pos_x, pos_y, ds_x, ds_y, auto=True)

    def _reload_visible_region(self):
        if self.multires_image is None:
            return

        try:
            view_scale, view_box = self._compute_view_scale()
            level0_shape = self.multires_image.get_level_shape(0)

            optimal_level = self.multires_image.get_optimal_level(
                view_scale,
                axes=_SEL_AXES[self.axis_name],
                current_level=self.current_level,
            )
            curr_shape = self.multires_image.get_level_shape(optimal_level)
            if not level0_shape or not curr_shape:
                return

            ds_x, ds_y = self._downsample(curr_shape, level0_shape)
            pos_x_off, pos_y_off = self._world_offset_xy(optimal_level)
            v_ax, h_ax, fixed_ax = self._axes()
            width_L = curr_shape[h_ax]
            height_L = curr_shape[v_ax]

            # Fixed-slice index at this level.
            axis_size_L = curr_shape[fixed_ax]
            axis_size_0 = level0_shape[fixed_ax]
            slice_scale = axis_size_0 / axis_size_L if axis_size_L > 0 else 1.0
            slice_idx_L = int(np.clip(
                self._fixed_slice_from_indices() / slice_scale,
                0, max(0, axis_size_L - 1),
            ))

            # View bounds -> level pixel indices, minus translation.
            vx0 = int(np.floor((view_box.left() - pos_x_off) / ds_x))
            vy0 = int(np.floor((view_box.top() - pos_y_off) / ds_y))
            vx1 = int(np.ceil((view_box.right() - pos_x_off) / ds_x))
            vy1 = int(np.ceil((view_box.bottom() - pos_y_off) / ds_y))
            vx0 = max(0, min(vx0, width_L))
            vy0 = max(0, min(vy0, height_L))
            vx1 = max(0, min(vx1, width_L))
            vy1 = max(0, min(vy1, height_L))
            if vx1 <= vx0 or vy1 <= vy0:
                vx0, vy0, vx1, vy1 = 0, 0, width_L, height_L

            # Safety stride if the fetched region would be enormous.
            max_fetch = 2_000_000
            want = (vy1 - vy0) * (vx1 - vx0)
            step = max(1, int(np.ceil(np.sqrt(want / max_fetch)))) if want > max_fetch else 1

            load_params = (optimal_level, slice_idx_L, vy0, vy1, vx0, vx1, step, self.axis_name)
            if load_params == self._last_load_params:
                return
            level_changed = (optimal_level != self.current_level)
            self._last_load_params = load_params
            self.current_level = optimal_level

            chunk = self.multires_image.get_slice_2d_region(
                optimal_level, self.axis_name.lower(), slice_idx_L,
                (vy0, vy1), (vx0, vx1),
            )
            if chunk is None or chunk.size == 0:
                return
            if step > 1:
                chunk = chunk[::step, ::step]

            x0_world = vx0 * ds_x + pos_x_off
            y0_world = vy0 * ds_y + pos_y_off
            self._set_image_with_transform(
                chunk, x0_world, y0_world, ds_x * step, ds_y * step, auto=False,
            )

            if level_changed:
                print(f"[{self.axis_name}] level -> {optimal_level}, shape {curr_shape}")
                self.levelChanged.emit(optimal_level, tuple(curr_shape))

        except Exception as e:
            print(f"[{self.axis_name}] reload error: {e}")
            traceback.print_exc()
