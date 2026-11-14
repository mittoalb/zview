"""Axis-oriented ortho view (XY / XZ / YZ) with crosshair + click-to-sync."""

import traceback

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore

from zview import _log
from zview.sview import DynamicImageView
from zview.theme import AXIS_PLANE, PLANE_COLORS


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

AXIS_LETTERS = ('z', 'y', 'x')


class OrthoImageView(DynamicImageView):
    """A single ortho pane: XY, XZ, or YZ, with crosshair, click-to-move-slice,
    and the same chunk-aligned lazy loading as sview."""

    # Left click picked a voxel: full (z, y, x) in level-0 coords. The pane's
    # fixed axis keeps its current value.
    positionPicked = QtCore.pyqtSignal(int, int, int)
    # Legacy: (h, v) world ints of the click, in this pane's on-screen axes.
    crosshairMoved = QtCore.pyqtSignal(int, int)

    def __init__(self, axis_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if axis_name not in _AXIS_GEOM:
            raise ValueError(f"axis_name must be XY/XZ/YZ, got {axis_name!r}")
        self.axis_name = axis_name
        self.slice_indices = [0, 0, 0]  # [z, y, x] in level-0 coords
        # Draw voxels with their physical proportions (from OME-NGFF scale).
        self.physical_aspect = True

        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.hLine = pg.InfiniteLine(angle=0, movable=False)
        self.addItem(self.vLine, ignoreBounds=True)
        self.addItem(self.hLine, ignoreBounds=True)
        self._apply_crosshair_pens()

        self.scene.sigMouseClicked.connect(self._on_mouse_clicked)

        self.view.setMouseEnabled(x=True, y=True)
        self.view.setMenuEnabled(False)
        self.view.setMouseMode(pg.ViewBox.PanMode)

    # ---------- geometry helpers ----------

    def _axes(self):
        return _AXIS_GEOM[self.axis_name]

    @property
    def fixed_axis(self):
        """Index into (z, y, x) of the axis this pane slices through."""
        return self._axes()[2]

    @property
    def h_axis(self):
        return self._axes()[1]

    @property
    def v_axis(self):
        return self._axes()[0]

    def _fixed_slice_from_indices(self):
        return self.slice_indices[self.fixed_axis]

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

    def _apply_aspect(self):
        """Lock the view so one voxel has its physical h:v proportions.

        pyqtgraph's ratio is (screen size of an x unit) / (of a y unit), i.e.
        voxel_h / voxel_v. Clamped so broken metadata can't flatten the pane.
        """
        ratio = 1.0
        if self.physical_aspect and self.multires_image is not None:
            vs = self.multires_image.voxel_size()
            sh, sv = vs[self.h_axis], vs[self.v_axis]
            if sh > 0 and sv > 0:
                ratio = min(20.0, max(0.05, sh / sv))
        self.view.setAspectLocked(True, ratio=ratio)

    def set_physical_aspect(self, on):
        self.physical_aspect = bool(on)
        self._apply_aspect()
        if self.image is not None:
            self.view.autoRange(padding=0)

    # ---------- public ----------

    def set_multires_image(self, multires_image, slice_indices=None):
        self._cancel_pending()
        self.multires_image = multires_image
        self._last_load_params = None

        initial_level = len(self.multires_image.levels) - 1
        self.current_level = initial_level

        level0_shape = self.multires_image.get_level_shape(0)
        if slice_indices is not None:
            self.slice_indices = [int(i) for i in slice_indices]
        elif level0_shape:
            self.slice_indices = [int(level0_shape[i] // 2) for i in range(3)]
        self.current_slice = self._fixed_slice_from_indices()

        self._apply_aspect()
        # Show the coarsest plane synchronously (instant feedback), then
        # refine to the screen-matched level in the background. Refining used
        # to be skipped because it blocked the GUI thread at startup; region
        # fetches are asynchronous now.
        self._show_initial_frame(initial_level, auto_levels=True)
        self._initial_load_done = True
        self._sync_crosshair()
        lvl_shape = self.multires_image.get_level_shape(initial_level)
        if lvl_shape is not None:
            self.levelChanged.emit(initial_level, tuple(lvl_shape))
        self.schedule_refine()

    def set_axis(self, axis_name):
        """Switch this pane to another plane, keeping levels/LUT, fitting view."""
        if axis_name not in _AXIS_GEOM:
            raise ValueError(f"axis_name must be XY/XZ/YZ, got {axis_name!r}")
        if axis_name == self.axis_name:
            return
        self.axis_name = axis_name
        self.current_slice = self._fixed_slice_from_indices()
        self._last_load_params = None
        self._cancel_pending()
        self._apply_crosshair_pens()
        if self.multires_image is None:
            return
        self._apply_aspect()
        # Restart from the coarsest level: the old view rect is in the old
        # plane's coordinates and would select a wrong region/level.
        coarsest = len(self.multires_image.levels) - 1
        self._initial_load_done = False
        self.current_level = coarsest
        self._show_initial_frame(coarsest, auto_levels=False)
        self._initial_load_done = True
        self._sync_crosshair()
        lvl_shape = self.multires_image.get_level_shape(coarsest)
        if lvl_shape is not None:
            self.levelChanged.emit(coarsest, tuple(lvl_shape))
        self.schedule_refine()

    def set_slice_indices(self, z, y, x):
        new = [int(z), int(y), int(x)]
        if self.slice_indices == new:
            return
        old_fixed = self._fixed_slice_from_indices()
        self.slice_indices = new
        self.current_slice = self._fixed_slice_from_indices()
        self._sync_crosshair()
        # Moving the crosshair within the plane doesn't change the image.
        if self.current_slice != old_fixed:
            self._last_load_params = None
            self._reload_visible_region()

    def set_crosshair_visible(self, visible):
        self.vLine.setVisible(visible)
        self.hLine.setVisible(visible)

    def _apply_crosshair_pens(self):
        """Colour each line by the plane it marks: in XY, the vertical line
        (x = const) is the YZ plane and the horizontal one the XZ plane."""
        for line, ax in ((self.vLine, self.h_axis), (self.hLine, self.v_axis)):
            r, g, b = (int(c * 255) for c in PLANE_COLORS[AXIS_PLANE[ax]])
            line.setPen(pg.mkPen((r, g, b, 210), width=1))

    def _sync_crosshair(self):
        # +0.5: voxel i spans [i, i+1) in world space; mark its centre.
        self.vLine.setPos(self.slice_indices[self.h_axis] + 0.5)
        self.hLine.setPos(self.slice_indices[self.v_axis] + 0.5)

    def update_crosshair(self, pos_x, pos_y):
        """Legacy API: place the crosshair at explicit pane coords."""
        self.vLine.setPos(pos_x)
        self.hLine.setPos(pos_y)

    # ---------- mouse ----------

    def _on_mouse_clicked(self, event):
        if self.image is None or self.multires_image is None:
            return
        if event.button() != QtCore.Qt.LeftButton or event.double():
            return
        level0_shape = self.multires_image.get_level_shape(0)
        if level0_shape is None:
            return
        p = self.view.mapSceneToView(event.scenePos())
        # Clamp to world (level-0) extent, not current-level pixel extent —
        # at coarse levels that collapses the clamp range.
        x_max, y_max = self._world_bounds_xy(level0_shape)
        h = max(0, min(int(np.floor(p.x())), x_max - 1))
        v = max(0, min(int(np.floor(p.y())), y_max - 1))
        idx = list(self.slice_indices)
        idx[self.h_axis] = h
        idx[self.v_axis] = v
        self.crosshairMoved.emit(h, v)
        self.positionPicked.emit(*idx)

    # ---------- rendering ----------

    def _show_initial_frame(self, level, auto_levels=True):
        lvl_shape = self.multires_image.get_level_shape(level)
        if lvl_shape is None:
            return
        level0_shape = self.multires_image.get_level_shape(0)

        fixed = self.fixed_axis
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
        self._last_load_params = None
        self._set_image_with_transform(
            img, pos_x, pos_y, ds_x, ds_y, auto_levels=auto_levels, auto_range=True,
        )

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

            axis_size_L = curr_shape[fixed_ax]
            axis_size_0 = level0_shape[fixed_ax]
            slice_scale = axis_size_0 / axis_size_L if axis_size_L > 0 else 1.0
            slice_idx_L = int(np.clip(
                self._fixed_slice_from_indices() / slice_scale,
                0, max(0, axis_size_L - 1),
            ))

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
            self._last_load_params = load_params

            mi = self.multires_image
            axis = self.axis_name

            def fetch():
                chunk = mi.get_slice_2d_region(
                    optimal_level, axis.lower(), slice_idx_L, (vy0, vy1), (vx0, vx1),
                )
                if chunk is not None and step > 1:
                    chunk = chunk[::step, ::step]
                return chunk

            def apply(chunk):
                if chunk is None or chunk.size == 0 or axis != self.axis_name:
                    return
                level_changed = optimal_level != self.current_level
                self.current_level = optimal_level
                self._set_image_with_transform(
                    chunk, vx0 * ds_x + pos_x_off, vy0 * ds_y + pos_y_off,
                    ds_x * step, ds_y * step,
                )
                if level_changed:
                    self.levelChanged.emit(optimal_level, tuple(curr_shape))

            self._submit_fetch(fetch, apply)

        except Exception as e:
            _log.error(f"[{self.axis_name}] reload error: {e}")
            traceback.print_exc()
