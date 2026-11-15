"""2D slice panes: XY / XZ / YZ views with lazy, asynchronous pyramid loading.

A `SlicePane` shows one plane through the volume. It loads only the chunks
covering the visible region, at the pyramid level that samples it at about
one voxel per screen pixel, on a background thread (newer requests supersede
older ones). It draws the crosshair of the shared (z, y, x) position, reports
the cursor's world coordinate and pixel value, and turns clicks into
position picks.
"""

# Bind PyOpenGL to the same GL libraries as Qt before pyqtgraph imports it.
from zview.core.sysinfo import apply_gl_compat as _apply_gl_compat
_apply_gl_compat()

import math  # noqa: E402
import traceback  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
import pyqtgraph as pg  # noqa: E402
from PyQt5 import QtCore, QtGui  # noqa: E402

from zview.core import log as _log  # noqa: E402
from zview.viewer.theme import AXIS_PLANE, PLANE_COLORS  # noqa: E402

# dtypes pyqtgraph renders natively (with levels + LUT); anything else is
# converted to float32 once per frame.
_NATIVE_DTYPES = {np.dtype(t) for t in (np.uint8, np.uint16, np.int8, np.int16,
                                        np.float32, np.float64)}

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


class SlicePane(pg.ImageView):
    """One plane (XY, XZ or YZ) with crosshair, click-to-pick and lazy loading."""

    levelChanged = QtCore.pyqtSignal(int, tuple)
    # World coords (h, v) under the cursor and the displayed pixel value
    # there (NaN when outside the image).
    cursorMoved = QtCore.pyqtSignal(float, float, float)
    # Ctrl+wheel over the pane: request a slice step of +/-1.
    sliceStepRequested = QtCore.pyqtSignal(int)
    # Left click picked a voxel: full (z, y, x) in level-0 coords; the pane's
    # fixed axis keeps its current value.
    positionPicked = QtCore.pyqtSignal(int, int, int)

    # Internal: worker thread -> GUI thread hand-off of fetched pixels.
    _fetched = QtCore.pyqtSignal(int, object)

    _RELOAD_DEBOUNCE_MS = 100

    def _init_loading(self):
        """State + timers for lazy, asynchronous region loading."""
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


    def detach(self):
        """Drop the dataset and blank the pane (e.g. on File > Close)."""
        self._reload_timer.stop()
        self._cancel_pending()
        self.multires_image = None
        self._last_load_params = None
        self._initial_load_done = False
        self._current_display_data = None
        self.clear()

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

    def _compute_view_scale(self):
        view_box = self.view.viewRect()
        widget_size = self.view.size()
        if widget_size.width() <= 0 or view_box.width() <= 0 or view_box.height() <= 0:
            return 1.0, view_box
        ppx = widget_size.width() / view_box.width()
        ppy = widget_size.height() / view_box.height()
        return min(ppx, ppy), view_box

    def get_current_image_data(self):
        """Return the pixel data currently on screen (for contrast/stats)."""
        return self._current_display_data

    def __init__(self, axis_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_loading()
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

    def set_dataset(self, multires_image, slice_indices=None):
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
