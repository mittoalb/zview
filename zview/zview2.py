"""Unified multi-resolution Zarr viewer: single/dual/quad/3D layouts + metadata tab."""

# Bind PyOpenGL to the same GL libraries as Qt before pyqtgraph imports it.
from zview.sysinfo import apply_gl_compat as _apply_gl_compat
_apply_gl_compat()

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

from zview import _log, sysinfo
from zview.io import open_dataset as _open_dataset
from zview.meta import MetadataViewer
from zview.msres import MultiResolutionImage, io_workers
from zview.oview import AXIS_LETTERS, OrthoImageView
from zview.theme import AXIS_PLANE, PLANE_COLORS
from zview.view3d import VISPY_AVAILABLE, Volume3DView, VolumeLoader

pg.setConfigOptions(imageAxisOrder='row-major')


PLANES = ('XY', 'XZ', 'YZ')
PLANE_LABELS = {'XY': 'XY · axial', 'XZ': 'XZ · coronal', 'YZ': 'YZ · sagittal'}
LAYOUTS = ('single', 'dual', 'quad', '3d')
LAYOUT_LABELS = {'single': '1 view', 'dual': '2 views', 'quad': '4 views', '3d': '3D'}

CONTRAST_PRESETS = [
    ("Percentile 0.5–99.5 %", (0.5, 99.5)),
    ("Percentile 1–99 %", (1.0, 99.0)),
    ("Percentile 2–98 %", (2.0, 98.0)),
    ("Percentile 5–95 %", (5.0, 95.0)),
    ("Min / Max", (0.0, 100.0)),
]


def _qcolor(rgb, alpha=255):
    return QtGui.QColor(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255), alpha)


def _fmt_shape(shape):
    return " × ".join(str(s) for s in shape)


# ============================================================ pane chrome

class _PaneHeader(QtWidgets.QWidget):
    doubleClicked = QtCore.pyqtSignal()

    def mouseDoubleClickEvent(self, event):
        self.doubleClicked.emit()
        super().mouseDoubleClickEvent(event)


class PaneFrame(QtWidgets.QWidget):
    """A view plus a slim header: colour accent, title or plane picker, a
    live info readout (slice / pyramid level), and an optional maximize
    button. Double-clicking the header toggles maximize."""

    maximizeToggled = QtCore.pyqtSignal()
    planeChanged = QtCore.pyqtSignal(str)

    _BTN_STYLE = """
        QToolButton { background: transparent; color: #ddd; border: 1px solid transparent;
                      border-radius: 3px; padding: 0px 4px; }
        QToolButton:hover { background: #4a4a4a; border-color: #666; }
    """

    def __init__(self, body, title="", plane=None, plane_selectable=False,
                 maximizable=False, parent=None):
        super().__init__(parent)
        self.body = body
        self.plane = plane
        self._maximized = False

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        self.header = _PaneHeader()
        self.header.setObjectName("paneHeader")
        self.header.setFixedHeight(24)
        h = QtWidgets.QHBoxLayout(self.header)
        h.setContentsMargins(8, 0, 4, 0)
        h.setSpacing(6)

        self.plane_combo = None
        self.title_label = None
        if plane_selectable:
            self.plane_combo = QtWidgets.QComboBox()
            for p in PLANES:
                self.plane_combo.addItem(PLANE_LABELS[p], p)
            self.plane_combo.setCurrentIndex(PLANES.index(plane or 'XY'))
            self.plane_combo.currentIndexChanged.connect(self._on_plane_combo)
            self.plane_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
            self.plane_combo.setStyleSheet("QComboBox { padding: 0px 6px; }")
            h.addWidget(self.plane_combo)
        else:
            self.title_label = QtWidgets.QLabel(title or PLANE_LABELS.get(plane, ""))
            self.title_label.setStyleSheet("font-weight: bold; color: #e8e8e8;")
            h.addWidget(self.title_label)

        h.addStretch(1)
        self.info_label = QtWidgets.QLabel("")
        self.info_label.setStyleSheet("color: #9a9a9a; font-family: monospace; font-size: 11px;")
        h.addWidget(self.info_label)

        self.max_btn = None
        if maximizable:
            self.max_btn = QtWidgets.QToolButton()
            self.max_btn.setText("⛶")
            self.max_btn.setToolTip("Maximize pane (double-click header)")
            self.max_btn.setStyleSheet(self._BTN_STYLE)
            self.max_btn.clicked.connect(self.maximizeToggled.emit)
            h.addWidget(self.max_btn)
            self.header.doubleClicked.connect(self.maximizeToggled.emit)

        v.addWidget(self.header)
        v.addWidget(body, 1)
        self._apply_accent()

    def _apply_accent(self):
        rgb = PLANE_COLORS.get(self.plane, (0.55, 0.55, 0.55))
        c = _qcolor(rgb).name()
        self.header.setStyleSheet(
            f"#paneHeader {{ background: #2a2a2a; border-left: 3px solid {c};"
            f" border-bottom: 1px solid #3a3a3a; }}"
        )

    def _on_plane_combo(self, idx):
        self.plane = PLANES[idx]
        self._apply_accent()
        self.planeChanged.emit(self.plane)

    def set_info(self, text):
        self.info_label.setText(text)

    def set_maximized(self, on):
        self._maximized = on
        if self.max_btn is not None:
            self.max_btn.setText("⧉" if on else "⛶")
            self.max_btn.setToolTip("Restore 4-pane view" if on else "Maximize pane (double-click header)")


# ============================================================ main window

class UnifiedZarrViewer(QtWidgets.QMainWindow):
    """Main window: toolbar + layouts (1/2/4/3D) + shared controls dock."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.zarr_store = None
        self.zarr_group = None
        self.multires_image = None
        self.dataset_path = None

        self.position = [0, 0, 0]          # (z, y, x) level-0
        self.layout_mode = 'single'
        self._levels = None                # shared display levels (lo, hi)
        self._cache_mb = None              # None = auto
        self._attached = set()             # views holding the current dataset
        self._active_view = None
        self._quad_maximized = None
        self._syncing_splitters = False
        self._auto_contrast_pending = False   # redo auto contrast once panes refine
        self._setting_levels = False

        self.settings = QtCore.QSettings("zview", "zview")

        self.setWindowTitle("zview")
        self.resize(1700, 1000)
        self.setAcceptDrops(True)

        self.loader = VolumeLoader(self)
        self._build_ui()
        self._restore_settings()

    # ---------------------------------------------------------------- views

    def _make_ortho(self, plane):
        view = OrthoImageView(plane)
        view.ui.roiBtn.hide()
        view.ui.menuBtn.hide()
        view.ui.histogram.hide()   # one shared histogram lives in the dock
        view.positionPicked.connect(self._on_position_picked)
        view.cursorMoved.connect(lambda h, v, val, vw=view: self._on_cursor(vw, h, v, val))
        view.levelChanged.connect(lambda lvl, shape, vw=view: self._update_pane_info(vw))
        view.sliceStepRequested.connect(lambda d, vw=view: self._step_view(vw, d))
        return view

    def _all_views(self):
        return [self.single_view, *self.dual_views, self.xy_view, self.xz_view, self.yz_view]

    def _page_views(self, mode=None):
        mode = mode or self.layout_mode
        if mode == 'single':
            return [self.single_view]
        if mode == 'dual':
            return list(self.dual_views)
        if mode == 'quad':
            return [self.xy_view, self.xz_view, self.yz_view]
        return []

    def _primary_view(self):
        views = self._page_views()
        return views[0] if views else None

    def _frame_of(self, view):
        return self._frames.get(id(view))

    # ---------------------------------------------------------------- UI

    def _build_ui(self):
        self._frames = {}
        self._build_menubar()
        self._build_main_toolbar()
        self.addToolBar(QtCore.Qt.TopToolBarArea, self._build_buttonbar())

        self.main_tabs = QtWidgets.QTabWidget()
        self.main_tabs.setDocumentMode(True)
        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(self._build_single_page())
        self.stack.addWidget(self._build_dual_page())
        self.stack.addWidget(self._build_quad_page())
        self.view_3d_only = Volume3DView(self.loader, compact=False)
        self.stack.addWidget(self.view_3d_only)
        self.main_tabs.addTab(self.stack, "Viewer")
        self.metadata_viewer = MetadataViewer()
        self.main_tabs.addTab(self.metadata_viewer, "Metadata")
        self.setCentralWidget(self.main_tabs)

        self._build_dock()
        self._build_statusbar()

        self.loader.volumeReady.connect(self._on_volume_ready)

        self._stats_timer = QtCore.QTimer(self)
        self._stats_timer.setSingleShot(True)
        self._stats_timer.timeout.connect(self._update_statistics)

        self.cache_stats_timer = QtCore.QTimer(self)
        self.cache_stats_timer.timeout.connect(self._update_cache_stats)
        self.cache_stats_timer.start(2000)

    def _build_single_page(self):
        self.single_view = self._make_ortho('XY')
        self.single_view.set_crosshair_visible(False)
        frame = PaneFrame(self.single_view, plane='XY', plane_selectable=True)
        frame.planeChanged.connect(lambda p: self._on_pane_plane_changed(self.single_view, p))
        self._frames[id(self.single_view)] = frame
        return frame

    def _build_dual_page(self):
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.dual_views = []
        for plane in ('XY', 'XZ'):
            view = self._make_ortho(plane)
            frame = PaneFrame(view, plane=plane, plane_selectable=True)
            frame.planeChanged.connect(lambda p, vw=view: self._on_pane_plane_changed(vw, p))
            self._frames[id(view)] = frame
            self.dual_views.append(view)
            splitter.addWidget(frame)
        splitter.setSizes([1, 1])
        # Legacy attribute names.
        self.dual_view1, self.dual_view2 = self.dual_views
        return splitter

    def _build_quad_page(self):
        """2×2 grid arranged so neighbours share an axis:

            XY  | 3D          XY and XZ share the x column,
            ----+----         XZ and YZ share the z row
            XZ  | YZ          (z runs down in both).
        """
        self.xy_view = self._make_ortho('XY')
        self.xz_view = self._make_ortho('XZ')
        self.yz_view = self._make_ortho('YZ')
        self.view_3d = Volume3DView(self.loader, compact=True)

        def frame(key, body, plane=None, title=""):
            f = PaneFrame(body, title=title, plane=plane, maximizable=True)
            f.maximizeToggled.connect(lambda k=key: self._toggle_pane_max(k))
            return f

        self.xy_container = frame('xy', self.xy_view, 'XY')
        self.xz_container = frame('xz', self.xz_view, 'XZ')
        self.yz_container = frame('yz', self.yz_view, 'YZ')
        self.view_3d_container = frame('3d', self.view_3d, title="3D volume")
        for view, f in ((self.xy_view, self.xy_container), (self.xz_view, self.xz_container),
                        (self.yz_view, self.yz_container)):
            self._frames[id(view)] = f

        self._top_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self._top_splitter.addWidget(self.xy_container)
        self._top_splitter.addWidget(self.view_3d_container)
        self._bottom_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self._bottom_splitter.addWidget(self.xz_container)
        self._bottom_splitter.addWidget(self.yz_container)
        self._main_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self._main_splitter.addWidget(self._top_splitter)
        self._main_splitter.addWidget(self._bottom_splitter)
        for s in (self._top_splitter, self._bottom_splitter, self._main_splitter):
            s.setChildrenCollapsible(False)
            s.setSizes([1, 1])
            s.setHandleWidth(4)

        # Keep the columns aligned so the shared x axis lines up.
        self._top_splitter.splitterMoved.connect(
            lambda *_: self._sync_splitters(self._top_splitter, self._bottom_splitter))
        self._bottom_splitter.splitterMoved.connect(
            lambda *_: self._sync_splitters(self._bottom_splitter, self._top_splitter))

        self._quad_panes = {
            'xy': self.xy_container, '3d': self.view_3d_container,
            'xz': self.xz_container, 'yz': self.yz_container,
        }
        self._pane_row = {'xy': self._top_splitter, '3d': self._top_splitter,
                          'xz': self._bottom_splitter, 'yz': self._bottom_splitter}
        return self._main_splitter

    def _sync_splitters(self, src, dst):
        if self._syncing_splitters:
            return
        self._syncing_splitters = True
        dst.setSizes(src.sizes())
        self._syncing_splitters = False

    # ---------------------------------------------------------------- dock

    def _build_dock(self):
        dock = QtWidgets.QDockWidget("Controls", self)
        dock.setObjectName("controlsDock")
        dock.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable
                         | QtWidgets.QDockWidget.DockWidgetClosable
                         | QtWidgets.QDockWidget.DockWidgetFloatable)
        dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)

        panel = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(panel)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(8)

        # --- Dataset
        ds_group = QtWidgets.QGroupBox("Dataset")
        form = QtWidgets.QFormLayout(ds_group)
        form.setLabelAlignment(QtCore.Qt.AlignRight)
        self.ds_labels = {}
        for key, label in (("name", "Name"), ("shape", "Shape (z,y,x)"), ("dtype", "Type"),
                           ("chunks", "Chunks"), ("levels", "Levels"), ("voxel", "Voxel size")):
            w = QtWidgets.QLabel("—")
            w.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            w.setWordWrap(True)
            self.ds_labels[key] = w
            form.addRow(label + ":", w)
        lay.addWidget(ds_group)

        # --- Position
        pos_group = QtWidgets.QGroupBox("Position")
        grid = QtWidgets.QGridLayout(pos_group)
        grid.setHorizontalSpacing(6)
        self.pos_sliders, self.pos_spins = [], []
        axis_plane = AXIS_PLANE   # plane whose fixed axis this is
        for i, letter in enumerate(AXIS_LETTERS):
            lab = QtWidgets.QLabel(letter.upper())
            lab.setStyleSheet(f"font-weight:bold; color:{_qcolor(PLANE_COLORS[axis_plane[i]]).name()};")
            lab.setToolTip(f"{letter} — the slice shown in the {axis_plane[i]} plane")
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            spin = QtWidgets.QSpinBox()
            spin.setMinimumWidth(70)
            spin.setKeyboardTracking(False)
            slider.valueChanged.connect(lambda val, ax=i: self._on_axis_control(ax, val))
            spin.valueChanged.connect(lambda val, ax=i: self._on_axis_control(ax, val))
            grid.addWidget(lab, i, 0)
            grid.addWidget(slider, i, 1)
            grid.addWidget(spin, i, 2)
            self.pos_sliders.append(slider)
            self.pos_spins.append(spin)
        center_btn = QtWidgets.QPushButton("Center")
        center_btn.setToolTip("Move the crosshair to the volume centre")
        center_btn.clicked.connect(self._center_position)
        grid.addWidget(center_btn, 3, 1, 1, 2)
        lay.addWidget(pos_group)
        self._set_position_controls_enabled(False)

        # --- Display (shared histogram / LUT)
        disp_group = QtWidgets.QGroupBox("Display")
        dl = QtWidgets.QVBoxLayout(disp_group)
        self.hist = pg.HistogramLUTWidget(orientation='horizontal')
        self.hist.setMinimumHeight(110)
        self.hist.setMaximumHeight(150)
        self.hist.setToolTip("Drag the handles to set contrast; right-click the "
                             "colour bar for colormaps. Applies to every pane.")
        self.hist.item.sigLevelsChanged.connect(self._on_hist_levels_changed)
        self.hist.item.sigLookupTableChanged.connect(self._on_hist_lut_changed)
        dl.addWidget(self.hist)
        row = QtWidgets.QHBoxLayout()
        self.contrast_combo = QtWidgets.QComboBox()
        for label, _ in CONTRAST_PRESETS:
            self.contrast_combo.addItem(label)
        self.contrast_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.contrast_combo.setMinimumContentsLength(12)
        row.addWidget(self.contrast_combo, 1)
        auto_btn = QtWidgets.QPushButton("Auto")
        auto_btn.setToolTip("Set contrast from what's currently visible in all panes (A)")
        auto_btn.clicked.connect(self.auto_contrast)
        row.addWidget(auto_btn)
        dl.addLayout(row)
        self.stats_label = QtWidgets.QLabel("")
        self.stats_label.setStyleSheet("color:#9a9a9a; font-family: monospace; font-size: 11px;")
        self.stats_label.setWordWrap(True)
        dl.addWidget(self.stats_label)
        lay.addWidget(disp_group)

        # --- Pyramid
        pyr_group = QtWidgets.QGroupBox("Pyramid")
        pl = QtWidgets.QVBoxLayout(pyr_group)
        self.pyramid_label = QtWidgets.QLabel("—")
        self.pyramid_label.setStyleSheet("font-family: monospace; font-size: 11px;")
        self.pyramid_label.setTextFormat(QtCore.Qt.RichText)
        pl.addWidget(self.pyramid_label)
        lay.addWidget(pyr_group)

        # --- System
        sys_group = QtWidgets.QGroupBox("System")
        sl = QtWidgets.QVBoxLayout(sys_group)
        self.system_label = QtWidgets.QLabel("")
        self.system_label.setWordWrap(True)
        self.system_label.setStyleSheet("color:#9a9a9a; font-size: 11px;")
        sl.addWidget(self.system_label)
        lay.addWidget(sys_group)
        self.loader.capsChanged.connect(self._update_system_label)
        self._update_system_label()

        lay.addStretch(1)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(330)
        dock.setWidget(scroll)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
        self.controls_dock = dock
        self._act_toggle_dock = dock.toggleViewAction()
        self._act_toggle_dock.setText("Controls panel")
        self._act_toggle_dock.setShortcut("Ctrl+D")
        self._view_menu.insertAction(self._view_menu_dock_anchor, self._act_toggle_dock)

    def _build_statusbar(self):
        sb = self.statusBar()
        self.status_label = QtWidgets.QLabel("Open a Zarr/N5 dataset (Ctrl+O) or drop a folder here")
        self.status_label.setStyleSheet("font-family: monospace; font-size: 11px; padding: 0 6px;")
        sb.addWidget(self.status_label, 1)
        self.cache_stats_label = QtWidgets.QLabel("")
        self.cache_stats_label.setStyleSheet("font-size: 11px; color: #8c8; padding: 0 6px;")
        sb.addPermanentWidget(self.cache_stats_label)

    # ---------------------------------------------------------------- menus / toolbars

    def _build_menubar(self):
        mb = self.menuBar()

        file_menu = mb.addMenu("&File")
        act_open = file_menu.addAction(
            self.style().standardIcon(QtWidgets.QStyle.SP_DirOpenIcon), "&Open…", self._load_file)
        act_open.setShortcut("Ctrl+O")
        self._act_open = act_open
        self._recent_menu = file_menu.addMenu("Open &Recent")
        self._rebuild_recent_menu()
        file_menu.addSeparator()
        self._act_render = file_menu.addAction("Render 3D…", self._open_render_dialog)
        self._act_render.setShortcut("Ctrl+Shift+R")
        self._act_render.setToolTip("Presentation-quality stills, turntable, tour video (VTK)")
        file_menu.addSeparator()
        act_close = file_menu.addAction("&Close Dataset", self._close_store)
        act_close.setShortcut("Ctrl+W")
        file_menu.addSeparator()
        act_exit = file_menu.addAction("E&xit", self.close)
        act_exit.setShortcut("Ctrl+Q")

        view_menu = mb.addMenu("&View")
        self._view_menu = view_menu
        self._layout_group = QtWidgets.QActionGroup(self)
        self._layout_actions = {}
        for i, mode in enumerate(LAYOUTS):
            act = QtWidgets.QAction(LAYOUT_LABELS[mode], self._layout_group, checkable=True)
            act.setShortcut(f"Ctrl+{i + 1}")
            act.setData(mode)
            view_menu.addAction(act)
            self._layout_actions[mode] = act
        self._layout_actions['single'].setChecked(True)
        self._layout_group.triggered.connect(lambda a: self.set_layout(a.data()))
        # Legacy attribute names.
        self._act_view_1 = self._layout_actions['single']
        self._act_view_2 = self._layout_actions['dual']
        self._act_view_4 = self._layout_actions['quad']
        self._act_view_3d = self._layout_actions['3d']

        view_menu.addSeparator()
        act = view_menu.addAction("Reset view", self._reset_active_view)
        act.setShortcut("R")
        act = view_menu.addAction("Reset all views", self._reset_all_views)
        act.setShortcut("Shift+R")
        act = view_menu.addAction("Auto contrast", self.auto_contrast)
        act.setShortcut("A")
        self._act_aspect = view_menu.addAction("Physical voxel aspect")
        self._act_aspect.setCheckable(True)
        self._act_aspect.setChecked(True)
        self._act_aspect.setToolTip("Draw anisotropic voxels with their real proportions")
        self._act_aspect.toggled.connect(
            lambda on: [v.set_physical_aspect(on) for v in self._all_views()])
        view_menu.addSeparator()
        for text, key, fn in (("Next slice", "PgDown", lambda: self._step_active(+1)),
                              ("Previous slice", "PgUp", lambda: self._step_active(-1)),
                              ("Next slice ×10", "Shift+PgDown", lambda: self._step_active(+10)),
                              ("Previous slice ×10", "Shift+PgUp", lambda: self._step_active(-10)),
                              ("First slice", "Home", lambda: self._jump_active(0)),
                              ("Last slice", "End", lambda: self._jump_active(None))):
            a = view_menu.addAction(text, fn)
            a.setShortcut(key)
        view_menu.addSeparator()
        self._view_menu_dock_anchor = view_menu.addSeparator()
        self._fullscreen_action = view_menu.addAction("Full screen", self._toggle_fullscreen)
        self._fullscreen_action.setShortcut("F11")
        self._fullscreen_action.setShortcutContext(QtCore.Qt.ApplicationShortcut)

        view_menu.addSeparator()
        cache_menu = view_menu.addMenu("Chunk cache size")
        cache_group = QtWidgets.QActionGroup(self)
        auto_mb = sysinfo.default_cache_mb()
        total, _ = sysinfo.system_memory()
        options = [(None, f"Auto ({auto_mb} MB)")]
        for mb_ in (256, 1024, 2048, 4096, 8192, 16384):
            if total is None or mb_ * 2**20 <= total * 0.5:
                options.append((mb_, f"{mb_ // 1024} GB" if mb_ >= 1024 else f"{mb_} MB"))
        for size_mb, label in options:
            a = QtWidgets.QAction(label, cache_group, checkable=True)
            a.setChecked(size_mb is None)
            a.triggered.connect(lambda _c=False, s=size_mb: self._set_cache_size(s))
            cache_menu.addAction(a)

        plugins_menu = mb.addMenu("&Plugins")
        self._plugins_menu = plugins_menu
        placeholder = plugins_menu.addAction("No plugins loaded")
        placeholder.setEnabled(False)

        help_menu = mb.addMenu("&Help")
        help_menu.addAction("Keyboard && mouse…", self._show_shortcuts)
        help_menu.addAction("System info…", self._show_system_info)
        help_menu.addAction("Performance tips…", self._show_performance_tips)
        help_menu.addSeparator()
        help_menu.addAction("About…", self._show_about)

    def _build_main_toolbar(self):
        tb = QtWidgets.QToolBar("Main", self)
        tb.setObjectName("mainToolbar")
        tb.setMovable(False)
        tb.setIconSize(QtCore.QSize(18, 18))
        tb.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        tb.addAction(self._act_open)
        tb.addSeparator()

        seg = QtWidgets.QWidget()
        sl = QtWidgets.QHBoxLayout(seg)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(0)
        self._layout_buttons = {}
        for i, mode in enumerate(LAYOUTS):
            b = QtWidgets.QToolButton()
            b.setDefaultAction(self._layout_actions[mode])
            b.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
            b.setObjectName("segFirst" if i == 0 else ("segLast" if i == len(LAYOUTS) - 1 else "segMid"))
            sl.addWidget(b)
            self._layout_buttons[mode] = b
        seg.setStyleSheet("""
            QToolButton { background:#3c3c3c; color:#ddd; border:1px solid #5a5a5a;
                          padding:3px 10px; margin:0px; }
            QToolButton:checked { background:#2a82da; color:white; border-color:#2a82da; }
            QToolButton:hover:!checked { background:#4a4a4a; }
            #segFirst { border-top-left-radius:4px; border-bottom-left-radius:4px; }
            #segLast  { border-top-right-radius:4px; border-bottom-right-radius:4px; }
        """)
        tb.addWidget(seg)
        tb.addSeparator()
        reset = tb.addAction("Reset view", self._reset_all_views)
        reset.setToolTip("Fit every pane to its image (Shift+R; R resets the pane under the mouse)")
        contrast = tb.addAction("Auto contrast", self.auto_contrast)
        contrast.setToolTip("Contrast from what's visible (A)")
        tb.addSeparator()
        tb.addAction(self._act_render)

        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        tb.addWidget(spacer)
        self._fullscreen_btn = QtWidgets.QToolButton()
        self._fullscreen_btn.setDefaultAction(self._fullscreen_action)
        self._fullscreen_btn.setText("⛶ Full screen")
        tb.addWidget(self._fullscreen_btn)
        self.addToolBar(QtCore.Qt.TopToolBarArea, tb)
        self.main_toolbar = tb

    def _build_buttonbar(self) -> QtWidgets.QToolBar:
        """Second toolbar for plugin action buttons"""
        tb = QtWidgets.QToolBar("Plugin Actions", self)
        tb.setObjectName("pluginToolbar")
        tb.setIconSize(QtCore.QSize(18, 18))
        tb.setMovable(False)
        tb.setFloatable(False)
        tb.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self._plugin_toolbar = tb

        # Example plugin buttons (disabled by default, will be enabled by plugins)
        for icon, text, tip in (
            (QtWidgets.QStyle.SP_DialogYesButton, "Measure", "Measure distances and areas (plugin not loaded)"),
            (QtWidgets.QStyle.SP_FileDialogDetailedView, "Segment", "Segmentation tools (plugin not loaded)"),
            (QtWidgets.QStyle.SP_FileDialogListView, "Annotate", "Add annotations (plugin not loaded)"),
        ):
            act = QtWidgets.QAction(self.style().standardIcon(icon), text, self)
            act.setEnabled(False)
            act.setToolTip(tip)
            tb.addAction(act)

        tb.addSeparator()
        act_export = QtWidgets.QAction(
            self.style().standardIcon(QtWidgets.QStyle.SP_DialogSaveButton), "Export", self)
        act_export.setEnabled(False)
        act_export.setToolTip("Export current view (plugin not loaded)")
        tb.addAction(act_export)

        # Flexible spacer marks where plugin buttons are inserted.
        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        tb.addWidget(spacer)
        return tb

    def add_plugin_button(self, text: str, slot=None, icon: QtGui.QIcon = None,
                          tooltip: str = "", checkable: bool = False) -> QtWidgets.QAction:
        """
        Public API for plugins to add buttons to the plugin toolbar.

        Example usage from a plugin:
            action = viewer.add_plugin_button(
                text="My Tool",
                slot=self.run_tool,
                icon=QtGui.QIcon("path/to/icon.png"),
                tooltip="Run my custom tool",
                checkable=False
            )

        Returns:
            QAction: The created action (can be stored for later removal)
        """
        if not hasattr(self, "_plugin_toolbar"):
            return None
        if icon is None:
            icon = self.style().standardIcon(QtWidgets.QStyle.SP_CommandLink)
        action = QtWidgets.QAction(icon, text, self)
        action.setCheckable(checkable)
        action.setProperty("zview_plugin", True)
        if tooltip:
            action.setToolTip(tooltip)
        if slot:
            action.triggered.connect(slot)

        # Insert before the spacer (keeps plugin buttons left of it).
        for act in self._plugin_toolbar.actions():
            widget = self._plugin_toolbar.widgetForAction(act)
            if widget is not None and widget.sizePolicy().horizontalPolicy() == QtWidgets.QSizePolicy.Expanding:
                self._plugin_toolbar.insertAction(act, action)
                return action
        self._plugin_toolbar.addAction(action)
        return action

    def remove_plugin_button(self, action: QtWidgets.QAction):
        """Remove a plugin button from the toolbar"""
        if hasattr(self, "_plugin_toolbar") and action:
            self._plugin_toolbar.removeAction(action)

    def clear_plugin_buttons(self):
        """Remove all plugin-added actions (keeps the default placeholders)."""
        if not hasattr(self, "_plugin_toolbar"):
            return
        for action in self._plugin_toolbar.actions():
            if action.property("zview_plugin"):
                self._plugin_toolbar.removeAction(action)

    # ---------------------------------------------------------------- layout

    def set_layout(self, mode):
        if mode not in LAYOUTS:
            return
        self.layout_mode = mode
        self._layout_actions[mode].setChecked(True)
        self.stack.setCurrentIndex(LAYOUTS.index(mode))
        self.main_tabs.setCurrentIndex(0)
        if self.multires_image is not None:
            self._attach_page_views()
            self._link_histogram()
            if self._levels is None:
                self.auto_contrast()
        self._update_pyramid_label()

    # Legacy entry point (index-based).
    def _on_layout_changed(self, index):
        self.set_layout(LAYOUTS[index])

    def _toggle_pane_max(self, key):
        if self._quad_maximized == key:
            self._restore_panes()
        else:
            self._maximize_pane(key)

    def _maximize_pane(self, key):
        if key not in self._quad_panes:
            return
        row = self._pane_row[key]
        for k, pane in self._quad_panes.items():
            pane.setVisible(k == key)
            pane.set_maximized(k == key)
        self._top_splitter.setVisible(row is self._top_splitter)
        self._bottom_splitter.setVisible(row is self._bottom_splitter)
        self._quad_maximized = key

    def _restore_panes(self):
        if self._quad_maximized is None:
            return
        for pane in self._quad_panes.values():
            pane.show()
            pane.set_maximized(False)
        self._top_splitter.show()
        self._bottom_splitter.show()
        for s in (self._top_splitter, self._bottom_splitter, self._main_splitter):
            s.setSizes([1, 1])
        self._quad_maximized = None

    # ---------------------------------------------------------------- dataset

    def _load_file(self):
        start = self.settings.value("last_dir", str(Path.home()))
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Open Zarr / N5 dataset (select the store folder)", start)
        if path:
            self.open_path(path)

    def open_path(self, path):
        """Open a dataset at `path` (directory store)."""
        path = str(path)
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            self._close_store(quiet=True)
            store, backend = _open_dataset(path)
            mi = MultiResolutionImage(store, cache_size_mb=self._cache_mb)
        except Exception as e:
            QtWidgets.QApplication.restoreOverrideCursor()
            import traceback  # noqa: PLC0415
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Could not open dataset", f"{path}\n\n{e}")
            return
        _log.success(f"[zview] opened {Path(path).name} via {backend}")

        self.zarr_store = self.zarr_group = store
        self.multires_image = mi
        self.dataset_path = path
        self.settings.setValue("last_dir", str(Path(path).parent))
        self._add_recent(path)

        shape = mi.get_level_shape(0)
        self.position = [s // 2 for s in shape]
        self._set_position_controls_range(shape)
        self._set_position_controls_enabled(True)
        self._fill_dataset_info(Path(path).name)

        self._attached.clear()
        self._levels = None
        self._attach_page_views()
        self._link_histogram()
        self.auto_contrast()
        self._auto_contrast_pending = True

        self.loader.set_dataset(mi)
        for v3 in (self.view_3d, self.view_3d_only):
            v3.set_dataset(mi)
            v3.set_position(*self.position)
        self.view_3d_container.set_info("")

        self.metadata_viewer.load_metadata(store)
        self.setWindowTitle(f"zview — {Path(path).name}")
        self._update_system_label()
        self._update_pyramid_label()
        QtWidgets.QApplication.restoreOverrideCursor()
        self.status_label.setText(
            f"Opened {Path(path).name}: {_fmt_shape(shape)} {mi.levels[0].dtype}, "
            f"{mi.get_num_levels()} levels ({backend})")

    def _fill_dataset_info(self, name):
        mi = self.multires_image
        lvl0 = mi.levels[0]
        self.ds_labels["name"].setText(name)
        self.ds_labels["name"].setToolTip(self.dataset_path or "")
        raw = "" if lvl0.raw_shape == lvl0.shape else f"  (raw {_fmt_shape(lvl0.raw_shape)})"
        self.ds_labels["shape"].setText(_fmt_shape(lvl0.shape) + raw)
        self.ds_labels["dtype"].setText(str(lvl0.dtype))
        self.ds_labels["chunks"].setText(_fmt_shape(lvl0.chunks) if lvl0.chunks else "—")
        self.ds_labels["levels"].setText(str(mi.get_num_levels()))
        vs = mi.voxel_size()
        units = mi.axis_units or [None] * 3
        unit = units[-1] or ""
        self.ds_labels["voxel"].setText(" × ".join(f"{v:g}" for v in vs) + (f" {unit}" if unit else ""))

    def _attach_page_views(self):
        """Give the current page's views the dataset (lazily, once per dataset)
        and bring them to the shared position / contrast."""
        mi = self.multires_image
        if mi is None:
            return
        for view in self._page_views():
            if id(view) not in self._attached:
                view.set_multires_image(mi, slice_indices=self.position)
                self._attached.add(id(view))
            else:
                view.set_slice_indices(*self.position)
            self._apply_display_to(view)
            self._update_pane_info(view)

    def _close_store(self, quiet=False):
        """Close the current dataset and blank every view."""
        self.loader.set_dataset(None)
        for v3 in (self.view_3d, self.view_3d_only):
            v3.set_dataset(None)
        for view in self._all_views():
            view.detach()
            frame = self._frame_of(view)
            if frame is not None:
                frame.set_info("")
        self._attached.clear()
        if self.zarr_store is not None:
            close = getattr(self.zarr_store, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        self.zarr_store = self.zarr_group = None
        if self.multires_image is not None:
            self.multires_image.chunk_cache.clear()
        self.multires_image = None
        self.dataset_path = None
        self._levels = None
        self._auto_contrast_pending = False
        self.metadata_viewer.clear()
        for w in self.ds_labels.values():
            w.setText("—")
        self._set_position_controls_enabled(False)
        self.stats_label.setText("")
        self.pyramid_label.setText("—")
        self.cache_stats_label.setText("")
        self.setWindowTitle("zview")
        if not quiet:
            self.status_label.setText("No dataset")

    # ---------------------------------------------------------------- position

    def _set_position_controls_range(self, shape):
        for ax in range(3):
            for w in (self.pos_sliders[ax], self.pos_spins[ax]):
                w.blockSignals(True)
                w.setRange(0, max(0, shape[ax] - 1))
                w.setValue(self.position[ax])
                w.blockSignals(False)
            self.pos_sliders[ax].setPageStep(max(1, shape[ax] // 20))

    def _set_position_controls_enabled(self, on):
        for w in (*self.pos_sliders, *self.pos_spins):
            w.setEnabled(on)

    def _on_axis_control(self, ax, value):
        pos = list(self.position)
        pos[ax] = value
        self.set_position(*pos)

    def _on_position_picked(self, z, y, x):
        self.set_position(z, y, x)

    def _center_position(self):
        if self.multires_image is not None:
            self.set_position(*(s // 2 for s in self.multires_image.get_level_shape(0)))

    def set_position(self, z, y, x):
        """Move the shared crosshair (level-0 voxel indices) everywhere."""
        if self.multires_image is None:
            return
        shape = self.multires_image.get_level_shape(0)
        new = [int(min(max(v, 0), s - 1)) for v, s in zip((z, y, x), shape)]
        if new == self.position:
            return
        self.position = new
        for ax in range(3):
            for w in (self.pos_sliders[ax], self.pos_spins[ax]):
                if w.value() != new[ax]:
                    w.blockSignals(True)
                    w.setValue(new[ax])
                    w.blockSignals(False)
        for view in self._page_views():
            if id(view) in self._attached:
                view.set_slice_indices(*new)
                self._update_pane_info(view)
        for v3 in (self.view_3d, self.view_3d_only):
            v3.set_position(*new)
        # Legacy attributes.
        self.z_pos, self.y_pos, self.x_pos = new

    def _step_view(self, view, delta):
        if self.multires_image is None:
            return
        pos = list(self.position)
        pos[view.fixed_axis] += delta
        self.set_position(*pos)

    def _target_view(self):
        """Pane that keyboard actions act on: last hovered on this page."""
        views = self._page_views()
        if self._active_view in views:
            return self._active_view
        return views[0] if views else None

    def _step_active(self, delta):
        view = self._target_view()
        if view is not None:
            self._step_view(view, delta)

    def _jump_active(self, value):
        view = self._target_view()
        if view is None or self.multires_image is None:
            return
        shape = self.multires_image.get_level_shape(0)
        pos = list(self.position)
        ax = view.fixed_axis
        pos[ax] = shape[ax] - 1 if value is None else value
        self.set_position(*pos)

    # Legacy names used by older shortcuts.
    def _step_z(self, delta):
        self.set_position(self.position[0] + delta, self.position[1], self.position[2])

    def _jump_z(self, value):
        self.set_position(value, self.position[1], self.position[2])

    # ---------------------------------------------------------------- pane info / status

    def _update_pane_info(self, view):
        frame = self._frame_of(view)
        mi = self.multires_image
        if frame is None or mi is None:
            return
        ax = view.fixed_axis
        n = mi.get_level_shape(0)[ax]
        lvl = view.current_level
        ds = mi.level_downsample(lvl)
        ds_v = (ds[view.h_axis] + ds[view.v_axis]) / 2
        frame.set_info(f"{AXIS_LETTERS[ax]} {self.position[ax]}/{n - 1} · L{lvl} ({ds_v:.3g}×)")
        self._stats_timer.start(150)

    def _update_pyramid_label(self):
        mi = self.multires_image
        if mi is None:
            self.pyramid_label.setText("—")
            return
        in_use = {}
        for view in self._page_views():
            if id(view) in self._attached:
                in_use.setdefault(view.current_level, []).append(view.axis_name)
        res = self.loader.result
        if res is not None and self.layout_mode in ('quad', '3d'):
            in_use.setdefault(res.level, []).append("3D")
        rows = []
        for i in range(mi.get_num_levels()):
            shape = mi.get_level_shape(i)
            ds = mi.level_downsample(i)
            tag = ", ".join(in_use.get(i, []))
            style = "color:#e8e8e8; font-weight:bold;" if tag else "color:#888;"
            rows.append(
                f"<tr style='{style}'><td>L{i}&nbsp;</td><td>{_fmt_shape(shape)}&nbsp;</td>"
                f"<td>{ds[2]:.3g}×&nbsp;</td><td style='color:#2a82da'>{tag}</td></tr>")
        self.pyramid_label.setText("<table cellspacing=0>" + "".join(rows) + "</table>")

    def _on_cursor(self, view, h, v, value):
        self._active_view = view
        mi = self.multires_image
        if mi is None:
            return
        shape = mi.get_level_shape(0)
        idx = list(self.position)
        idx[view.h_axis] = int(math.floor(h))
        idx[view.v_axis] = int(math.floor(v))
        z, y, x = idx
        inside = all(0 <= c < s for c, s in zip(idx, shape))
        if not inside:
            self.status_label.setText(f"[{view.axis_name}] outside volume")
            return
        val = "" if math.isnan(value) else (f"{value:.6g}")
        self.status_label.setText(
            f"[{view.axis_name}] z={z:<6d} y={y:<6d} x={x:<6d} value={val:<10s} "
            f"L{view.current_level}   volume {_fmt_shape(shape)}")

    # Legacy signature (plane, h, v) used by older code paths.
    def _update_status(self, plane, h, v):
        view = {'XY': self.xy_view, 'XZ': self.xz_view, 'YZ': self.yz_view}.get(plane)
        if view is not None:
            self._on_cursor(view, h, v, math.nan)

    def _on_volume_ready(self, res):
        self.view_3d_container.set_info(
            f"L{res.level} · {_fmt_shape(res.data.shape)}" + ("" if res.final else " (preview)"))
        self._update_pyramid_label()

    # ---------------------------------------------------------------- contrast

    def _link_histogram(self):
        """Point the shared histogram at the current page's primary pane.

        HistogramLUTItem.setImageItem() auto-levels to that image, which would
        clobber the shared contrast; suppress propagation while linking and
        restore the saved levels afterwards.
        """
        view = self._primary_view()
        if view is None or view.imageItem is None:
            return
        saved = self._levels
        img = view.imageItem
        prev = getattr(self, "_hist_image", None)
        if prev is not img:
            # setImageItem() connects but never disconnects; drop the old
            # item's connections so recomputes don't pile up per relink.
            if prev is not None:
                pairs = [("sigImageChanged", "imageChanged"),
                         ("sigLevelsChanged", "imageLevelsChanged")]  # latter: newer pyqtgraph
                for sig_name, slot_name in pairs:
                    sig = getattr(prev, sig_name, None)
                    slot = getattr(self.hist.item, slot_name, None)
                    if sig is None or slot is None:
                        continue
                    try:
                        sig.disconnect(slot)
                    except (TypeError, RuntimeError):
                        pass
            self._linking_hist = True
            try:
                self.hist.item.setImageItem(img)
            finally:
                self._linking_hist = False
            self._hist_image = img
        if saved is not None:
            self._levels = saved
            self._setting_levels = True
            try:
                self.hist.item.setLevels(*saved)
            finally:
                self._setting_levels = False
            for v in self._page_views():
                if id(v) in self._attached:
                    v.imageItem.setLevels(saved)

    def _apply_display_to(self, view):
        if self._levels is not None:
            view.imageItem.setLevels(self._levels)
        view.imageItem.setLookupTable(self.hist.item.getLookupTable)

    def _on_hist_levels_changed(self, *_):
        if getattr(self, "_linking_hist", False):
            return
        if not self._setting_levels:
            self._auto_contrast_pending = False   # user took over
        lo, hi = self.hist.item.getLevels()
        if not (math.isfinite(lo) and math.isfinite(hi)):
            return
        self._levels = (float(lo), float(hi))
        for view in self._all_views():
            if id(view) in self._attached and view.image is not None:
                view.imageItem.setLevels(self._levels)

    def _on_hist_lut_changed(self, *_):
        for view in self._all_views():
            if id(view) in self._attached:
                view.imageItem.setLookupTable(self.hist.item.getLookupTable)

    def auto_contrast(self):
        """Set shared contrast from the pixels currently visible in all panes."""
        samples = []
        for view in self._page_views():
            data = view.get_current_image_data()
            if data is None or data.size == 0:
                continue
            flat = data.reshape(-1)
            samples.append(flat[::max(1, flat.size // 500_000)])
        if not samples:
            return
        s = np.concatenate([np.asarray(a, dtype=np.float64) for a in samples])
        s = s[np.isfinite(s)]
        if s.size == 0:
            return
        p_lo, p_hi = CONTRAST_PRESETS[self.contrast_combo.currentIndex()][1]
        lo, hi = (float(v) for v in np.percentile(s, [p_lo, p_hi]))
        if hi <= lo:
            lo, hi = float(s.min()), float(s.max())
            if hi <= lo:
                hi = lo + 1.0
        self._levels = (lo, hi)
        self._setting_levels = True
        try:
            self.hist.item.setLevels(lo, hi)
        finally:
            self._setting_levels = False
        span = hi - lo
        self.hist.item.setHistogramRange(lo - 0.25 * span, hi + 0.25 * span)
        for view in self._all_views():
            if id(view) in self._attached:
                self._apply_display_to(view)

    def _maybe_refine_contrast(self):
        """The first auto contrast runs on the coarsest frames; once every
        pane has refined, redo it on the real pixels (unless the user has
        touched the contrast in the meantime)."""
        if not self._auto_contrast_pending or self.multires_image is None:
            return
        views = [v for v in self._page_views() if id(v) in self._attached]
        coarsest = self.multires_image.get_num_levels() - 1
        if views and all(v.has_refined or v.current_level < coarsest for v in views):
            self._auto_contrast_pending = False
            self.auto_contrast()

    def _update_statistics(self):
        self._update_pyramid_label()
        self._maybe_refine_contrast()
        view = self._primary_view()
        data = view.get_current_image_data() if view is not None else None
        if data is None or data.size == 0:
            self.stats_label.setText("")
            return
        flat = data.reshape(-1)
        s = flat[::max(1, flat.size // 1_000_000)].astype(np.float64)
        s = s[np.isfinite(s)]
        if s.size == 0:
            return
        self.stats_label.setText(
            f"visible {view.axis_name}: min {s.min():.5g}  max {s.max():.5g}  mean {s.mean():.5g}")

    # ---------------------------------------------------------------- view actions

    def _reset_active_view(self):
        if self.layout_mode == '3d':
            self.view_3d_only.reset_camera()
            return
        if self.layout_mode == 'quad' and self.view_3d.underMouse():
            self.view_3d.reset_camera()
            return
        view = self._target_view()
        if view is not None and view.image is not None:
            view.view.autoRange(padding=0)

    def _reset_all_views(self):
        for view in self._page_views():
            if view.image is not None:
                view.view.autoRange(padding=0)
        if self.layout_mode == 'quad':
            self.view_3d.reset_camera()
        elif self.layout_mode == '3d':
            self.view_3d_only.reset_camera()

    def _on_pane_plane_changed(self, view, plane):
        view.set_axis(plane)
        if id(view) in self._attached:
            self._apply_display_to(view)
            self._update_pane_info(view)
        self._update_pyramid_label()

    def _toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
            self._fullscreen_btn.setText("⛶ Full screen")
        else:
            self.showFullScreen()
            self._fullscreen_btn.setText("⤢ Exit full screen")

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape and self.isFullScreen():
            self._toggle_fullscreen()
            return
        super().keyPressEvent(event)

    # ---------------------------------------------------------------- cache / system

    def _set_cache_size(self, size_mb):
        self._cache_mb = size_mb
        if self.multires_image is not None:
            target = size_mb if size_mb is not None else sysinfo.default_cache_mb()
            self.multires_image.chunk_cache.set_max_memory(target)
            _log.info(f"Chunk cache limit set to {target} MB")
        self._update_cache_stats()
        self._update_system_label()

    def _update_cache_stats(self):
        if self.multires_image is None:
            self.cache_stats_label.setText("")
            return
        st = self.multires_image.chunk_cache.get_stats()
        self.cache_stats_label.setText(
            f"cache {st['memory_mb']:.0f}/{st['max_mb']:.0f} MB · {st['chunks']} chunks · "
            f"{st['hit_rate'] * 100:.0f}% hits")

    def _update_system_label(self):
        cache = (self.multires_image.chunk_cache.max_memory_mb if self.multires_image is not None
                 else (self._cache_mb or sysinfo.default_cache_mb()))
        lines = [sysinfo.describe_machine(self.loader.gl_caps),
                 f"I/O threads {io_workers()} · chunk cache {cache:.0f} MB"]
        if VISPY_AVAILABLE and self.loader.multires is not None:
            lines.append(f"3D budget (auto): {self.loader.budget('auto').summary()}")
        self.system_label.setText("\n".join(lines))

    # ---------------------------------------------------------------- recent files / dnd

    def _recent(self):
        val = self.settings.value("recent", [])
        if isinstance(val, str):
            val = [val]
        return [v for v in (val or []) if v]

    def _add_recent(self, path):
        rec = [p for p in self._recent() if p != path]
        rec.insert(0, path)
        self.settings.setValue("recent", rec[:10])
        self._rebuild_recent_menu()

    def _rebuild_recent_menu(self):
        self._recent_menu.clear()
        rec = self._recent()
        for p in rec:
            self._recent_menu.addAction(p, lambda _=False, pp=p: self.open_path(pp))
        if not rec:
            a = self._recent_menu.addAction("(empty)")
            a.setEnabled(False)
        else:
            self._recent_menu.addSeparator()
            self._recent_menu.addAction("Clear list", lambda: (self.settings.setValue("recent", []),
                                                               self._rebuild_recent_menu()))

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            urls = [u for u in event.mimeData().urls() if u.isLocalFile()]
            if urls and Path(urls[0].toLocalFile()).is_dir():
                event.acceptProposedAction()

    def dropEvent(self, event):
        urls = [u for u in event.mimeData().urls() if u.isLocalFile()]
        if urls:
            self.open_path(urls[0].toLocalFile())

    # ---------------------------------------------------------------- settings

    def _restore_settings(self):
        geom = self.settings.value("geometry")
        if geom is not None:
            self.restoreGeometry(geom)
        state = self.settings.value("windowState")
        if state is not None:
            self.restoreState(state)
        mode = self.settings.value("layout", "single")
        self.set_layout(mode if mode in LAYOUTS else "single")

    def closeEvent(self, event):
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
        self.settings.setValue("layout", self.layout_mode)
        self.loader.cancel()
        self._close_store(quiet=True)
        super().closeEvent(event)

    # ---------------------------------------------------------------- dialogs

    def _open_render_dialog(self):
        if self.multires_image is None:
            QtWidgets.QMessageBox.information(self, "Render 3D", "Open a dataset first.")
            return
        from zview.render_dialog import RenderDialog  # noqa: PLC0415
        dlg = getattr(self, "_render_dialog", None)
        if dlg is None or dlg.viewer.dataset_path != self.dataset_path:
            dlg = self._render_dialog = RenderDialog(self)
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _show_shortcuts(self):
        QtWidgets.QMessageBox.information(
            self, "Keyboard & mouse",
            "<table cellpadding=3>"
            "<tr><td><b>Ctrl+O</b></td><td>Open dataset (or drop a folder on the window)</td></tr>"
            "<tr><td><b>Ctrl+1…4</b></td><td>1 view / 2 views / 4 views / 3D</td></tr>"
            "<tr><td><b>PgUp / PgDn</b></td><td>Previous / next slice of the pane under the mouse"
            " (Shift: ×10)</td></tr>"
            "<tr><td><b>Ctrl+wheel</b></td><td>Scroll through slices</td></tr>"
            "<tr><td><b>Home / End</b></td><td>First / last slice</td></tr>"
            "<tr><td><b>R / Shift+R</b></td><td>Reset pane under mouse / all panes</td></tr>"
            "<tr><td><b>A</b></td><td>Auto contrast from what's visible</td></tr>"
            "<tr><td><b>Ctrl+D</b></td><td>Show / hide the controls panel</td></tr>"
            "<tr><td><b>F11 / Esc</b></td><td>Full screen / leave full screen</td></tr>"
            "<tr><td><b>Wheel, drag</b></td><td>Zoom, pan (2D) · rotate (3D; Shift+drag pans)</td></tr>"
            "<tr><td><b>Click</b></td><td>Move the crosshair; other panes follow</td></tr>"
            "<tr><td><b>Double-click header</b></td><td>Maximize / restore a pane (4 views)</td></tr>"
            "</table>")

    def _show_system_info(self):
        caps = self.loader.gl_caps
        rows = [("Machine", sysinfo.describe_machine()),
                ("I/O threads", str(io_workers())),
                ("Chunk cache", f"{(self._cache_mb or sysinfo.default_cache_mb())} MB"
                                f"{' (auto)' if self._cache_mb is None else ''}")]
        if caps is not None:
            rows += [("GPU", caps.renderer), ("OpenGL", caps.version),
                     ("Max 3D texture", str(caps.max_3d_texture)),
                     ("VRAM", f"{caps.vram_free_mb} MB free / {caps.vram_total_mb} MB"
                      if caps.vram_total_mb else (f"{caps.vram_free_mb} MB free" if caps.vram_free_mb
                                                  else "not reported"))]
        else:
            rows.append(("GPU", "not queried yet (open the 3D view once)"))
        if VISPY_AVAILABLE:
            itemsize = (np.dtype(self.multires_image.levels[0].dtype).itemsize
                        if self.multires_image is not None else 2)
            for q in ("auto", "medium", "high"):
                b = sysinfo.volume_budget(caps, itemsize, q)
                rows.append((f"3D budget ({q})", f"{b.summary()}<br><small>{'; '.join(b.reasons)}</small>"))
        html = "<table cellpadding=3>" + "".join(
            f"<tr><td valign=top><b>{k}</b></td><td>{v}</td></tr>" for k, v in rows) + "</table>"
        QtWidgets.QMessageBox.information(self, "System info", html)

    def _show_about(self):
        QtWidgets.QMessageBox.information(
            self, "About zview",
            "<h3>zview</h3>"
            "<p>Fast viewer for large multi-resolution Zarr (v2/v3) and N5 volumes.</p>"
            "<ul><li>Chunk-aligned, parallel, asynchronous loading</li>"
            "<li>Automatic pyramid-level selection per pane</li>"
            "<li>1 / 2 / 4-pane orthogonal views with linked crosshair</li>"
            "<li>Low-resolution 3D volume rendering sized to this machine</li></ul>")

    def _show_performance_tips(self):
        QtWidgets.QMessageBox.information(
            self, "Performance tips",
            "<p><b>Everything is sized automatically</b> from this machine's CPU count, free RAM "
            "and GPU; see <i>Help › System info</i>.</p>"
            "<ul>"
            "<li><b>I/O threads</b> default to the CPU count (4–32). Override with "
            "<code>ZVIEW_IO_THREADS</code>; raise it on high-latency network storage.</li>"
            "<li><b>Chunk cache</b> defaults to 15% of free RAM (256 MB–4 GB). Change it in "
            "<i>View › Chunk cache size</i> or with <code>ZVIEW_CACHE_MB</code>.</li>"
            "<li><b>3D</b> picks the finest pyramid level that fits the GPU's 3D-texture limit, "
            "half the free VRAM and a quarter of free RAM. Use the quality menu in the 3D pane, "
            "or <code>ZVIEW_3D_MAX_AXIS</code> / <code>ZVIEW_3D_MAX_VOXELS</code>.</li>"
            "<li>Datasets without a deep pyramid still work in 3D (strided read), but load faster "
            "with 5–6 levels.</li>"
            "</ul>")


# ============================================================ app setup

def apply_dark_theme(app):
    """Apply dark theme"""
    app.setStyle('Fusion')
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(48, 48, 48))
    palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor(32, 32, 32))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(48, 48, 48))
    palette.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor(40, 40, 40))
    palette.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(42, 130, 218))
    palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.white)
    for role in (QtGui.QPalette.Text, QtGui.QPalette.ButtonText, QtGui.QPalette.WindowText):
        palette.setColor(QtGui.QPalette.Disabled, role, QtGui.QColor(120, 120, 120))
    app.setPalette(palette)

    app.setStyleSheet("""
        QGroupBox {
            border: 1px solid #4a4a4a;
            border-radius: 5px;
            margin-top: 12px;
            font-weight: bold;
            padding: 8px 4px 4px 4px;
        }
        QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; color: #cfcfcf; }
        QPushButton {
            background-color: #454545; border: 1px solid #5f5f5f;
            border-radius: 4px; padding: 4px 12px;
        }
        QPushButton:hover { background-color: #505050; }
        QPushButton:disabled { color: #777; }
        QComboBox {
            background-color: #3f3f3f; border: 1px solid #5f5f5f;
            border-radius: 3px; padding: 2px 6px;
        }
        QToolBar { background: #383838; border: none; border-bottom: 1px solid #2a2a2a;
                   spacing: 4px; padding: 2px; }
        QToolBar QToolButton { padding: 3px 6px; border-radius: 3px; }
        QToolBar QToolButton:hover { background: #4a4a4a; }
        QToolBar QToolButton:disabled { color: #6a6a6a; }
        QDockWidget::title { background: #383838; padding: 4px 8px; }
        QStatusBar { background: #262626; color: #cccccc; }
        QStatusBar::item { border: none; }
        QSplitter::handle { background: #1e1e1e; }
        QTabWidget::pane { border: none; }
        QTabBar::tab { background: #333; color: #bbb; padding: 4px 14px; border: none; }
        QTabBar::tab:selected { background: #484848; color: white; }
    """)


def main(argv=None):
    """Run the unified viewer: `zview [PATH] [--layout quad]`."""
    parser = argparse.ArgumentParser(prog="zview", description="Multi-resolution Zarr/N5 viewer")
    parser.add_argument("path", nargs="?", help="Zarr (v2/v3) or N5 store directory")
    parser.add_argument("--layout", choices=LAYOUTS, help="initial layout")
    parser.add_argument("--cache-mb", type=int, help="chunk cache size (default: auto)")
    args, qt_args = parser.parse_known_args(sys.argv[1:] if argv is None else argv)

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([sys.argv[0], *qt_args])
    app.setApplicationName("zview")
    app.setOrganizationName("zview")
    apply_dark_theme(app)

    viewer = UnifiedZarrViewer()
    if args.cache_mb:
        viewer._set_cache_size(args.cache_mb)
    if args.layout:
        viewer.set_layout(args.layout)
    viewer.show()
    if args.path:
        QtCore.QTimer.singleShot(0, lambda: viewer.open_path(args.path))

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
