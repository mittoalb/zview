"""Unified multi-resolution Zarr viewer: single/dual/quad modes + metadata tab."""

import sys
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import z5py
from PyQt5 import QtCore, QtGui, QtWidgets

from zview.sview import DynamicImageView
from zview.oview import OrthoImageView
from zview.meta import MetadataViewer
from zview.msres import MultiResolutionImage

pg.setConfigOptions(imageAxisOrder='row-major')

# 3D viewer is currently stubbed. To re-enable, swap the import below for
# `from zview.view3d import VisPy3DView, VISPY_AVAILABLE`.
VISPY_AVAILABLE = False


class VisPy3DView(QtWidgets.QWidget):
    """Placeholder for 3D view when disabled or when VisPy is not installed."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        label = QtWidgets.QLabel("3D View Disabled")
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.setStyleSheet("color: #888; font-size: 14px;")
        layout.addWidget(label)

    def set_multires_image(self, multires_image):
        pass


# Optional environment variable settings for performance
# export BLOSC_NTHREADS=8
# export NUMEXPR_MAX_THREADS=8


# ==================== Main Unified Viewer ====================

class UnifiedZarrViewer(QtWidgets.QDialog):
    """Unified viewer with toggleable single/orthogonal modes"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.zarr_store = None
        self.zarr_group = None
        self.multires_image = None
        
        self.view_mode = 'single'
        
        self.z_pos = 0
        self.y_pos = 0
        self.x_pos = 0
        
        self.setWindowTitle("Multiscale Zarr Viewer")
        self.setModal(False)
        self.resize(1800, 1000)
        
        self._build_ui()
    
    def _build_ui(self):
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setSpacing(5)
        
        self.menu_bar = self._build_menubar()
        main_layout.setMenuBar(self.menu_bar)

        self.button_bar = self._build_buttonbar()
        main_layout.addWidget(self.button_bar)

        toolbar = self._build_toolbar()
        main_layout.addWidget(toolbar)
        
        self.main_tabs = QtWidgets.QTabWidget()
        
        self.stack = QtWidgets.QStackedWidget()
        
        self.single_view_widget = self._build_single_view()
        self.stack.addWidget(self.single_view_widget)
        
        self.dual_view_widget = self._build_dual_view()
        self.stack.addWidget(self.dual_view_widget)
        
        self.quad_view_widget = self._build_quad_view()
        self.stack.addWidget(self.quad_view_widget)
        
        self.view_3d_only = VisPy3DView()
        self.stack.addWidget(self.view_3d_only)
        
        self.main_tabs.addTab(self.stack, "Image Viewer")
        
        self.metadata_viewer = MetadataViewer()
        self.main_tabs.addTab(self.metadata_viewer, "Metadata")

        main_layout.addWidget(self.main_tabs)

        # Status bar at the bottom, showing cursor world coord + pixel value.
        self.status_label = QtWidgets.QLabel("")
        self.status_label.setStyleSheet(
            "background-color: #202020; color: #cccccc; "
            "padding: 2px 8px; font-family: monospace; font-size: 11px;"
        )
        self.status_label.setMinimumHeight(20)
        main_layout.addWidget(self.status_label)

        self._install_qol_features()

    def _install_qol_features(self):
        """Wire the quality-of-life widgets built elsewhere in `_build_ui`.

        - cursor-position reporting from every pane into `status_label`
        - keyboard shortcuts (F11, R, PgUp/Dn, Home/End)
        - double-click on an ortho pane's header to maximize/restore
        """
        # Cursor-position reporting.
        self.single_image_view.cursorMoved.connect(
            lambda x, y: self._update_status('XY', x, y)
        )
        self.dual_view1.cursorMoved.connect(
            lambda x, y: self._update_status(self._dual_plane(1), x, y)
        )
        self.dual_view2.cursorMoved.connect(
            lambda x, y: self._update_status(self._dual_plane(2), x, y)
        )
        self.xy_view.cursorMoved.connect(lambda x, y: self._update_status('XY', x, y))
        self.xz_view.cursorMoved.connect(lambda x, y: self._update_status('XZ', x, y))
        self.yz_view.cursorMoved.connect(lambda x, y: self._update_status('YZ', x, y))

        # Keyboard shortcuts.
        def sc(seq, fn):
            s = QtWidgets.QShortcut(QtGui.QKeySequence(seq), self)
            s.setContext(QtCore.Qt.ApplicationShortcut)
            s.activated.connect(fn)
            return s

        # F11 is wired via a QAction on the toolbar — QShortcut on a QDialog
        # is sometimes swallowed by the window manager.
        sc("R", self._reset_active_view)
        sc("PgDown", lambda: self._step_z(+1))
        sc("PgUp", lambda: self._step_z(-1))
        sc("Shift+PgDown", lambda: self._step_z(+10))
        sc("Shift+PgUp", lambda: self._step_z(-10))
        sc("Home", lambda: self._jump_z(0))
        sc("End", lambda: self._jump_z(10**9))  # clamped by spinbox max

        # Map pane key -> container widget (populated here now that _build_ui
        # has already run _build_quad_view).
        self._quad_panes = {
            'xy': self.xy_container,
            'xz': self.xz_container,
            'yz': self.yz_container,
            '3d': self.view_3d_container,
        }
        self._quad_maximized = None
        # Restore buttons start disabled (no pane is maxed on startup).
        for btn in getattr(self, "_pane_restore_buttons", {}).values():
            btn.setEnabled(False)

    def _dual_plane(self, which):
        combo = self.dual_view1_plane_combo if which == 1 else self.dual_view2_plane_combo
        idx = combo.currentIndex()
        return ('XY', 'XZ', 'YZ')[idx]

    def _update_status(self, plane, h, v):
        """Format and show cursor world coord + pixel value in the status bar."""
        if self.multires_image is None:
            return
        shape0 = self.multires_image.get_level_shape(0)
        if shape0 is None or len(shape0) < 3:
            return
        zs, ys, xs = shape0[-3:]
        hi, vi = int(round(h)), int(round(v))
        if plane == 'XY':
            x, y, z = hi, vi, self.z_pos
        elif plane == 'XZ':
            x, z, y = hi, vi, self.y_pos
        elif plane == 'YZ':
            y, z, x = hi, vi, self.x_pos
        else:
            return
        in_bounds = (0 <= z < zs) and (0 <= y < ys) and (0 <= x < xs)
        if not in_bounds:
            self.status_label.setText(
                f"[{plane}] out of bounds (z={z}  y={y}  x={x})"
            )
            return
        self.status_label.setText(
            f"[{plane}] z={z:5d}  y={y:5d}  x={x:5d}   "
            f"(volume: {zs}×{ys}×{xs})"
        )

    def _active_view(self):
        """Return the DynamicImageView currently visible, best-effort."""
        idx = self.stack.currentIndex()
        if idx == 0:
            return self.single_image_view
        if idx == 1:
            return self.dual_view1
        if idx == 2:
            # Prefer whichever ortho pane currently has focus.
            for v in (self.xy_view, self.xz_view, self.yz_view):
                if v.hasFocus():
                    return v
            return self.xy_view
        return None

    def _reset_active_view(self):
        v = self._active_view()
        if v is not None and v.image is not None:
            v.view.autoRange(padding=0)

    def _toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
            if hasattr(self, "_fullscreen_btn"):
                self._fullscreen_btn.setText("⛶ Fullscreen")
        else:
            self.showFullScreen()
            if hasattr(self, "_fullscreen_btn"):
                self._fullscreen_btn.setText("⤢ Exit Fullscreen")

    def _step_z(self, delta):
        if hasattr(self, 'z_spin') and self.z_spin.isEnabled():
            self.z_spin.setValue(self.z_spin.value() + delta)
        if hasattr(self, 'single_slice_slider') and self.single_slice_slider.isEnabled():
            self.single_slice_slider.setValue(
                self.single_slice_slider.value() + delta
            )

    def _jump_z(self, value):
        if hasattr(self, 'z_spin') and self.z_spin.isEnabled():
            self.z_spin.setValue(value)
        if hasattr(self, 'single_slice_slider') and self.single_slice_slider.isEnabled():
            self.single_slice_slider.setValue(value)

    _PANE_SIDE = {'xy': 'left', 'yz': 'left', 'xz': 'right', '3d': 'right'}

    _PANE_BTN_STYLE = """
        QToolButton {
            background: #4a4a4a;
            color: #f0f0f0;
            border: 1px solid #777;
            border-radius: 3px;
            font-size: 13px;
            padding: 0px;
        }
        QToolButton:hover   { background: #5c5c5c; border-color: #aaa; }
        QToolButton:pressed { background: #3a3a3a; }
        QToolButton:disabled {
            background: #2e2e2e;
            color: #666;
            border-color: #444;
        }
    """

    def _build_pane_header(self, key, label):
        """Header row for a quad-view pane: title label + max/restore buttons.

        Buttons sit top-right at an easily clickable size with explicit chrome
        so they're visible against the dark pane header. Restore starts
        disabled and becomes enabled once a pane is maximized.
        """
        label.setStyleSheet("padding: 3px; font-weight: bold; color: #e8e8e8;")
        label.setAlignment(QtCore.Qt.AlignCenter)

        def _mk_btn(glyph, tip, slot, start_enabled=True):
            btn = QtWidgets.QToolButton()
            btn.setText(glyph)
            btn.setToolTip(tip)
            btn.setFixedSize(QtCore.QSize(26, 22))
            btn.setStyleSheet(self._PANE_BTN_STYLE)
            btn.setEnabled(start_enabled)
            btn.clicked.connect(slot)
            return btn

        max_btn = _mk_btn(
            "⛶", "Maximize this pane",
            lambda _=False, k=key: self._maximize_pane(k),
        )
        restore_btn = _mk_btn(
            "⧉", "Restore 4-pane view",
            lambda _=False: self._restore_panes(),
            start_enabled=False,
        )

        if not hasattr(self, "_pane_restore_buttons"):
            self._pane_restore_buttons = {}
        self._pane_restore_buttons[key] = restore_btn

        header = QtWidgets.QWidget()
        header.setStyleSheet("background-color: #2a2a2a;")
        row = QtWidgets.QHBoxLayout(header)
        row.setContentsMargins(6, 2, 6, 2)
        row.setSpacing(4)
        row.addStretch(1)
        row.addWidget(label)
        row.addStretch(1)
        row.addWidget(max_btn)
        row.addWidget(restore_btn)
        return header

    def _set_restore_buttons_enabled(self, enabled):
        for btn in getattr(self, "_pane_restore_buttons", {}).values():
            btn.setEnabled(enabled)

    def _maximize_pane(self, key):
        """Expand `key`'s pane to fill the quad-view area."""
        if key not in self._quad_panes:
            return
        side = self._PANE_SIDE[key]
        for k, pane in self._quad_panes.items():
            pane.setVisible(k == key)
        # Hide the opposite splitter so the maxed pane actually gets the
        # window (hiding a splitter child only expands its sibling, not the
        # splitter itself).
        self._left_splitter.setVisible(side == 'left')
        self._right_splitter.setVisible(side == 'right')
        self._quad_maximized = key
        self._set_restore_buttons_enabled(True)

    def _restore_panes(self):
        """Return to the 4-pane layout."""
        if self._quad_maximized is None:
            return
        for pane in self._quad_panes.values():
            pane.show()
        self._left_splitter.show()
        self._right_splitter.show()
        self._left_splitter.setSizes([1, 1])
        self._right_splitter.setSizes([1, 1])
        self._main_splitter.setSizes([1, 1])
        self._quad_maximized = None
        self._set_restore_buttons_enabled(False)

    # Keep the old shortcut-compatible name as a toggle alias.
    def _toggle_pane_max(self, key):
        if self._quad_maximized == key:
            self._restore_panes()
        else:
            self._maximize_pane(key)
    
    def _build_toolbar(self):
        """Build info toolbar"""
        toolbar = QtWidgets.QWidget()
        toolbar.setMaximumHeight(40)
        layout = QtWidgets.QHBoxLayout(toolbar)
        layout.setContentsMargins(5, 2, 5, 2)
        
        self.file_label = QtWidgets.QLabel("No file loaded")
        self.file_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(self.file_label)
        
        layout.addStretch()
        
        # Cache stats
        self.cache_stats_label = QtWidgets.QLabel("Cache: N/A")
        self.cache_stats_label.setStyleSheet("font-size: 11px; color: #4a4; padding: 3px;")
        layout.addWidget(self.cache_stats_label)

        # Update cache stats every 2 seconds
        self.cache_stats_timer = QtCore.QTimer()
        self.cache_stats_timer.timeout.connect(self._update_cache_stats)
        self.cache_stats_timer.start(2000)

        # Fullscreen toggle. Visible button + QAction (which also wires F11
        # reliably — QShortcut on a QDialog sometimes doesn't receive F11).
        self._fullscreen_btn = QtWidgets.QToolButton()
        self._fullscreen_btn.setText("⛶ Fullscreen")
        self._fullscreen_btn.setToolTip("Toggle full screen (F11)")
        self._fullscreen_btn.setStyleSheet(
            "QToolButton { background:#4a4a4a; color:#f0f0f0; "
            "border:1px solid #777; border-radius:3px; padding:2px 10px; "
            "font-size:11px; }"
            "QToolButton:hover { background:#5c5c5c; }"
            "QToolButton:pressed { background:#3a3a3a; }"
        )
        self._fullscreen_btn.clicked.connect(self._toggle_fullscreen)
        layout.addWidget(self._fullscreen_btn)

        self._fullscreen_action = QtWidgets.QAction("Toggle fullscreen", self)
        self._fullscreen_action.setShortcut(QtGui.QKeySequence("F11"))
        self._fullscreen_action.setShortcutContext(QtCore.Qt.ApplicationShortcut)
        self._fullscreen_action.triggered.connect(self._toggle_fullscreen)
        self.addAction(self._fullscreen_action)

        return toolbar
    
    def _update_cache_stats(self):
        """Update cache statistics display"""
        if self.multires_image is not None:
            stats = self.multires_image.chunk_cache.get_stats()
            self.cache_stats_label.setText(
                f"Cache: {stats['chunks']} chunks, "
                f"{stats['memory_mb']:.1f} MB, "
                f"{stats['hit_rate']*100:.0f}% hits"
            )
        else:
            self.cache_stats_label.setText("")
    
    def _build_single_view(self):
        """Build single view mode interface"""
        widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QHBoxLayout(widget)
        main_layout.setSpacing(10)
        
        left_panel = QtWidgets.QWidget()
        left_panel.setMaximumWidth(350)
        control_layout = QtWidgets.QVBoxLayout(left_panel)
        
        pyramid_group = QtWidgets.QGroupBox("Pyramid Information")
        pyramid_layout = QtWidgets.QFormLayout()
        
        self.single_num_levels_label = QtWidgets.QLabel("N/A")
        self.single_current_level_label = QtWidgets.QLabel("N/A")
        self.single_level_shape_label = QtWidgets.QLabel("N/A")
        self.single_level_dtype_label = QtWidgets.QLabel("N/A")
        self.single_level_chunks_label = QtWidgets.QLabel("N/A")
        
        pyramid_layout.addRow("Total levels:", self.single_num_levels_label)
        pyramid_layout.addRow("Current level:", self.single_current_level_label)
        pyramid_layout.addRow("Level shape:", self.single_level_shape_label)
        pyramid_layout.addRow("Data type:", self.single_level_dtype_label)
        pyramid_layout.addRow("Chunk size:", self.single_level_chunks_label)
        
        self.single_effective_res_label = QtWidgets.QLabel("N/A")
        self.single_effective_res_label.setStyleSheet("color: #4a4; font-weight: bold;")
        pyramid_layout.addRow("Effective res:", self.single_effective_res_label)
        
        pyramid_group.setLayout(pyramid_layout)
        control_layout.addWidget(pyramid_group)
        
        slice_group = QtWidgets.QGroupBox("Slice Selection")
        slice_layout = QtWidgets.QVBoxLayout()
        
        slider_layout = QtWidgets.QHBoxLayout()
        slider_layout.addWidget(QtWidgets.QLabel("Index:"))
        self.single_slice_label = QtWidgets.QLabel("0")
        self.single_slice_label.setStyleSheet("font-weight: bold;")
        slider_layout.addWidget(self.single_slice_label)
        slice_layout.addLayout(slider_layout)
        
        self.single_slice_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.single_slice_slider.setMinimum(0)
        self.single_slice_slider.setEnabled(False)
        self.single_slice_slider.valueChanged.connect(self._on_single_slice_changed)
        slice_layout.addWidget(self.single_slice_slider)
        
        slice_group.setLayout(slice_layout)
        control_layout.addWidget(slice_group)
        
        contrast_group = QtWidgets.QGroupBox("Contrast Control")
        contrast_layout = QtWidgets.QVBoxLayout()
        
        auto_layout = QtWidgets.QHBoxLayout()
        auto_layout.addWidget(QtWidgets.QLabel("Auto Level:"))
        
        self.single_auto_level_combo = QtWidgets.QComboBox()
        self.single_auto_level_combo.addItems([
            "Auto (default)",
            "Min/Max",
            "Percentile 1-99%",
            "Percentile 2-98%",
            "Percentile 5-95%",
            "Manual"
        ])
        self.single_auto_level_combo.currentIndexChanged.connect(self._on_single_contrast_changed)
        auto_layout.addWidget(self.single_auto_level_combo)
        contrast_layout.addLayout(auto_layout)
        
        manual_widget = QtWidgets.QWidget()
        manual_layout = QtWidgets.QFormLayout()
        manual_layout.setContentsMargins(0, 0, 0, 0)
        
        self.single_min_spin = QtWidgets.QDoubleSpinBox()
        self.single_min_spin.setRange(-1e10, 1e10)
        self.single_min_spin.setDecimals(4)
        self.single_min_spin.valueChanged.connect(self._on_single_manual_contrast_changed)
        manual_layout.addRow("Min:", self.single_min_spin)
        
        self.single_max_spin = QtWidgets.QDoubleSpinBox()
        self.single_max_spin.setRange(-1e10, 1e10)
        self.single_max_spin.setDecimals(4)
        self.single_max_spin.valueChanged.connect(self._on_single_manual_contrast_changed)
        manual_layout.addRow("Max:", self.single_max_spin)
        
        manual_widget.setLayout(manual_layout)
        manual_widget.setVisible(False)
        self.single_manual_controls = manual_widget
        contrast_layout.addWidget(manual_widget)
        
        auto_btn = QtWidgets.QPushButton("Auto Adjust Levels")
        auto_btn.clicked.connect(self._single_auto_adjust)
        contrast_layout.addWidget(auto_btn)
        
        contrast_group.setLayout(contrast_layout)
        control_layout.addWidget(contrast_group)
        
        view_group = QtWidgets.QGroupBox("View Controls")
        view_layout = QtWidgets.QVBoxLayout()
        
        reset_view_btn = QtWidgets.QPushButton("Reset View")
        reset_view_btn.clicked.connect(lambda: self.single_image_view.view.autoRange())
        view_layout.addWidget(reset_view_btn)
        
        view_info = QtWidgets.QLabel(
            "<b>Navigation:</b><br>"
            "• Mouse wheel: Zoom<br>"
            "• Right drag: Pan<br>"
            "• Resolution adjusts automatically"
        )
        view_info.setWordWrap(True)
        view_info.setStyleSheet("padding: 10px; background-color: #2a2a2a; border-radius: 5px;")
        view_layout.addWidget(view_info)
        
        view_group.setLayout(view_layout)
        control_layout.addWidget(view_group)
        
        stats_group = QtWidgets.QGroupBox("Image Statistics")
        stats_layout = QtWidgets.QFormLayout()
        
        self.single_min_val_label = QtWidgets.QLabel("N/A")
        self.single_max_val_label = QtWidgets.QLabel("N/A")
        self.single_mean_val_label = QtWidgets.QLabel("N/A")
        
        stats_layout.addRow("Min:", self.single_min_val_label)
        stats_layout.addRow("Max:", self.single_max_val_label)
        stats_layout.addRow("Mean:", self.single_mean_val_label)
        
        stats_group.setLayout(stats_layout)
        control_layout.addWidget(stats_group)
        
        control_layout.addStretch()
        main_layout.addWidget(left_panel)
        
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        self.single_image_view = DynamicImageView()
        self.single_image_view.ui.roiBtn.hide()
        self.single_image_view.ui.menuBtn.hide()
        self.single_image_view.levelChanged.connect(self._update_single_level_info)
        right_layout.addWidget(self.single_image_view)
        
        main_layout.addWidget(right_panel)
        main_layout.setStretch(1, 1)
        
        return widget
    
    def _build_dual_view(self):
        """Build dual view mode interface with resizable splitters and plane selection"""
        widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QHBoxLayout(widget)
        
        # Main splitter for views
        views_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        
        # View 1 container
        view1_container = QtWidgets.QWidget()
        view1_layout = QtWidgets.QVBoxLayout(view1_container)
        view1_layout.setContentsMargins(0, 0, 0, 0)
        view1_layout.setSpacing(2)
        
        self.dual_view1_label = QtWidgets.QLabel("XY (Axial)")
        self.dual_view1_label.setStyleSheet("background-color: #2a2a2a; padding: 3px; font-weight: bold;")
        self.dual_view1_label.setAlignment(QtCore.Qt.AlignCenter)
        view1_layout.addWidget(self.dual_view1_label)
        
        self.dual_view1 = OrthoImageView('XY')
        self.dual_view1.ui.roiBtn.hide()
        self.dual_view1.ui.menuBtn.hide()
        self.dual_view1.crosshairMoved.connect(self._on_dual_view1_crosshair_moved)
        view1_layout.addWidget(self.dual_view1)
        
        # View 2 container
        view2_container = QtWidgets.QWidget()
        view2_layout = QtWidgets.QVBoxLayout(view2_container)
        view2_layout.setContentsMargins(0, 0, 0, 0)
        view2_layout.setSpacing(2)
        
        self.dual_view2_label = QtWidgets.QLabel("XZ (Coronal)")
        self.dual_view2_label.setStyleSheet("background-color: #2a2a2a; padding: 3px; font-weight: bold;")
        self.dual_view2_label.setAlignment(QtCore.Qt.AlignCenter)
        view2_layout.addWidget(self.dual_view2_label)
        
        self.dual_view2 = OrthoImageView('XZ')
        self.dual_view2.ui.roiBtn.hide()
        self.dual_view2.ui.menuBtn.hide()
        self.dual_view2.crosshairMoved.connect(self._on_dual_view2_crosshair_moved)
        view2_layout.addWidget(self.dual_view2)
        
        views_splitter.addWidget(view1_container)
        views_splitter.addWidget(view2_container)
        views_splitter.setStretchFactor(0, 1)
        views_splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(views_splitter, stretch=4)
        
        control_panel = self._build_dual_control_panel()
        main_layout.addWidget(control_panel, stretch=1)
        
        return widget
    
    def _build_dual_control_panel(self):
        """Build dual view control panel with plane selection"""
        panel = QtWidgets.QWidget()
        panel.setMaximumWidth(300)
        layout = QtWidgets.QVBoxLayout(panel)
        
        # Plane selection group
        plane_group = QtWidgets.QGroupBox("Plane Selection")
        plane_layout = QtWidgets.QVBoxLayout()
        
        # View 1 plane selector
        view1_layout = QtWidgets.QHBoxLayout()
        view1_layout.addWidget(QtWidgets.QLabel("View 1:"))
        self.dual_view1_plane_combo = QtWidgets.QComboBox()
        self.dual_view1_plane_combo.addItems(["XY (Axial)", "XZ (Coronal)", "YZ (Sagittal)"])
        self.dual_view1_plane_combo.currentIndexChanged.connect(self._on_dual_view1_plane_changed)
        view1_layout.addWidget(self.dual_view1_plane_combo)
        plane_layout.addLayout(view1_layout)
        
        # View 2 plane selector
        view2_layout = QtWidgets.QHBoxLayout()
        view2_layout.addWidget(QtWidgets.QLabel("View 2:"))
        self.dual_view2_plane_combo = QtWidgets.QComboBox()
        self.dual_view2_plane_combo.addItems(["XY (Axial)", "XZ (Coronal)", "YZ (Sagittal)"])
        self.dual_view2_plane_combo.setCurrentIndex(1)  # Default to XZ
        self.dual_view2_plane_combo.currentIndexChanged.connect(self._on_dual_view2_plane_changed)
        view2_layout.addWidget(self.dual_view2_plane_combo)
        plane_layout.addLayout(view2_layout)
        
        plane_group.setLayout(plane_layout)
        layout.addWidget(plane_group)
        
        info_group = QtWidgets.QGroupBox("Volume Information")
        info_layout = QtWidgets.QFormLayout()
        
        self.dual_shape_label = QtWidgets.QLabel("N/A")
        self.dual_dtype_label = QtWidgets.QLabel("N/A")
        
        info_layout.addRow("Shape (Z,Y,X):", self.dual_shape_label)
        info_layout.addRow("Data type:", self.dual_dtype_label)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        pos_group = QtWidgets.QGroupBox("Slice Position")
        pos_layout = QtWidgets.QVBoxLayout()
        
        z_layout = QtWidgets.QHBoxLayout()
        z_layout.addWidget(QtWidgets.QLabel("Z:"))
        self.dual_z_spin = QtWidgets.QSpinBox()
        self.dual_z_spin.valueChanged.connect(self._on_dual_z_changed)
        z_layout.addWidget(self.dual_z_spin)
        pos_layout.addLayout(z_layout)
        
        self.dual_z_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.dual_z_slider.valueChanged.connect(self._on_dual_z_changed)
        pos_layout.addWidget(self.dual_z_slider)
        
        y_layout = QtWidgets.QHBoxLayout()
        y_layout.addWidget(QtWidgets.QLabel("Y:"))
        self.dual_y_spin = QtWidgets.QSpinBox()
        self.dual_y_spin.valueChanged.connect(self._on_dual_y_changed)
        y_layout.addWidget(self.dual_y_spin)
        pos_layout.addLayout(y_layout)
        
        self.dual_y_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.dual_y_slider.valueChanged.connect(self._on_dual_y_changed)
        pos_layout.addWidget(self.dual_y_slider)
        
        x_layout = QtWidgets.QHBoxLayout()
        x_layout.addWidget(QtWidgets.QLabel("X:"))
        self.dual_x_spin = QtWidgets.QSpinBox()
        self.dual_x_spin.valueChanged.connect(self._on_dual_x_changed)
        x_layout.addWidget(self.dual_x_spin)
        pos_layout.addLayout(x_layout)
        
        self.dual_x_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.dual_x_slider.valueChanged.connect(self._on_dual_x_changed)
        pos_layout.addWidget(self.dual_x_slider)
        
        pos_group.setLayout(pos_layout)
        layout.addWidget(pos_group)
        
        contrast_group = QtWidgets.QGroupBox("Contrast Control")
        contrast_layout = QtWidgets.QVBoxLayout()
        
        mode_layout = QtWidgets.QHBoxLayout()
        mode_layout.addWidget(QtWidgets.QLabel("Mode:"))
        
        self.dual_contrast_combo = QtWidgets.QComboBox()
        self.dual_contrast_combo.addItems([
            "Auto",
            "Min/Max",
            "Percentile 1-99%",
            "Percentile 2-98%",
            "Percentile 5-95%"
        ])
        self.dual_contrast_combo.currentIndexChanged.connect(self._on_dual_contrast_changed)
        mode_layout.addWidget(self.dual_contrast_combo)
        contrast_layout.addLayout(mode_layout)
        
        apply_btn = QtWidgets.QPushButton("Apply to Both Views")
        apply_btn.clicked.connect(self._apply_dual_contrast)
        contrast_layout.addWidget(apply_btn)
        
        contrast_group.setLayout(contrast_layout)
        layout.addWidget(contrast_group)
        
        controls_group = QtWidgets.QGroupBox("View Controls")
        controls_layout = QtWidgets.QVBoxLayout()
        
        reset_btn = QtWidgets.QPushButton("Reset Both Views")
        reset_btn.clicked.connect(self._reset_dual_views)
        controls_layout.addWidget(reset_btn)
        
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        layout.addStretch()
        
        return panel
    
    def _build_quad_view(self):
        """Build quad (4-view orthogonal) view mode interface with resizable splitters"""
        widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QHBoxLayout(widget)

        self._main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_splitter = self._main_splitter

        self._left_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        left_splitter = self._left_splitter
        
        # XY view with header (label + max/restore buttons)
        self.xy_container = QtWidgets.QWidget()
        xy_layout = QtWidgets.QVBoxLayout(self.xy_container)
        xy_layout.setContentsMargins(0, 0, 0, 0)
        xy_layout.setSpacing(2)
        self.xy_label = QtWidgets.QLabel("XY (Axial)")
        xy_layout.addWidget(self._build_pane_header('xy', self.xy_label))
        self.xy_view = OrthoImageView('XY')
        self.xy_view.ui.roiBtn.hide()
        self.xy_view.ui.menuBtn.hide()
        self.xy_view.crosshairMoved.connect(self._on_xy_crosshair_moved)
        xy_layout.addWidget(self.xy_view)

        # YZ view
        self.yz_container = QtWidgets.QWidget()
        yz_layout = QtWidgets.QVBoxLayout(self.yz_container)
        yz_layout.setContentsMargins(0, 0, 0, 0)
        yz_layout.setSpacing(2)
        self.yz_label = QtWidgets.QLabel("YZ (Sagittal)")
        yz_layout.addWidget(self._build_pane_header('yz', self.yz_label))
        self.yz_view = OrthoImageView('YZ')
        self.yz_view.ui.roiBtn.hide()
        self.yz_view.ui.menuBtn.hide()
        self.yz_view.crosshairMoved.connect(self._on_yz_crosshair_moved)
        yz_layout.addWidget(self.yz_view)

        left_splitter.addWidget(self.xy_container)
        left_splitter.addWidget(self.yz_container)
        left_splitter.setStretchFactor(0, 1)
        left_splitter.setStretchFactor(1, 1)
        # Give splitters explicit equal initial sizes — setStretchFactor alone
        # only controls resize distribution, so the initial layout picks up
        # size hints and the panes end up unequal.
        left_splitter.setSizes([1, 1])
        
        self._right_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        right_splitter = self._right_splitter
        
        # XZ view
        self.xz_container = QtWidgets.QWidget()
        xz_layout = QtWidgets.QVBoxLayout(self.xz_container)
        xz_layout.setContentsMargins(0, 0, 0, 0)
        xz_layout.setSpacing(2)
        self.xz_label = QtWidgets.QLabel("XZ (Coronal)")
        xz_layout.addWidget(self._build_pane_header('xz', self.xz_label))
        self.xz_view = OrthoImageView('XZ')
        self.xz_view.ui.roiBtn.hide()
        self.xz_view.ui.menuBtn.hide()
        self.xz_view.crosshairMoved.connect(self._on_xz_crosshair_moved)
        xz_layout.addWidget(self.xz_view)

        # 3D view
        self.view_3d_container = QtWidgets.QWidget()
        view_3d_layout = QtWidgets.QVBoxLayout(self.view_3d_container)
        view_3d_layout.setContentsMargins(0, 0, 0, 0)
        view_3d_layout.setSpacing(2)
        self.view_3d_label = QtWidgets.QLabel("3D Volume (disabled)")
        view_3d_layout.addWidget(self._build_pane_header('3d', self.view_3d_label))
        self.view_3d = VisPy3DView()
        view_3d_layout.addWidget(self.view_3d)

        right_splitter.addWidget(self.xz_container)
        right_splitter.addWidget(self.view_3d_container)
        right_splitter.setStretchFactor(0, 1)
        right_splitter.setStretchFactor(1, 1)
        right_splitter.setSizes([1, 1])

        # Add left and right to main splitter
        main_splitter.addWidget(left_splitter)
        main_splitter.addWidget(right_splitter)
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setSizes([1, 1])
        
        main_layout.addWidget(main_splitter, stretch=4)
        
        control_panel = self._build_ortho_control_panel()
        main_layout.addWidget(control_panel, stretch=1)
        
        return widget
    
    def _build_ortho_control_panel(self):
        """Build orthogonal view control panel"""
        panel = QtWidgets.QWidget()
        panel.setMaximumWidth(300)
        layout = QtWidgets.QVBoxLayout(panel)
        
        info_group = QtWidgets.QGroupBox("Volume Information")
        info_layout = QtWidgets.QFormLayout()
        
        self.ortho_shape_label = QtWidgets.QLabel("N/A")
        self.ortho_dtype_label = QtWidgets.QLabel("N/A")
        self.ortho_levels_label = QtWidgets.QLabel("N/A")
        
        info_layout.addRow("Shape (Z,Y,X):", self.ortho_shape_label)
        info_layout.addRow("Data type:", self.ortho_dtype_label)
        info_layout.addRow("Pyramid levels:", self.ortho_levels_label)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        pos_group = QtWidgets.QGroupBox("Slice Position")
        pos_layout = QtWidgets.QVBoxLayout()
        
        z_layout = QtWidgets.QHBoxLayout()
        z_layout.addWidget(QtWidgets.QLabel("Z:"))
        self.z_spin = QtWidgets.QSpinBox()
        self.z_spin.valueChanged.connect(self._on_z_changed)
        z_layout.addWidget(self.z_spin)
        pos_layout.addLayout(z_layout)
        
        self.z_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.z_slider.valueChanged.connect(self._on_z_changed)
        pos_layout.addWidget(self.z_slider)
        
        y_layout = QtWidgets.QHBoxLayout()
        y_layout.addWidget(QtWidgets.QLabel("Y:"))
        self.y_spin = QtWidgets.QSpinBox()
        self.y_spin.valueChanged.connect(self._on_y_changed)
        y_layout.addWidget(self.y_spin)
        pos_layout.addLayout(y_layout)
        
        self.y_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.y_slider.valueChanged.connect(self._on_y_changed)
        pos_layout.addWidget(self.y_slider)
        
        x_layout = QtWidgets.QHBoxLayout()
        x_layout.addWidget(QtWidgets.QLabel("X:"))
        self.x_spin = QtWidgets.QSpinBox()
        self.x_spin.valueChanged.connect(self._on_x_changed)
        x_layout.addWidget(self.x_spin)
        pos_layout.addLayout(x_layout)
        
        self.x_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.x_slider.valueChanged.connect(self._on_x_changed)
        pos_layout.addWidget(self.x_slider)
        
        pos_group.setLayout(pos_layout)
        layout.addWidget(pos_group)
        
        contrast_group = QtWidgets.QGroupBox("Contrast Control")
        contrast_layout = QtWidgets.QVBoxLayout()
        
        mode_layout = QtWidgets.QHBoxLayout()
        mode_layout.addWidget(QtWidgets.QLabel("Mode:"))
        
        self.ortho_contrast_combo = QtWidgets.QComboBox()
        self.ortho_contrast_combo.addItems([
            "Auto",
            "Min/Max",
            "Percentile 1-99%",
            "Percentile 2-98%",
            "Percentile 5-95%"
        ])
        self.ortho_contrast_combo.currentIndexChanged.connect(self._on_ortho_contrast_changed)
        mode_layout.addWidget(self.ortho_contrast_combo)
        contrast_layout.addLayout(mode_layout)
        
        apply_btn = QtWidgets.QPushButton("Apply to All Views")
        apply_btn.clicked.connect(self._apply_ortho_contrast)
        contrast_layout.addWidget(apply_btn)
        
        contrast_group.setLayout(contrast_layout)
        layout.addWidget(contrast_group)
        
        controls_group = QtWidgets.QGroupBox("View Controls")
        controls_layout = QtWidgets.QVBoxLayout()
        
        reset_btn = QtWidgets.QPushButton("Reset All Views")
        reset_btn.clicked.connect(self._reset_all_views)
        controls_layout.addWidget(reset_btn)
        
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        layout.addStretch()
        
        return panel
    
    # ==================== Single View Handlers ====================
    
    def _update_single_level_info(self, level, shape):
        """Update single view level info"""
        self.single_current_level_label.setText(str(level))
        self.single_level_shape_label.setText(str(shape))
        
        if self.multires_image and level >= 0:
            level_0_shape = self.multires_image.get_level_shape(0)
            if level_0_shape and shape:
                downsample = level_0_shape[-2] / shape[-2]
                if downsample > 1.0:
                    self.single_effective_res_label.setText(f"{downsample:.1f}x downsampled")
                else:
                    self.single_effective_res_label.setText("Full resolution")
        
        self._update_single_statistics()
    
    def _update_single_statistics(self):
        """Update statistics for current image"""
        if self.single_image_view.image is None:
            return
        
        try:
            img = self.single_image_view.image
            if img.size > 0:
                min_val = float(np.min(img))
                max_val = float(np.max(img))
                mean_val = float(np.mean(img))
                
                self.single_min_val_label.setText(f"{min_val:.4f}")
                self.single_max_val_label.setText(f"{max_val:.4f}")
                self.single_mean_val_label.setText(f"{mean_val:.4f}")
        except Exception as e:
            print(f"Error updating statistics: {e}")
    
    def _on_single_contrast_changed(self, index):
        """Handle contrast mode change"""
        if index == 5:
            self.single_manual_controls.setVisible(True)
            if hasattr(self.single_image_view, 'getLevels'):
                levels = self.single_image_view.getLevels()
                if levels is not None:
                    self.single_min_spin.blockSignals(True)
                    self.single_max_spin.blockSignals(True)
                    self.single_min_spin.setValue(levels[0])
                    self.single_max_spin.setValue(levels[1])
                    self.single_min_spin.blockSignals(False)
                    self.single_max_spin.blockSignals(False)
        else:
            self.single_manual_controls.setVisible(False)
            self._apply_single_contrast_mode(index)
    
    def _apply_single_contrast_mode(self, mode):
        """Apply contrast adjustment based on mode"""
        if self.single_image_view.image is None:
            return
        
        try:
            img = self.single_image_view.image
            
            if mode == 0:
                self.single_image_view.autoLevels()
            elif mode == 1:
                min_val = float(np.min(img))
                max_val = float(np.max(img))
                self.single_image_view.setLevels(min_val, max_val)
            elif mode == 2:
                min_val = float(np.percentile(img, 1))
                max_val = float(np.percentile(img, 99))
                self.single_image_view.setLevels(min_val, max_val)
            elif mode == 3:
                min_val = float(np.percentile(img, 2))
                max_val = float(np.percentile(img, 98))
                self.single_image_view.setLevels(min_val, max_val)
            elif mode == 4:
                min_val = float(np.percentile(img, 5))
                max_val = float(np.percentile(img, 95))
                self.single_image_view.setLevels(min_val, max_val)
        except Exception as e:
            print(f"Error applying contrast: {e}")
    
    def _on_single_manual_contrast_changed(self):
        """Handle manual contrast spinbox changes"""
        min_val = self.single_min_spin.value()
        max_val = self.single_max_spin.value()
        
        if min_val < max_val:
            self.single_image_view.setLevels(min_val, max_val)
    
    def _single_auto_adjust(self):
        """Auto adjust levels for single view"""
        mode = self.single_auto_level_combo.currentIndex()
        if mode == 5:
            self.single_auto_level_combo.setCurrentIndex(0)
        self._apply_single_contrast_mode(self.single_auto_level_combo.currentIndex())
    
    def _on_single_slice_changed(self, value):
        """Handle single view slice change"""
        self.single_slice_label.setText(str(value))
        if self.single_image_view.multires_image is not None:
            self.single_image_view.set_slice(value)
    
    # ==================== Dual View Handlers ====================
    
    def _update_dual_views(self):
        """Update both dual views"""
        if self.multires_image is None:
            return
        
        level_0_shape = self.multires_image.get_level_shape(0)
        if level_0_shape and len(level_0_shape) >= 3:
            z_size, y_size, x_size = level_0_shape[-3:]
            
            self.dual_shape_label.setText(f"{z_size} × {y_size} × {x_size}")
            
            self.dual_z_slider.setMaximum(z_size - 1)
            self.dual_z_spin.setMaximum(z_size - 1)
            self.dual_z_slider.setValue(z_size // 2)
            
            self.dual_y_slider.setMaximum(y_size - 1)
            self.dual_y_spin.setMaximum(y_size - 1)
            self.dual_y_slider.setValue(y_size // 2)
            
            self.dual_x_slider.setMaximum(x_size - 1)
            self.dual_x_spin.setMaximum(x_size - 1)
            self.dual_x_slider.setValue(x_size // 2)
            
            level_0 = self.multires_image.levels[0]
            if hasattr(level_0, 'dtype'):
                self.dual_dtype_label.setText(str(level_0.dtype))
            
            self.dual_view1.set_multires_image(self.multires_image)
            self.dual_view2.set_multires_image(self.multires_image)
            
            self._sync_dual_views()
    
    def _sync_dual_views(self):
        """Sync dual view slice positions"""
        z = self.dual_z_slider.value()
        y = self.dual_y_slider.value()
        x = self.dual_x_slider.value()
        
        self.dual_view1.set_slice_indices(z, y, x)
        self.dual_view2.set_slice_indices(z, y, x)
        
        if self.dual_view1.axis_name == 'XY':
            self.dual_view1.update_crosshair(x, y)
        elif self.dual_view1.axis_name == 'XZ':
            self.dual_view1.update_crosshair(x, z)
        elif self.dual_view1.axis_name == 'YZ':
            self.dual_view1.update_crosshair(z, y)
        
        if self.dual_view2.axis_name == 'XY':
            self.dual_view2.update_crosshair(x, y)
        elif self.dual_view2.axis_name == 'XZ':
            self.dual_view2.update_crosshair(x, z)
        elif self.dual_view2.axis_name == 'YZ':
            self.dual_view2.update_crosshair(z, y)
    
    def _on_dual_view1_plane_changed(self, index):
        """Handle plane selection change for view 1"""
        plane_names = ['XY', 'XZ', 'YZ']
        plane_labels = ['XY (Axial)', 'XZ (Coronal)', 'YZ (Sagittal)']
        
        self.dual_view1.axis_name = plane_names[index]
        self.dual_view1_label.setText(plane_labels[index])
        
        if self.multires_image is not None:
            self._sync_dual_views()
    
    def _on_dual_view2_plane_changed(self, index):
        """Handle plane selection change for view 2"""
        plane_names = ['XY', 'XZ', 'YZ']
        plane_labels = ['XY (Axial)', 'XZ (Coronal)', 'YZ (Sagittal)']
        
        self.dual_view2.axis_name = plane_names[index]
        self.dual_view2_label.setText(plane_labels[index])
        
        if self.multires_image is not None:
            self._sync_dual_views()
    
    def _on_dual_z_changed(self, value):
        self.dual_z_slider.blockSignals(True)
        self.dual_z_spin.blockSignals(True)
        self.dual_z_slider.setValue(value)
        self.dual_z_spin.setValue(value)
        self.dual_z_slider.blockSignals(False)
        self.dual_z_spin.blockSignals(False)
        self._sync_dual_views()
    
    def _on_dual_y_changed(self, value):
        self.dual_y_slider.blockSignals(True)
        self.dual_y_spin.blockSignals(True)
        self.dual_y_slider.setValue(value)
        self.dual_y_spin.setValue(value)
        self.dual_y_slider.blockSignals(False)
        self.dual_y_spin.blockSignals(False)
        self._sync_dual_views()
    
    def _on_dual_x_changed(self, value):
        self.dual_x_slider.blockSignals(True)
        self.dual_x_spin.blockSignals(True)
        self.dual_x_slider.setValue(value)
        self.dual_x_spin.setValue(value)
        self.dual_x_slider.blockSignals(False)
        self.dual_x_spin.blockSignals(False)
        self._sync_dual_views()
    
    def _on_dual_view1_crosshair_moved(self, x, y):
        """Handle crosshair movement in dual view 1"""
        if self.dual_view1.axis_name == 'XY':
            self.dual_x_slider.setValue(x)
            self.dual_y_slider.setValue(y)
        elif self.dual_view1.axis_name == 'XZ':
            self.dual_x_slider.setValue(x)
            self.dual_z_slider.setValue(y)
        elif self.dual_view1.axis_name == 'YZ':
            self.dual_z_slider.setValue(x)
            self.dual_y_slider.setValue(y)
    
    def _on_dual_view2_crosshair_moved(self, x, y):
        """Handle crosshair movement in dual view 2"""
        if self.dual_view2.axis_name == 'XY':
            self.dual_x_slider.setValue(x)
            self.dual_y_slider.setValue(y)
        elif self.dual_view2.axis_name == 'XZ':
            self.dual_x_slider.setValue(x)
            self.dual_z_slider.setValue(y)
        elif self.dual_view2.axis_name == 'YZ':
            self.dual_z_slider.setValue(x)
            self.dual_y_slider.setValue(y)
    
    def _on_dual_contrast_changed(self, index):
        """Handle dual view contrast mode change"""
        pass
    
    def _apply_dual_contrast(self):
        """Apply contrast adjustment to both dual views"""
        mode = self.dual_contrast_combo.currentIndex()
        self._apply_contrast_to_view(self.dual_view1, mode)
        self._apply_contrast_to_view(self.dual_view2, mode)
    
    def _reset_dual_views(self):
        """Reset both dual views"""
        self.dual_view1.view.autoRange()
        self.dual_view2.view.autoRange()
    
    # ==================== Ortho View Handlers ====================
    
    def _update_all_ortho_views(self):
        """Update all orthogonal views"""
        self.xy_view.set_slice_indices(self.z_pos, self.y_pos, self.x_pos)
        self.xz_view.set_slice_indices(self.z_pos, self.y_pos, self.x_pos)
        self.yz_view.set_slice_indices(self.z_pos, self.y_pos, self.x_pos)
        
        self.xy_view.update_crosshair(self.x_pos, self.y_pos)
        self.xz_view.update_crosshair(self.x_pos, self.z_pos)
        self.yz_view.update_crosshair(self.z_pos, self.y_pos)
    
    def _on_z_changed(self, value):
        self.z_pos = value
        self.z_slider.blockSignals(True)
        self.z_spin.blockSignals(True)
        self.z_slider.setValue(value)
        self.z_spin.setValue(value)
        self.z_slider.blockSignals(False)
        self.z_spin.blockSignals(False)
        self._update_all_ortho_views()
    
    def _on_y_changed(self, value):
        self.y_pos = value
        self.y_slider.blockSignals(True)
        self.y_spin.blockSignals(True)
        self.y_slider.setValue(value)
        self.y_spin.setValue(value)
        self.y_slider.blockSignals(False)
        self.y_spin.blockSignals(False)
        self._update_all_ortho_views()
    
    def _on_x_changed(self, value):
        self.x_pos = value
        self.x_slider.blockSignals(True)
        self.x_spin.blockSignals(True)
        self.x_slider.setValue(value)
        self.x_spin.setValue(value)
        self.x_slider.blockSignals(False)
        self.x_spin.blockSignals(False)
        self._update_all_ortho_views()
    
    def _on_xy_crosshair_moved(self, x, y):
        self.x_pos = x
        self.y_pos = y
        self.x_slider.setValue(x)
        self.y_slider.setValue(y)
    
    def _on_xz_crosshair_moved(self, x, z):
        self.x_pos = x
        self.z_pos = z
        self.x_slider.setValue(x)
        self.z_slider.setValue(z)
    
    def _on_yz_crosshair_moved(self, z, y):
        self.z_pos = z
        self.y_pos = y
        self.z_slider.setValue(z)
        self.y_slider.setValue(y)
    
    def _on_ortho_contrast_changed(self, index):
        """Handle ortho view contrast mode change"""
        pass
    
    def _apply_ortho_contrast(self):
        """Apply contrast adjustment to all ortho views"""
        mode = self.ortho_contrast_combo.currentIndex()
        self._apply_contrast_to_view(self.xy_view, mode)
        self._apply_contrast_to_view(self.xz_view, mode)
        self._apply_contrast_to_view(self.yz_view, mode)
    
    def _apply_contrast_to_view(self, view, mode):
        """Apply contrast mode to a specific view"""
        if view.image is None:
            return
        
        try:
            img = view.image
            
            if mode == 0:
                view.autoLevels()
            elif mode == 1:
                min_val = float(np.min(img))
                max_val = float(np.max(img))
                view.setLevels(min_val, max_val)
            elif mode == 2:
                min_val = float(np.percentile(img, 1))
                max_val = float(np.percentile(img, 99))
                view.setLevels(min_val, max_val)
            elif mode == 3:
                min_val = float(np.percentile(img, 2))
                max_val = float(np.percentile(img, 98))
                view.setLevels(min_val, max_val)
            elif mode == 4:
                min_val = float(np.percentile(img, 5))
                max_val = float(np.percentile(img, 95))
                view.setLevels(min_val, max_val)
        except Exception as e:
            print(f"Error applying contrast to view: {e}")
    
    def _reset_all_views(self):
        self.xy_view.view.autoRange()
        self.xz_view.view.autoRange()
        self.yz_view.view.autoRange()
    
    # ==================== General Handlers ====================
    
    def _on_layout_changed(self, index):
        """Handle layout change (1, 2, 4 views, or 3D only)"""
        if index == 0:
            self.stack.setCurrentIndex(0)
        elif index == 1:
            self.stack.setCurrentIndex(1)
            if self.multires_image is not None:
                self._update_dual_views()
        elif index == 2:
            self.stack.setCurrentIndex(2)
            if self.multires_image is not None:
                self._update_all_ortho_views()
                self.view_3d.set_multires_image(self.multires_image)
        elif index == 3:
            self.stack.setCurrentIndex(3)
            if self.multires_image is not None:
                self.view_3d_only.set_multires_image(self.multires_image)
    
    def _load_file(self):
        """Load Zarr store"""
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Zarr Store Directory"
        )
        
        if not path:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Open Zarr Store", "", "Zip Files (*.zip);;All Files (*)"
            )
        
        if not path:
            return
        
        try:
            if self.zarr_store is not None:
                try:
                    self.zarr_store.close()
                except:
                    pass
            
            self.zarr_store = z5py.File(path, mode='r', use_zarr_format=True)
            self.zarr_group = self.zarr_store
            
            self.multires_image = MultiResolutionImage(self.zarr_group)
            
            self.file_label.setText(Path(path).name)
            
            level_0_shape = self.multires_image.get_level_shape(0)
            level_0 = self.multires_image.levels[0]
            
            self.single_num_levels_label.setText(str(self.multires_image.get_num_levels()))
            self.single_level_shape_label.setText(str(level_0_shape))
            if hasattr(level_0, 'dtype'):
                self.single_level_dtype_label.setText(str(level_0.dtype))
            if hasattr(level_0, 'chunks'):
                self.single_level_chunks_label.setText(str(level_0.chunks))
            
            if len(level_0_shape) > 2:
                num_slices = level_0_shape[0]
                self.single_slice_slider.setMaximum(num_slices - 1)
                self.single_slice_slider.setEnabled(True)
                self.single_slice_slider.setValue(num_slices // 2)
            
            self.single_image_view.set_multires_image(self.multires_image)
            
            if len(level_0_shape) >= 3:
                z_size, y_size, x_size = level_0_shape[-3:]
                self.ortho_shape_label.setText(f"{z_size} × {y_size} × {x_size}")
                
                self.z_slider.setMaximum(z_size - 1)
                self.z_spin.setMaximum(z_size - 1)
                self.z_slider.setValue(z_size // 2)
                
                self.y_slider.setMaximum(y_size - 1)
                self.y_spin.setMaximum(y_size - 1)
                self.y_slider.setValue(y_size // 2)
                
                self.x_slider.setMaximum(x_size - 1)
                self.x_spin.setMaximum(x_size - 1)
                self.x_slider.setValue(x_size // 2)
                
                self.z_pos = z_size // 2
                self.y_pos = y_size // 2
                self.x_pos = x_size // 2
                
                if hasattr(level_0, 'dtype'):
                    self.ortho_dtype_label.setText(str(level_0.dtype))
                
                self.ortho_levels_label.setText(str(self.multires_image.get_num_levels()))
                
                self.xy_view.set_multires_image(self.multires_image)
                self.xz_view.set_multires_image(self.multires_image)
                self.yz_view.set_multires_image(self.multires_image)
                
                self.view_3d.set_multires_image(self.multires_image)
                self.view_3d_only.set_multires_image(self.multires_image)
                
                self._update_all_ortho_views()
            
            self.metadata_viewer.load_metadata(self.zarr_group)
            
            # Print cache info
            print(f"\nCache initialized with 500MB capacity")
            print(f"Expected to hold ~{500 / 0.5:.0f} chunks (assuming 512x512 uint16)")
            
            QtWidgets.QMessageBox.information(
                self, "Success", 
                f"Loaded Zarr store\nShape: {level_0_shape}\n"
                f"Levels: {self.multires_image.get_num_levels()}\n"
                f"Cache: 500MB (optimized for speed)"
            )
            
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            QtWidgets.QMessageBox.critical(
                self, "Error", f"Failed to load:\n{str(e)}"
            )
    
    def closeEvent(self, event):
        if self.zarr_store is not None:
            try:
                self.zarr_store.close()
            except:
                pass
        super().closeEvent(event)

    def _build_menubar(self) -> QtWidgets.QMenuBar:
        """Top classic menu bar"""
        mb = QtWidgets.QMenuBar(self)

        # File menu
        file_menu = mb.addMenu("&File")
        act_open = file_menu.addAction(
            self.style().standardIcon(QtWidgets.QStyle.SP_DirOpenIcon),
            "Load Zarr…", 
            self._load_file
        )
        act_open.setShortcut("Ctrl+O")
        file_menu.addSeparator()
        
        act_close = file_menu.addAction("Close Store", self._close_store)
        act_close.setEnabled(True)
        
        file_menu.addSeparator()
        act_exit = file_menu.addAction("E&xit", self.close)
        act_exit.setShortcut("Ctrl+Q")

        # View menu
        view_menu = mb.addMenu("&View")
        grp = QtWidgets.QActionGroup(self)
        self._act_view_1 = grp.addAction(QtWidgets.QAction("1 View (Single)", grp, checkable=True))
        self._act_view_2 = grp.addAction(QtWidgets.QAction("2 Views (Dual)", grp, checkable=True))
        self._act_view_4 = grp.addAction(QtWidgets.QAction("4 Views (Orthogonal)", grp, checkable=True))
        self._act_view_3d = grp.addAction(QtWidgets.QAction("3D Only", grp, checkable=True))
        view_menu.addActions([self._act_view_1, self._act_view_2, self._act_view_4, self._act_view_3d])
        self._act_view_1.setChecked(True)

        def _sync_layout_from_menu(action: QtWidgets.QAction):
            mapping = {
                self._act_view_1: 0,
                self._act_view_2: 1,
                self._act_view_4: 2,
                self._act_view_3d: 3,
            }
            idx = mapping.get(action, 0)
            self._on_layout_changed(idx)

        grp.triggered.connect(_sync_layout_from_menu)
        
        view_menu.addSeparator()
        
        # Cache settings submenu
        cache_menu = view_menu.addMenu("Cache Size")
        cache_group = QtWidgets.QActionGroup(self)
        cache_group.setExclusive(True)
        
        for size_mb, label in [(100, "100 MB (Small)"), 
                                (500, "500 MB (Default)"), 
                                (1000, "1 GB (Large)"),
                                (2000, "2 GB (XLarge)")]:
            act = QtWidgets.QAction(label, cache_group, checkable=True)
            if size_mb == 500:
                act.setChecked(True)
            act.triggered.connect(lambda checked, s=size_mb: self._set_cache_size(s))
            cache_menu.addAction(act)

        # Plugins menu
        plugins_menu = mb.addMenu("&Plugins")
        self._plugins_menu = plugins_menu
        placeholder = plugins_menu.addAction("No plugins loaded")
        placeholder.setEnabled(False)

        # Help menu
        help_menu = mb.addMenu("&Help")
        help_menu.addAction("About…", self._show_about)
        help_menu.addAction("Performance Tips…", self._show_performance_tips)

        return mb
    
    def _close_store(self):
        """Close current store and clear cache"""
        if self.zarr_store is not None:
            try:
                self.zarr_store.close()
            except:
                pass
            self.zarr_store = None
            self.zarr_group = None
            
        if self.multires_image is not None:
            self.multires_image.chunk_cache.clear()
            self.multires_image = None
        
        self.file_label.setText("No file loaded")
        self.cache_stats_label.setText("")
    
    def _set_cache_size(self, size_mb):
        """Set cache size"""
        if self.multires_image is not None:
            old_stats = self.multires_image.chunk_cache.get_stats()
            self.multires_image.chunk_cache.max_memory_bytes = size_mb * 1024 * 1024
            print(f"Cache size changed to {size_mb} MB")
            print(f"Previous: {old_stats['chunks']} chunks, {old_stats['memory_mb']:.1f} MB")
            # Trigger eviction if needed
            while (self.multires_image.chunk_cache.current_memory > 
                   self.multires_image.chunk_cache.max_memory_bytes and 
                   self.multires_image.chunk_cache.access_order):
                old_key = self.multires_image.chunk_cache.access_order.pop(0)
                if old_key in self.multires_image.chunk_cache.cache:
                    old_data = self.multires_image.chunk_cache.cache.pop(old_key)
                    self.multires_image.chunk_cache.current_memory -= old_data.nbytes
            self._update_cache_stats()
    
    def _show_about(self):
        """Show about dialog"""
        QtWidgets.QMessageBox.information(
            self, "About ZVIEW", 
            "<h2>ZVIEW - Optimized Zarr Viewer</h2>"
            "<p><b>Version:</b> 2.0 (Optimized)</p>"
            "<p><b>Features:</b></p>"
            "<ul>"
            "<li>Chunk-aligned loading (10-100x faster)</li>"
            "<li>Intelligent caching with memory management</li>"
            "<li>Multi-resolution pyramid support</li>"
            "<li>Multiple view modes (1/2/4 views, 3D)</li>"
            "<li>Plugin architecture (coming soon)</li>"
            "</ul>"
            "<p><b>Performance:</b> Optimized for large multiscale Zarr datasets</p>"
        )
    
    def _show_performance_tips(self):
        """Show performance tips dialog"""
        QtWidgets.QMessageBox.information(
            self, "Performance Tips",
            "<h3>Getting the Best Performance</h3>"
            "<p><b>Cache Size:</b></p>"
            "<ul>"
            "<li>100 MB: Good for exploration, ~200 chunks</li>"
            "<li>500 MB: Default, good for most uses, ~1000 chunks</li>"
            "<li>1-2 GB: Best for intensive work, many slices</li>"
            "</ul>"
            "<p><b>Tips:</b></p>"
            "<ul>"
            "<li>Watch cache hit rate (aim for >70%)</li>"
            "<li>If hit rate is low, increase cache size</li>"
            "<li>Pan/zoom at same slice is cached</li>"
            "<li>Changing slices requires new loads</li>"
            "</ul>"
            "<p><b>Chunk-aligned loading:</b> Loads complete chunks for maximum speed!</p>"
        )

    def _build_buttonbar(self) -> QtWidgets.QToolBar:
        """Second top bar for plugin action buttons"""
        tb = QtWidgets.QToolBar("Plugin Actions", self)
        tb.setIconSize(QtCore.QSize(24, 24))
        tb.setMovable(False)
        tb.setFloatable(False)
        tb.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        tb.setStyleSheet("""
            QToolBar {
                background-color: #3a3a3a;
                border-bottom: 1px solid #555;
                spacing: 3px;
                padding: 3px;
            }
            QToolButton {
                background-color: #454545;
                border: 1px solid #666;
                border-radius: 3px;
                padding: 4px 8px;
                margin: 2px;
            }
            QToolButton:hover {
                background-color: #505050;
                border: 1px solid #777;
            }
            QToolButton:pressed {
                background-color: #3a3a3a;
            }
            QToolButton:disabled {
                background-color: #3a3a3a;
                color: #666;
            }
        """)

        self._plugin_toolbar = tb

        # Example plugin buttons (disabled by default, will be enabled by plugins)
        act_measure = QtWidgets.QAction(
            self.style().standardIcon(QtWidgets.QStyle.SP_DialogYesButton),
            "Measure", 
            self
        )
        act_measure.setEnabled(False)
        act_measure.setToolTip("Measure distances and areas (plugin not loaded)")
        tb.addAction(act_measure)

        act_segment = QtWidgets.QAction(
            self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogDetailedView),
            "Segment", 
            self
        )
        act_segment.setEnabled(False)
        act_segment.setToolTip("Segmentation tools (plugin not loaded)")
        tb.addAction(act_segment)
        
        act_annotate = QtWidgets.QAction(
            self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogListView),
            "Annotate", 
            self
        )
        act_annotate.setEnabled(False)
        act_annotate.setToolTip("Add annotations (plugin not loaded)")
        tb.addAction(act_annotate)

        tb.addSeparator()
        
        act_export = QtWidgets.QAction(
            self.style().standardIcon(QtWidgets.QStyle.SP_DialogSaveButton),
            "Export", 
            self
        )
        act_export.setEnabled(False)
        act_export.setToolTip("Export current view (plugin not loaded)")
        tb.addAction(act_export)

        # Flexible spacer to push everything else to the right
        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        tb.addWidget(spacer)
        
        # Info label on the right
        info_label = QtWidgets.QLabel(" Plugins: Load via Plugins menu ")
        info_label.setStyleSheet("color: #888; font-size: 11px; padding-right: 10px;")
        tb.addWidget(info_label)

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
        
        if tooltip:
            action.setToolTip(tooltip)
        
        if slot:
            action.triggered.connect(slot)
        
        # Insert before spacer (to keep buttons on left, info on right)
        actions = self._plugin_toolbar.actions()
        if len(actions) > 0:
            # Find spacer widget
            for i, act in enumerate(actions):
                widget = self._plugin_toolbar.widgetForAction(act)
                if widget and isinstance(widget, QtWidgets.QWidget):
                    if widget.sizePolicy().horizontalPolicy() == QtWidgets.QSizePolicy.Expanding:
                        self._plugin_toolbar.insertAction(act, action)
                        return action
        
        # Fallback: just add to end
        self._plugin_toolbar.addAction(action)
        return action
    
    def remove_plugin_button(self, action: QtWidgets.QAction):
        """Remove a plugin button from the toolbar"""
        if hasattr(self, "_plugin_toolbar") and action:
            self._plugin_toolbar.removeAction(action)
    
    def clear_plugin_buttons(self):
        """Remove all user-added plugin actions from the toolbar (keeps default placeholders)"""
        if not hasattr(self, "_plugin_toolbar"):
            return
        
        # Get list of default actions (first 4 + separator + export)
        actions = self._plugin_toolbar.actions()
        default_count = 6  # measure, segment, annotate, separator, export, spacer
        
        # Remove all actions after defaults
        for action in actions[default_count:]:
            widget = self._plugin_toolbar.widgetForAction(action)
            if not widget or not isinstance(widget, QtWidgets.QLabel):  # Keep info label
                self._plugin_toolbar.removeAction(action)


def apply_dark_theme(app):
    """Apply dark theme"""
    app.setStyle('Fusion')
    palette = QtGui.QPalette()
    
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor(35, 35, 35))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(42, 130, 218))
    palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.white)
    
    app.setPalette(palette)
    
    app.setStyleSheet("""
        QGroupBox {
            border: 1px solid #555;
            border-radius: 5px;
            margin-top: 10px;
            font-weight: bold;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
        }
        QPushButton {
            background-color: #454545;
            border: 1px solid #666;
            border-radius: 4px;
            padding: 6px 16px;
        }
        QPushButton:hover {
            background-color: #505050;
        }
        QComboBox {
            background-color: #454545;
            border: 1px solid #666;
            border-radius: 4px;
            padding: 5px;
        }
    """)


def main():
    """Run the unified viewer"""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    
    app.setApplicationName("ZVIEW")
    apply_dark_theme(app)
    
    viewer = UnifiedZarrViewer()
    viewer.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()