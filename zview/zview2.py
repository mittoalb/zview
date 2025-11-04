#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Multi-Resolution Zarr Viewer with Enhanced Contrast Controls - OPTIMIZED
----------------------------------------------------------------------------------
Performance improvements:
- Chunk-aligned loading for maximum speed (10-100x faster)
- Intelligent caching with memory management
- Reduced redundant loads

Features:
- Single view mode: Full-screen viewer with comprehensive contrast controls
- Orthogonal mode: 4-quadrant XY/XZ/YZ + 3D view with contrast controls
- Dynamic resolution switching
- Metadata viewer
- Toggle between modes
- Enhanced contrast adjustment (auto, percentile, manual)
"""

import sys
import z5py
import numpy as np
from typing import Optional, List, Tuple
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
from pathlib import Path
import json
import threading

# Set pyqtgraph configuration
pg.setConfigOptions(imageAxisOrder='row-major')

# Import custom components
from sview import DynamicImageView
from oview import OrthoImageView
from meta import MetadataViewer, ZarrMetadataExtractor
from msres import MultiResolutionImage

# 3D Viewer (optional)
try:
    from view3d import VisPy3DView, VISPY_AVAILABLE
except ImportError:
    print("Warning: view3d module not found. 3D viewing will be disabled.")
    VISPY_AVAILABLE = False
    
    class VisPy3DView(QtWidgets.QWidget):
        """Placeholder for 3D view when vispy is not available"""
        def __init__(self, parent=None):
            super().__init__(parent)
            layout = QtWidgets.QVBoxLayout(self)
            label = QtWidgets.QLabel("3D View Not Available\n\nInstall VisPy to enable 3D visualization")
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
        
        self.setWindowTitle("Multiscale Zarr Viewer - OPTIMIZED")
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
        
        # Main vertical splitter for left/right
        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        
        # Left side - vertical splitter for XY and YZ
        left_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        
        # XY view with label
        xy_container = QtWidgets.QWidget()
        xy_layout = QtWidgets.QVBoxLayout(xy_container)
        xy_layout.setContentsMargins(0, 0, 0, 0)
        xy_layout.setSpacing(2)
        
        xy_label = QtWidgets.QLabel("XY (Axial)")
        xy_label.setStyleSheet("background-color: #2a2a2a; padding: 3px; font-weight: bold;")
        xy_label.setAlignment(QtCore.Qt.AlignCenter)
        xy_layout.addWidget(xy_label)
        
        self.xy_view = OrthoImageView('XY')
        self.xy_view.ui.roiBtn.hide()
        self.xy_view.ui.menuBtn.hide()
        self.xy_view.crosshairMoved.connect(self._on_xy_crosshair_moved)
        xy_layout.addWidget(self.xy_view)
        
        # YZ view with label
        yz_container = QtWidgets.QWidget()
        yz_layout = QtWidgets.QVBoxLayout(yz_container)
        yz_layout.setContentsMargins(0, 0, 0, 0)
        yz_layout.setSpacing(2)
        
        yz_label = QtWidgets.QLabel("YZ (Sagittal)")
        yz_label.setStyleSheet("background-color: #2a2a2a; padding: 3px; font-weight: bold;")
        yz_label.setAlignment(QtCore.Qt.AlignCenter)
        yz_layout.addWidget(yz_label)
        
        self.yz_view = OrthoImageView('YZ')
        self.yz_view.ui.roiBtn.hide()
        self.yz_view.ui.menuBtn.hide()
        self.yz_view.crosshairMoved.connect(self._on_yz_crosshair_moved)
        yz_layout.addWidget(self.yz_view)
        
        left_splitter.addWidget(xy_container)
        left_splitter.addWidget(yz_container)
        left_splitter.setStretchFactor(0, 1)
        left_splitter.setStretchFactor(1, 1)
        
        # Right side - vertical splitter for XZ and 3D
        right_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        
        # XZ view with label
        xz_container = QtWidgets.QWidget()
        xz_layout = QtWidgets.QVBoxLayout(xz_container)
        xz_layout.setContentsMargins(0, 0, 0, 0)
        xz_layout.setSpacing(2)
        
        xz_label = QtWidgets.QLabel("XZ (Coronal)")
        xz_label.setStyleSheet("background-color: #2a2a2a; padding: 3px; font-weight: bold;")
        xz_label.setAlignment(QtCore.Qt.AlignCenter)
        xz_layout.addWidget(xz_label)
        
        self.xz_view = OrthoImageView('XZ')
        self.xz_view.ui.roiBtn.hide()
        self.xz_view.ui.menuBtn.hide()
        self.xz_view.crosshairMoved.connect(self._on_xz_crosshair_moved)
        xz_layout.addWidget(self.xz_view)
        
        # 3D view with label
        view_3d_container = QtWidgets.QWidget()
        view_3d_layout = QtWidgets.QVBoxLayout(view_3d_container)
        view_3d_layout.setContentsMargins(0, 0, 0, 0)
        view_3d_layout.setSpacing(2)
        
        view_3d_label = QtWidgets.QLabel("3D Volume (VisPy)")
        view_3d_label.setStyleSheet("background-color: #2a2a2a; padding: 3px; font-weight: bold;")
        view_3d_label.setAlignment(QtCore.Qt.AlignCenter)
        view_3d_layout.addWidget(view_3d_label)
        
        self.view_3d = VisPy3DView()
        view_3d_layout.addWidget(self.view_3d)
        
        right_splitter.addWidget(xz_container)
        right_splitter.addWidget(view_3d_container)
        right_splitter.setStretchFactor(0, 1)
        right_splitter.setStretchFactor(1, 1)
        
        # Add left and right to main splitter
        main_splitter.addWidget(left_splitter)
        main_splitter.addWidget(right_splitter)
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 1)
        
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
    
    app.setApplicationName("ZVIEW - OPTIMIZED")
    apply_dark_theme(app)
    
    viewer = UnifiedZarrViewer()
    viewer.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()