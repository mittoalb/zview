#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VisPy 3D Volume Viewer Module
------------------------------
Enhanced 3D visualization widget with advanced controls for transparency,
colormaps, intensity ranges, and more.

Usage:
    from vispy_viewer import VisPy3DView, VISPY_AVAILABLE
"""

import numpy as np
from PyQt5 import QtWidgets, QtCore

# Try to import VisPy
try:
    from vispy import scene
    from vispy.color import get_colormap, ColorArray
    VISPY_AVAILABLE = True
except ImportError:
    VISPY_AVAILABLE = False


class VisPy3DView(QtWidgets.QWidget):
    """Ultra-fast 3D visualization using VisPy with advanced controls"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.multires_image = None
        self.volume_data = None
        self.canvas = None
        self.volume_visual = None
        self.view = None
        self._build_ui()
    
    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        if not VISPY_AVAILABLE:
            error_label = QtWidgets.QLabel(
                "⚠️ 3D Visualization Not Available\n\n"
                "VisPy library is required for 3D visualization.\n\n"
                "Install with:\n"
                "pip install vispy PyOpenGL PyOpenGL_accelerate"
            )
            error_label.setAlignment(QtCore.Qt.AlignCenter)
            error_label.setStyleSheet("font-size: 14px; color: #f88; padding: 40px;")
            layout.addWidget(error_label)
            return
        
        # Control panel
        controls = QtWidgets.QGroupBox("3D Visualization Controls")
        controls_layout = QtWidgets.QVBoxLayout()
        
        # === Render Mode ===
        mode_layout = QtWidgets.QHBoxLayout()
        mode_layout.addWidget(QtWidgets.QLabel("Render Mode:"))
        
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems([
            "Volume (MIP)",
            "Volume (Translucent)",
            "Volume (Additive)",
            "Isosurface"
        ])
        self.mode_combo.setMinimumWidth(180)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(self.mode_combo)
        mode_layout.addStretch()
        controls_layout.addLayout(mode_layout)
        
        # === Colormap Selection ===
        cmap_layout = QtWidgets.QHBoxLayout()
        cmap_layout.addWidget(QtWidgets.QLabel("Colormap:"))
        
        self.cmap_combo = QtWidgets.QComboBox()
        # Safe colormaps that work in all VisPy versions
        self.cmap_combo.addItems([
            "grays",
            "hot",
            "cool",
            "spring",
            "summer",
            "autumn",
            "winter",
            "fire",
            "ice"
        ])
        self.cmap_combo.currentIndexChanged.connect(self._on_colormap_changed)
        cmap_layout.addWidget(self.cmap_combo)
        cmap_layout.addStretch()
        controls_layout.addLayout(cmap_layout)
        
        # === Transparency Control ===
        alpha_layout = QtWidgets.QHBoxLayout()
        alpha_layout.addWidget(QtWidgets.QLabel("Opacity:"))
        self.alpha_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.alpha_slider.setRange(1, 100)
        self.alpha_slider.setValue(100)
        self.alpha_slider.valueChanged.connect(self._on_alpha_changed)
        alpha_layout.addWidget(self.alpha_slider)
        self.alpha_label = QtWidgets.QLabel("100%")
        self.alpha_label.setMinimumWidth(45)
        alpha_layout.addWidget(self.alpha_label)
        controls_layout.addLayout(alpha_layout)
        
        # === Intensity Range Controls ===
        intensity_group = QtWidgets.QGroupBox("Intensity Range")
        intensity_layout = QtWidgets.QVBoxLayout()
        
        # Min intensity
        min_layout = QtWidgets.QHBoxLayout()
        min_layout.addWidget(QtWidgets.QLabel("Min:"))
        self.intensity_min_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.intensity_min_slider.setRange(0, 100)
        self.intensity_min_slider.setValue(0)
        self.intensity_min_slider.valueChanged.connect(self._on_intensity_changed)
        min_layout.addWidget(self.intensity_min_slider)
        self.intensity_min_label = QtWidgets.QLabel("0%")
        self.intensity_min_label.setMinimumWidth(45)
        min_layout.addWidget(self.intensity_min_label)
        intensity_layout.addLayout(min_layout)
        
        # Max intensity
        max_layout = QtWidgets.QHBoxLayout()
        max_layout.addWidget(QtWidgets.QLabel("Max:"))
        self.intensity_max_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.intensity_max_slider.setRange(0, 100)
        self.intensity_max_slider.setValue(100)
        self.intensity_max_slider.valueChanged.connect(self._on_intensity_changed)
        max_layout.addWidget(self.intensity_max_slider)
        self.intensity_max_label = QtWidgets.QLabel("100%")
        self.intensity_max_label.setMinimumWidth(45)
        max_layout.addWidget(self.intensity_max_label)
        intensity_layout.addLayout(max_layout)
        
        # Auto adjust button
        auto_intensity_btn = QtWidgets.QPushButton("Auto Adjust")
        auto_intensity_btn.clicked.connect(self._auto_adjust_intensity)
        intensity_layout.addWidget(auto_intensity_btn)
        
        intensity_group.setLayout(intensity_layout)
        controls_layout.addWidget(intensity_group)
        
        # === Threshold (for isosurface and volume) ===
        threshold_layout = QtWidgets.QHBoxLayout()
        threshold_layout.addWidget(QtWidgets.QLabel("Threshold:"))
        self.threshold_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(50)
        self.threshold_slider.valueChanged.connect(self._on_threshold_changed)
        threshold_layout.addWidget(self.threshold_slider)
        self.threshold_label = QtWidgets.QLabel("50%")
        self.threshold_label.setMinimumWidth(45)
        threshold_layout.addWidget(self.threshold_label)
        controls_layout.addLayout(threshold_layout)
        
        # === Background Color ===
        bg_layout = QtWidgets.QHBoxLayout()
        bg_layout.addWidget(QtWidgets.QLabel("Background:"))
        
        self.bg_combo = QtWidgets.QComboBox()
        self.bg_combo.addItems(["Black", "White", "Gray", "Dark Blue"])
        self.bg_combo.currentIndexChanged.connect(self._on_background_changed)
        bg_layout.addWidget(self.bg_combo)
        bg_layout.addStretch()
        controls_layout.addLayout(bg_layout)
        
        # === Resolution and Data Loading ===
        res_group = QtWidgets.QGroupBox("Data Resolution")
        res_layout = QtWidgets.QVBoxLayout()
        
        level_layout = QtWidgets.QHBoxLayout()
        level_layout.addWidget(QtWidgets.QLabel("Pyramid Level:"))
        self.level_spin = QtWidgets.QSpinBox()
        self.level_spin.setMinimum(0)
        self.level_spin.setValue(0)
        self.level_spin.setToolTip("0 = highest resolution")
        level_layout.addWidget(self.level_spin)
        level_layout.addStretch()
        res_layout.addLayout(level_layout)
        
        down_layout = QtWidgets.QHBoxLayout()
        down_layout.addWidget(QtWidgets.QLabel("Downsample:"))
        self.downsample_spin = QtWidgets.QSpinBox()
        self.downsample_spin.setRange(1, 8)
        self.downsample_spin.setValue(1)
        self.downsample_spin.setToolTip("Additional downsampling (1 = none)")
        down_layout.addWidget(self.downsample_spin)
        down_layout.addStretch()
        res_layout.addLayout(down_layout)
        
        res_group.setLayout(res_layout)
        controls_layout.addWidget(res_group)
        
        # === Action Buttons ===
        button_layout = QtWidgets.QHBoxLayout()
        
        self.render_btn = QtWidgets.QPushButton("🚀 Render 3D View")
        self.render_btn.clicked.connect(self._render_3d)
        self.render_btn.setEnabled(False)
        self.render_btn.setMinimumHeight(40)
        self.render_btn.setStyleSheet(
            "background-color: #2a5; font-weight: bold; font-size: 13px;"
        )
        button_layout.addWidget(self.render_btn)
        
        self.update_btn = QtWidgets.QPushButton("Update View")
        self.update_btn.clicked.connect(self._update_render_mode)
        self.update_btn.setEnabled(False)
        self.update_btn.setToolTip("Update rendering without reloading data")
        button_layout.addWidget(self.update_btn)
        
        self.clear_btn = QtWidgets.QPushButton("Clear")
        self.clear_btn.clicked.connect(self._clear_view)
        self.clear_btn.setEnabled(False)
        button_layout.addWidget(self.clear_btn)
        
        controls_layout.addLayout(button_layout)
        
        controls.setLayout(controls_layout)
        layout.addWidget(controls)
        
        # VisPy canvas container (lazy init)
        self.canvas_container = QtWidgets.QWidget()
        self.canvas_container.setMinimumHeight(400)
        layout.addWidget(self.canvas_container, stretch=1)
        
        # Info label
        self.info_label = QtWidgets.QLabel("Load a volume to begin")
        self.info_label.setStyleSheet(
            "background-color: rgba(40, 40, 40, 200); "
            "color: white; padding: 8px; font-size: 11px;"
        )
        layout.addWidget(self.info_label)
    
    def _initialize_canvas(self):
        """Lazy initialization of VisPy canvas"""
        if self.canvas is not None:
            return
        
        try:
            # Create VisPy canvas with Qt backend
            self.canvas = scene.SceneCanvas(keys='interactive', show=False, parent=self.canvas_container)
            self.canvas.native.setParent(self.canvas_container)
            
            # Layout canvas in container
            canvas_layout = QtWidgets.QVBoxLayout(self.canvas_container)
            canvas_layout.setContentsMargins(0, 0, 0, 0)
            canvas_layout.addWidget(self.canvas.native)
            
            # Create view
            self.view = self.canvas.central_widget.add_view()
            self.view.camera = 'turntable'
            self.view.camera.fov = 60
            self.view.camera.distance = 500
            
            # Set initial background
            self.canvas.bgcolor = 'black'
            
            print("✓ VisPy canvas initialized")
        except Exception as e:
            print(f"Failed to initialize VisPy canvas: {e}")
            import traceback
            traceback.print_exc()
    
    def set_multires_image(self, multires_image):
        """Set the multi-resolution image"""
        self.multires_image = multires_image
        
        if not VISPY_AVAILABLE:
            return
        
        # Update level spinner maximum
        num_levels = multires_image.get_num_levels()
        self.level_spin.setMaximum(num_levels - 1)
        
        # Get volume info
        level_0_shape = multires_image.get_level_shape(0)
        if level_0_shape and len(level_0_shape) >= 3:
            z, y, x = level_0_shape[-3:]
            total_voxels = z * y * x
            
            # Auto-suggest settings based on size
            if total_voxels > 100_000_000:  # 100M voxels
                self.level_spin.setValue(min(1, num_levels - 1))
                self.downsample_spin.setValue(2)
                recommendation = "Large volume - using level 1 + 2x downsample"
            elif total_voxels > 50_000_000:  # 50M voxels
                self.downsample_spin.setValue(2)
                recommendation = "Using 2x downsample for performance"
            else:
                recommendation = "Volume size OK for full resolution"
            
            self.info_label.setText(
                f"Shape: {z}×{y}×{x} | Voxels: {total_voxels:,} | "
                f"Levels: {num_levels} | {recommendation}"
            )
            
            self.render_btn.setEnabled(True)
    
    def _on_mode_changed(self, index):
        """Handle mode change"""
        if self.volume_data is not None:
            self._update_render_mode()
    
    def _on_colormap_changed(self, index):
        """Handle colormap change"""
        if self.volume_visual is not None:
            cmap_name = self.cmap_combo.currentText()
            
            mode_idx = self.mode_combo.currentIndex()
            if mode_idx in [0, 1, 2]:  # Volume rendering
                try:
                    self.volume_visual.cmap = cmap_name
                    self.canvas.update()
                except Exception as e:
                    print(f"Failed to set colormap '{cmap_name}': {e}")
                    # Fall back to grays
                    try:
                        self.volume_visual.cmap = 'grays'
                        self.canvas.update()
                        self.cmap_combo.blockSignals(True)
                        self.cmap_combo.setCurrentText('grays')
                        self.cmap_combo.blockSignals(False)
                    except:
                        pass
            elif mode_idx == 3:  # Isosurface
                self._update_render_mode()
    
    def _on_alpha_changed(self, value):
        """Handle opacity slider change"""
        self.alpha_label.setText(f"{value}%")
        if self.volume_data is not None and self.volume_visual is not None:
            self._update_render_mode()
    
    def _on_intensity_changed(self):
        """Handle intensity range slider changes"""
        min_val = self.intensity_min_slider.value()
        max_val = self.intensity_max_slider.value()
        
        # Ensure min < max
        if min_val >= max_val:
            if self.sender() == self.intensity_min_slider:
                min_val = max_val - 1
                self.intensity_min_slider.blockSignals(True)
                self.intensity_min_slider.setValue(min_val)
                self.intensity_min_slider.blockSignals(False)
            else:
                max_val = min_val + 1
                self.intensity_max_slider.blockSignals(True)
                self.intensity_max_slider.setValue(max_val)
                self.intensity_max_slider.blockSignals(False)
        
        self.intensity_min_label.setText(f"{min_val}%")
        self.intensity_max_label.setText(f"{max_val}%")
        
        if self.volume_visual is not None:
            # Update intensity window
            min_norm = min_val / 100.0
            max_norm = max_val / 100.0
            
            if self.mode_combo.currentIndex() in [0, 1, 2]:  # Volume modes
                try:
                    self.volume_visual.clim = (min_norm, max_norm)
                    self.canvas.update()
                except Exception as e:
                    print(f"Error setting intensity range: {e}")
    
    def _auto_adjust_intensity(self):
        """Auto-adjust intensity range based on percentiles (robust, fast)."""
        if self.volume_data is None:
            return
        try:
            # work in float, optionally sample if huge for speed
            v = np.asarray(self.volume_data, dtype=np.float32)
            if v.size > 8_000_000:
                # sample 1M voxels for speed
                idx = np.random.default_rng(0).choice(v.size, size=1_000_000, replace=False)
                sample = v.reshape(-1)[idx]
            else:
                sample = v

            # robust percentiles
            p1  = float(np.percentile(sample, 1))
            p99 = float(np.percentile(sample, 99))

            data_min = float(np.nanmin(v))
            data_max = float(np.nanmax(v))
            rng = data_max - data_min

            if rng > 0 and np.isfinite(rng):
                min_percent = int(np.clip(((p1  - data_min) / rng) * 100.0, 0, 100))
                max_percent = int(np.clip(((p99 - data_min) / rng) * 100.0, 0, 100))
                if max_percent <= min_percent:
                    max_percent = min(100, min_percent + 1)

                self.intensity_min_slider.setValue(min_percent)
                self.intensity_max_slider.setValue(max_percent)
        except Exception as e:
            print(f"Error auto-adjusting intensity: {e}")
    
    def _on_threshold_changed(self, value):
        """Handle threshold slider change"""
        self.threshold_label.setText(f"{value}%")
        if self.volume_data is not None:
            self._update_render_mode()
    
    def _on_background_changed(self, index):
        """Handle background color change"""
        if self.canvas is None:
            return
        
        bg_colors = {
            0: 'black',
            1: 'white',
            2: (0.3, 0.3, 0.3, 1.0),
            3: (0.1, 0.1, 0.2, 1.0)
        }
        
        self.canvas.bgcolor = bg_colors.get(index, 'black')
        self.canvas.update()
    
    def _render_3d(self):
        """Render the 3D volume using VisPy"""
        if not VISPY_AVAILABLE or self.multires_image is None:
            return
        
        try:
            # Lazy init canvas
            if self.canvas is None:
                self._initialize_canvas()
                if self.canvas is None:
                    return
            
            # Get parameters
            level = self.level_spin.value()
            downsample = self.downsample_spin.value()
            
            # Load volume
            self.info_label.setText("Loading volume...")
            QtWidgets.QApplication.processEvents()
            
            self.volume_data = self.multires_image.get_full_volume(level, downsample)
            
            if self.volume_data is None:
                QtWidgets.QMessageBox.critical(
                    self, "Error", "Failed to load volume data"
                )
                return
            
            self.info_label.setText("Rendering...")
            QtWidgets.QApplication.processEvents()
            
            # Create visualization
            self._create_volume_visual()
            
            self.clear_btn.setEnabled(True)
            self.update_btn.setEnabled(True)
            
            # Enable auto-adjust
            self._auto_adjust_intensity()
            
            self.info_label.setText(
                f"✓ Rendered | Shape: {self.volume_data.shape} | "
                f"Mouse: Drag=Rotate, Wheel=Zoom, Right-drag=Pan"
            )
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Rendering Error",
                f"Failed to render 3D volume:\n{str(e)}"
            )
            import traceback
            traceback.print_exc()
            self.info_label.setText(f"✗ Error: {str(e)}")
    
    def _create_volume_visual(self):
        """Create VisPy volume visual with current settings"""
        # Clear existing visuals
        if self.volume_visual is not None:
            self.volume_visual.parent = None
            self.volume_visual = None
        
        # Normalize volume data (robust, no overflow)
        v = np.asarray(self.volume_data, dtype=np.float32)  # upcast once, no copy if already float
        vol_min = float(np.nanmin(v))
        vol_max = float(np.nanmax(v))
        rng = vol_max - vol_min

        if not np.isfinite(rng) or rng <= 0.0:
            vol_normalized = np.zeros_like(v, dtype=np.float32)
        else:
            # do math in float32 explicitly; clip to [0,1]
            vol_normalized = np.subtract(v, vol_min, dtype=np.float32)
            vol_normalized /= rng
            np.clip(vol_normalized, 0.0, 1.0, out=vol_normalized)

        # Apply threshold (works on float32 safely)
        threshold = self.threshold_slider.value() / 100.0
        if threshold > 0:
            vol_normalized[vol_normalized < threshold] = 0.0

        
        # Apply threshold
        threshold = self.threshold_slider.value() / 100.0
        vol_normalized[vol_normalized < threshold] = 0
        
        mode_idx = self.mode_combo.currentIndex()
        cmap_name = self.cmap_combo.currentText()
        
        if mode_idx in [0, 1, 2]:  # Volume rendering modes
            # Create volume
            self.volume_visual = scene.visuals.Volume(
                vol_normalized,
                parent=self.view.scene,
            )
            
            # Set rendering method
            if mode_idx == 0:  # MIP
                self.volume_visual.method = 'mip'
            elif mode_idx == 1:  # Translucent
                self.volume_visual.method = 'translucent'
            elif mode_idx == 2:  # Additive
                self.volume_visual.method = 'additive'
            
            # Set colormap
            try:
                self.volume_visual.cmap = cmap_name
            except Exception as e:
                print(f"Failed to set colormap '{cmap_name}': {e}, using grays")
                self.volume_visual.cmap = 'grays'
            
            # Set intensity range
            min_intensity = self.intensity_min_slider.value() / 100.0
            max_intensity = self.intensity_max_slider.value() / 100.0
            try:
                self.volume_visual.clim = (min_intensity, max_intensity)
            except Exception as e:
                print(f"Failed to set intensity range: {e}")
            
            # Center the volume
            self.volume_visual.transform = scene.STTransform(
                translate=(-vol_normalized.shape[0] / 2,
                          -vol_normalized.shape[1] / 2,
                          -vol_normalized.shape[2] / 2)
            )
            
        elif mode_idx == 3:  # Isosurface
            from vispy.scene.visuals import Isosurface
            
            # Create isosurface visual
            self.volume_visual = Isosurface(
                vol_normalized,
                level=threshold,
                parent=self.view.scene
            )
            
            # Set color based on colormap
            try:
                cm = get_colormap(cmap_name)
                color = cm[0.5].rgba
            except Exception as e:
                print(f"Failed to get colormap '{cmap_name}': {e}, using gray")
                color = (0.8, 0.8, 0.8, 1.0)
            
            # Apply opacity
            alpha = self.alpha_slider.value() / 100.0
            color = (*color[:3], alpha)
            
            if hasattr(self.volume_visual, 'color'):
                self.volume_visual.color = color
            
            # Center the isosurface
            self.volume_visual.transform = scene.STTransform(
                translate=(-vol_normalized.shape[0] / 2,
                          -vol_normalized.shape[1] / 2,
                          -vol_normalized.shape[2] / 2)
            )
        
        # Reset camera
        self.view.camera.set_range()
        
        print(f"✓ VisPy rendered: {self.mode_combo.currentText()}")
    
    def _update_render_mode(self):
        """Update render mode without reloading data"""
        if self.volume_data is not None and self.canvas is not None:
            self._create_volume_visual()
            self.info_label.setText(
                f"✓ Updated | {self.mode_combo.currentText()} | {self.cmap_combo.currentText()}"
            )
    
    def _clear_view(self):
        """Clear the 3D view"""
        if self.volume_visual is not None:
            self.volume_visual.parent = None
            self.volume_visual = None
        
        if self.canvas is not None:
            self.canvas.update()
        
        self.volume_data = None
        self.info_label.setText("View cleared")
        self.clear_btn.setEnabled(False)
        self.update_btn.setEnabled(False)
    
    def closeEvent(self, event):
        """Clean up when closing"""
        if self.canvas is not None:
            try:
                self.canvas.close()
            except:
                pass
        super().closeEvent(event)
