#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Multi-Resolution Zarr Viewer with Enhanced Contrast Controls
---------------------------------------------------------------------
Combines single view and orthogonal (4-quadrant) view modes with toggle.

Features:
- Single view mode: Full-screen viewer with comprehensive contrast controls
- Orthogonal mode: 4-quadrant XY/XZ/YZ + 3D view with contrast controls
- Dynamic resolution switching
- Metadata viewer
- Toggle between modes
- Enhanced contrast adjustment (auto, percentile, manual)
- FIXED: Properly fills field of view at all resolution levels
- FIXED: 3D viewer properly embedded in GUI (no external windows)
"""

import z5py
import numpy as np
from typing import Optional, List, Tuple
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
from pathlib import Path
import json

#pyqtgraph config
pg.setConfigOptions(imageAxisOrder='row-major')

# 3D Viewer
from view3d import VisPy3DView, VISPY_AVAILABLE



# ==================== Metadata Classes ====================

class ZarrMetadataExtractor:
    """Extract metadata from Zarr files (z5py compatible)"""
    
    @staticmethod
    def extract_metadata(zarr_group):
        """Extract metadata from Zarr group and arrays"""
        metadata = []
        
        if hasattr(zarr_group, 'attrs'):
            try:
                attrs_dict = dict(zarr_group.attrs)
                for key, value in attrs_dict.items():
                    value_str = json.dumps(value, indent=2) if isinstance(value, (dict, list)) else str(value)
                    if len(value_str) > 500:
                        value_str = value_str[:500] + "..."
                    metadata.append((f"/.zattrs/{key}", value_str, type(value).__name__))
            except:
                pass
        
        def visit_items(name, obj):
            if hasattr(obj, 'shape') and hasattr(obj, 'dtype'):
                if hasattr(obj, 'attrs'):
                    try:
                        attrs_dict = dict(obj.attrs)
                        for key, value in attrs_dict.items():
                            value_str = json.dumps(value, indent=2) if isinstance(value, (dict, list)) else str(value)
                            if len(value_str) > 500:
                                value_str = value_str[:500] + "..."
                            metadata.append((f"/{name}/.zattrs/{key}", value_str, type(value).__name__))
                    except:
                        pass
                
                metadata.append((f"/{name}/shape", str(obj.shape), "tuple"))
                metadata.append((f"/{name}/dtype", str(obj.dtype), "dtype"))
                if hasattr(obj, 'chunks'):
                    metadata.append((f"/{name}/chunks", str(obj.chunks), "tuple"))
                if hasattr(obj, 'compression'):
                    metadata.append((f"/{name}/compression", str(obj.compression), "str"))
        
        try:
            zarr_group.visititems(visit_items)
        except AttributeError:
            for key in zarr_group.keys():
                try:
                    obj = zarr_group[key]
                    visit_items(key, obj)
                except:
                    pass
        
        return metadata
    
    @staticmethod
    def extract_tree_structure(zarr_group):
        """Extract tree structure of Zarr hierarchy"""
        structure = []
        
        def visit_items(name, obj):
            if hasattr(obj, 'shape') and hasattr(obj, 'dtype'):
                structure.append((name, 'Dataset', obj.shape, obj.dtype))
            elif hasattr(obj, 'keys'):
                structure.append((name, 'Group', None, None))
        
        try:
            zarr_group.visititems(visit_items)
        except AttributeError:
            for key in zarr_group.keys():
                try:
                    obj = zarr_group[key]
                    visit_items(key, obj)
                except:
                    pass
        
        return structure


class MetadataViewer(QtWidgets.QWidget):
    """Widget for displaying Zarr metadata"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
    
    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        
        self.tab_widget = QtWidgets.QTabWidget()
        
        # Attributes tab
        metadata_widget = QtWidgets.QWidget()
        metadata_layout = QtWidgets.QVBoxLayout(metadata_widget)
        
        filter_layout = QtWidgets.QHBoxLayout()
        filter_layout.addWidget(QtWidgets.QLabel("Filter:"))
        self.filter_input = QtWidgets.QLineEdit()
        self.filter_input.setPlaceholderText("Type to filter...")
        self.filter_input.textChanged.connect(self._filter_metadata)
        filter_layout.addWidget(self.filter_input)
        metadata_layout.addLayout(filter_layout)
        
        self.metadata_table = QtWidgets.QTableWidget()
        self.metadata_table.setColumnCount(3)
        self.metadata_table.setHorizontalHeaderLabels(['Path/Attribute', 'Value', 'Type'])
        self.metadata_table.horizontalHeader().setStretchLastSection(False)
        self.metadata_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.metadata_table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Interactive)
        self.metadata_table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        self.metadata_table.setAlternatingRowColors(True)
        self.metadata_table.setSortingEnabled(True)
        metadata_layout.addWidget(self.metadata_table)
        
        self.tab_widget.addTab(metadata_widget, "Attributes")
        
        # Structure tab
        structure_widget = QtWidgets.QWidget()
        structure_layout = QtWidgets.QVBoxLayout(structure_widget)
        
        self.structure_tree = QtWidgets.QTreeWidget()
        self.structure_tree.setHeaderLabels(['Path', 'Type', 'Shape', 'Dtype'])
        self.structure_tree.setAlternatingRowColors(True)
        structure_layout.addWidget(self.structure_tree)
        
        self.tab_widget.addTab(structure_widget, "Structure")
        
        layout.addWidget(self.tab_widget)
        
        self.status_label = QtWidgets.QLabel("No metadata loaded")
        self.status_label.setStyleSheet("color: #999; padding: 5px;")
        layout.addWidget(self.status_label)
    
    def load_metadata(self, zarr_group):
        """Load metadata from Zarr group"""
        try:
            metadata = ZarrMetadataExtractor.extract_metadata(zarr_group)
            self._all_metadata = metadata
            self._populate_metadata_table(metadata)
            
            structure = ZarrMetadataExtractor.extract_tree_structure(zarr_group)
            self._populate_structure_tree(structure)
            
            self.status_label.setText(f"Loaded {len(metadata)} attributes from Zarr store")
            self.status_label.setStyleSheet("color: #4a4; padding: 5px;")
        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")
            self.status_label.setStyleSheet("color: #f44; padding: 5px;")
    
    def _populate_metadata_table(self, metadata):
        self.metadata_table.setSortingEnabled(False)
        self.metadata_table.setRowCount(len(metadata))
        
        for row, (path, value, dtype) in enumerate(metadata):
            path_item = QtWidgets.QTableWidgetItem(path)
            path_item.setFlags(path_item.flags() & ~QtCore.Qt.ItemIsEditable)
            self.metadata_table.setItem(row, 0, path_item)
            
            value_item = QtWidgets.QTableWidgetItem(str(value))
            value_item.setFlags(value_item.flags() & ~QtCore.Qt.ItemIsEditable)
            value_item.setToolTip(str(value))
            self.metadata_table.setItem(row, 1, value_item)
            
            type_item = QtWidgets.QTableWidgetItem(dtype)
            type_item.setFlags(type_item.flags() & ~QtCore.Qt.ItemIsEditable)
            self.metadata_table.setItem(row, 2, type_item)
        
        self.metadata_table.setSortingEnabled(True)
        self.metadata_table.resizeColumnsToContents()
    
    def _populate_structure_tree(self, structure):
        self.structure_tree.clear()
        root = QtWidgets.QTreeWidgetItem(self.structure_tree)
        root.setText(0, '/')
        root.setText(1, 'Group')
        root.setExpanded(True)
        
        for path, obj_type, shape, dtype in sorted(structure):
            item = QtWidgets.QTreeWidgetItem()
            item.setText(0, path)
            item.setText(1, obj_type)
            if shape is not None:
                item.setText(2, str(shape))
            if dtype is not None:
                item.setText(3, str(dtype))
            root.addChild(item)
        
        self.structure_tree.expandAll()
        self.structure_tree.resizeColumnToContents(0)
    
    def _filter_metadata(self, text):
        if not hasattr(self, '_all_metadata'):
            return
        if not text:
            self._populate_metadata_table(self._all_metadata)
        else:
            filtered = [item for item in self._all_metadata if text.lower() in item[0].lower()]
            self._populate_metadata_table(filtered)
    
    def clear(self):
        self.metadata_table.setRowCount(0)
        self.structure_tree.clear()
        self.status_label.setText("No metadata loaded")
        self.status_label.setStyleSheet("color: #999; padding: 5px;")


# ==================== Multi-Resolution Image Handler ====================

class MultiResolutionImage:
    """Manages multi-resolution Zarr pyramid with chunk-based loading"""
    
    def __init__(self, zarr_group):
        self.zarr_group = zarr_group
        self.levels = []
        self.scales = []
        self._discover_pyramid()
    
    def _discover_pyramid(self):
        """Discover pyramid levels in Zarr group"""
        try:
            attrs = dict(self.zarr_group.attrs) if hasattr(self.zarr_group, 'attrs') else {}
            
            if 'multiscales' in attrs:
                multiscales = attrs['multiscales']
                if isinstance(multiscales, list) and len(multiscales) > 0:
                    datasets = multiscales[0].get('datasets', [])
                    for ds in datasets:
                        path = ds.get('path', '')
                        if path in self.zarr_group:
                            self.levels.append(self.zarr_group[path])
                            if 'coordinateTransformations' in ds:
                                transforms = ds['coordinateTransformations']
                                scale = None
                                for t in transforms:
                                    if t.get('type') == 'scale':
                                        scale_vals = t.get('scale', [1.0, 1.0, 1.0])
                                        scale = scale_vals[-2:]
                                        break
                                self.scales.append(scale or [1.0, 1.0])
                            else:
                                self.scales.append([1.0, 1.0])
        except Exception as e:
            print(f"Note: Could not parse multiscales metadata: {e}")
        
        # Fallback: numeric pyramid
        if not self.levels:
            level_idx = 0
            while str(level_idx) in self.zarr_group:
                try:
                    array = self.zarr_group[str(level_idx)]
                    if hasattr(array, 'shape') and hasattr(array, 'dtype'):
                        self.levels.append(array)
                        if level_idx == 0:
                            self.scales.append([1.0, 1.0])
                        else:
                            prev_shape = self.levels[0].shape
                            curr_shape = array.shape
                            scale_y = prev_shape[-2] / curr_shape[-2] if len(curr_shape) >= 2 else 1.0
                            scale_x = prev_shape[-1] / curr_shape[-1] if len(curr_shape) >= 1 else 1.0
                            self.scales.append([scale_y, scale_x])
                        level_idx += 1
                    else:
                        break
                except Exception as e:
                    print(f"Warning: Could not load level {level_idx}: {e}")
                    break
        
        if not self.levels:
            raise ValueError("No pyramid levels found in Zarr group")
        
        print(f"Found {len(self.levels)} pyramid levels")
        for i, level in enumerate(self.levels):
            print(f"  Level {i}: shape={level.shape}")
    
    def get_num_levels(self):
        return len(self.levels)
    
    def get_level_shape(self, level):
        if 0 <= level < len(self.levels):
            return self.levels[level].shape
        return None
    
    def get_scale_factors(self, level):
        """Get scale factors for a given level relative to level 0"""
        if level < 0 or level >= len(self.levels):
            return (1.0, 1.0)
        
        level_0_shape = self.levels[0].shape
        current_shape = self.levels[level].shape
        
        scale_y = level_0_shape[-2] / current_shape[-2]
        scale_x = level_0_shape[-1] / current_shape[-1]
        
        return (scale_y, scale_x)
    
    def get_optimal_level(self, view_scale):
        """
        Determine optimal level based on view scale.
        
        Strategy: Find the level where approximately 1 screen pixel = 1 data pixel
        This ensures we're always using the most appropriate resolution.
        
        view_scale: pixels per data point in current view
        - view_scale > 1.0: zoomed in (use high res)
        - view_scale = 1.0: 1:1 view
        - view_scale < 1.0: zoomed out (use lower res)
        """
        # If we're zoomed in or at 1:1, always use highest resolution
        if view_scale >= 0.8:
            return 0
        
        # Find the level where we're closest to 1 screen pixel per data pixel
        best_level = len(self.levels) - 1  # Default to lowest resolution
        best_score = 0.0
        
        for i in range(len(self.levels)):
            level_shape = self.levels[i].shape
            level_0_shape = self.levels[0].shape
            
            downsample_y = level_0_shape[-2] / level_shape[-2] 
            downsample_x = level_0_shape[-1] / level_shape[-1]
            avg_downsample = (downsample_y + downsample_x) / 2
            
            # Effective scale = screen pixels per data pixel at this level
            effective_scale = view_scale * avg_downsample
            
            # We want effective_scale close to 1.0 (but prefer slightly higher)
            # Score: prefer effective_scale between 0.8 and 1.5
            if effective_scale >= 0.8:
                # Good range - close to 1:1
                score = min(effective_scale, 1.5)
                if score > best_score:
                    best_score = score
                    best_level = i
            elif effective_scale > 0.4:
                # Acceptable - slightly undersampled but still okay
                score = effective_scale * 0.8
                if score > best_score:
                    best_score = score
                    best_level = i
        
        return best_level
    
    def get_slice_2d(self, level, axis, slice_idx):
        """
        Get a 2D slice from the 3D volume at specified level
        
        Parameters:
        -----------
        level : int
            Pyramid level
        axis : str
            'xy', 'xz', or 'yz'
        slice_idx : int
            Slice index along the perpendicular axis
        
        Returns:
        --------
        2D numpy array
        """
        if level < 0 or level >= len(self.levels):
            return None
        
        array = self.levels[level]
        shape = array.shape
        
        # Assume shape is (z, y, x) for 3D data
        if len(shape) < 3:
            return None
        
        z_size, y_size, x_size = shape[-3], shape[-2], shape[-1]
        
        try:
            if axis == 'xy':
                slice_idx = min(slice_idx, z_size - 1)
                return np.array(array[slice_idx, :, :])
            elif axis == 'xz':
                slice_idx = min(slice_idx, y_size - 1)
                return np.array(array[:, slice_idx, :])
            elif axis == 'yz':
                slice_idx = min(slice_idx, x_size - 1)
                return np.array(array[:, :, slice_idx])
        except Exception as e:
            print(f"Error loading slice: {e}")
            return None
        
        return None
    
    def get_chunk_data(self, level, slice_idx, y_range, x_range):
        """Load data chunk for single view mode"""
        if level < 0 or level >= len(self.levels):
            return None
        
        array = self.levels[level]
        shape = array.shape
        ndim = len(shape)
        
        y_start, y_end = y_range
        x_start, x_end = x_range
        
        y_start = max(0, min(y_start, shape[-2]))
        y_end = max(0, min(y_end, shape[-2]))
        x_start = max(0, min(x_start, shape[-1]))
        x_end = max(0, min(x_end, shape[-1]))
        
        if y_start >= y_end or x_start >= x_end:
            return np.zeros((y_end - y_start, x_end - x_start), dtype=array.dtype)
        
        try:
            if ndim == 2:
                return np.array(array[y_start:y_end, x_start:x_end])
            elif ndim == 3:
                slice_idx = min(slice_idx, shape[0] - 1)
                return np.array(array[slice_idx, y_start:y_end, x_start:x_end])
            elif ndim == 4:
                slice_idx = min(slice_idx, shape[0] - 1)
                return np.array(array[slice_idx, 0, y_start:y_end, x_start:x_end])
            elif ndim == 5:
                slice_idx = min(slice_idx, shape[0] - 1)
                return np.array(array[slice_idx, 0, 0, y_start:y_end, x_start:x_end])
        except Exception as e:
            print(f"Error loading chunk: {e}")
            return np.zeros((y_end - y_start, x_end - x_start), dtype=array.dtype)
        
        return None
    
    def get_full_volume(self, level=0, downsample=1):
        """Load full 3D volume for 3D visualization
        
        Parameters:
        -----------
        level : int
            Pyramid level to load (0 = highest resolution)
        downsample : int
            Additional downsampling factor (1 = no downsampling)
        
        Returns:
        --------
        3D numpy array
        """
        if level < 0 or level >= len(self.levels):
            level = 0
        
        array = self.levels[level]
        shape = array.shape
        
        print(f"Loading full volume from level {level}, shape {shape}")
        
        # Extract 3D volume based on dimensionality
        try:
            if len(shape) == 3:
                volume = np.array(array[:, :, :])
            elif len(shape) == 4:
                # Assume (t/c, z, y, x) - take first timepoint/channel
                volume = np.array(array[0, :, :, :])
            elif len(shape) == 5:
                # Assume (t, c, z, y, x) - take first timepoint and channel
                volume = np.array(array[0, 0, :, :, :])
            else:
                print(f"Unsupported shape: {shape}")
                return None
            
            # Apply additional downsampling if requested
            if downsample > 1:
                volume = volume[::downsample, ::downsample, ::downsample]
                print(f"Downsampled to {volume.shape}")
            
            return volume
            
        except Exception as e:
            print(f"Error loading full volume: {e}")
            import traceback
            traceback.print_exc()
            return None


# ==================== Single View Components ====================

class DynamicImageView(pg.ImageView):
    """Extended ImageView with dynamic chunk loading for single view mode"""
    
    levelChanged = QtCore.pyqtSignal(int, tuple)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.multires_image = None
        self.current_level = 0
        self.current_slice = 0
        self.view_changed_timer = QtCore.QTimer()
        self.view_changed_timer.setSingleShot(True)
        self.view_changed_timer.timeout.connect(self._reload_visible_region)
        
        self.view.sigRangeChanged.connect(self._on_view_changed)
    
    def set_multires_image(self, multires_image):
        self.multires_image = multires_image
        self.current_level = 0
        self.current_slice = 0
        
        # Load and display initial image at level 0
        level_0 = self.multires_image.levels[0]
        level_shape = level_0.shape
        
        # Get initial slice
        if len(level_shape) >= 3:
            slice_idx = level_shape[0] // 2
            img_data = self.multires_image.get_chunk_data(0, slice_idx, (0, level_shape[-2]), (0, level_shape[-1]))
        else:
            img_data = np.array(level_0[:])
        
        if img_data is not None:
            display_img = self._prepare_for_display(img_data)
            # Set image at level 0 coordinates with identity transform
            self.setImage(display_img, autoRange=True, autoLevels=True)
            
            # Ensure the image item has an identity transform initially
            if hasattr(self, 'imageItem') and self.imageItem is not None:
                self.imageItem.resetTransform()
        
        # Now enable dynamic loading for pan/zoom
        self._reload_visible_region()
    
    def set_slice(self, slice_idx):
        self.current_slice = slice_idx
        self._reload_visible_region()
    
    def _on_view_changed(self):
        self.view_changed_timer.start(100)
    
    def _prepare_for_display(self, image):
        """Prepare image for display - handle all data types"""
        if image is None or image.size == 0:
            return image
        
        dtype = image.dtype
        
        if dtype in [np.float32, np.float64]:
            return image.astype(np.float32)
        
        if np.issubdtype(dtype, np.integer):
            return image.astype(np.float32)
        
        return image.astype(np.float32)
    
    def _reload_visible_region(self):
        """Reload visible region at appropriate resolution - FIXED to fill field of view"""
        if self.multires_image is None:
            return
        
        try:
            view_box = self.view.viewRect()
            widget_size = self.view.size()
            
            if widget_size.width() > 0 and view_box.width() > 0:
                pixels_per_data_x = widget_size.width() / view_box.width()
                pixels_per_data_y = widget_size.height() / view_box.height()
                view_scale = min(pixels_per_data_x, pixels_per_data_y)
            else:
                view_scale = 1.0
            
            optimal_level = self.multires_image.get_optimal_level(view_scale)
            self.current_level = optimal_level
            
            # Get shapes and calculate scale factors
            level_0_shape = self.multires_image.get_level_shape(0)
            current_shape = self.multires_image.get_level_shape(optimal_level)
            
            if level_0_shape is None or current_shape is None:
                return
            
            # Calculate the scale factor from current level to level 0
            scale_y, scale_x = self.multires_image.get_scale_factors(optimal_level)
            
            # Debug output
            downsample = scale_y
            effective = view_scale * downsample
            print(f"View scale: {view_scale:.3f} | Level: {optimal_level}/{self.multires_image.get_num_levels()-1} | "
                  f"Downsample: {downsample:.1f}x | Effective scale: {effective:.3f} | Scales: ({scale_x:.2f}, {scale_y:.2f})")
            
            height, width = current_shape[-2:]
            
            # Load the full resolution level data
            # We always load the complete image at the selected resolution level
            chunk_data = self.multires_image.get_chunk_data(
                optimal_level, 
                self.current_slice,
                (0, height),
                (0, width)
            )
            
            if chunk_data is not None and chunk_data.size > 0:
                display_image = self._prepare_for_display(chunk_data)
                
                # Create a transform to scale the image to level 0 coordinates
                # This ensures the image fills the correct space in the view
                transform = QtGui.QTransform()
                transform.scale(scale_x, scale_y)
                
                # Set the image without auto-adjusting (to preserve view)
                self.setImage(display_image, autoLevels=False, autoRange=False)
                
                # Apply the transform to the ImageItem to map it to level 0 coordinate space
                if hasattr(self, 'imageItem') and self.imageItem is not None:
                    self.imageItem.setTransform(transform)
                
                self.levelChanged.emit(optimal_level, current_shape)
                
        except Exception as e:
            print(f"Error loading region: {e}")
            import traceback
            traceback.print_exc()


# ==================== Orthogonal View Components ====================

class OrthoImageView(pg.ImageView):
    """Extended ImageView for orthogonal viewing with crosshairs"""
    
    crosshairMoved = QtCore.pyqtSignal(int, int)
    
    def __init__(self, axis_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.axis_name = axis_name
        self.multires_image = None
        self.current_level = 0
        self.slice_indices = [0, 0, 0]
        
        # Crosshairs
        self.vLine = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('y', width=1))
        self.hLine = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('y', width=1))
        self.addItem(self.vLine, ignoreBounds=True)
        self.addItem(self.hLine, ignoreBounds=True)
        
        self.scene.sigMouseMoved.connect(self.mouseMoved)
        self.scene.sigMouseClicked.connect(self.mouseClicked)
        
        self.view_changed_timer = QtCore.QTimer()
        self.view_changed_timer.setSingleShot(True)
        self.view_changed_timer.timeout.connect(self.reload_image)
        self.view.sigRangeChanged.connect(self._on_view_changed)
    
    def set_multires_image(self, multires_image):
        self.multires_image = multires_image
        self.reload_image(initial_load=True)
    
    def set_slice_indices(self, z, y, x):
        self.slice_indices = [z, y, x]
        self.reload_image()
    
    def update_crosshair(self, pos_x, pos_y):
        self.vLine.setPos(pos_x)
        self.hLine.setPos(pos_y)
    
    def mouseMoved(self, pos):
        if self.image is not None:
            mousePoint = self.view.mapSceneToView(pos)
            self.vLine.setPos(mousePoint.x())
            self.hLine.setPos(mousePoint.y())
    
    def mouseClicked(self, event):
        pos = event.scenePos()
        if self.image is not None:
            mousePoint = self.view.mapSceneToView(pos)
            x = int(mousePoint.x())
            y = int(mousePoint.y())
            
            if hasattr(self.image, 'shape'):
                x = max(0, min(x, self.image.shape[1] - 1))
                y = max(0, min(y, self.image.shape[0] - 1))
                self.crosshairMoved.emit(x, y)
    
    def _on_view_changed(self):
        self.view_changed_timer.start(100)
    
    def reload_image(self, initial_load=False):
        if self.multires_image is None:
            return
        
        try:
            view_box = self.view.viewRect()
            widget_size = self.view.size()
            
            if widget_size.width() > 0 and view_box.width() > 0:
                pixels_per_data_x = widget_size.width() / view_box.width()
                pixels_per_data_y = widget_size.height() / view_box.height()
                view_scale = min(pixels_per_data_x, pixels_per_data_y)
            else:
                view_scale = 1.0
            
            optimal_level = self.multires_image.get_optimal_level(view_scale)
            self.current_level = optimal_level
            
            z_idx, y_idx, x_idx = self.slice_indices
            
            level_0_shape = self.multires_image.get_level_shape(0)
            current_shape = self.multires_image.get_level_shape(optimal_level)
            
            if level_0_shape and current_shape:
                scale_z = current_shape[-3] / level_0_shape[-3]
                scale_y = current_shape[-2] / level_0_shape[-2]
                scale_x = current_shape[-1] / level_0_shape[-1]
                
                z_idx = int(z_idx * scale_z)
                y_idx = int(y_idx * scale_y)
                x_idx = int(x_idx * scale_x)
            
            if self.axis_name == 'XY':
                slice_data = self.multires_image.get_slice_2d(optimal_level, 'xy', z_idx)
            elif self.axis_name == 'XZ':
                slice_data = self.multires_image.get_slice_2d(optimal_level, 'xz', y_idx)
            elif self.axis_name == 'YZ':
                slice_data = self.multires_image.get_slice_2d(optimal_level, 'yz', x_idx)
            else:
                slice_data = None
            
            if slice_data is not None:
                display_image = slice_data.astype(np.float32)
                
                # Get scale factors for this level
                scale_y, scale_x = self.multires_image.get_scale_factors(optimal_level)
                
                # Create transform to map to level 0 coordinates
                transform = QtGui.QTransform()
                
                # Apply appropriate scales based on the view axis
                if self.axis_name == 'XY':
                    transform.scale(scale_x, scale_y)
                elif self.axis_name == 'XZ':
                    scale_z = level_0_shape[-3] / current_shape[-3] if current_shape[-3] > 0 else 1.0
                    transform.scale(scale_x, scale_z)
                elif self.axis_name == 'YZ':
                    scale_z = level_0_shape[-3] / current_shape[-3] if current_shape[-3] > 0 else 1.0
                    transform.scale(scale_z, scale_y)
                
                if initial_load:
                    self.setImage(display_image, autoLevels=True, autoRange=True)
                else:
                    self.setImage(display_image, autoLevels=False, autoRange=False)
                
                # Apply transform to maintain proper coordinate space
                if hasattr(self, 'imageItem') and self.imageItem is not None:
                    self.imageItem.setTransform(transform)
        
        except Exception as e:
            print(f"Error reloading {self.axis_name} view: {e}")
            import traceback
            traceback.print_exc()



# ==================== Main Unified Viewer ====================

class UnifiedZarrViewer(QtWidgets.QDialog):
    """Unified viewer with toggleable single/orthogonal modes"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.zarr_store = None
        self.zarr_group = None
        self.multires_image = None
        
        # Current mode
        self.view_mode = 'single'  # 'single' or 'ortho'
        
        # Ortho mode state
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
        
        # Top toolbar
        toolbar = self._build_toolbar()
        main_layout.addWidget(toolbar)
        
        # Main tabs
        self.main_tabs = QtWidgets.QTabWidget()
        
        # Stacked widget to switch between view modes
        self.stack = QtWidgets.QStackedWidget()
        
        # Single view mode
        self.single_view_widget = self._build_single_view()
        self.stack.addWidget(self.single_view_widget)
        
        # Dual view mode
        self.dual_view_widget = self._build_dual_view()
        self.stack.addWidget(self.dual_view_widget)
        
        # Quad view mode
        self.quad_view_widget = self._build_quad_view()
        self.stack.addWidget(self.quad_view_widget)
        
        # 3D Only view mode
        self.view_3d_only = VisPy3DView()
        self.stack.addWidget(self.view_3d_only)
        
        self.main_tabs.addTab(self.stack, "Image Viewer")
        
        # Metadata viewer tab
        self.metadata_viewer = MetadataViewer()
        self.main_tabs.addTab(self.metadata_viewer, "Metadata")
        
        main_layout.addWidget(self.main_tabs)
    
    def _build_toolbar(self):
        """Build top toolbar with mode toggle"""
        toolbar = QtWidgets.QWidget()
        toolbar.setMaximumHeight(60)
        layout = QtWidgets.QHBoxLayout(toolbar)
        
        # File info
        self.file_label = QtWidgets.QLabel("No file loaded")
        self.file_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(self.file_label)
        
        layout.addStretch()
        
        # View layout selector
        layout_label = QtWidgets.QLabel("Layout:")
        layout.addWidget(layout_label)
        
        self.layout_combo = QtWidgets.QComboBox()
        self.layout_combo.addItems(["1 View", "2 Views", "4 Views (Orthogonal)", "3D Only"])
        self.layout_combo.currentIndexChanged.connect(self._on_layout_changed)
        self.layout_combo.setMinimumWidth(200)
        layout.addWidget(self.layout_combo)
        
        # View selection (for 1 and 2 view modes)
        self.view_select_label = QtWidgets.QLabel("Show:")
        layout.addWidget(self.view_select_label)
        
        self.view1_combo = QtWidgets.QComboBox()
        self.view1_combo.addItems(["XY (Axial)", "XZ (Coronal)", "YZ (Sagittal)"])
        self.view1_combo.currentIndexChanged.connect(self._on_view_selection_changed)
        self.view1_combo.setMinimumWidth(120)
        layout.addWidget(self.view1_combo)
        
        self.view2_combo = QtWidgets.QComboBox()
        self.view2_combo.addItems(["XY (Axial)", "XZ (Coronal)", "YZ (Sagittal)"])
        self.view2_combo.setCurrentIndex(1)  # Default to XZ
        self.view2_combo.currentIndexChanged.connect(self._on_view_selection_changed)
        self.view2_combo.setMinimumWidth(120)
        self.view2_combo.setVisible(False)
        layout.addWidget(self.view2_combo)
        
        # Load button
        load_btn = QtWidgets.QPushButton("Load Zarr")
        load_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DirOpenIcon))
        load_btn.clicked.connect(self._load_file)
        load_btn.setMinimumHeight(40)
        layout.addWidget(load_btn)
        
        return toolbar
    
    def _build_single_view(self):
        """Build single view mode interface"""
        widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QHBoxLayout(widget)
        main_layout.setSpacing(10)
        
        # Left control panel
        left_panel = QtWidgets.QWidget()
        left_panel.setMaximumWidth(350)
        control_layout = QtWidgets.QVBoxLayout(left_panel)
        
        # Pyramid info
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
        
        # Add effective resolution indicator
        self.single_effective_res_label = QtWidgets.QLabel("N/A")
        self.single_effective_res_label.setStyleSheet("color: #4a4; font-weight: bold;")
        pyramid_layout.addRow("Effective res:", self.single_effective_res_label)
        
        pyramid_group.setLayout(pyramid_layout)
        control_layout.addWidget(pyramid_group)
        
        # Slice selection (for 3D data)
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
        
        # Contrast controls
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
        
        # Manual controls
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
        
        # View controls
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
        
        # Statistics
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
        
        # Right: image view
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
        """Build dual view mode interface (2 views side by side)"""
        widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QHBoxLayout(widget)
        
        # Left: 2 views stacked or side-by-side
        views_widget = QtWidgets.QWidget()
        views_layout = QtWidgets.QHBoxLayout(views_widget)
        views_layout.setSpacing(5)
        
        # View 1
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
        
        # View 2
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
        
        views_layout.addWidget(view1_container)
        views_layout.addWidget(view2_container)
        
        main_layout.addWidget(views_widget)
        
        # Right: control panel
        control_panel = self._build_dual_control_panel()
        main_layout.addWidget(control_panel)
        
        main_layout.setStretch(0, 4)
        main_layout.setStretch(1, 1)
        
        return widget
    
    def _build_dual_control_panel(self):
        """Build dual view control panel"""
        panel = QtWidgets.QWidget()
        panel.setMaximumWidth(300)
        layout = QtWidgets.QVBoxLayout(panel)
        
        # Volume info
        info_group = QtWidgets.QGroupBox("Volume Information")
        info_layout = QtWidgets.QFormLayout()
        
        self.dual_shape_label = QtWidgets.QLabel("N/A")
        self.dual_dtype_label = QtWidgets.QLabel("N/A")
        
        info_layout.addRow("Shape (Z,Y,X):", self.dual_shape_label)
        info_layout.addRow("Data type:", self.dual_dtype_label)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Slice position
        pos_group = QtWidgets.QGroupBox("Slice Position")
        pos_layout = QtWidgets.QVBoxLayout()
        
        # Z
        z_layout = QtWidgets.QHBoxLayout()
        z_layout.addWidget(QtWidgets.QLabel("Z:"))
        self.dual_z_spin = QtWidgets.QSpinBox()
        self.dual_z_spin.valueChanged.connect(self._on_dual_z_changed)
        z_layout.addWidget(self.dual_z_spin)
        pos_layout.addLayout(z_layout)
        
        self.dual_z_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.dual_z_slider.valueChanged.connect(self._on_dual_z_changed)
        pos_layout.addWidget(self.dual_z_slider)
        
        # Y
        y_layout = QtWidgets.QHBoxLayout()
        y_layout.addWidget(QtWidgets.QLabel("Y:"))
        self.dual_y_spin = QtWidgets.QSpinBox()
        self.dual_y_spin.valueChanged.connect(self._on_dual_y_changed)
        y_layout.addWidget(self.dual_y_spin)
        pos_layout.addLayout(y_layout)
        
        self.dual_y_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.dual_y_slider.valueChanged.connect(self._on_dual_y_changed)
        pos_layout.addWidget(self.dual_y_slider)
        
        # X
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
        
        # Contrast controls for dual view
        contrast_group = QtWidgets.QGroupBox("Contrast Control")
        contrast_layout = QtWidgets.QVBoxLayout()
        
        # Contrast mode selector
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
        
        # Apply button
        apply_btn = QtWidgets.QPushButton("Apply to Both Views")
        apply_btn.clicked.connect(self._apply_dual_contrast)
        contrast_layout.addWidget(apply_btn)
        
        contrast_group.setLayout(contrast_layout)
        layout.addWidget(contrast_group)
        
        # Controls
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
        """Build quad (4-view orthogonal) view mode interface"""
        widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QHBoxLayout(widget)
        
        # Left: 2x2 grid
        views_widget = QtWidgets.QWidget()
        views_layout = QtWidgets.QGridLayout(views_widget)
        views_layout.setSpacing(5)
        
        # Create views
        self.xy_view = OrthoImageView('XY')
        self.xy_view.ui.roiBtn.hide()
        self.xy_view.ui.menuBtn.hide()
        self.xy_view.crosshairMoved.connect(self._on_xy_crosshair_moved)
        
        self.xz_view = OrthoImageView('XZ')
        self.xz_view.ui.roiBtn.hide()
        self.xz_view.ui.menuBtn.hide()
        self.xz_view.crosshairMoved.connect(self._on_xz_crosshair_moved)
        
        self.yz_view = OrthoImageView('YZ')
        self.yz_view.ui.roiBtn.hide()
        self.yz_view.ui.menuBtn.hide()
        self.yz_view.crosshairMoved.connect(self._on_yz_crosshair_moved)
        
        # 3D view using vedo - EMBEDDED
        self.view_3d = VisPy3DView()
        
        # Labels
        xy_label = QtWidgets.QLabel("XY (Axial)")
        xy_label.setStyleSheet("background-color: #2a2a2a; padding: 3px; font-weight: bold;")
        xy_label.setAlignment(QtCore.Qt.AlignCenter)
        
        xz_label = QtWidgets.QLabel("XZ (Coronal)")
        xz_label.setStyleSheet("background-color: #2a2a2a; padding: 3px; font-weight: bold;")
        xz_label.setAlignment(QtCore.Qt.AlignCenter)
        
        yz_label = QtWidgets.QLabel("YZ (Sagittal)")
        yz_label.setStyleSheet("background-color: #2a2a2a; padding: 3px; font-weight: bold;")
        yz_label.setAlignment(QtCore.Qt.AlignCenter)
        
        view_3d_label = QtWidgets.QLabel("3D Volume (VisPy)")
        view_3d_label.setStyleSheet("background-color: #2a2a2a; padding: 3px; font-weight: bold;")
        view_3d_label.setAlignment(QtCore.Qt.AlignCenter)
        
        # Add to grid
        views_layout.addWidget(xy_label, 0, 0)
        views_layout.addWidget(self.xy_view, 1, 0)
        views_layout.addWidget(xz_label, 0, 1)
        views_layout.addWidget(self.xz_view, 1, 1)
        views_layout.addWidget(yz_label, 2, 0)
        views_layout.addWidget(self.yz_view, 3, 0)
        views_layout.addWidget(view_3d_label, 2, 1)
        views_layout.addWidget(self.view_3d, 3, 1)
        
        main_layout.addWidget(views_widget)
        
        # Right: control panel
        control_panel = self._build_ortho_control_panel()
        main_layout.addWidget(control_panel)
        
        main_layout.setStretch(0, 4)
        main_layout.setStretch(1, 1)
        
        return widget
    
    def _build_ortho_control_panel(self):
        """Build orthogonal view control panel"""
        panel = QtWidgets.QWidget()
        panel.setMaximumWidth(300)
        layout = QtWidgets.QVBoxLayout(panel)
        
        # Volume info
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
        
        # Slice position
        pos_group = QtWidgets.QGroupBox("Slice Position")
        pos_layout = QtWidgets.QVBoxLayout()
        
        # Z
        z_layout = QtWidgets.QHBoxLayout()
        z_layout.addWidget(QtWidgets.QLabel("Z:"))
        self.z_spin = QtWidgets.QSpinBox()
        self.z_spin.valueChanged.connect(self._on_z_changed)
        z_layout.addWidget(self.z_spin)
        pos_layout.addLayout(z_layout)
        
        self.z_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.z_slider.valueChanged.connect(self._on_z_changed)
        pos_layout.addWidget(self.z_slider)
        
        # Y
        y_layout = QtWidgets.QHBoxLayout()
        y_layout.addWidget(QtWidgets.QLabel("Y:"))
        self.y_spin = QtWidgets.QSpinBox()
        self.y_spin.valueChanged.connect(self._on_y_changed)
        y_layout.addWidget(self.y_spin)
        pos_layout.addLayout(y_layout)
        
        self.y_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.y_slider.valueChanged.connect(self._on_y_changed)
        pos_layout.addWidget(self.y_slider)
        
        # X
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
        
        # Contrast controls for ortho view
        contrast_group = QtWidgets.QGroupBox("Contrast Control")
        contrast_layout = QtWidgets.QVBoxLayout()
        
        # Contrast mode selector
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
        
        # Apply button
        apply_btn = QtWidgets.QPushButton("Apply to All Views")
        apply_btn.clicked.connect(self._apply_ortho_contrast)
        contrast_layout.addWidget(apply_btn)
        
        contrast_group.setLayout(contrast_layout)
        layout.addWidget(contrast_group)
        
        # Controls
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
        
        # Calculate and show effective resolution
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
        if index == 5:  # Manual
            self.single_manual_controls.setVisible(True)
            # Set current levels to spinboxes
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
            
            if mode == 0:  # Auto (default)
                self.single_image_view.autoLevels()
            elif mode == 1:  # Min/Max
                min_val = float(np.min(img))
                max_val = float(np.max(img))
                self.single_image_view.setLevels(min_val, max_val)
            elif mode == 2:  # Percentile 1-99%
                min_val = float(np.percentile(img, 1))
                max_val = float(np.percentile(img, 99))
                self.single_image_view.setLevels(min_val, max_val)
            elif mode == 3:  # Percentile 2-98%
                min_val = float(np.percentile(img, 2))
                max_val = float(np.percentile(img, 98))
                self.single_image_view.setLevels(min_val, max_val)
            elif mode == 4:  # Percentile 5-95%
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
        if mode == 5:  # Manual mode
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
            
            # Set up sliders
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
        
        # Update crosshairs based on view axes
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
        pass  # Will be applied when button is clicked
    
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
        pass  # Will be applied when button is clicked
    
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
            
            if mode == 0:  # Auto
                view.autoLevels()
            elif mode == 1:  # Min/Max
                min_val = float(np.min(img))
                max_val = float(np.max(img))
                view.setLevels(min_val, max_val)
            elif mode == 2:  # Percentile 1-99%
                min_val = float(np.percentile(img, 1))
                max_val = float(np.percentile(img, 99))
                view.setLevels(min_val, max_val)
            elif mode == 3:  # Percentile 2-98%
                min_val = float(np.percentile(img, 2))
                max_val = float(np.percentile(img, 98))
                view.setLevels(min_val, max_val)
            elif mode == 4:  # Percentile 5-95%
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
        if index == 0:  # 1 view
            self.stack.setCurrentIndex(0)
            self.view1_combo.setVisible(True)
            self.view2_combo.setVisible(False)
            self.view_select_label.setVisible(True)
        elif index == 1:  # 2 views
            self.stack.setCurrentIndex(1)
            self.view1_combo.setVisible(True)
            self.view2_combo.setVisible(True)
            self.view_select_label.setVisible(True)
            if self.multires_image is not None:
                self._update_dual_views()
        elif index == 2:  # 4 views
            self.stack.setCurrentIndex(2)
            self.view1_combo.setVisible(False)
            self.view2_combo.setVisible(False)
            self.view_select_label.setVisible(False)
            if self.multires_image is not None:
                self._update_all_ortho_views()
                # Update 3D view in quad mode
                self.view_3d.set_multires_image(self.multires_image)
        elif index == 3:  # 3D Only
            self.stack.setCurrentIndex(3)
            self.view1_combo.setVisible(False)
            self.view2_combo.setVisible(False)
            self.view_select_label.setVisible(False)
            if self.multires_image is not None:
                self.view_3d_only.set_multires_image(self.multires_image)
    
    def _on_view_selection_changed(self):
        """Handle view selection change for 1 and 2 view modes"""
        if self.multires_image is None:
            return
        
        current_layout = self.layout_combo.currentIndex()
        
        if current_layout == 0:  # Single view
            # Update single view based on selection
            view_type = self.view1_combo.currentIndex()
            # Will be handled by switching to appropriate ortho view
            pass
        elif current_layout == 1:  # Dual view
            # Update which views are shown
            view1_type = self.view1_combo.currentIndex()
            view2_type = self.view2_combo.currentIndex()
            
            # Map selection to axis
            axis_map = ['XY', 'XZ', 'YZ']
            self.dual_view1.axis_name = axis_map[view1_type]
            self.dual_view2.axis_name = axis_map[view2_type]
            
            # Update labels
            labels = ["XY (Axial)", "XZ (Coronal)", "YZ (Sagittal)"]
            self.dual_view1_label.setText(labels[view1_type])
            self.dual_view2_label.setText(labels[view2_type])
            
            self._update_dual_views()
    
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
            
            # Update single view
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
            
            # Update ortho view
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
                
                # Set for 3D view in quad mode
                self.view_3d.set_multires_image(self.multires_image)
                
                # Set for standalone 3D view
                self.view_3d_only.set_multires_image(self.multires_image)
                
                self._update_all_ortho_views()
            
            # Load metadata
            self.metadata_viewer.load_metadata(self.zarr_group)
            
            QtWidgets.QMessageBox.information(
                self, "Success", 
                f"Loaded Zarr store\nShape: {level_0_shape}\n"
                f"Levels: {self.multires_image.get_num_levels()}"
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
    import sys
    
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    
    app.setApplicationName("Unified Zarr Viewer - 2D/3D Multi-Resolution (VisPy 3D)")
    apply_dark_theme(app)
    
    viewer = UnifiedZarrViewer()
    viewer.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
