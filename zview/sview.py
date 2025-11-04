#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic Image View Component
Handles multi-resolution image viewing with chunk-aligned loading
"""

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui


class DynamicImageView(pg.ImageView):
    """Extended ImageView with dynamic chunk loading - OPTIMIZED"""
    
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
        self._last_load_params = None

    def set_multires_image(self, multires_image):
        self.multires_image = multires_image
        self.current_level = 0
        self._last_load_params = None
        
        level_0 = self.multires_image.levels[0]
        level_shape = level_0.shape
        
        if len(level_shape) >= 3:
            slice_idx = level_shape[0] // 2
            self.current_slice = slice_idx  # Set to the slice we're loading
            img_data = self.multires_image.get_chunk_data_aligned(
                0, slice_idx, (0, level_shape[-2]), (0, level_shape[-1])
            )
        else:
            self.current_slice = 0
            img_data = np.array(level_0[:])
        
        if img_data is not None:
            display_img = self._prepare_for_display(img_data)
            self.setImage(display_img, autoRange=True, autoLevels=True)
            
            if hasattr(self, 'imageItem') and self.imageItem is not None:
                self.imageItem.resetTransform()
        
        self._reload_visible_region()
    
    def set_slice(self, slice_idx):
        self.current_slice = slice_idx
        self._last_load_params = None
        self._reload_visible_region()
    
    def _on_view_changed(self):
        self.view_changed_timer.start(200)  # Increased from 100ms to 200ms
    
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

            level_0_shape = self.multires_image.get_level_shape(0)
            current_shape = self.multires_image.get_level_shape(optimal_level)
            if level_0_shape is None or current_shape is None:
                return

            scale_y, scale_x = self.multires_image.get_scale_factors(optimal_level)

            vx0 = max(0, int(view_box.left()  / scale_x))
            vy0 = max(0, int(view_box.top()   / scale_y))
            vx1 = min(current_shape[-1], int(view_box.right()  / scale_x) + 1)
            vy1 = min(current_shape[-2], int(view_box.bottom() / scale_y) + 1)

            if vx0 >= vx1 or vy0 >= vy1:
                vx0, vy0 = 0, 0
                vx1, vy1 = current_shape[-1], current_shape[-2]

            max_fetch = 2_000_000
            want = (vy1 - vy0) * (vx1 - vx0)
            step = 1
            if want > max_fetch:
                step = int(np.ceil(np.sqrt(want / max_fetch)))

            # Scale slice index to current level
            # If current_slice is in level 0 coords and we're viewing level 1 (2x downsampled),
            # we need slice_idx = current_slice / 2
            level_0_shape = self.multires_image.get_level_shape(0)
            if level_0_shape and len(level_0_shape) >= 3:
                scale_factor = current_shape[-3] / level_0_shape[-3]
                scaled_slice_idx = int(self.current_slice * scale_factor)
                scaled_slice_idx = max(0, min(scaled_slice_idx, current_shape[-3] - 1))
            else:
                scaled_slice_idx = self.current_slice

            # Check if we need to reload
            load_params = (optimal_level, scaled_slice_idx, vy0, vy1, vx0, vx1, step)
            if load_params == self._last_load_params:
                return
            self._last_load_params = load_params

            # Use chunk-aligned loading
            chunk = self.multires_image.get_chunk_data_aligned(
                optimal_level,
                scaled_slice_idx,
                (vy0, vy1),
                (vx0, vx1),
            )
            
            if chunk is None or chunk.size == 0:
                return
            
            if step > 1:
                chunk = chunk[::step, ::step]

            display_image = self._prepare_for_display(chunk)

            # Get coordinate transformations from metadata
            transform_info = self.multires_image.get_transform(optimal_level)
            translation = transform_info['translation']
            
            # Apply transform with translation
            # Translation must be applied in the DISPLAY coordinate space (after scaling)
            t = QtGui.QTransform()
            # First translate viewport position and translation offset
            t.translate(vx0 * scale_x + translation[-1] * scale_x, 
                       vy0 * scale_y + translation[-2] * scale_y)
            # Then scale
            t.scale(scale_x * step, scale_y * step)

            self.setImage(display_image, autoLevels=False, autoRange=False)

            if hasattr(self, 'imageItem') and self.imageItem is not None:
                self.imageItem.setTransform(t)
                try:
                    self.imageItem.setAutoDownsample(True)
                except Exception:
                    pass

            self.levelChanged.emit(optimal_level, current_shape)

        except Exception as e:
            print(f"Error loading region: {e}")
            import traceback
            traceback.print_exc()