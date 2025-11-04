#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed Orthogonal Image View Component - NON-PANNABLE
Handles orthogonal slice viewing with crosshairs and optimized loading
NO PANNING - Image is locked to view for proper chunked loading
"""

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui
from collections import OrderedDict
from typing import Tuple, Optional
import time


class ImageCache:
    """LRU cache for loaded image data"""
    
    def __init__(self, max_size=20):
        self.cache = OrderedDict()
        self.max_size = max_size
    
    def _make_key(self, level, axis, slice_idx):
        """Create hashable cache key"""
        return (level, axis, slice_idx)
    
    def get(self, level, axis, slice_idx):
        """Get cached image data"""
        key = self._make_key(level, axis, slice_idx)
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def put(self, level, axis, slice_idx, data):
        """Store image data in cache"""
        key = self._make_key(level, axis, slice_idx)
        
        # Remove oldest if at capacity
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
        
        self.cache[key] = data.copy() if isinstance(data, np.ndarray) else data
    
    def clear(self):
        """Clear all cached data"""
        self.cache.clear()
    
    def get_memory_usage(self):
        """Estimate memory usage in MB"""
        total_bytes = 0
        for data in self.cache.values():
            if isinstance(data, np.ndarray):
                total_bytes += data.nbytes
        return total_bytes / (1024 * 1024)


class ImageLoadWorker(QtCore.QThread):
    """Background worker for loading image slices"""
    
    imageLoaded = QtCore.pyqtSignal(object, int, str)  # data, level, request_id
    loadFailed = QtCore.pyqtSignal(str, str)  # error_msg, request_id
    
    def __init__(self, multires_image, axis_name, cache):
        super().__init__()
        self.multires_image = multires_image
        self.axis_name = axis_name
        self.cache = cache
        
        # Parameters for current request
        self.optimal_level = 0
        self.slice_indices = [0, 0, 0]
        self.request_id = ""
        self._should_cancel = False
    
    def set_load_params(self, optimal_level, slice_indices, request_id=""):
        """Set parameters for the next load operation"""
        self.optimal_level = optimal_level
        self.slice_indices = list(slice_indices)
        self.request_id = request_id
        self._should_cancel = False
    
    def cancel(self):
        """Cancel current operation"""
        self._should_cancel = True
    
    def run(self):
        """Load image data in background thread"""
        try:
            if self._should_cancel:
                return
            
            z_idx, y_idx, x_idx = self.slice_indices
            level = self.optimal_level
            
            # Get scaling information
            level_0_shape = self.multires_image.get_level_shape(0)
            current_shape = self.multires_image.get_level_shape(level)
            
            if level_0_shape and current_shape:
                scale_z = current_shape[-3] / level_0_shape[-3]
                scale_y = current_shape[-2] / level_0_shape[-2]
                scale_x = current_shape[-1] / level_0_shape[-1]
                
                z_idx_scaled = int(z_idx * scale_z)
                y_idx_scaled = int(y_idx * scale_y)
                x_idx_scaled = int(x_idx * scale_x)
            else:
                z_idx_scaled = z_idx
                y_idx_scaled = y_idx
                x_idx_scaled = x_idx
            
            if self._should_cancel:
                return
            
            # Check cache first
            cached_data = self.cache.get(level, self.axis_name, self._get_slice_idx())
            if cached_data is not None:
                # Emit cached data
                if not self._should_cancel:
                    self.imageLoaded.emit(cached_data, level, self.request_id)
                return
            
            # Load full slice at optimal level (no partial loading)
            slice_data = None
            
            if self.axis_name == 'XY':
                slice_data = self.multires_image.get_slice_2d(level, 'xy', z_idx_scaled)
            elif self.axis_name == 'XZ':
                slice_data = self.multires_image.get_slice_2d(level, 'xz', y_idx_scaled)
            elif self.axis_name == 'YZ':
                slice_data = self.multires_image.get_slice_2d(level, 'yz', x_idx_scaled)
            
            if self._should_cancel:
                return
            
            if slice_data is not None:
                display_image = slice_data.astype(np.float32)
                
                # Cache the loaded data
                self.cache.put(level, self.axis_name, self._get_slice_idx(), display_image)
                
                if not self._should_cancel:
                    self.imageLoaded.emit(display_image, level, self.request_id)
            else:
                self.loadFailed.emit("Failed to load slice data", self.request_id)
        
        except Exception as e:
            if not self._should_cancel:
                self.loadFailed.emit(str(e), self.request_id)
    
    def _get_slice_idx(self):
        """Get the relevant slice index for current axis"""
        z_idx, y_idx, x_idx = self.slice_indices
        if self.axis_name == 'XY':
            return z_idx
        elif self.axis_name == 'XZ':
            return y_idx
        elif self.axis_name == 'YZ':
            return x_idx
        return 0


class OrthoImageView(pg.ImageView):
    """Extended ImageView for orthogonal viewing - LOCKED (non-pannable)"""
    
    crosshairMoved = QtCore.pyqtSignal(int, int)
    
    def __init__(self, axis_name, shared_cache=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.axis_name = axis_name
        self.multires_image = None
        self.current_level = 0
        self.slice_indices = [0, 0, 0]
        
        # Caching
        self.cache = shared_cache if shared_cache is not None else ImageCache(max_size=20)
        
        # Background loading
        self.load_worker = None
        self.pending_load = False
        self.current_request_id = ""
        
        # Crosshairs
        self.vLine = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('y', width=1))
        self.hLine = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('y', width=1))
        self.addItem(self.vLine, ignoreBounds=True)
        self.addItem(self.hLine, ignoreBounds=True)
        
        # Mouse interaction
        self.scene.sigMouseMoved.connect(self.mouseMoved)
        self.scene.sigMouseClicked.connect(self.mouseClicked)
        
        # CRITICAL: Disable panning and lock view
        self._lock_view()
        
        # Debounced view updates (only for zoom changes)
        self.view_changed_timer = QtCore.QTimer()
        self.view_changed_timer.setSingleShot(True)
        self.view_changed_timer.timeout.connect(self._reload_image_debounced)
        self.view.sigRangeChanged.connect(self._on_view_changed)
        
        # Track if initial load is complete
        self.initial_load_complete = False
        
        # Track current zoom level to detect actual zoom changes
        self.last_zoom_level = None
    
    def _lock_view(self):
        """Lock the view to prevent panning while allowing zoom"""
        # Disable mouse panning but keep zoom
        self.view.setMouseEnabled(x=False, y=False)
        
        # Disable menu (right-click context menu)
        self.view.setMenuEnabled(False)
        
        # Keep mouse mode for zoom with wheel
        self.view.setMouseMode(pg.ViewBox.PanMode)
        
        # Re-enable wheel zoom only
        self.view.setMouseEnabled(x=False, y=False)
        
        # Override the wheelEvent to handle zoom without panning
        original_wheelEvent = self.view.wheelEvent
        
        def locked_wheelEvent(ev, axis=None):
            # Get current center
            center = self.view.viewRect().center()
            
            # Do the zoom
            original_wheelEvent(ev, axis)
            
            # Force center back to image center after zoom
            QtCore.QTimer.singleShot(0, lambda: self._recenter_view())
        
        self.view.wheelEvent = locked_wheelEvent
    
    def _recenter_view(self):
        """Recenter view to image after zoom"""
        if self.image is not None and hasattr(self.image, 'shape'):
            # Get image bounds in data coordinates
            height, width = self.image.shape[:2]
            
            # Get current view range
            view_rect = self.view.viewRect()
            
            # Calculate center of image
            img_center_x = width / 2.0
            img_center_y = height / 2.0
            
            # Calculate new view rect centered on image
            new_x = img_center_x - view_rect.width() / 2.0
            new_y = img_center_y - view_rect.height() / 2.0
            
            # Set the view range without emitting signals (to avoid reload loop)
            self.view.blockSignals(True)
            self.view.setRange(
                xRange=(new_x, new_x + view_rect.width()),
                yRange=(new_y, new_y + view_rect.height()),
                padding=0
            )
            self.view.blockSignals(False)
    
    def set_multires_image(self, multires_image):
        """Set the multi-resolution image source"""
        self.multires_image = multires_image
        self.cache.clear()
        self.initial_load_complete = False
        self.last_zoom_level = None
        self.reload_image(initial_load=True)
    
    def set_slice_indices(self, z, y, x):
        """Update which slice to display"""
        if self.slice_indices != [z, y, x]:
            self.slice_indices = [z, y, x]
            self.reload_image()
    
    def update_crosshair(self, pos_x, pos_y):
        """Update crosshair position"""
        self.vLine.setPos(pos_x)
        self.hLine.setPos(pos_y)
    
    def mouseMoved(self, pos):
        """Handle mouse movement for crosshair"""
        if self.image is not None:
            mousePoint = self.view.mapSceneToView(pos)
            self.vLine.setPos(mousePoint.x())
            self.hLine.setPos(mousePoint.y())
    
    def mouseClicked(self, event):
        """Handle mouse clicks for slice navigation"""
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
        """Handle view changes (zoom only, no pan) with debouncing"""
        if not self.initial_load_complete:
            return
        
        # Calculate current zoom level
        current_zoom = self._get_zoom_level()
        
        # Only trigger reload if zoom actually changed significantly
        if self.last_zoom_level is None or abs(current_zoom - self.last_zoom_level) > 0.1:
            self.last_zoom_level = current_zoom
            self.view_changed_timer.start(150)
    
    def _get_zoom_level(self):
        """Get current zoom level"""
        if self.image is None:
            return 1.0
        
        view_box = self.view.viewRect()
        widget_size = self.view.size()
        
        if widget_size.width() > 0 and view_box.width() > 0:
            pixels_per_data_x = widget_size.width() / view_box.width()
            return pixels_per_data_x
        return 1.0
    
    def _reload_image_debounced(self):
        """Called after view change debounce period"""
        if self.initial_load_complete:
            self.reload_image(initial_load=False)
    
    def _calculate_optimal_level(self):
        """Calculate optimal resolution level based on view zoom"""
        view_box = self.view.viewRect()
        widget_size = self.view.size()
        
        if widget_size.width() > 0 and view_box.width() > 0:
            pixels_per_data_x = widget_size.width() / view_box.width()
            pixels_per_data_y = widget_size.height() / view_box.height()
            view_scale = min(pixels_per_data_x, pixels_per_data_y)
        else:
            view_scale = 1.0
        
        return self.multires_image.get_optimal_level(view_scale)
    
    def reload_image(self, initial_load=False):
        """Reload image data - main entry point for loading"""
        if self.multires_image is None:
            return
        
        # Generate unique request ID
        self.current_request_id = f"{self.axis_name}_{time.time()}"
        
        try:
            if initial_load:
                # For initial load, load full slice at level 0 for autoRange
                self._load_initial_image()
            else:
                # For subsequent updates, load optimal level (full slice)
                self._load_optimized_image()
        
        except Exception as e:
            print(f"Error reloading {self.axis_name} view: {e}")
            import traceback
            traceback.print_exc()
    
    def _load_initial_image(self):
        """Load full slice at base level for initial display"""
        z_idx, y_idx, x_idx = self.slice_indices
        
        # Load full slice at level 0
        if self.axis_name == 'XY':
            slice_data = self.multires_image.get_slice_2d(0, 'xy', z_idx)
        elif self.axis_name == 'XZ':
            slice_data = self.multires_image.get_slice_2d(0, 'xz', y_idx)
        elif self.axis_name == 'YZ':
            slice_data = self.multires_image.get_slice_2d(0, 'yz', x_idx)
        else:
            slice_data = None
        
        if slice_data is not None:
            display_image = slice_data.astype(np.float32)
            self.setImage(display_image, autoLevels=True, autoRange=True)
            if hasattr(self, 'imageItem') and self.imageItem is not None:
                self.imageItem.resetTransform()
            
            # Center the view on the image
            self._recenter_view()
            self.initial_load_complete = True
            self.last_zoom_level = self._get_zoom_level()
    
    def _load_optimized_image(self):
        """Load image with optimal resolution (full slice, no panning)"""
        # Calculate optimal level
        optimal_level = self._calculate_optimal_level()
        self.current_level = optimal_level
        
        # Cancel any pending load
        if self.load_worker is not None and self.load_worker.isRunning():
            self.load_worker.cancel()
            self.load_worker.wait(100)
        
        # Create worker if needed
        if self.load_worker is None:
            self.load_worker = ImageLoadWorker(self.multires_image, self.axis_name, self.cache)
            self.load_worker.imageLoaded.connect(self._on_image_loaded)
            self.load_worker.loadFailed.connect(self._on_load_failed)
        
        # Start background load (full slice at optimal level)
        self.load_worker.set_load_params(
            optimal_level=optimal_level,
            slice_indices=self.slice_indices,
            request_id=self.current_request_id
        )
        
        if not self.load_worker.isRunning():
            self.load_worker.start()
    
    def _on_image_loaded(self, display_image, level, request_id):
        """Handle loaded image data from worker thread"""
        # Ignore outdated requests
        if request_id != self.current_request_id:
            return
        
        try:
            # Update display - no transform needed since view is locked
            self.setImage(display_image, autoLevels=False, autoRange=False)
            
            # Reset transform to identity (no scaling or translation)
            if hasattr(self, 'imageItem') and self.imageItem is not None:
                self.imageItem.resetTransform()
            
            # Re-center view (in case zoom changed it)
            self._recenter_view()
        
        except Exception as e:
            print(f"Error updating {self.axis_name} display: {e}")
            import traceback
            traceback.print_exc()
    
    def _on_load_failed(self, error_msg, request_id):
        """Handle load failure"""
        # Ignore outdated requests
        if request_id != self.current_request_id:
            return
        
        print(f"Load failed for {self.axis_name}: {error_msg}")
    
    def cleanup(self):
        """Cleanup resources"""
        if self.load_worker is not None:
            self.load_worker.cancel()
            self.load_worker.wait()
            self.load_worker = None


class ParallelOrthoViewer:
    """Coordinator for parallel loading across multiple orthogonal views"""
    
    def __init__(self, xy_view: OrthoImageView, xz_view: OrthoImageView, 
                 yz_view: OrthoImageView, shared_cache=None):
        self.xy_view = xy_view
        self.xz_view = xz_view
        self.yz_view = yz_view
        
        # Use shared cache across all views
        if shared_cache is None:
            shared_cache = ImageCache(max_size=30)
        
        self.xy_view.cache = shared_cache
        self.xz_view.cache = shared_cache
        self.yz_view.cache = shared_cache
    
    def set_multires_image(self, multires_image):
        """Set image for all views - triggers parallel loading"""
        self.xy_view.set_multires_image(multires_image)
        self.xz_view.set_multires_image(multires_image)
        self.yz_view.set_multires_image(multires_image)
    
    def set_slice_indices(self, z, y, x):
        """Update slice indices for all views - triggers parallel loading"""
        self.xy_view.set_slice_indices(z, y, x)
        self.xz_view.set_slice_indices(z, y, x)
        self.yz_view.set_slice_indices(z, y, x)
    
    def cleanup(self):
        """Cleanup all views"""
        self.xy_view.cleanup()
        self.xz_view.cleanup()
        self.yz_view.cleanup()
    
    def get_cache_stats(self):
        """Get cache statistics"""
        return {
            'size': len(self.xy_view.cache.cache),
            'memory_mb': self.xy_view.cache.get_memory_usage()
        }