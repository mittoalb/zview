#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Resolution Image Handler
Handles pyramid-style Zarr data with chunk-aligned loading and caching
"""

import numpy as np
from typing import Optional, Tuple
import threading


class ChunkCache:
    """Simple LRU cache for chunks with memory management"""
    
    def __init__(self, max_memory_mb=500):
        self.cache = {}
        self.access_order = []
        self.current_memory = 0
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.hits = 0
        self.misses = 0
        self.lock = threading.Lock()
    
    def get_key(self, level, slice_idx, chunk_pos):
        """Generate cache key"""
        return (level, slice_idx, chunk_pos)
    
    def get(self, key):
        """Get cached chunk with minimal lock time"""
        with self.lock:
            if key in self.cache:
                # Update access order
                self.access_order.remove(key)
                self.access_order.append(key)
                self.hits += 1
                # Return reference (caller should not modify)
                return self.cache[key]
            self.misses += 1
            return None
    
    def get_many(self, keys):
        """Get multiple cached chunks at once (reduced lock contention)"""
        results = {}
        with self.lock:
            for key in keys:
                if key in self.cache:
                    self.access_order.remove(key)
                    self.access_order.append(key)
                    self.hits += 1
                    results[key] = self.cache[key]
                else:
                    self.misses += 1
                    results[key] = None
        return results
    
    def put(self, key, data):
        """Store chunk in cache with minimal lock time"""
        with self.lock:
            # Skip if already cached (race condition)
            if key in self.cache:
                return
            
            data_size = data.nbytes
            
            # Evict old entries if needed (FIFO for speed)
            while (self.current_memory + data_size > self.max_memory_bytes and 
                   self.access_order):
                old_key = self.access_order.pop(0)
                if old_key in self.cache:
                    old_data = self.cache.pop(old_key)
                    self.current_memory -= old_data.nbytes
            
            # Only cache if it fits
            if data_size <= self.max_memory_bytes:
                self.cache[key] = data
                self.access_order.append(key)
                self.current_memory += data_size
    
    def get_stats(self):
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'chunks': len(self.cache),
            'memory_mb': self.current_memory / (1024 * 1024)
        }
    
    def clear(self):
        """Clear cache"""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            self.current_memory = 0
            self.hits = 0
            self.misses = 0
    
    def get_stats(self):
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'chunks': len(self.cache),
            'memory_mb': self.current_memory / (1024 * 1024)
        }
    
    def clear(self):
        """Clear cache"""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            self.current_memory = 0
            self.hits = 0
            self.misses = 0


class MultiResolutionImage:
    """Manages multi-resolution Zarr pyramid with efficient chunk-based loading"""
    
    def __init__(self, zarr_group, cache_size_mb=500):
        self.zarr_group = zarr_group
        self.levels = []
        self.scales = []
        self.transforms = []
        self.chunk_cache = ChunkCache(max_memory_mb=cache_size_mb)
        self._discover_pyramid()

    def get_transform(self, level):
        """Get scale and translation for a level"""
        if 0 <= level < len(self.transforms):
            return self.transforms[level]
        return {'scale': [1.0, 1.0, 1.0], 'translation': [0.0, 0.0, 0.0]}

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
                            
                            # Parse BOTH scale and translation
                            scale = [1.0, 1.0, 1.0]
                            translation = [0.0, 0.0, 0.0]
                            
                            if 'coordinateTransformations' in ds:
                                transforms = ds['coordinateTransformations']
                                for t in transforms:
                                    if t.get('type') == 'scale':
                                        scale_vals = t.get('scale', [1.0, 1.0, 1.0])
                                        if len(scale_vals) >= 3:
                                            scale = scale_vals[-3:]  # Z, Y, X
                                        elif len(scale_vals) >= 2:
                                            scale = [1.0] + scale_vals[-2:]  # Y, X -> Z, Y, X
                                    elif t.get('type') == 'translation':
                                        trans_vals = t.get('translation', [0.0, 0.0, 0.0])
                                        if len(trans_vals) >= 3:
                                            translation = trans_vals[-3:]
                            
                            self.transforms.append({
                                'scale': scale,
                                'translation': translation
                            })
                            
                            # Also store 2D scale for backward compatibility
                            self.scales.append(scale[-2:])
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
                            self.transforms.append({
                                'scale': [1.0, 1.0, 1.0],
                                'translation': [0.0, 0.0, 0.0]
                            })
                        else:
                            prev_shape = self.levels[0].shape
                            curr_shape = array.shape
                            scale_y = prev_shape[-2] / curr_shape[-2] if len(curr_shape) >= 2 else 1.0
                            scale_x = prev_shape[-1] / curr_shape[-1] if len(curr_shape) >= 1 else 1.0
                            scale_z = prev_shape[-3] / curr_shape[-3] if len(curr_shape) >= 3 and len(prev_shape) >= 3 else 1.0
                            
                            self.scales.append([scale_y, scale_x])
                            self.transforms.append({
                                'scale': [scale_z, scale_y, scale_x],
                                'translation': [0.0, 0.0, 0.0]
                            })
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
            chunks = getattr(level, 'chunks', None)
            print(f"  Level {i}: shape={level.shape}, chunks={chunks}")
    
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
        Determine optimal pyramid level based on view scale.
        
        view_scale: pixels per data unit (how zoomed in we are)
        - High value (>1): zoomed in, need high resolution
        - Low value (<1): zoomed out, can use lower resolution
        
        Returns the level that best matches the needed resolution.
        """
        if view_scale >= 1.0:
            # Fully zoomed in or more, always use full resolution
            return 0
        
        # Calculate what downsample factor we need
        # If view_scale = 0.5, we're showing 1 screen pixel per 2 data pixels
        # So we can use 2x downsampled data
        needed_downsample = 1.0 / view_scale
        
        # Find the level whose downsample is closest to what we need
        # But prefer slightly higher resolution (lower downsample) to avoid blur
        best_level = 0
        best_diff = float('inf')
        
        level_0_shape = self.levels[0].shape
        
        for i in range(len(self.levels)):
            level_shape = self.levels[i].shape
            
            # Calculate actual downsample factor for this level
            downsample_y = level_0_shape[-2] / level_shape[-2]
            downsample_x = level_0_shape[-1] / level_shape[-1]
            avg_downsample = (downsample_y + downsample_x) / 2.0
            
            # How far is this level from what we need?
            # Prefer slightly UNDER-downsampled (higher quality) over OVER-downsampled
            if avg_downsample <= needed_downsample:
                # This level has enough detail
                diff = needed_downsample - avg_downsample
            else:
                # This level doesn't have enough detail (penalize more)
                diff = (avg_downsample - needed_downsample) * 2.0
            
            if diff < best_diff:
                best_diff = diff
                best_level = i
        
        # Debug output (throttled - only print if level changes)
        if not hasattr(self, '_last_printed_level'):
            self._last_printed_level = -1
        
        if best_level != self._last_printed_level:
            level_shape = self.levels[best_level].shape
            downsample = level_0_shape[-2] / level_shape[-2]
            print(f"Level changed: {self._last_printed_level} → {best_level} ({downsample:.0f}x downsample, view_scale={view_scale:.3f})")
            self._last_printed_level = best_level
        
        return best_level
    
    def get_slice_2d(self, level, axis, slice_idx):
        """Get a 2D slice from the 3D volume at specified level - CACHED"""
        if level < 0 or level >= len(self.levels):
            return None
        
        array = self.levels[level]
        shape = array.shape
        
        if len(shape) < 3:
            return None
        
        z_size, y_size, x_size = shape[-3], shape[-2], shape[-1]
        
        try:
            if axis == 'xy':
                slice_idx = min(slice_idx, z_size - 1)
                # For XY slices, use chunk-aligned loading for speed
                return self.get_chunk_data_aligned(level, slice_idx, (0, y_size), (0, x_size))
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
    
    def get_chunk_data_aligned(self, level, slice_idx, y_range, x_range):
        """
        Load data using TRUE chunk-aligned reads for maximum performance.
        Reads complete chunks and extracts needed portions.
        """
        if level < 0 or level >= len(self.levels):
            return None
        
        array = self.levels[level]
        shape = array.shape
        ndim = len(shape)
        
        # Get chunk size
        if not hasattr(array, 'chunks'):
            return self.get_chunk_data(level, slice_idx, y_range, x_range)
        
        chunks = array.chunks
        if ndim >= 2:
            chunk_y, chunk_x = chunks[-2], chunks[-1]
        else:
            return self.get_chunk_data(level, slice_idx, y_range, x_range)
        
        y_start, y_end = y_range
        x_start, x_end = x_range
        
        # Clamp to array bounds
        y_start = max(0, min(y_start, shape[-2]))
        y_end = max(0, min(y_end, shape[-2]))
        x_start = max(0, min(x_start, shape[-1]))
        x_end = max(0, min(x_end, shape[-1]))
        
        if y_start >= y_end or x_start >= x_end:
            return np.zeros((y_end - y_start, x_end - x_start), dtype=array.dtype)
        
        # Determine which chunks we need
        chunk_y_start = y_start // chunk_y
        chunk_y_end = (y_end - 1) // chunk_y + 1
        chunk_x_start = x_start // chunk_x
        chunk_x_end = (x_end - 1) // chunk_x + 1
        
        # Clamp slice_idx to valid range
        slice_idx = min(slice_idx, shape[0] - 1) if ndim >= 3 else 0
        
        # Initialize output array
        result = np.zeros((y_end - y_start, x_end - x_start), dtype=array.dtype)
        
        # Collect all cache keys for bulk lookup
        all_cache_keys = []
        chunk_coords = []
        for cy in range(chunk_y_start, chunk_y_end):
            for cx in range(chunk_x_start, chunk_x_end):
                cache_key = self.chunk_cache.get_key(level, slice_idx, (cy, cx))
                all_cache_keys.append(cache_key)
                chunk_coords.append((cy, cx))
        
        # Bulk cache lookup (single lock acquisition)
        cached_chunks = self.chunk_cache.get_many(all_cache_keys)
        
        # Separate cached vs needs-loading
        chunks_to_load = []
        chunk_positions = []
        
        for (cy, cx), cache_key in zip(chunk_coords, all_cache_keys):
            chunk_data = cached_chunks[cache_key]
            if chunk_data is None:
                chunks_to_load.append((cy, cx, cache_key))
            else:
                chunk_positions.append((cy, cx, chunk_data))
        
        # Load missing chunks (TRUE CHUNK-ALIGNED: read complete chunks)
        if chunks_to_load:
            for cy, cx, cache_key in chunks_to_load:
                # Calculate FULL chunk boundaries
                cy_chunk_start = cy * chunk_y
                cy_chunk_end = min((cy + 1) * chunk_y, shape[-2])
                cx_chunk_start = cx * chunk_x
                cx_chunk_end = min((cx + 1) * chunk_x, shape[-1])
                
                try:
                    # CHUNK-ALIGNED READ: Read full chunks in Y,X but only the slice we need in Z
                    # Loading full Z-chunks would be wasteful (64+ slices when we need 1!)
                    if ndim == 2:
                        chunk_data = np.asarray(array[cy_chunk_start:cy_chunk_end, cx_chunk_start:cx_chunk_end])
                    elif ndim == 3:
                        # Read just the slice we need, but aligned in Y,X
                        chunk_data = np.asarray(array[slice_idx, cy_chunk_start:cy_chunk_end, cx_chunk_start:cx_chunk_end])
                    elif ndim == 4:
                        chunk_data = np.asarray(array[slice_idx, 0, cy_chunk_start:cy_chunk_end, cx_chunk_start:cx_chunk_end])
                    elif ndim == 5:
                        chunk_data = np.asarray(array[slice_idx, 0, 0, cy_chunk_start:cy_chunk_end, cx_chunk_start:cx_chunk_end])
                    else:
                        continue
                    
                    # Make a copy for caching (asarray may give view)
                    chunk_data = np.array(chunk_data)
                    
                    # Cache it (release lock quickly)
                    self.chunk_cache.put(cache_key, chunk_data)
                    chunk_positions.append((cy, cx, chunk_data))
                    
                except Exception as e:
                    print(f"Error loading chunk ({cy}, {cx}): {e}")
                    continue
        
        # Assemble result from all chunks
        for cy, cx, chunk_data in chunk_positions:
            chunk_y_start_abs = cy * chunk_y
            chunk_x_start_abs = cx * chunk_x
            
            # Source region in chunk
            src_y_start = max(0, y_start - chunk_y_start_abs)
            src_y_end = min(chunk_data.shape[0], y_end - chunk_y_start_abs)
            src_x_start = max(0, x_start - chunk_x_start_abs)
            src_x_end = min(chunk_data.shape[1], x_end - chunk_x_start_abs)
            
            # Destination region in result
            dst_y_start = max(0, chunk_y_start_abs - y_start + src_y_start)
            dst_y_end = dst_y_start + (src_y_end - src_y_start)
            dst_x_start = max(0, chunk_x_start_abs - x_start + src_x_start)
            dst_x_end = dst_x_start + (src_x_end - src_x_start)
            
            if dst_y_end > dst_y_start and dst_x_end > dst_x_start:
                result[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                    chunk_data[src_y_start:src_y_end, src_x_start:src_x_end]
        
        return result
    
    def get_chunk_data(self, level, slice_idx, y_range, x_range):
        """Load data chunk - fallback for non-chunked arrays"""
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
                return np.asarray(array[y_start:y_end, x_start:x_end])
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
        """Load full 3D volume for 3D visualization"""
        if level < 0 or level >= len(self.levels):
            level = 0
        
        array = self.levels[level]
        shape = array.shape
        
        print(f"Loading full volume from level {level}, shape {shape}")
        
        try:
            if len(shape) == 3:
                volume = np.asarray(array[:, :, :])
            elif len(shape) == 4:
                volume = np.array(array[0, :, :, :])
            elif len(shape) == 5:
                volume = np.array(array[0, 0, :, :, :])
            else:
                print(f"Unsupported shape: {shape}")
                return None
            
            if downsample > 1:
                volume = volume[::downsample, ::downsample, ::downsample]
                print(f"Downsampled to {volume.shape}")
            
            return volume
            
        except Exception as e:
            print(f"Error loading full volume: {e}")
            import traceback
            traceback.print_exc()
            return None