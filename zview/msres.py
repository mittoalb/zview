"""Multi-resolution Zarr pyramid loader with LRU chunk cache + parallel I/O."""

import os
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# Chunk-fetch concurrency. z5py releases the GIL during I/O so a thread pool
# gives a straight 4-8x speedup on local disk; override with ZVIEW_IO_THREADS.
_IO_WORKERS = max(1, int(os.environ.get("ZVIEW_IO_THREADS", "8")))
_IO_POOL = ThreadPoolExecutor(max_workers=_IO_WORKERS, thread_name_prefix="zview-io")

# Honor per-level `translation` from OME-NGFF coordinateTransformations.
# Set ZVIEW_NO_TRANSLATION=1 to disable (use if your dataset writes
# translations in a non-standard convention and levels visibly jump).
_APPLY_TRANSLATION = os.environ.get("ZVIEW_NO_TRANSLATION", "").strip() not in ("1", "true", "True")


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
        return (level, slice_idx, chunk_pos)
    
    def get(self, key):
        with self.lock:
            if key in self.cache:
                self.access_order.remove(key)
                self.access_order.append(key)
                self.hits += 1
                return self.cache[key]
            self.misses += 1
            return None
    
    def get_many(self, keys):
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
        with self.lock:
            if key in self.cache:
                return
            
            data_size = data.nbytes
            
            while (self.current_memory + data_size > self.max_memory_bytes and 
                   self.access_order):
                old_key = self.access_order.pop(0)
                if old_key in self.cache:
                    old_data = self.cache.pop(old_key)
                    self.current_memory -= old_data.nbytes
            
            if data_size <= self.max_memory_bytes:
                self.cache[key] = data
                self.access_order.append(key)
                self.current_memory += data_size
    
    def get_stats(self):
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

    def get_world_offset(self, level):
        """Per-axis translation of `level` relative to level 0, in level-0 pixel units.

        OME-NGFF (and Neuroglancer's zarr reader) store `translation` in the
        same units as `scale`, which may be physical (nm/µm) rather than pixels.
        This viewer uses level-0 pixel space as its world space, so the raw
        translation difference must be divided by level-0's per-axis scale.
        Returned as (z_offset, y_offset, x_offset).

        If ZVIEW_NO_TRANSLATION=1 is set, returns zeros — use this if your data
        writes per-level translations with a non-standard convention and levels
        visibly jump when switching.
        """
        if not _APPLY_TRANSLATION:
            return (0.0, 0.0, 0.0)
        if level < 0 or level >= len(self.transforms):
            return (0.0, 0.0, 0.0)
        t_level = self.transforms[level].get('translation', [0.0, 0.0, 0.0])
        t_0 = self.transforms[0].get('translation', [0.0, 0.0, 0.0])
        s_0 = self.transforms[0].get('scale', [1.0, 1.0, 1.0])

        def axis(i):
            denom = s_0[i] if s_0[i] not in (0, 0.0) else 1.0
            return float((t_level[i] - t_0[i]) / denom)

        return (axis(0), axis(1), axis(2))

    def _discover_pyramid(self):
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
                            
                            scale = [1.0, 1.0, 1.0]
                            translation = [0.0, 0.0, 0.0]
                            
                            if 'coordinateTransformations' in ds:
                                transforms = ds['coordinateTransformations']
                                for t in transforms:
                                    if t.get('type') == 'scale':
                                        scale_vals = t.get('scale', [1.0, 1.0, 1.0])
                                        if len(scale_vals) >= 3:
                                            scale = scale_vals[-3:]
                                        elif len(scale_vals) >= 2:
                                            scale = [1.0] + scale_vals[-2:]
                                    elif t.get('type') == 'translation':
                                        trans_vals = t.get('translation', [0.0, 0.0, 0.0])
                                        if len(trans_vals) >= 3:
                                            translation = trans_vals[-3:]
                            
                            self.transforms.append({
                                'scale': scale,
                                'translation': translation
                            })
                            
                            self.scales.append(scale[-2:])
        except Exception as e:
            print(f"Note: Could not parse multiscales metadata: {e}")
        
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
            level0_shape = self.levels[0].shape
            downsample_y = level0_shape[-2] / level.shape[-2]
            downsample_x = level0_shape[-1] / level.shape[-1]
            avg_ds = (downsample_y + downsample_x) / 2.0
            tr = self.transforms[i] if i < len(self.transforms) else {}
            print(
                f"  Level {i}: shape={level.shape}, chunks={chunks}, "
                f"downsample={avg_ds:.2f}x, scale={tr.get('scale')}, "
                f"translation={tr.get('translation')}"
            )
        if not _APPLY_TRANSLATION:
            print("  (ZVIEW_NO_TRANSLATION=1 — per-level translations are IGNORED)")
    
    def get_num_levels(self):
        return len(self.levels)

    def get_level_shape(self, level):
        if 0 <= level < len(self.levels):
            return self.levels[level].shape
        return None

    def get_optimal_level(self, view_scale, axes=(-2, -1), current_level=None):
        """Pick the best pyramid level for a given view scale.

        axes: pair of axis indices whose downsample drives selection. XY uses
            (-2, -1); XZ needs (-3, -1); YZ needs (-3, -2). Anisotropic volumes
            otherwise pick an over-coarse level on XZ/YZ panes.
        current_level: when supplied, apply one-octave hysteresis around the
            current level so small pans don't thrash between levels.
        """
        level_0_shape = self.levels[0].shape
        required_downsample = 1.0 / view_scale

        def level_downsample(level):
            level_shape = self.levels[level].shape
            vals = [level_0_shape[a] / level_shape[a] for a in axes]
            return sum(vals) / len(vals)

        if current_level is not None and 0 <= current_level < len(self.levels):
            ds_curr = level_downsample(current_level)
            if ds_curr * 0.5 <= required_downsample <= ds_curr * 1.5:
                return current_level

        for level in range(len(self.levels)):
            if level_downsample(level) >= required_downsample * 0.67:
                return level

        return len(self.levels) - 1
    
    def get_slice_2d(self, level, axis, slice_idx):
        """Load a full 2D plane (used for the initial frame only).

        For per-frame reloads use `get_slice_2d_region`, which fetches only the
        visible sub-region; loading the entire XZ/YZ plane at a fine level can
        pull hundreds of MB per view change.
        """
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

    def get_slice_2d_region(self, level, axis, slice_idx, v_range, h_range):
        """Load a sub-rectangle of a 2D slice without materializing the whole plane.

        v_range/h_range are (start, end) in the level's own pixel space for the
        on-screen vertical and horizontal axes:
            axis='xy' -> v=y, h=x, fixed=z=slice_idx
            axis='xz' -> v=z, h=x, fixed=y=slice_idx
            axis='yz' -> v=z, h=y, fixed=x=slice_idx
        """
        if level < 0 or level >= len(self.levels):
            return None

        array = self.levels[level]
        shape = array.shape
        if len(shape) < 3:
            return None

        z_size, y_size, x_size = shape[-3], shape[-2], shape[-1]
        v0, v1 = v_range
        h0, h1 = h_range

        try:
            if axis == 'xy':
                z = min(max(slice_idx, 0), z_size - 1)
                v0 = max(0, min(v0, y_size)); v1 = max(v0, min(v1, y_size))
                h0 = max(0, min(h0, x_size)); h1 = max(h0, min(h1, x_size))
                # Chunk-aligned path benefits from the LRU chunk cache
                return self.get_chunk_data_aligned(level, z, (v0, v1), (h0, h1))

            elif axis == 'xz':
                y = min(max(slice_idx, 0), y_size - 1)
                v0 = max(0, min(v0, z_size)); v1 = max(v0, min(v1, z_size))
                h0 = max(0, min(h0, x_size)); h1 = max(h0, min(h1, x_size))
                if v1 <= v0 or h1 <= h0:
                    return np.zeros((max(0, v1 - v0), max(0, h1 - h0)), dtype=array.dtype)
                return self._load_slice_chunked(array, 'xz', v0, v1, h0, h1, y)

            elif axis == 'yz':
                x = min(max(slice_idx, 0), x_size - 1)
                v0 = max(0, min(v0, z_size)); v1 = max(v0, min(v1, z_size))
                h0 = max(0, min(h0, y_size)); h1 = max(h0, min(h1, y_size))
                if v1 <= v0 or h1 <= h0:
                    return np.zeros((max(0, v1 - v0), max(0, h1 - h0)), dtype=array.dtype)
                return self._load_slice_chunked(array, 'yz', v0, v1, h0, h1, x)

        except Exception as e:
            print(f"Error loading slice region ({axis}): {e}")
            return None

        return None

    def _load_slice_chunked(self, array, axis, v0, v1, h0, h1, fixed_idx):
        """Chunk-aligned, parallel loader for an XZ or YZ sub-region.

        Replaces direct zarr slicing (`array[v0:v1, y, h0:h1]`) which reads
        intersecting chunks serially inside z5py. Loading in parallel yields a
        straight 4-8x speedup on local disk.
        """
        chunks = getattr(array, 'chunks', None)
        if chunks is None or len(chunks) < 3:
            if axis == 'xz':
                return np.asarray(array[v0:v1, fixed_idx, h0:h1])
            return np.asarray(array[v0:v1, h0:h1, fixed_idx])

        chunk_z, chunk_y, chunk_x = chunks[-3], chunks[-2], chunks[-1]
        if axis == 'xz':
            cv = chunk_z
            ch = chunk_x
        else:  # yz
            cv = chunk_z
            ch = chunk_y

        cv0 = v0 // cv
        cv1 = (v1 - 1) // cv + 1
        ch0 = h0 // ch
        ch1 = (h1 - 1) // ch + 1

        result = np.zeros((v1 - v0, h1 - h0), dtype=array.dtype)
        tasks = []
        for cvi in range(cv0, cv1):
            for chi in range(ch0, ch1):
                tasks.append((cvi, chi))

        def _fetch(coord):
            cvi, chi = coord
            v_s = cvi * cv
            v_e = min(v_s + cv, array.shape[-3])
            h_s_abs = chi * ch
            h_axis_size = array.shape[-1] if axis == 'xz' else array.shape[-2]
            h_e = min(h_s_abs + ch, h_axis_size)
            try:
                if axis == 'xz':
                    data = np.asarray(array[v_s:v_e, fixed_idx, h_s_abs:h_e])
                else:  # yz
                    data = np.asarray(array[v_s:v_e, h_s_abs:h_e, fixed_idx])
                return (cvi, chi, np.array(data))
            except Exception as e:
                print(f"Error loading {axis} chunk ({cvi},{chi}): {e}")
                return (cvi, chi, None)

        for cvi, chi, data in _IO_POOL.map(_fetch, tasks):
            if data is None:
                continue
            v_s_abs = cvi * cv
            h_s_abs = chi * ch
            # Overlap of chunk's absolute range with the requested [v0,v1)×[h0,h1).
            src_v_s = max(0, v0 - v_s_abs)
            src_v_e = min(data.shape[0], v1 - v_s_abs)
            src_h_s = max(0, h0 - h_s_abs)
            src_h_e = min(data.shape[1], h1 - h_s_abs)
            if src_v_e <= src_v_s or src_h_e <= src_h_s:
                continue
            dst_v_s = max(0, v_s_abs - v0 + src_v_s)
            dst_h_s = max(0, h_s_abs - h0 + src_h_s)
            result[
                dst_v_s:dst_v_s + (src_v_e - src_v_s),
                dst_h_s:dst_h_s + (src_h_e - src_h_s),
            ] = data[src_v_s:src_v_e, src_h_s:src_h_e]
        return result

    def get_chunk_data_aligned(self, level, slice_idx, y_range, x_range):
        if level < 0 or level >= len(self.levels):
            return None
        
        array = self.levels[level]
        shape = array.shape
        ndim = len(shape)
        
        if not hasattr(array, 'chunks'):
            return self.get_chunk_data(level, slice_idx, y_range, x_range)
        
        chunks = array.chunks
        if ndim >= 2:
            chunk_y, chunk_x = chunks[-2], chunks[-1]
        else:
            return self.get_chunk_data(level, slice_idx, y_range, x_range)
        
        y_start, y_end = y_range
        x_start, x_end = x_range
        
        y_start = max(0, min(y_start, shape[-2]))
        y_end = max(0, min(y_end, shape[-2]))
        x_start = max(0, min(x_start, shape[-1]))
        x_end = max(0, min(x_end, shape[-1]))
        
        if y_start >= y_end or x_start >= x_end:
            return np.zeros((y_end - y_start, x_end - x_start), dtype=array.dtype)
        
        chunk_y_start = y_start // chunk_y
        chunk_y_end = (y_end - 1) // chunk_y + 1
        chunk_x_start = x_start // chunk_x
        chunk_x_end = (x_end - 1) // chunk_x + 1
        
        slice_idx = min(slice_idx, shape[-3] - 1) if ndim >= 3 else 0
        
        result = np.zeros((y_end - y_start, x_end - x_start), dtype=array.dtype)
        
        all_cache_keys = []
        chunk_coords = []
        for cy in range(chunk_y_start, chunk_y_end):
            for cx in range(chunk_x_start, chunk_x_end):
                cache_key = self.chunk_cache.get_key(level, slice_idx, (cy, cx))
                all_cache_keys.append(cache_key)
                chunk_coords.append((cy, cx))
        
        cached_chunks = self.chunk_cache.get_many(all_cache_keys)
        
        chunks_to_load = []
        chunk_positions = []
        
        for (cy, cx), cache_key in zip(chunk_coords, all_cache_keys):
            chunk_data = cached_chunks[cache_key]
            if chunk_data is None:
                chunks_to_load.append((cy, cx, cache_key))
            else:
                chunk_positions.append((cy, cx, chunk_data))
        
        if chunks_to_load:
            def _fetch(args):
                cy, cx, cache_key = args
                cy_s = cy * chunk_y
                cy_e = min((cy + 1) * chunk_y, shape[-2])
                cx_s = cx * chunk_x
                cx_e = min((cx + 1) * chunk_x, shape[-1])
                try:
                    if ndim == 2:
                        data = np.asarray(array[cy_s:cy_e, cx_s:cx_e])
                    elif ndim == 3:
                        data = np.asarray(array[slice_idx, cy_s:cy_e, cx_s:cx_e])
                    elif ndim == 4:
                        data = np.asarray(array[slice_idx, 0, cy_s:cy_e, cx_s:cx_e])
                    elif ndim == 5:
                        data = np.asarray(array[slice_idx, 0, 0, cy_s:cy_e, cx_s:cx_e])
                    else:
                        return (cy, cx, cache_key, None)
                    return (cy, cx, cache_key, np.array(data))
                except Exception as e:
                    print(f"Error loading chunk ({cy}, {cx}): {e}")
                    return (cy, cx, cache_key, None)

            # Fetch in parallel — z5py releases the GIL during I/O.
            for cy, cx, cache_key, chunk_data in _IO_POOL.map(_fetch, chunks_to_load):
                if chunk_data is None:
                    continue
                self.chunk_cache.put(cache_key, chunk_data)
                chunk_positions.append((cy, cx, chunk_data))
        
        for cy, cx, chunk_data in chunk_positions:
            chunk_y_start_abs = cy * chunk_y
            chunk_x_start_abs = cx * chunk_x
            
            src_y_start = max(0, y_start - chunk_y_start_abs)
            src_y_end = min(chunk_data.shape[0], y_end - chunk_y_start_abs)
            src_x_start = max(0, x_start - chunk_x_start_abs)
            src_x_end = min(chunk_data.shape[1], x_end - chunk_x_start_abs)
            
            dst_y_start = max(0, chunk_y_start_abs - y_start + src_y_start)
            dst_y_end = dst_y_start + (src_y_end - src_y_start)
            dst_x_start = max(0, chunk_x_start_abs - x_start + src_x_start)
            dst_x_end = dst_x_start + (src_x_end - src_x_start)
            
            if dst_y_end > dst_y_start and dst_x_end > dst_x_start:
                result[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                    chunk_data[src_y_start:src_y_end, src_x_start:src_x_end]
        
        return result
    
    def get_chunk_data(self, level, slice_idx, y_range, x_range):
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
            traceback.print_exc()
            return None