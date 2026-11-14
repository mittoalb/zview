"""Multi-resolution Zarr pyramid loader with LRU chunk cache + parallel I/O."""

import math
import os
import threading
import time
import traceback
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

import numpy as np

from zview import _log
from zview.sysinfo import default_cache_mb, default_io_workers, system_memory

# Chunk-fetch concurrency, sized to the machine (override: ZVIEW_IO_THREADS).
# zarr/z5py release the GIL during decompression, so threads scale well.
_IO_WORKERS = default_io_workers()
_IO_POOL = ThreadPoolExecutor(max_workers=_IO_WORKERS, thread_name_prefix="zview-io")

# Honor per-level `translation` from OME-NGFF coordinateTransformations.
# Set ZVIEW_NO_TRANSLATION=1 to disable (use if your dataset writes
# translations in a non-standard convention and levels visibly jump).
_APPLY_TRANSLATION = os.environ.get("ZVIEW_NO_TRANSLATION", "").strip() not in ("1", "true", "True")


def io_workers():
    return _IO_WORKERS


class Volume3D:
    """Present any 2D-5D array as (z, y, x).

    OME-NGFF arrays are commonly (t, c, z, y, x) or (c, z, y, x). The loaders
    index z/y/x positionally, so leading axes are pinned to index 0 here
    instead of in every call site (previously the 4D/5D branches put the
    slice index on the *time* axis). 2D arrays become a single-slice volume.
    """

    def __init__(self, array):
        self.array = array
        shape = tuple(int(s) for s in array.shape)
        if len(shape) < 2:
            raise ValueError(f"Need at least a 2D array, got shape {shape}")
        self._lead = max(0, len(shape) - 3)
        self._promote = len(shape) == 2
        self.shape = (1,) + shape if self._promote else shape[-3:]
        self.ndim = 3
        self.dtype = array.dtype
        chunks = getattr(array, "chunks", None)
        if chunks is not None:
            chunks = tuple(int(c) for c in chunks)
            chunks = (1,) + chunks if self._promote else chunks[-3:]
        self.chunks = chunks
        self.raw_shape = shape
        self.attrs = getattr(array, "attrs", {})

    @property
    def nbytes(self):
        return math.prod(self.shape) * np.dtype(self.dtype).itemsize

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        key = key + (slice(None),) * (3 - len(key))
        if self._promote:
            z, y, x = key
            out = np.asarray(self.array[y, x])
            # Re-insert the z axis if the caller asked for a z-range.
            return out[np.newaxis] if isinstance(z, slice) else out
        return np.asarray(self.array[(0,) * self._lead + key])


class ChunkCache:
    """Thread-safe LRU cache for decoded chunks, bounded by bytes."""

    def __init__(self, max_memory_mb=500):
        self.cache = OrderedDict()
        self.current_memory = 0
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self.hits = 0
        self.misses = 0
        self.lock = threading.Lock()

    @staticmethod
    def get_key(level, slice_idx, chunk_pos):
        return (level, slice_idx, chunk_pos)

    def get(self, key):
        with self.lock:
            data = self.cache.get(key)
            if data is None:
                self.misses += 1
                return None
            self.cache.move_to_end(key)
            self.hits += 1
            return data

    def get_many(self, keys):
        results = {}
        with self.lock:
            for key in keys:
                data = self.cache.get(key)
                if data is None:
                    self.misses += 1
                else:
                    self.cache.move_to_end(key)
                    self.hits += 1
                results[key] = data
        return results

    def _evict_locked(self, incoming=0):
        while self.cache and self.current_memory + incoming > self.max_memory_bytes:
            _, old = self.cache.popitem(last=False)
            self.current_memory -= old.nbytes

    def put(self, key, data):
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return
            size = data.nbytes
            if size > self.max_memory_bytes:
                return
            self._evict_locked(size)
            self.cache[key] = data
            self.current_memory += size

    def set_max_memory(self, max_memory_mb):
        with self.lock:
            self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
            self._evict_locked()

    @property
    def max_memory_mb(self):
        return self.max_memory_bytes / (1024 * 1024)

    def get_stats(self):
        with self.lock:
            total = self.hits + self.misses
            return {
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': self.hits / total if total > 0 else 0,
                'chunks': len(self.cache),
                'memory_mb': self.current_memory / (1024 * 1024),
                'max_mb': self.max_memory_bytes / (1024 * 1024),
            }

    def clear(self):
        with self.lock:
            self.cache.clear()
            self.current_memory = 0
            self.hits = 0
            self.misses = 0


class VolumeReadCancelled(Exception):
    pass


class MultiResolutionImage:
    """Manages multi-resolution Zarr pyramid with efficient chunk-based loading"""

    def __init__(self, zarr_group, cache_size_mb=None):
        self.zarr_group = zarr_group
        self.levels = []      # Volume3D per level, finest first
        self.scales = []
        self.transforms = []
        self.axis_names = None
        self.axis_units = None
        if cache_size_mb is None:
            cache_size_mb = default_cache_mb()
        self.chunk_cache = ChunkCache(max_memory_mb=cache_size_mb)
        self._discover_pyramid()

    # ------------------------------------------------------------------ meta

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

    def voxel_size(self):
        """Level-0 physical voxel size (z, y, x) from metadata (1s if absent)."""
        if not self.transforms:
            return (1.0, 1.0, 1.0)
        s = self.transforms[0].get('scale', [1.0, 1.0, 1.0])
        out = []
        for v in s[-3:]:
            try:
                v = float(v)
            except (TypeError, ValueError):
                v = 1.0
            out.append(v if v > 0 and math.isfinite(v) else 1.0)
        return tuple(out)

    def level_downsample(self, level):
        """(z, y, x) size ratio of level 0 to `level`."""
        s0 = self.levels[0].shape
        sl = self.levels[level].shape
        return tuple(a / b if b else 1.0 for a, b in zip(s0, sl))

    def _discover_pyramid(self):
        try:
            attrs = dict(self.zarr_group.attrs) if hasattr(self.zarr_group, 'attrs') else {}
            # OME-NGFF 0.5 nests metadata under "ome".
            if 'multiscales' not in attrs and isinstance(attrs.get('ome'), dict):
                attrs = attrs['ome']

            if 'multiscales' in attrs:
                multiscales = attrs['multiscales']
                if isinstance(multiscales, list) and len(multiscales) > 0:
                    ms = multiscales[0]
                    axes = ms.get('axes')
                    if isinstance(axes, list) and axes:
                        names = [a.get('name') if isinstance(a, dict) else str(a) for a in axes]
                        units = [a.get('unit') if isinstance(a, dict) else None for a in axes]
                        self.axis_names = names[-3:]
                        self.axis_units = units[-3:]
                    for ds in ms.get('datasets', []):
                        path = ds.get('path', '')
                        if path not in self.zarr_group:
                            continue
                        self.levels.append(Volume3D(self.zarr_group[path]))

                        scale = [1.0, 1.0, 1.0]
                        translation = [0.0, 0.0, 0.0]
                        for t in ds.get('coordinateTransformations', []):
                            if t.get('type') == 'scale':
                                vals = t.get('scale', [1.0, 1.0, 1.0])
                                if len(vals) >= 3:
                                    scale = list(vals[-3:])
                                elif len(vals) >= 2:
                                    scale = [1.0] + list(vals[-2:])
                            elif t.get('type') == 'translation':
                                vals = t.get('translation', [0.0, 0.0, 0.0])
                                if len(vals) >= 3:
                                    translation = list(vals[-3:])
                                elif len(vals) >= 2:
                                    translation = [0.0] + list(vals[-2:])

                        self.transforms.append({'scale': scale, 'translation': translation})
                        self.scales.append(scale[-2:])
        except Exception as e:
            _log.warn(f"Could not parse multiscales metadata: {e}")

        if not self.levels:
            self.transforms = []
            self.scales = []
            level_idx = 0
            while str(level_idx) in self.zarr_group:
                try:
                    array = self.zarr_group[str(level_idx)]
                    if not (hasattr(array, 'shape') and hasattr(array, 'dtype')):
                        break
                    vol = Volume3D(array)
                    self.levels.append(vol)
                    s0 = self.levels[0].shape
                    sc = [a / b for a, b in zip(s0, vol.shape)]
                    self.scales.append(sc[-2:])
                    self.transforms.append({'scale': sc, 'translation': [0.0, 0.0, 0.0]})
                    level_idx += 1
                except Exception as e:
                    _log.warn(f"Could not load level {level_idx}: {e}")
                    break

        if not self.levels:
            raise ValueError("No pyramid levels found in Zarr group")

        _log.info(f"Found {len(self.levels)} pyramid levels")
        for i, level in enumerate(self.levels):
            ds = self.level_downsample(i)
            tr = self.transforms[i] if i < len(self.transforms) else {}
            raw = level.raw_shape
            raw_note = f" (raw {raw})" if raw != level.shape else ""
            _log.info(
                f"  Level {i}: shape={level.shape}{raw_note}, chunks={level.chunks}, "
                f"downsample={ds[1]:.2f}x, scale={tr.get('scale')}, "
                f"translation={tr.get('translation')}"
            )
        if not _APPLY_TRANSLATION:
            _log.info("  (ZVIEW_NO_TRANSLATION=1 — per-level translations are IGNORED)")

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
        required_downsample = 1.0 / max(view_scale, 1e-9)

        def level_downsample(level):
            # Coarsest of the pane's two axes. Averaging under-weights the
            # axis that *is* downsampled when the other isn't (e.g. z kept at
            # full resolution), which picked a level ~2x too coarse in XZ/YZ.
            level_shape = self.levels[level].shape
            return max(level_0_shape[a] / level_shape[a] for a in axes)

        if current_level is not None and 0 <= current_level < len(self.levels):
            ds_curr = level_downsample(current_level)
            if ds_curr * 0.5 <= required_downsample <= ds_curr * 1.5:
                return current_level

        for level in range(len(self.levels)):
            if level_downsample(level) >= required_downsample * 0.67:
                return level

        return len(self.levels) - 1

    # ------------------------------------------------------------------ 2D

    def get_slice_2d(self, level, axis, slice_idx):
        """Load a full 2D plane (used for the initial frame only)."""
        shape = self.get_level_shape(level)
        if shape is None:
            return None
        z_size, y_size, x_size = shape
        if axis == 'xy':
            return self.get_slice_2d_region(level, axis, slice_idx, (0, y_size), (0, x_size))
        if axis == 'xz':
            return self.get_slice_2d_region(level, axis, slice_idx, (0, z_size), (0, x_size))
        if axis == 'yz':
            return self.get_slice_2d_region(level, axis, slice_idx, (0, z_size), (0, y_size))
        return None

    def get_slice_2d_region(self, level, axis, slice_idx, v_range, h_range):
        """Load a sub-rectangle of a 2D slice without materializing the whole plane.

        v_range/h_range are (start, end) in the level's own pixel space for the
        on-screen vertical and horizontal axes::

            axis='xy' -> v=y, h=x, fixed=z=slice_idx
            axis='xz' -> v=z, h=x, fixed=y=slice_idx
            axis='yz' -> v=z, h=y, fixed=x=slice_idx
        """
        if level < 0 or level >= len(self.levels):
            return None

        array = self.levels[level]
        z_size, y_size, x_size = array.shape
        v0, v1 = v_range
        h0, h1 = h_range

        try:
            if axis == 'xy':
                z = min(max(slice_idx, 0), z_size - 1)
                return self.get_chunk_data_aligned(level, z, (v0, v1), (h0, h1))

            if axis == 'xz':
                y = min(max(slice_idx, 0), y_size - 1)
                v0 = max(0, min(v0, z_size)); v1 = max(v0, min(v1, z_size))
                h0 = max(0, min(h0, x_size)); h1 = max(h0, min(h1, x_size))
                if v1 <= v0 or h1 <= h0:
                    return np.zeros((max(0, v1 - v0), max(0, h1 - h0)), dtype=array.dtype)
                return self._load_slice_chunked(array, 'xz', v0, v1, h0, h1, y)

            if axis == 'yz':
                x = min(max(slice_idx, 0), x_size - 1)
                v0 = max(0, min(v0, z_size)); v1 = max(v0, min(v1, z_size))
                h0 = max(0, min(h0, y_size)); h1 = max(h0, min(h1, y_size))
                if v1 <= v0 or h1 <= h0:
                    return np.zeros((max(0, v1 - v0), max(0, h1 - h0)), dtype=array.dtype)
                return self._load_slice_chunked(array, 'yz', v0, v1, h0, h1, x)

        except Exception as e:
            _log.error(f"Error loading slice region ({axis}): {e}")
            return None

        return None

    def _load_slice_chunked(self, array, axis, v0, v1, h0, h1, fixed_idx):
        """Chunk-aligned, parallel loader for an XZ or YZ sub-region.

        Replaces direct zarr slicing (`array[v0:v1, y, h0:h1]`) which reads
        intersecting chunks serially. Loading in parallel yields a straight
        4-8x speedup on local disk.
        """
        chunks = array.chunks
        if chunks is None:
            if axis == 'xz':
                return np.asarray(array[v0:v1, fixed_idx, h0:h1])
            return np.asarray(array[v0:v1, h0:h1, fixed_idx])

        chunk_z, chunk_y, chunk_x = chunks
        cv = chunk_z
        ch = chunk_x if axis == 'xz' else chunk_y
        h_axis_size = array.shape[2] if axis == 'xz' else array.shape[1]

        result = np.zeros((v1 - v0, h1 - h0), dtype=array.dtype)
        tasks = [
            (cvi, chi)
            for cvi in range(v0 // cv, (v1 - 1) // cv + 1)
            for chi in range(h0 // ch, (h1 - 1) // ch + 1)
        ]

        def _fetch(coord):
            cvi, chi = coord
            # Only read the part of the chunk that's inside the request.
            v_s = max(cvi * cv, v0)
            v_e = min((cvi + 1) * cv, v1, array.shape[0])
            h_s = max(chi * ch, h0)
            h_e = min((chi + 1) * ch, h1, h_axis_size)
            try:
                if axis == 'xz':
                    data = array[v_s:v_e, fixed_idx, h_s:h_e]
                else:
                    data = array[v_s:v_e, h_s:h_e, fixed_idx]
                return v_s, h_s, data
            except Exception as e:
                _log.error(f"Error loading {axis} chunk ({cvi},{chi}): {e}")
                return v_s, h_s, None

        for v_s, h_s, data in _IO_POOL.map(_fetch, tasks):
            if data is None:
                continue
            result[v_s - v0:v_s - v0 + data.shape[0], h_s - h0:h_s - h0 + data.shape[1]] = data
        return result

    def get_chunk_data_aligned(self, level, slice_idx, y_range, x_range):
        """XY sub-region at z=slice_idx, assembled from (cached) whole chunks."""
        if level < 0 or level >= len(self.levels):
            return None

        array = self.levels[level]
        shape = array.shape
        if array.chunks is None:
            return self.get_chunk_data(level, slice_idx, y_range, x_range)
        chunk_y, chunk_x = array.chunks[1], array.chunks[2]

        y_start = max(0, min(y_range[0], shape[1]))
        y_end = max(0, min(y_range[1], shape[1]))
        x_start = max(0, min(x_range[0], shape[2]))
        x_end = max(0, min(x_range[1], shape[2]))
        if y_start >= y_end or x_start >= x_end:
            return np.zeros((max(0, y_end - y_start), max(0, x_end - x_start)), dtype=array.dtype)

        slice_idx = min(max(slice_idx, 0), shape[0] - 1)
        result = np.zeros((y_end - y_start, x_end - x_start), dtype=array.dtype)

        coords = [
            (cy, cx)
            for cy in range(y_start // chunk_y, (y_end - 1) // chunk_y + 1)
            for cx in range(x_start // chunk_x, (x_end - 1) // chunk_x + 1)
        ]
        keys = [self.chunk_cache.get_key(level, slice_idx, c) for c in coords]
        cached = self.chunk_cache.get_many(keys)

        placed = []
        to_load = []
        for (cy, cx), key in zip(coords, keys):
            data = cached[key]
            if data is None:
                to_load.append((cy, cx, key))
            else:
                placed.append((cy, cx, data))

        if to_load:
            def _fetch(args):
                cy, cx, key = args
                try:
                    data = array[slice_idx,
                                 cy * chunk_y:min((cy + 1) * chunk_y, shape[1]),
                                 cx * chunk_x:min((cx + 1) * chunk_x, shape[2])]
                    return cy, cx, key, np.ascontiguousarray(data)
                except Exception as e:
                    _log.error(f"Error loading chunk ({cy}, {cx}): {e}")
                    return cy, cx, key, None

            for cy, cx, key, data in _IO_POOL.map(_fetch, to_load):
                if data is None:
                    continue
                self.chunk_cache.put(key, data)
                placed.append((cy, cx, data))

        for cy, cx, data in placed:
            cy0 = cy * chunk_y
            cx0 = cx * chunk_x
            sy0 = max(0, y_start - cy0)
            sy1 = min(data.shape[0], y_end - cy0)
            sx0 = max(0, x_start - cx0)
            sx1 = min(data.shape[1], x_end - cx0)
            if sy1 <= sy0 or sx1 <= sx0:
                continue
            dy0 = cy0 + sy0 - y_start
            dx0 = cx0 + sx0 - x_start
            result[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)] = data[sy0:sy1, sx0:sx1]

        return result

    def get_chunk_data(self, level, slice_idx, y_range, x_range):
        """Direct (unchunked) XY sub-region read; fallback for chunkless arrays."""
        if level < 0 or level >= len(self.levels):
            return None
        array = self.levels[level]
        shape = array.shape
        y_start = max(0, min(y_range[0], shape[1]))
        y_end = max(0, min(y_range[1], shape[1]))
        x_start = max(0, min(x_range[0], shape[2]))
        x_end = max(0, min(x_range[1], shape[2]))
        if y_start >= y_end or x_start >= x_end:
            return np.zeros((max(0, y_end - y_start), max(0, x_end - x_start)), dtype=array.dtype)
        try:
            z = min(max(slice_idx, 0), shape[0] - 1)
            return np.array(array[z, y_start:y_end, x_start:x_end])
        except Exception as e:
            _log.error(f"Error loading chunk: {e}")
            return np.zeros((y_end - y_start, x_end - x_start), dtype=array.dtype)

    # ------------------------------------------------------------------ 3D

    def estimate_read_bytes(self, level, stride=(1, 1, 1)):
        """Bytes `read_volume(level, stride)` must decode.

        Chunks are decoded whole, so along an axis where the stride is smaller
        than the chunk every chunk is touched; only strides larger than the
        chunk skip chunks. Used to decide whether a preview is worth it and to
        tell the user what a load costs.
        """
        array = self.levels[level]
        chunks = array.chunks or array.shape
        total = np.dtype(array.dtype).itemsize
        for n, c, s in zip(array.shape, chunks, stride):
            c = max(1, min(int(c), n))
            s = max(1, int(s))
            if s <= c:
                touched = -(-n // c)
            else:
                touched = len(np.unique(np.arange(0, n, s) // c))
            total *= min(n, touched * c)
        return int(total)

    def read_volume(self, level, stride=(1, 1, 1), progress=None, cancel=None,
                    executor=None, target_block_bytes=64 * 2**20):
        """Read `level` subsampled by `stride` into a (z, y, x) numpy array.

        Works for arbitrarily large arrays: reads happen block-by-block (whole
        chunks grouped into ~64 MB blocks), in parallel, with a bounded number
        of blocks in flight, so peak memory is the output plus a few blocks.
        Blocks containing no sampled voxel are skipped entirely, so a large
        stride also cuts I/O, not just memory.

        progress(done, total) is called from the calling thread; cancel is a
        threading.Event checked between blocks (raises VolumeReadCancelled).
        """
        array = self.levels[level]
        shape = array.shape
        sz, sy, sx = (max(1, int(s)) for s in stride)
        out_shape = tuple(math.ceil(d / s) for d, s in zip(shape, (sz, sy, sx)))
        out = np.zeros(out_shape, dtype=array.dtype)
        itemsize = np.dtype(array.dtype).itemsize

        # Block size: whole chunks, grown along x, then y, then z until the
        # block is ~target_block_bytes. Tiny chunks would otherwise make
        # millions of tasks; giant chunks are used as-is.
        chunks = list(array.chunks or (min(shape[0], 64), min(shape[1], 256), min(shape[2], 256)))
        chunks = [max(1, min(c, d)) for c, d in zip(chunks, shape)]
        block = list(chunks)
        strides = (sz, sy, sx)
        for ax in (2, 1, 0):
            # Along axes where the stride exceeds the chunk, keep one chunk
            # per block so chunks without a sampled voxel are never decoded
            # (matches estimate_read_bytes).
            if strides[ax] > chunks[ax]:
                continue
            while (block[ax] < shape[ax]
                   and math.prod(block) * itemsize * 2 <= target_block_bytes):
                block[ax] = min(shape[ax], block[ax] + chunks[ax])

        def _ranges(ax, s):
            """Block [start, end) ranges along `ax` that contain a sampled index."""
            n, b = shape[ax], block[ax]
            out_r = []
            for start in range(0, n, b):
                end = min(n, start + b)
                first = -(-start // s) * s  # first multiple of s >= start
                if first < end:
                    out_r.append((start, end, first - start))
            return out_r

        zr, yr, xr = _ranges(0, sz), _ranges(1, sy), _ranges(2, sx)
        tasks = [(z, y, x) for z in zr for y in yr for x in xr]
        total = len(tasks)

        def _fetch(task):
            (z0, z1, oz), (y0, y1, oy), (x0, x1, ox) = task
            if cancel is not None and cancel.is_set():
                return task, None
            data = array[z0:z1, y0:y1, x0:x1]
            if (sz, sy, sx) != (1, 1, 1):
                # Copy the sample so the full decoded block can be freed now
                # rather than when the result is consumed.
                data = np.ascontiguousarray(data[oz::sz, oy::sy, ox::sx])
            return task, data

        pool = executor or _IO_POOL
        # Bound decoded bytes in flight (not just task count): with 64 MB
        # chunks, "2 per worker" could otherwise hold several GB at once.
        block_bytes = max(1, math.prod(block) * itemsize)
        _, avail = system_memory()
        inflight_budget = min(1 << 30, int(avail * 0.1)) if avail else 512 << 20
        max_inflight = max(2, min(getattr(pool, "_max_workers", 4) * 2,
                                  inflight_budget // block_bytes))
        it = iter(tasks)
        inflight = set()
        done_count = 0

        def _submit_more():
            while len(inflight) < max_inflight:
                try:
                    t = next(it)
                except StopIteration:
                    return
                inflight.add(pool.submit(_fetch, t))

        try:
            _submit_more()
            while inflight:
                finished, _ = wait(inflight, timeout=0.25, return_when=FIRST_COMPLETED)
                if cancel is not None and cancel.is_set():
                    raise VolumeReadCancelled()
                for fut in finished:
                    inflight.discard(fut)
                    task, data = fut.result()
                    if data is None:
                        continue
                    (z0, _, oz), (y0, _, oy), (x0, _, ox) = task
                    dz, dy, dx = (z0 + oz) // sz, (y0 + oy) // sy, (x0 + ox) // sx
                    out[dz:dz + data.shape[0], dy:dy + data.shape[1], dx:dx + data.shape[2]] = data
                    done_count += 1
                    if progress is not None:
                        progress(done_count, total)
                _submit_more()
        finally:
            for fut in inflight:
                fut.cancel()
        return out

    def get_full_volume(self, level=0, downsample=1):
        """Whole level as a numpy array (kept for API compatibility)."""
        if level < 0 or level >= len(self.levels):
            level = 0
        _log.info(f"Loading full volume from level {level}, shape {self.levels[level].shape}")
        try:
            t0 = time.perf_counter()
            vol = self.read_volume(level, (downsample,) * 3)
            _log.info(f"  -> {vol.shape} in {time.perf_counter() - t0:.1f}s")
            return vol
        except Exception as e:
            _log.error(f"Error loading full volume: {e}")
            traceback.print_exc()
            return None
