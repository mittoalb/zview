"""HDF5 volumes (no Qt): one stored resolution, coarser levels by striding.

An HDF5 file normally holds a single full-resolution dataset, with no
pyramid. zview opens it as a pyramid anyway: level k takes one voxel every
2**k (per axis, see `pyramid.strided_factors`), read straight from the file
with a strided selection. Nothing is averaged, nothing is written, and only
the sampled rows are read, so even very large files open at once.

    spec = "scan.h5"                 # dataset auto-detected
    spec = "scan.h5:/exchange/data"  # dataset named explicitly
"""

import itertools
import os
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

SUFFIXES = (".h5", ".hdf5", ".hdf", ".he5", ".nxs", ".nx5")
# Checked in order before falling back to the largest dataset in the file.
STANDARD_DATASETS = ("/exchange/data", "/data", "/volume", "/entry/data/data")
# Voxel-size attributes (on the dataset, then on the file), all in µm.
_VOXEL_ATTRS = ("element_size_um", "voxel_size_um", "voxel_size", "pixel_size_um", "pixel_size",
                "resolution", "spacing")
# DXchange stores the pixel size (µm) as a dataset.
_VOXEL_DATASETS = ("/measurement/instrument/detection_system/objective/resolution",)


def split_spec(spec):
    """'file.h5:/path/in/file' -> (file, '/path/in/file'); 'file.h5' -> (file, None)."""
    spec = str(spec)
    if Path(spec).is_file():
        return spec, None
    i = spec.rfind(":")
    while i > 0:
        if Path(spec[:i]).is_file():
            dset = spec[i + 1:].strip()
            return spec[:i], ("/" + dset.lstrip("/")) if dset else None
        i = spec.rfind(":", 0, i)
    return spec, None


def is_hdf5(spec):
    """True if `spec` (a path, optionally with ':/dataset') is an HDF5 file."""
    path, _ = split_spec(spec)
    p = Path(path)
    if not p.is_file():
        return False
    if p.suffix.lower() in SUFFIXES:
        return True
    try:
        import h5py  # noqa: PLC0415
        return bool(h5py.is_hdf5(str(p)))
    except Exception:
        return False


def open_file(path, cache_mb=256):
    """Open read-only with a chunk cache big enough for whole compressed slices.

    The default 1 MB HDF5 chunk cache would decompress a (1, Y, X) chunk again
    for every tile of the same slice. File locking is off so files on NFS, or
    still being written by an acquisition, open too.
    """
    import h5py  # noqa: PLC0415
    kw = dict(rdcc_nbytes=int(cache_mb) << 20, rdcc_nslots=100_003)
    try:
        return h5py.File(str(path), "r", locking=False, **kw)
    except TypeError:  # h5py < 3.5 has no `locking`
        return h5py.File(str(path), "r", **kw)


def candidates(f):
    """[(path, shape, dtype)] of datasets that can be viewed (2D or more), 3D+ and largest first."""
    import h5py  # noqa: PLC0415
    out = []

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset) and obj.ndim >= 2 and min(obj.shape[-2:]) >= 2 \
                and obj.dtype.kind in "uifb":
            out.append(("/" + name, tuple(obj.shape), obj.dtype))

    f.visititems(visit)
    out.sort(key=lambda c: (len(c[1]) < 3, -_size(c[1])))
    return out


def _size(shape):
    n = 1
    for s in shape:
        n *= int(s)
    return n


def find_dataset(f, name=None):
    """The dataset to view: `name` if given, a standard path, else the largest 3D one."""
    import h5py  # noqa: PLC0415
    if name:
        if name not in f or not isinstance(f[name], h5py.Dataset):
            raise KeyError(f"no dataset {name!r} in {f.filename}")
        return name
    for c in STANDARD_DATASETS:
        if c in f and isinstance(f[c], h5py.Dataset) and f[c].ndim >= 3:
            return c
    cands = candidates(f)
    if not cands:
        raise ValueError(f"{f.filename}: no 2D or 3D numeric dataset found")
    return cands[0][0]


def voxel_size(f, dset):
    """(z, y, x) voxel size in µm from common attributes, or None if absent."""
    def parse(v):
        try:
            import numpy as np  # noqa: PLC0415
            a = np.asarray(v, dtype=float).ravel()
        except (TypeError, ValueError):
            return None
        if a.size == 1 and a[0] > 0:
            return (float(a[0]),) * 3
        if a.size >= 3 and (a[-3:] > 0).all():
            return tuple(float(x) for x in a[-3:])
        return None

    for holder in (f[dset], f):
        for key in _VOXEL_ATTRS:
            if key in holder.attrs:
                vs = parse(holder.attrs[key])
                if vs:
                    return vs
    for path in _VOXEL_DATASETS:
        try:
            if path in f:
                vs = parse(f[path][()])
                if vs:
                    return vs
        except Exception:
            pass
    return None


# ---------------------------------------------------------------- direct reads

# Reads of an uncompressed contiguous dataset bypass HDF5: h5py serialises
# every call under one global lock and HDF5 reads a strided selection row by
# row, so on network storage each sampled row costs a full round trip, one
# after the other. Here rows are read with os.preadv straight into the output
# buffer from many threads at once (measured on NFS: a 16x-strided volume of a
# 100 GB file in 5 s instead of 44 s; a strided YZ slice in 0.5 s instead of 4 s).
_PREAD_THREADS = max(8, min(64, int(os.environ.get("ZVIEW_PREAD_THREADS", 32))))
_PREAD_POOL = ThreadPoolExecutor(max_workers=_PREAD_THREADS, thread_name_prefix="zview-pread")
_RUN_MAX = 8 << 20          # longest single read when consecutive rows merge
_ROWS_PER_TASK = 16          # small row reads grouped per pool task


def direct_reader(f, dset):
    """ContiguousDataset for `dset` if its bytes can be read directly, else None."""
    import h5py  # noqa: PLC0415
    d = f[dset]
    try:
        if (d.chunks is not None or d.compression or d.shuffle or d.fletcher32 or d.scaleoffset
                or d.external or getattr(d, "is_virtual", False) or f.driver != "sec2"
                or d.dtype.kind not in "uifb" or d.dtype.fields is not None):
            return None
        if d.id.get_create_plist().get_layout() != h5py.h5d.CONTIGUOUS:
            return None
        offset = d.id.get_offset()
        if offset is None:          # never written: HDF5 returns the fill value
            return None
        return ContiguousDataset(f.filename, int(offset), d.shape, d.dtype, d.attrs)
    except Exception:
        return None


class ContiguousDataset:
    """Read-only numpy-style view of an uncompressed contiguous dataset.

    Supports ints and positive-step slices on every axis. The last axis is
    read as one span per row (then strided in numpy); rows are read in
    parallel, and consecutive full rows merge into large reads.
    """

    chunks = None

    def __init__(self, filename, offset, shape, dtype, attrs=None):
        self.filename = str(filename)
        self.offset = offset
        self.shape = tuple(int(s) for s in shape)
        self.ndim = len(self.shape)
        self.dtype = np.dtype(dtype)
        self.attrs = attrs if attrs is not None else {}
        self._fd = os.open(self.filename, os.O_RDONLY)
        self._closed = threading.Event()
        self._finalizer = weakref.finalize(self, os.close, self._fd)
        isz = self.dtype.itemsize
        self._strides = [isz * int(np.prod(self.shape[i + 1:], dtype=np.int64)) for i in range(self.ndim)]

    def close(self):
        self._closed.set()
        self._finalizer()

    def __getitem__(self, key):
        if self._closed.is_set():
            raise ValueError("dataset is closed")
        if not isinstance(key, tuple):
            key = (key,)
        if len(key) > self.ndim:
            raise IndexError("too many indices")
        key = key + (slice(None),) * (self.ndim - len(key))
        sel = []                      # per axis: range of indices
        keep = []                     # axes that stay in the output
        for ax, (k, n) in enumerate(zip(key, self.shape)):
            if isinstance(k, slice):
                r = range(*k.indices(n))
                if r.step < 1:
                    raise IndexError("negative steps are not supported")
                keep.append(ax)
            else:
                i = int(k)
                i += n if i < 0 else 0
                if not 0 <= i < n:
                    raise IndexError(f"index {k} out of range for axis {ax} of size {n}")
                r = range(i, i + 1)
            sel.append(r)
        out_shape = [len(r) for r in sel]
        if 0 in out_shape:
            return np.zeros(tuple(out_shape[a] for a in keep), dtype=self.dtype.newbyteorder("="))

        xr = sel[-1]
        x0, span = xr.start, xr[-1] - xr.start + 1
        lead = sel[:-1]
        buf = np.empty(tuple(len(r) for r in lead) + (span,), dtype=self.dtype)
        flat = buf.reshape(-1, span)
        row_bytes = span * self.dtype.itemsize

        # Runs of (file offset, first output row, rows). Consecutive rows of
        # the second-to-last axis are adjacent on disk when the span is the
        # whole row and the step is 1: merge them.
        runs = []
        mergeable = self.ndim >= 2 and span == self.shape[-1] and lead[-1].step == 1
        per = max(1, _RUN_MAX // max(1, row_bytes)) if mergeable else 1
        outer = lead[:-1] if mergeable else lead
        row = 0
        for idx in itertools.product(*outer):
            base = self.offset + sum(i * s for i, s in zip(idx, self._strides)) + x0 * self._strides[-1]
            if mergeable:
                yr = lead[-1]
                for j in range(0, len(yr), per):
                    cnt = min(per, len(yr) - j)
                    runs.append((base + (yr.start + j) * self._strides[-2], row, cnt))
                    row += cnt
            else:
                runs.append((base, row, 1))
                row += 1

        fd = self._fd

        def read(batch):
            for off, r0, cnt in batch:
                view = memoryview(flat[r0:r0 + cnt]).cast("B")
                got = 0
                while got < len(view):
                    n = os.preadv(fd, [view[got:]], off + got)
                    if n <= 0:
                        raise OSError(f"short read in {self.filename} at offset {off + got}")
                    got += n

        group = 1 if mergeable else _ROWS_PER_TASK
        batches = [runs[i:i + group] for i in range(0, len(runs), group)]
        if len(batches) == 1:
            read(batches[0])
        else:
            for fut in [_PREAD_POOL.submit(read, b) for b in batches]:
                fut.result()

        data = buf[..., ::xr.step] if xr.step > 1 else buf
        idx = tuple(slice(None) if ax in keep else 0 for ax in range(self.ndim))
        data = data[idx]
        if not self.dtype.isnative:
            data = data.astype(self.dtype.newbyteorder("="))
        data = np.asarray(data)
        return np.ascontiguousarray(data) if data.ndim else data   # ascontiguousarray makes 0-d 1-d


def open_image(spec, dataset=None, cache_size_mb=None, direct=True):
    """Open an HDF5 dataset as a MultiResolutionImage with strided virtual levels.

    Returns (multires, h5py.File, dataset_path). Uncompressed contiguous
    datasets are read directly (ContiguousDataset); everything else through
    h5py. Close with multires.close() and file.close().
    """
    from zview.core.pyramid import MultiResolutionImage  # noqa: PLC0415
    path, spec_dset = split_spec(spec)
    f = open_file(path)
    try:
        dset = find_dataset(f, dataset or spec_dset)
        vs = voxel_size(f, dset)
        raw = direct_reader(f, dset) if direct and os.environ.get("ZVIEW_H5_DIRECT", "1") != "0" else None
        mi = MultiResolutionImage.from_array(
            raw if raw is not None else f[dset], voxel_size=vs, units=["micrometer"] * 3 if vs else None,
            group=f, cache_size_mb=cache_size_mb)
        if raw is not None:
            mi.add_closer(raw.close)
    except Exception:
        f.close()
        raise
    return mi, f, dset
