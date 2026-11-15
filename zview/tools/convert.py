"""HDF5 or TIFF-stack (reconstructed tomography volume) -> multiscale sharded OME-Zarr v3.

Modeled on DataTools' tiff2zarr but:
  - reads from an HDF5 dataset (streamed in Z-slabs so an 8 TB volume fits)
  - WRITES ACTUAL SHARDED Zarr v3 (ShardingCodec wired up, fixing the dead
    --shard_size parameter in tiff2zarr)
  - builds the multiscale pyramid level-by-level from the finest level, so
    peak RAM = one slab at mip 0 plus the full downsampled mips 1..N

Output layout (OME-Zarr v0.4 multiscales):
  dataset.zarr/
    .zgroup / zarr.json
    0/   zarr.json ...   (finest, sharded)
    1/   ...
    ...

Example:
    zview-convert /path/to/rec.h5 \\
        /path/to/rec.zarr \\
        --dataset /exchange/data \\
        --shard-size 512 512 512 \\
        --chunk-size 128 128 128 \\
        --pixel-size 1.1 \\
        --dtype uint16 \\
        --slab 128
"""

import argparse
import os
import shutil
import time
from pathlib import Path

import numpy as np


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"


def find_dataset(h5file, requested: str | None):
    """Pick an HDF5 dataset path.
    If `requested` is given, use it. Otherwise auto-detect by trying common
    tomography conventions, then falling back to the largest 3-D dataset.
    """
    import h5py

    if requested:
        if requested not in h5file:
            raise SystemExit(f"Dataset {requested!r} not found in file.")
        node = h5file[requested]
        if not isinstance(node, h5py.Dataset):
            raise SystemExit(f"{requested!r} is a group, not a dataset.")
        return requested, node

    candidates = ["/exchange/data", "/data", "/volume", "/entry/data/data"]
    for name in candidates:
        if name in h5file and isinstance(h5file[name], h5py.Dataset) and h5file[name].ndim == 3:
            return name, h5file[name]

    # Walk the file for the largest 3-D dataset
    best = {"path": None, "size": 0, "node": None}

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset) and obj.ndim == 3:
            size = int(np.prod(obj.shape)) * obj.dtype.itemsize
            if size > best["size"]:
                best.update(path="/" + name, size=size, node=obj)

    h5file.visititems(visit)
    if best["node"] is None:
        raise SystemExit("No 3-D dataset found in HDF5 file. Use --dataset to point at one.")
    print(f"Auto-selected dataset: {best['path']}  shape={best['node'].shape} dtype={best['node'].dtype}")
    return best["path"], best["node"]


class TiffStack:
    """A folder of 2-D TIFF slices (sorted by name) behaving like a read-only 3-D dataset."""

    def __init__(self, folder):
        import tifffile
        self.files = sorted(p for p in Path(folder).iterdir()
                            if p.suffix.lower() in (".tif", ".tiff"))
        if not self.files:
            raise SystemExit(f"no .tif/.tiff files in {folder}")
        first = tifffile.imread(self.files[0])
        self.shape = (len(self.files),) + first.shape
        self.dtype = first.dtype
        self.ndim = 3

    def __getitem__(self, key):
        import tifffile
        from concurrent.futures import ThreadPoolExecutor
        if not isinstance(key, tuple):
            key = (key,)
        key = key + (slice(None),) * (3 - len(key))
        zs = range(*key[0].indices(self.shape[0])) if isinstance(key[0], slice) else [key[0]]
        with ThreadPoolExecutor(16) as ex:
            sl = list(ex.map(lambda z: tifffile.imread(self.files[z])[key[1], key[2]], zs))
        out = np.stack(sl)
        return out if isinstance(key[0], slice) else out[0]


def open_source(path, dataset, stack):
    """(dataset-like, closer) for an HDF5 file or a folder of TIFF slices."""
    path = Path(path)
    if path.is_dir():
        return TiffStack(path), None
    import h5py
    h5 = stack.enter_context(h5py.File(path, "r"))
    return find_dataset(h5, dataset)[1], h5


def compute_contrast_from_hdf5(ds, lo_pct=0.5, hi_pct=99.5, max_voxels=10_000_000):
    """Estimate percentile contrast from a coarse subsample of the HDF5 dataset."""
    stride = max(1, int((int(np.prod(ds.shape)) / max_voxels) ** (1.0 / ds.ndim)))
    print(f"Contrast subsample stride: {stride} (reads approximately {int(np.prod(ds.shape)) / stride**3 / 1e6:.1f} M voxels)")
    sample = ds[::stride, ::stride, ::stride]
    lo = float(np.percentile(sample, lo_pct))
    hi = float(np.percentile(sample, hi_pct))
    return lo, hi


def scale_to_dtype(data, lo, hi, target_dtype):
    if lo >= hi:
        return np.zeros_like(data, dtype=target_dtype)
    clipped = np.clip(data, lo, hi)
    if np.issubdtype(target_dtype, np.integer):
        info = np.iinfo(target_dtype)
        if info.min < 0:
            scaled = ((clipped - lo) / (hi - lo) * (info.max - info.min) + info.min)
        else:
            scaled = (clipped - lo) / (hi - lo) * info.max
    else:
        scaled = (clipped - lo) / (hi - lo)
    return scaled.astype(target_dtype)


def calculate_levels(shape, max_levels=6, min_size=64):
    """Number of 2x downsamplings to build, stopping before any axis drops below
    `min_size`. Axes need not be divisible by 2: level i has shape s // 2**i
    (odd tails are trimmed), which the slab-streaming loop reproduces exactly as
    long as the slab Z is a multiple of 2**n_levels."""
    n = 0
    while n < max_levels and min(s // 2 ** (n + 1) for s in shape) >= min_size:
        n += 1
    return n


def read_scaled_slab(ds, z0, z1, lo, hi, target_dtype, sub=32):
    """Read ds[z0:z1] and scale it to target_dtype in `sub`-slice blocks, so peak
    extra RAM is one float block instead of several full-slab float temporaries."""
    out = np.empty((z1 - z0,) + tuple(ds.shape[1:]), dtype=target_dtype)
    for a in range(z0, z1, sub):
        b = min(a + sub, z1)
        out[a - z0:b - z0] = scale_to_dtype(ds[a:b], lo, hi, target_dtype)
    return out


def make_array(root, name, shape, chunk, shard, dtype, compression):
    """Create a Zarr v3 array, optionally sharded (inner chunks packed into shard files)."""
    from zarr.codecs import BloscCodec

    comp = BloscCodec(cname=compression, clevel=1, shuffle="shuffle")
    return root.create_array(
        name=name, shape=tuple(shape), dtype=dtype,
        chunks=tuple(chunk),
        shards=tuple(shard) if shard is not None else None,
        compressors=[comp],
    )


def downsample_slab(slab, factor=2):
    """Mean-downsample a 3-D slab by `factor` along each axis using strided averaging.
    Requires slab shape divisible by factor along each axis; if not, trims the tail.
    """
    s = slab
    # trim to multiples of factor
    s = s[: s.shape[0] - s.shape[0] % factor,
          : s.shape[1] - s.shape[1] % factor,
          : s.shape[2] - s.shape[2] % factor]
    if s.size == 0:
        return s
    new = s.reshape(
        s.shape[0] // factor, factor,
        s.shape[1] // factor, factor,
        s.shape[2] // factor, factor,
    )
    return new.mean(axis=(1, 3, 5), dtype=np.float32)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input_h5", type=Path, help="HDF5 file, or a folder of TIFF slices")
    p.add_argument("output_zarr", type=Path)
    p.add_argument("--dataset", default=None,
                   help="HDF5 dataset path (default: auto, trying /exchange/data, /data, /volume, ...)")
    p.add_argument("--dtype", default="uint16",
                   choices=["uint8", "int8", "uint16", "int16", "int32", "float32"],
                   help="Output dtype (default uint16; use uint8 for a smaller/faster render set)")
    p.add_argument("--chunk-size", nargs=3, type=int, default=[128, 128, 128],
                   metavar=("CZ", "CY", "CX"), help="Inner chunk size (default 128 128 128)")
    p.add_argument("--shard-size", nargs=3, type=int, default=[512, 512, 512],
                   metavar=("SZ", "SY", "SX"),
                   help="Shard size (outer chunk, Zarr v3). Set to 0 0 0 to disable sharding.")
    p.add_argument("--compression", default="blosclz",
                   choices=["blosclz", "lz4", "lz4hc", "zlib", "zstd"])
    p.add_argument("--max-levels", type=int, default=6, help="Max pyramid levels (default 6)")
    p.add_argument("--pixel-size", type=float, default=1.0, help="Voxel size in micrometers")
    p.add_argument("--slab", type=int, default=128,
                   help="Z-slab height in voxels for streaming (default 128). Multiple of shard Z.")
    p.add_argument("--min-val", type=float, default=None, help="Manual contrast lower bound")
    p.add_argument("--max-val", type=float, default=None, help="Manual contrast upper bound")
    p.add_argument("--min-pct", type=float, default=0.5)
    p.add_argument("--max-pct", type=float, default=99.5)
    p.add_argument("--concurrency", type=int, default=64,
                   help="Max concurrent shard reads/writes in zarr (default 64)")
    p.add_argument("--overwrite", action="store_true", help="Delete output_zarr if it already exists")
    args = p.parse_args(argv)

    import h5py
    import zarr

    print(f"zarr version: {zarr.__version__}")
    zarr.config.set({"async.concurrency": args.concurrency})
    if not zarr.__version__.startswith("3"):
        raise SystemExit(f"Need zarr-python >= 3.0 for v3 output; have {zarr.__version__}")

    # Resolve sharding
    shard = tuple(args.shard_size)
    if shard == (0, 0, 0):
        shard = None
    chunk = tuple(args.chunk_size)
    if shard is not None:
        for s, c in zip(shard, chunk):
            if s % c != 0:
                raise SystemExit(f"Shard size {shard} must be a multiple of chunk size {chunk} along every axis.")

    import contextlib
    src_size = (sum(f.stat().st_size for f in args.input_h5.iterdir()) if args.input_h5.is_dir()
                else os.path.getsize(args.input_h5))
    print(f"Opening {args.input_h5} ({human_bytes(src_size)})")
    with contextlib.ExitStack() as stack:
        ds, _ = open_source(args.input_h5, args.dataset, stack)
        shape = tuple(int(s) for s in ds.shape)
        src_dtype = ds.dtype
        raw_bytes = int(np.prod(shape)) * src_dtype.itemsize
        print(f"Source: shape={shape} dtype={src_dtype} raw={human_bytes(raw_bytes)}")

        # Output dtype
        dtype_map = {
            "uint8": np.uint8, "int8": np.int8, "uint16": np.uint16, "int16": np.int16,
            "int32": np.int32, "float32": np.float32,
        }
        target = dtype_map[args.dtype]

        # Contrast
        if args.min_val is not None and args.max_val is not None:
            lo, hi = args.min_val, args.max_val
        elif args.min_val is not None or args.max_val is not None:
            raise SystemExit("--min-val and --max-val must be provided together")
        elif np.issubdtype(src_dtype, np.floating) or args.dtype != str(src_dtype):
            print("Estimating contrast from subsample...")
            t0 = time.perf_counter()
            lo, hi = compute_contrast_from_hdf5(ds, args.min_pct, args.max_pct)
            print(f"  lo={lo:.4g}  hi={hi:.4g}  ({time.perf_counter() - t0:.1f}s)")
        else:
            lo = float(np.iinfo(src_dtype).min)
            hi = float(np.iinfo(src_dtype).max)

        # Output path
        if args.output_zarr.exists():
            if args.overwrite:
                print(f"Removing existing {args.output_zarr}")
                shutil.rmtree(args.output_zarr)
            else:
                raise SystemExit(f"{args.output_zarr} exists (use --overwrite)")
        args.output_zarr.parent.mkdir(parents=True, exist_ok=True)

        # Figure out pyramid levels
        n_levels = calculate_levels(shape, args.max_levels)
        if n_levels < 1:
            print("Volume too small for a pyramid; writing a single level.")
        level_shapes = [tuple(s // (2 ** i) for s in shape) for i in range(n_levels + 1)]
        print(f"Pyramid: {n_levels + 1} levels")
        for i, sh in enumerate(level_shapes):
            print(f"  L{i}: {sh}  ({human_bytes(int(np.prod(sh)) * np.dtype(target).itemsize)})")

        # Create root group and arrays
        root = zarr.open_group(str(args.output_zarr), mode="w", zarr_format=3)
        arrays = []
        for i, sh in enumerate(level_shapes):
            # Scale chunk/shard to fit if level is tiny
            lvl_chunk = tuple(min(c, s) for c, s in zip(chunk, sh))
            lvl_shard = None
            if shard is not None:
                lvl_shard = tuple(min(sh_, s) for sh_, s in zip(shard, sh))
                # Trim shard to a multiple of chunk
                lvl_shard = tuple((sh_ // c) * c if (sh_ // c) * c > 0 else c for sh_, c in zip(lvl_shard, lvl_chunk))
                if any(s_ == 0 for s_ in lvl_shard):
                    lvl_shard = None
            z = make_array(root, f"{i}", sh, lvl_chunk, lvl_shard, target, args.compression)
            arrays.append(z)
            print(f"  array {i}: chunk={lvl_chunk} shard={lvl_shard}")

        # OME-Zarr multiscales metadata
        datasets_meta = []
        for i in range(n_levels + 1):
            scale = 2 ** i
            datasets_meta.append({
                "path": f"{i}",
                "coordinateTransformations": [
                    {"type": "scale", "scale": [args.pixel_size * scale] * 3},
                ],
            })
        root.attrs["multiscales"] = [{
            "version": "0.4",
            "name": args.output_zarr.stem,
            "axes": [
                {"name": "z", "type": "space", "unit": "micrometer"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"},
            ],
            "datasets": datasets_meta,
            "type": "mean",
            "metadata": {"method": "strided_mean_downsample", "args": "[2]", "input_dtype": str(src_dtype),
                          "contrast": [lo, hi], "target_dtype": args.dtype},
        }]

        # Stream the volume in Z-slabs. Slab Z must be a multiple of the shard Z
        # (aligned level-0 writes) and of 2**n_levels (exact coarse-level offsets).
        Z = shape[0]
        align = 2 ** n_levels
        if shard is not None:
            align = int(np.lcm(align, shard[0]))
        slab_z = ((args.slab + align - 1) // align) * align
        if slab_z != args.slab:
            print(f"Rounded slab Z to {slab_z} (multiple of {align})")
        slab_bytes = slab_z * int(np.prod(shape[1:])) * np.dtype(target).itemsize
        print(f"Slab: {slab_z} slices = {human_bytes(slab_bytes)} per slab "
              f"(x2 with prefetch)")

        from concurrent.futures import ThreadPoolExecutor
        starts = list(range(0, Z, slab_z))
        reader = ThreadPoolExecutor(max_workers=1)  # prefetch next slab while writing this one

        def submit(z0):
            return reader.submit(read_scaled_slab, ds, z0, min(z0 + slab_z, Z), lo, hi, target)

        pending = submit(starts[0])
        total_bytes_read = 0
        t_start = time.perf_counter()
        for i_slab, z0 in enumerate(starts):
            z1 = min(z0 + slab_z, Z)
            slab = pending.result()
            pending = submit(starts[i_slab + 1]) if i_slab + 1 < len(starts) else None
            total_bytes_read += (z1 - z0) * int(np.prod(shape[1:])) * src_dtype.itemsize
            arrays[0][z0:z1, :, :] = slab

            # Build each coarser level from the previous pyramid level (not re-reading)
            # We maintain a sliding "current" that starts as the finest slab.
            cur = slab
            # To correctly align downsampled rows across slab boundaries, we require
            # slab_z to be a multiple of 2**n_levels. If not, we still proceed but
            # may lose one row at the tail per level.
            for lvl in range(1, n_levels + 1):
                cur = downsample_slab(cur, factor=2).astype(target)
                # Where to put this slab in the coarser level?
                coarse_z0 = z0 // (2 ** lvl)
                coarse_z1 = coarse_z0 + cur.shape[0]
                if cur.shape[0] == 0:
                    break
                arrays[lvl][coarse_z0:coarse_z1, :, :] = cur

            elapsed = time.perf_counter() - t_start
            rate = total_bytes_read / elapsed / 1024**2
            print(f"  slab z={z0:>6}..{z1:<6} ({(z1 - z0)} slices)  "
                  f"total {human_bytes(total_bytes_read)}  rate {rate:.1f} MB/s  "
                  f"ETA {(raw_bytes - total_bytes_read) / (rate * 1024**2):.0f} s")

        reader.shutdown()
        print(f"\nDone. Wrote {args.output_zarr}")
        print(f"Total read: {human_bytes(total_bytes_read)}  in {time.perf_counter() - t_start:.1f} s")


if __name__ == "__main__":
    main()
