"""Blood-vessel segmentation of a PRE-MASKED volume (brain already cut out, 0 = removed).

    zview-vessels segment INPUT --out vessels.zarr --pixel-size 15.04 [--tiff-out DIR]

INPUT: a folder of 2-D TIFF slices, a Zarr (multiscale group -> --level, or a plain array)
or an HDF5 file. Voxels equal to 0 are outside the brain. Vessels are dark tubes (empty
lumens). Defaults are tuned for an 8x-binned volume (~15 um voxels); for other voxel sizes
scale --tissue-sigma and --thick with the vessel size in voxels.

Per 256^3 tile (padded, all cores):
  tissue level  local mean of the brain voxels (normalised Gaussian, --tissue-sigma)
  darkness      D = (tissue - signal) / (tissue - lumen); lumen = darkest brain voxels
  main body     only the largest connected brain piece is used (detached debris dropped);
                debris that touches the brain is NOT removed - mask it in the input
  thick dark    regions that survive an opening of --thick voxels (ventricles, clefts, big
                voids that were not masked) are excluded, with a small margin
  vessels       hysteresis: D > --high are seeds, D > --low kept where connected to a seed,
                1-voxel closing to bridge gaps, inside the eroded brain, pieces < --min-size out

Output: multiscale Zarr v3 (uint8 0/1; levels 1..3 max-pooled), same grid as the input, plus
vessels_preview.png next to it; --tiff-out writes the full-resolution mask as 0/255 TIFFs.
Render with:  zview-vessels render vessels.zarr --image INPUT --out DIR
"""

import argparse
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
T = 256
_G = {}


def log(*a):
    print(*a, flush=True)


def open_volume(path, level="0", dataset=None):
    """3-D array-like for a TIFF folder, Zarr (group level or array) or HDF5 file."""
    path = Path(path)
    if path.suffix == ".zarr" or (path.is_dir() and (path / "zarr.json").exists()):
        import zarr
        z = zarr.open(str(path), mode="r")
        return z[level] if hasattr(z, "array_keys") else z
    if path.is_dir():
        from zview.tools.convert import TiffStack
        return TiffStack(path)
    import h5py
    from zview.tools.convert import find_dataset
    return find_dataset(h5py.File(path, "r"), dataset)[1]


def _vol():
    if "vol" not in _G:
        _G["vol"] = open_volume(_G["path"], _G["level"], _G["dataset"])
    return _G["vol"]


def _tile(job):
    z0, y0, x0 = job
    import zarr
    from scipy import ndimage as ndi
    from skimage.filters import apply_hysteresis_threshold
    from skimage.morphology import remove_small_objects
    vol = _vol()
    nz, ny, nx = vol.shape
    z1, y1, x1 = min(z0 + T, nz), min(y0 + T, ny), min(x0 + T, nx)
    p = _G["pad"]
    a0, a1, b0, b1, c0, c1 = max(0, z0 - p), min(nz, z1 + p), max(0, y0 - p), min(ny, y1 + p), max(0, x0 - p), min(nx, x1 + p)
    B = np.asarray(vol[a0:a1, b0:b1, c0:c1], np.float32)
    brain = B > 0
    m4 = _G["m4"]                                               # main brain body (no debris islands)
    brain &= m4[np.arange(a0, a1)[:, None, None] // 4, np.arange(b0, b1)[None, :, None] // 4,
                np.arange(c0, c1)[None, None, :] // 4]
    if brain.sum() < 1000:
        return job, 0
    b1s = ndi.generate_binary_structure(3, 1)
    inner = ndi.binary_erosion(brain, structure=b1s, iterations=_G["erode"])
    s = _G["tissue_sigma"]
    tissue = ndi.gaussian_filter(np.where(brain, B, 0), s) / np.maximum(ndi.gaussian_filter(brain.astype(np.float32), s), 1e-3)
    S = ndi.gaussian_filter(B, _G["smooth"])
    D = np.clip((tissue - S) / np.maximum(tissue - _G["lumen"], 1e-6), 0, 1)
    del tissue, S, B
    D[~inner] = 0
    if _G["thick"] > 0:
        dark = D > _G["low"]
        thick = ndi.binary_opening(dark, structure=b1s, iterations=_G["thick"])
        thick = ndi.binary_dilation(thick, structure=b1s, iterations=2)
        D[thick] = 0
    h = apply_hysteresis_threshold(D, _G["low"], _G["high"])
    h = ndi.binary_closing(h, structure=b1s, iterations=1) & inner
    if _G["thick"] > 0:
        h &= ~thick
    try:
        h = remove_small_objects(h, max_size=_G["min_size"] - 1, connectivity=3)
    except TypeError:
        h = remove_small_objects(h, min_size=_G["min_size"], connectivity=3)
    core = h[z0 - a0:z0 - a0 + (z1 - z0), y0 - b0:y0 - b0 + (y1 - y0), x0 - c0:x0 - c0 + (x1 - x0)]
    if "OUT" not in _G:
        _G["OUT"] = zarr.open_group(_G["out"], mode="r+")["0"]
    _G["OUT"][z0:z1, y0:y1, x0:x1] = core.astype(np.uint8)
    return job, int(core.sum())


def _pool_shard(job):
    lvl, z0, y0, x0 = job
    import zarr
    root = zarr.open_group(_G["out"], mode="r+")
    src, dst = root[str(lvl - 1)], root[str(lvl)]
    nz, ny, nx = dst.shape
    z1, y1, x1 = min(z0 + T, nz), min(y0 + T, ny), min(x0 + T, nx)
    a = np.asarray(src[2 * z0:2 * z1, 2 * y0:2 * y1, 2 * x0:2 * x1])
    pz, py, px = (-a.shape[0]) % 2, (-a.shape[1]) % 2, (-a.shape[2]) % 2
    a = np.pad(a, ((0, pz), (0, py), (0, px)))
    a = a.reshape(a.shape[0] // 2, 2, a.shape[1] // 2, 2, a.shape[2] // 2, 2).max(axis=(1, 3, 5))
    dst[z0:z0 + a.shape[0], y0:y0 + a.shape[1], x0:x0 + a.shape[2]] = a[:z1 - z0, :y1 - y0, :x1 - x0]
    return job


def _tiff_slice(z):
    import tifffile
    import zarr
    m = zarr.open_group(_G["out"], mode="r")["0"]
    tifffile.imwrite(Path(_G["tiff_out"]) / f"vessels_{z:05d}.tiff", (np.asarray(m[z]) * 255).astype(np.uint8))
    return z


def _m4_slab(z0):
    vol = _vol(); nz, ny, nx = vol.shape
    a = np.asarray(vol[z0:min(z0 + 4, nz)]) > 0
    a = np.pad(a, ((0, 4 - a.shape[0]), (0, (-ny) % 4), (0, (-nx) % 4)))
    return z0 // 4, a.reshape(4, a.shape[1] // 4, 4, a.shape[2] // 4, 4).max(axis=(0, 2, 4))


def main_body(vol, workers, keep_frac):
    """Brain mask at 1/4 resolution, keeping the largest connected piece (and pieces >= keep_frac
    of it), so loose debris that survived the masking is not segmented."""
    from scipy import ndimage as ndi
    nz, ny, nx = vol.shape
    m4 = np.zeros((-(-nz // 4), -(-ny // 4), -(-nx // 4)), bool)
    with Pool(workers) as pool:
        for k, sl in pool.imap_unordered(_m4_slab, range(0, nz, 4), chunksize=4):
            m4[k] = sl
    lab, n = ndi.label(m4, structure=ndi.generate_binary_structure(3, 1))
    if n == 0:
        return m4
    sz = np.bincount(lab.ravel()); sz[0] = 0
    keep = sz >= keep_frac * sz.max()
    log(f"main brain body: kept {int(keep.sum())} of {n} connected pieces")
    return keep[lab]


def preview(vol, out, path):
    import zarr
    from matplotlib.figure import Figure
    m = zarr.open_group(out, mode="r")["0"]
    nz = vol.shape[0]
    fig = Figure(figsize=(18, 12)); axs = fig.subplots(2, 3)
    for c, z in enumerate((nz // 4, nz // 2, 3 * nz // 4)):
        sl = np.asarray(vol[z], np.float32); v = np.asarray(m[z]).astype(bool)
        nzv = sl[sl > 0]
        lo, hi = np.percentile(nzv, [1, 99.5]) if nzv.size else (0, 1)
        axs[0, c].imshow(sl, cmap="gray", vmin=lo, vmax=hi); axs[0, c].set_title(f"slice {z}")
        axs[1, c].imshow(sl, cmap="gray", vmin=lo, vmax=hi)
        axs[1, c].imshow(np.ma.masked_where(~v, v), cmap="autumn", alpha=0.9); axs[1, c].set_title(f"vessels on slice {z}")
    for a in axs.ravel():
        a.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=60)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path, help="TIFF folder, Zarr or HDF5 (pre-masked: 0 = outside the brain)")
    p.add_argument("--out", type=Path, required=True, help="output vessel-mask Zarr")
    p.add_argument("--level", default="0", help="Zarr input: pyramid level (default 0)")
    p.add_argument("--dataset", default=None, help="HDF5 input: dataset path (default auto)")
    p.add_argument("--pixel-size", type=float, default=1.0, help="voxel size in um (metadata)")
    p.add_argument("--low", type=float, default=0.35, help="hysteresis low threshold on darkness (default 0.35)")
    p.add_argument("--high", type=float, default=0.55, help="hysteresis high threshold / seeds (default 0.55)")
    p.add_argument("--thick", type=int, default=2,
                   help="exclude dark regions thicker than ~2x this many voxels (opening; 0 = off; default 2)")
    p.add_argument("--tissue-sigma", type=float, default=8, help="local tissue level smoothing, voxels (default 8)")
    p.add_argument("--smooth", type=float, default=0.5, help="Gaussian on the data before thresholding (default 0.5)")
    p.add_argument("--erode", type=int, default=2, help="erode the brain mask by this many voxels (default 2)")
    p.add_argument("--min-size", type=int, default=30, help="drop pieces smaller than this, voxels (default 30)")
    p.add_argument("--keep-frac", type=float, default=0.05,
                   help="keep brain pieces >= this fraction of the largest one (drops debris; default 0.05)")
    p.add_argument("--tiff-out", type=Path, default=None, help="also write the mask as 0/255 TIFF slices here")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)
    t0 = time.time()

    import zarr
    from zarr.codecs import BloscCodec
    vol = open_volume(args.input, args.level, args.dataset)
    nz, ny, nx = vol.shape
    log(f"input {args.input}: {vol.shape} {vol.dtype}")

    rng = np.random.default_rng(0)                              # lumen level: darkest brain voxels
    smp = np.concatenate([np.asarray(vol[int(z)], np.float32)[::4, ::4].ravel()
                          for z in rng.choice(nz, size=min(40, nz), replace=False)])
    smp = smp[smp > 0]
    lumen = float(np.percentile(smp, 0.5)); tissue = float(np.median(smp))
    log(f"lumen level {lumen:.4g}, typical tissue {tissue:.4g}")

    if args.out.exists() and not args.overwrite:
        raise SystemExit(f"{args.out} exists (use --overwrite)")
    out = zarr.open_group(str(args.out), mode="w", zarr_format=3)
    shapes = [(nz, ny, nx)]
    for _ in range(3):
        shapes.append(tuple(-(-s // 2) for s in shapes[-1]))
    comp = [BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle")]
    for i, sh in enumerate(shapes):
        out.create_array(str(i), shape=sh, dtype=np.uint8, chunks=(64, 64, 64), shards=(T, T, T),
                         compressors=comp, fill_value=0)
    out.attrs["multiscales"] = [{
        "version": "0.4", "name": "vessels",
        "axes": [{"name": a, "type": "space", "unit": "micrometer"} for a in "zyx"],
        "datasets": [{"path": str(i), "coordinateTransformations": [{"type": "scale", "scale": [args.pixel_size * 2 ** i] * 3}]}
                     for i in range(4)],
        "metadata": {"method": "hysteresis darkness, pre-masked input", "source": str(args.input),
                     **{k: getattr(args, k) for k in ("low", "high", "thick", "tissue_sigma", "smooth", "erode", "min_size")}}}]

    _G.update(path=str(args.input), level=args.level, dataset=args.dataset, out=str(args.out), lumen=lumen,
              low=args.low, high=args.high, thick=args.thick, tissue_sigma=args.tissue_sigma, smooth=args.smooth,
              erode=args.erode, min_size=args.min_size, pad=int(max(3 * args.tissue_sigma, 16)))
    _G["m4"] = main_body(vol, args.workers, args.keep_frac)
    jobs = [(z, y, x) for z in range(0, nz, T) for y in range(0, ny, T) for x in range(0, nx, T)]
    log(f"{len(jobs)} tiles of {T}^3 on {args.workers} workers")
    total, t1 = 0, time.time()
    with Pool(args.workers) as pool:
        for i, (job, n) in enumerate(pool.imap_unordered(_tile, jobs, chunksize=1)):
            total += n
            if (i + 1) % 20 == 0 or i + 1 == len(jobs):
                el = time.time() - t1
                log(f"  tiles {i + 1}/{len(jobs)}  {el:.0f}s  ETA {el / (i + 1) * (len(jobs) - i - 1):.0f}s")
    for lvl in (1, 2, 3):
        sh = shapes[lvl]
        pj = [(lvl, z, y, x) for z in range(0, sh[0], T) for y in range(0, sh[1], T) for x in range(0, sh[2], T)]
        with Pool(args.workers) as pool:
            list(pool.imap_unordered(_pool_shard, pj, chunksize=1))
    nb = sum(int((np.asarray(vol[int(z)]) > 0).sum()) for z in np.linspace(0, nz - 1, 40).astype(int)) * nz / 40
    log(f"vessels: {total:,} voxels = {total / max(nb, 1) * 100:.2f}% of the brain  ({time.time() - t0:.0f}s)")
    preview(vol, str(args.out), args.out.parent / f"{args.out.stem}_preview.png")
    log(f"preview -> {args.out.parent / (args.out.stem + '_preview.png')}")
    if args.tiff_out:
        args.tiff_out.mkdir(parents=True, exist_ok=True)
        _G["tiff_out"] = str(args.tiff_out)
        with Pool(args.workers) as pool:
            list(pool.imap_unordered(_tiff_slice, range(nz), chunksize=8))
        log(f"TIFF mask (0/255) -> {args.tiff_out}/")
    log(f"done -> {args.out}")


if __name__ == "__main__":
    main()
