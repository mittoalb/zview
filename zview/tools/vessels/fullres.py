"""Whole-brain blood-vessel segmentation at FULL resolution (headless, many cores).

    zview-vessels fullres DIR/render.json --zarr /path/rec.zarr --out /path/vessels.zarr

Vessels are fine dark tubes (empty lumens). Per 512^3 tile of the Zarr full-resolution
level (aligned with the output shards, so every tile is one output file):

  darkness  D = (local tissue level - signal) / (local tissue level - lumen level), after a
            light Gaussian; tissue level = median of the tile, lumen level = darkest voxels of
            the brain (sampled once). Empty lumens are darker than the embedding medium.
  vessels   hysteresis on D: voxels above --high are seeds, voxels above --low are kept
            only where they connect to a seed (follows faint stretches, ignores faint noise),
            then a 1-voxel closing to bridge tiny gaps
  inside    the brain only (tissue mask from the render pipeline, eroded)
  cleanup   connected pieces smaller than --min-size voxels are removed

Output: OME-style Zarr v3, level 0 = full-res uint8 mask (0/1), levels 1..3 = max-pooled
(a voxel is 1 if any vessel voxel is inside) for rendering. Then render with:

    zview-vessels render /path/vessels.zarr --config DIR/render.json --level 3 --out DIR/vessels_video

Needs: render.json from zview-render prepare (cut, z-range, cached volume) and the Zarr from
zview-convert. No display needed. Write --out to a disk nothing else cleans up; finished tiles
are logged in OUT.progress so an interrupted run continues with --resume.
"""

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_G = {}
T = 512                                                   # tile = output shard edge


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
def _brain_tile(z0, z1, y0, y1, x0, x1):
    """Full-res brain mask for a region, from the step-16 mask (nearest)."""
    m16, z16_0, f = _G["m16"], _G["z16_0"], 16
    zi = np.clip(np.arange(z0, z1) // f - z16_0, -1, m16.shape[0])
    yi = np.clip(np.arange(y0, y1) // f, 0, m16.shape[1] - 1)
    xi = np.clip(np.arange(x0, x1) // f, 0, m16.shape[2] - 1)
    ok = (zi >= 0) & (zi < m16.shape[0])
    out = np.zeros((z1 - z0, y1 - y0, x1 - x0), bool)
    if ok.any():
        sub = m16[zi[ok]][:, yi][:, :, xi]
        out[ok] = sub.astype(bool)
    return out


def _tile(job):
    z0, y0, x0 = job
    from scipy import ndimage as ndi
    from skimage.filters import apply_hysteresis_threshold
    from skimage.morphology import remove_small_objects
    import zarr
    if "L0" not in _G:
        _G["L0"] = zarr.open_group(_G["zarr"], mode="r")["0"]
        _G["OUT"] = zarr.open_group(_G["out"], mode="r+")["0"]
    L0, OUT = _G["L0"], _G["OUT"]
    nz, ny, nx = L0.shape
    z1, y1, x1 = min(z0 + T, nz), min(y0 + T, ny), min(x0 + T, nx)
    p = _G["pad"]
    a0, a1, b0, b1, c0, c1 = max(0, z0 - p), min(nz, z1 + p), max(0, y0 - p), min(ny, y1 + p), max(0, x0 - p), min(nx, x1 + p)
    brain = _brain_tile(a0, a1, b0, b1, c0, c1)
    if brain.sum() < 1000:
        return job, 0
    S = ndi.gaussian_filter(np.asarray(L0[a0:a1, b0:b1, c0:c1], np.float32), 1.0)
    tissue = float(np.median(S[brain][::7]))                     # local tissue level (this tile)
    lumen = _G["lumen"]                                          # global lumen level (darkest vessels)
    D = (tissue - S) / max(tissue - lumen, 1e-6)                 # 0 = tissue, 1 = lumen
    del S
    D[~brain] = 0
    h = apply_hysteresis_threshold(D, _G["low"], _G["high"])
    h = ndi.binary_closing(h, structure=ndi.generate_binary_structure(3, 1), iterations=1) & brain
    try:
        h = remove_small_objects(h, max_size=_G["min_size"] - 1, connectivity=3)
    except TypeError:
        h = remove_small_objects(h, min_size=_G["min_size"], connectivity=3)
    core = h[z0 - a0:z0 - a0 + (z1 - z0), y0 - b0:y0 - b0 + (y1 - y0), x0 - c0:x0 - c0 + (x1 - x0)]
    OUT[z0:z1, y0:y1, x0:x1] = core.astype(np.uint8)
    n = int(core.sum())
    if n:                                                        # make sure the shard really landed
        shard = Path(_G["out"]) / "0" / "c" / str(z0 // T) / str(y0 // T) / str(x0 // T)
        if not shard.exists():
            raise RuntimeError(f"wrote tile {job} but {shard} does not exist - is something deleting "
                               f"files in {_G['out']}?")
    return job, n


def _pool_shard(job):
    lvl, z0, y0, x0 = job
    import zarr
    root = zarr.open_group(_G["out"], mode="r+")
    src, dst = root[str(lvl - 1)], root[str(lvl)]
    nz, ny, nx = dst.shape
    z1, y1, x1 = min(z0 + T, nz), min(y0 + T, ny), min(x0 + T, nx)
    a = np.asarray(src[2 * z0:2 * z1, 2 * y0:2 * y1, 2 * x0:2 * x1])
    a = a[: (a.shape[0] // 2) * 2, : (a.shape[1] // 2) * 2, : (a.shape[2] // 2) * 2]
    a = a.reshape(a.shape[0] // 2, 2, a.shape[1] // 2, 2, a.shape[2] // 2, 2).max(axis=(1, 3, 5))
    dst[z0:z0 + a.shape[0], y0:y0 + a.shape[1], x0:x0 + a.shape[2]] = a
    return job


# ---------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", type=Path, help="render.json (cut, z-range, cached volume)")
    p.add_argument("--zarr", type=Path, required=True, help="multiscale Zarr (e.g. from zview-convert)")
    p.add_argument("--out", type=Path, required=True, help="output vessel-mask Zarr (TB-scale disk not needed: "
                                                           "it compresses to a few % of the data)")
    p.add_argument("--low", type=float, default=0.40, help="hysteresis low threshold on darkness (default 0.40)")
    p.add_argument("--high", type=float, default=0.55, help="hysteresis high threshold (seeds, default 0.55)")
    p.add_argument("--min-size", type=int, default=500, help="drop pieces smaller than this (voxels, default 500)")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--z-window", type=int, nargs=2, default=None, metavar=("Z0", "Z1"),
                   help="only process full-res slices Z0..Z1 (for a test)")
    p.add_argument("--yx-window", type=int, nargs=4, default=None, metavar=("Y0", "Y1", "X0", "X1"),
                   help="only process this full-res Y/X window (for a test)")
    p.add_argument("--resume", action="store_true",
                   help="continue an interrupted run (tiles listed in OUT.progress are skipped)")
    p.add_argument("--overwrite", action="store_true", help="start from scratch, deleting OUT")
    args = p.parse_args(argv)
    cfg = json.loads(args.config.read_text())
    t0 = time.time()

    import zarr
    from zarr.codecs import BloscCodec
    from zview.render import scene as rv
    root = zarr.open_group(str(args.zarr), mode="r")
    L0, L3 = root["0"], root["3"]
    nz, ny, nx = L0.shape
    log(f"source {args.zarr}: level 0 {L0.shape} {L0.dtype}")

    m16, _, z0_full = rv.brain_mask_from_config(cfg, min(args.workers, 32))
    z16_0 = z0_full // 16
    log(f"brain mask {m16.shape} at step 16 from full-res z {z0_full}")

    # global lumen level: darkest voxels inside the brain, sampled from random full-res blocks
    zz_, yy_s, xx_s = np.nonzero(m16[::2, ::4, ::4])
    rng = np.random.default_rng(0)
    lum, tis = [], []
    for k in rng.choice(len(zz_), size=min(24, len(zz_)), replace=False):
        cz, cy, cx = (z16_0 + zz_[k] * 2) * 16, yy_s[k] * 64, xx_s[k] * 64
        cz = int(np.clip(cz, 64, nz - 64)); cy = int(np.clip(cy, 128, ny - 128)); cx = int(np.clip(cx, 128, nx - 128))
        b = np.asarray(L0[cz - 64:cz + 64, cy - 128:cy + 128, cx - 128:cx + 128], np.float32)
        from scipy import ndimage as ndi
        b = ndi.gaussian_filter(b, 1.0)
        lum.append(np.percentile(b, 0.3)); tis.append(np.median(b))
    lumen = float(np.median(lum))
    log(f"lumen level {lumen:.0f} (tissue ~{np.median(tis):.0f}) from {len(lum)} sample blocks ({time.time() - t0:.0f}s)")
    del L3

    shapes = [(nz, ny, nx)]
    for _ in range(3):
        shapes.append(tuple(-(-s // 2) for s in shapes[-1]))
    progress = Path(str(args.out) + ".progress")                 # finished tiles, one "z y x" per line
    done = set()
    if args.resume and (args.out / "zarr.json").exists() and (args.out / "0" / "zarr.json").exists():
        out = zarr.open_group(str(args.out), mode="r+")
        if progress.exists():
            done = {tuple(int(v) for v in ln.split()) for ln in progress.read_text().splitlines() if ln.strip()}
        log(f"resuming {args.out}: {len(done)} tiles already done")
    else:
        if args.resume:
            log(f"--resume: no valid store at {args.out}, starting fresh")
        elif args.out.exists() and not args.overwrite:
            raise SystemExit(f"{args.out} exists (use --resume to continue, or --overwrite)")
        progress.unlink(missing_ok=True)
        out = zarr.open_group(str(args.out), mode="w", zarr_format=3)
    comp = [BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle")]
    for i, sh in enumerate(shapes):
        if str(i) not in out:
            out.create_array(str(i), shape=sh, dtype=np.uint8, chunks=(128, 128, 128), shards=(T, T, T),
                             compressors=comp, fill_value=0)
    px = cfg.get("pixel_size_um", 1.0)
    out.attrs["multiscales"] = [{"version": "0.4", "name": "vessels",
                                 "axes": [{"name": a, "type": "space", "unit": "micrometer"} for a in "zyx"],
                                 "datasets": [{"path": str(i), "coordinateTransformations":
                                               [{"type": "scale", "scale": [px * 2 ** i] * 3}]} for i in range(4)],
                                 "metadata": {"method": "hysteresis darkness", "low": args.low, "high": args.high,
                                              "min_size": args.min_size, "source": str(args.zarr)}}]

    # brain bounding box in full-res -> tile list (aligned to shards)
    zz, yy_, xx_ = (np.flatnonzero(m16.any(axis=ax)) for ax in ((1, 2), (0, 2), (0, 1)))
    zb = ((z16_0 + zz[0]) * 16, (z16_0 + zz[-1] + 1) * 16)
    if args.z_window:
        zb = (max(zb[0], args.z_window[0]), min(zb[1], args.z_window[1]))
    yb, xb = (yy_[0] * 16, (yy_[-1] + 1) * 16), (xx_[0] * 16, (xx_[-1] + 1) * 16)
    if args.yx_window:
        yb = (max(yb[0], args.yx_window[0]), min(yb[1], args.yx_window[1]))
        xb = (max(xb[0], args.yx_window[2]), min(xb[1], args.yx_window[3]))
    jobs = [(z, y, x) for z in range(zb[0] // T * T, zb[1], T) for y in range(yb[0] // T * T, yb[1], T)
            for x in range(xb[0] // T * T, xb[1], T)]
    jobs = [j for j in jobs if tuple(int(v) for v in j) not in done]
    log(f"{len(jobs)} tiles of {T}^3 to do over z {zb}, y {yb}, x {xb}; {args.workers} workers")
    _G.update(zarr=str(args.zarr), out=str(args.out), lumen=lumen, low=args.low, high=args.high, m16=m16, z16_0=z16_0,
              min_size=args.min_size, pad=16)
    t1 = time.time(); total = 0
    with Pool(args.workers) as pool, open(progress, "a") as prog:
        for i, (job, n) in enumerate(pool.imap_unordered(_tile, jobs, chunksize=1)):
            total += n
            prog.write(f"{job[0]} {job[1]} {job[2]}\n"); prog.flush()
            if (i + 1) % 50 == 0:
                for meta in (args.out / "zarr.json", args.out / "0" / "zarr.json"):
                    if not meta.exists():
                        pool.terminate()
                        raise SystemExit(f"{meta} disappeared - something deleted files in {args.out}. "
                                         f"Fix that, then restart with --overwrite.")
            if (i + 1) % 50 == 0 or i + 1 == len(jobs):
                el = time.time() - t1
                log(f"  tiles {i + 1}/{len(jobs)}  {el:.0f}s  ETA {el / (i + 1) * (len(jobs) - i - 1):.0f}s  "
                    f"vessel voxels {total:,}")

    for lvl in (1, 2, 3):                                       # max-pool only the processed region
        sh = shapes[lvl]
        f = 2 ** lvl
        rng = [(lo_ // f // T * T, min(sh[k], -(-hi_ // f))) for k, (lo_, hi_) in
               enumerate((zb, yb, xb))]
        pj = [(lvl, z, y, x) for z in range(rng[0][0], rng[0][1], T) for y in range(rng[1][0], rng[1][1], T)
              for x in range(rng[2][0], rng[2][1], T)]
        with Pool(min(args.workers, 32)) as pool:
            list(pool.imap_unordered(_pool_shard, pj, chunksize=1))
        log(f"  level {lvl} {sh} done ({time.time() - t0:.0f}s)")
    log(f"done: {total:,} vessel voxels -> {args.out}  ({(time.time() - t0) / 3600:.2f} h)")


if __name__ == "__main__":
    main()
