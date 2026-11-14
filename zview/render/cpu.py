# Moved from zarr-auto-render/render_cpu.py.
"""CPU volume renderer for a multiscale OME-Zarr — no OpenGL, no display, no GPU.

Headless servers often can't give napari/vispy a working OpenGL context
(Qt 'offscreen' renders black frames, Xvfb/EGL crash). This renders the 3D
views with numpy/scipy instead, parallel over CPU cores:

  * turntable: the volume rotates about its vertical (Z) axis; each frame casts
    parallel rays through it, either
      - mip:  maximum-intensity projection (sharp, shows fine bright detail)
      - comp: front-to-back alpha compositing (looks solid / 3D, hides speckle)
  * views:     stills at 0/90/180/270 deg for each mode + a top-down MIP
  * slice sweep: fly-through of Z slices (MP4)

Sample-in-a-tube preprocessing (on by default):
  * subtract the radial background (embedding medium + reconstruction cupping)
  * mask everything outside --mask-radius (container walls)
so the tissue, not the tube, is what shows up.

Example:
    python render_cpu.py rec.zarr --out renders --frames 180 --modes mip,comp
"""

import argparse
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

_VOL = None  # preprocessed float32 volume, shared with workers via fork


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def open_levels(zarr_path):
    import zarr
    root = zarr.open_group(str(zarr_path), mode="r")
    keys = sorted(root.array_keys(), key=int)
    return root, keys


def pick_level(root, keys, budget_gb):
    for k in keys:
        a = root[k]
        if np.prod(a.shape) * 4 / 1024**3 <= budget_gb:   # float32 in RAM
            return k
    return keys[-1]


_BG = _MASK = _RING = _OFFS = None
_WALL = None     # per-slice (cy, cx, r_mask) arrays from fit_walls(); None = fixed circle


def _fit_circle(pts, iters=4, tol=6.0):
    """Algebraic circle fit with outlier rejection. pts: (N, 2) as (y, x)."""
    y, x = pts[:, 0].astype(float), pts[:, 1].astype(float)
    keep = np.ones(len(y), bool)
    cy = cx = r = None
    for _ in range(iters):
        if keep.sum() < 50:
            return None
        A = np.c_[2 * x[keep], 2 * y[keep], np.ones(keep.sum())]
        b = x[keep] ** 2 + y[keep] ** 2
        c = np.linalg.lstsq(A, b, rcond=None)[0]
        cx, cy = c[0], c[1]
        r = np.sqrt(max(c[2] + cx ** 2 + cy ** 2, 0))
        d = np.abs(np.hypot(y - cy, x - cx) - r)
        keep = d < max(tol, 2.5 * np.median(d[keep]))
    return cy, cx, r


def fit_walls(vol, bg, block, margin, fallback_r, log=print):
    """Fit the container's inner wall (a thin bright ring, persistent along Z)
    per block of slices; return per-slice (cy, cx, r_mask) with r_mask = wall - margin."""
    from scipy.ndimage import median_filter
    nz, ny, nx = vol.shape
    R0 = min(nx, ny) / 2
    yy, xx = np.indices((ny, nx))
    r_img = np.hypot(yy - (ny - 1) / 2, xx - (nx - 1) / 2)
    band = (r_img > 0.55 * R0) & (r_img < 0.97 * R0)
    fits = []
    for b0 in range(0, nz, block):
        zs = np.linspace(b0, min(b0 + block, nz) - 1, 8).astype(int)
        res = np.median(np.stack([np.maximum(vol[z] - bg, 0) for z in zs]), axis=0)
        # thin ridges: subtract a local background, keep the brightest 0.5 % in the band
        ridge = res - median_filter(res, size=9)
        thr = np.percentile(ridge[band], 99.0)
        pts = np.argwhere(band & (ridge > thr))
        # innermost ridge per 2-degree sector = the inner tube wall (outer walls are further out)
        ang = (np.degrees(np.arctan2(pts[:, 0] - (ny - 1) / 2, pts[:, 1] - (nx - 1) / 2)) % 360 // 2).astype(int)
        rad = r_img[pts[:, 0], pts[:, 1]]
        order = np.lexsort((rad, ang))
        first = np.r_[True, ang[order][1:] != ang[order][:-1]]
        f = _fit_circle(pts[order][first], tol=4.0) if first.sum() >= 50 else None
        ok = f is not None and 0.6 * R0 < f[2] < 0.99 * R0 and np.hypot(f[0] - ny / 2, f[1] - nx / 2) < 0.2 * R0
        fits.append(f if ok else None)
    good = [f for f in fits if f is not None]
    if not good:
        log("wall fit: no wall found, using fixed circular mask")
        return None
    # straight (possibly tilted) cylinder: centre linear in Z, constant radius.
    zc = np.array([b * block + block / 2 for b, f in enumerate(fits) if f is not None], float)
    F = np.array(good, float)                            # columns: cy, cx, r
    keep = np.ones(len(zc), bool)
    for _ in range(3):                                   # robust line fit: drop outlier blocks
        deg = 1 if keep.sum() >= 3 else 0
        py = np.polyfit(zc[keep], F[keep, 0], deg); px = np.polyfit(zc[keep], F[keep, 1], deg)
        resid = np.hypot(F[:, 0] - np.polyval(py, zc), F[:, 1] - np.polyval(px, zc))
        keep = resid < max(5.0, 3 * np.median(resid[keep]))
    z = np.arange(nz)
    cy, cx = np.polyval(py, z), np.polyval(px, z)
    r_wall = float(np.median(F[keep, 2]))
    rm = np.full(nz, r_wall - margin * R0)
    log(f"wall fit ({len(good)}/{len(fits)} blocks, {keep.sum()} used): straight cylinder, wall r={r_wall:.0f}, "
        f"cut r={rm[0]:.0f}; centre top y={cy[0]:.0f} x={cx[0]:.0f} -> bottom y={cy[-1]:.0f} x={cx[-1]:.0f} "
        f"(image centre {(ny - 1) / 2:.0f},{(nx - 1) / 2:.0f})")
    return cy.astype(np.float32), cx.astype(np.float32), rm.astype(np.float32)


def _slice_mask(z, ny, nx):
    cy, cx, rm = _WALL[0][z], _WALL[1][z], _WALL[2][z]
    yy, xx = np.ogrid[:ny, :nx]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) < rm * rm


_RING_PCT = 50.0


def _ring_median(z):
    """Percentile (_RING_PCT) of slice z inside the current ring/mask."""
    if _WALL is not None and _RING is None:
        ny, nx = _VOL.shape[1:]
        cy, cx, rm = _WALL[0][z], _WALL[1][z], _WALL[2][z]
        yy, xx = np.ogrid[:ny, :nx]
        d2 = (yy - cy) ** 2 + (xx - cx) ** 2
        R0 = min(ny, nx) / 2
        ring = (d2 < rm * rm) & (d2 >= (rm - 0.06 * R0) ** 2)
        return float(np.percentile(_VOL[z][ring], _RING_PCT))
    return float(np.percentile(_VOL[z][_RING], _RING_PCT))


def ring_percentile_profile(vol, zs, ri, pct):
    """Per-radius percentile of the pooled pixels of slices zs (ri = flat radius index).
    A LOW percentile tracks the embedding medium even where tissue fills most of a ring."""
    vals = np.concatenate([vol[z].ravel() for z in zs])
    rings = np.tile(ri, len(zs))
    order = np.lexsort((vals, rings))
    counts = np.bincount(rings, minlength=ri.max() + 1)
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    idx = np.minimum(starts + (counts * pct / 100).astype(np.int64), len(vals) - 1)
    prof = vals[order][idx].astype(np.float64)
    prof[counts == 0] = np.nan
    good = ~np.isnan(prof)
    return np.interp(np.arange(len(prof)), np.flatnonzero(good), prof[good])


_SCALE = None


_TISSUE_T = 0.0


def _slice_p99(z):
    """Tissue brightness of slice z: median of clear-tissue voxels (0 if too few)."""
    v = _VOL[z]
    v = v[v > _TISSUE_T]
    return float(np.median(v)) if v.size > 2000 else 0.0


def _scale_slice(z):
    _VOL[z] *= _SCALE[z]
    return z


_OTSU = False
_RI = None


def _otsu_background(sl, inside):
    """Per-slice background: split tissue/medium with Otsu on a smoothed copy, then fit the
    radial (cupping) profile from MEDIUM pixels only, so tissue never becomes 'background'."""
    from scipy.ndimage import gaussian_filter, median_filter
    from skimage.filters import threshold_otsu
    sm = gaussian_filter(sl, 2.0)
    vals = sm[inside]
    if vals.size < 1000:
        return None
    t = threshold_otsu(vals[:: max(1, vals.size // 200000)])
    med = inside & (sm < t)
    if med.sum() < 0.02 * inside.sum():
        return None
    ri = _RI
    cnt = np.bincount(ri[med], minlength=ri.max() + 1).astype(np.float64)
    tot = np.bincount(ri[med], weights=sm[med], minlength=ri.max() + 1)
    good = cnt > 20
    if good.sum() < 10:
        return None
    prof = np.interp(np.arange(len(cnt)), np.flatnonzero(good), tot[good] / cnt[good])
    prof = median_filter(prof, 31, mode="nearest")
    return prof[ri].astype(np.float32)


def _prep_slice(z):
    from scipy.ndimage import median_filter
    sl = _VOL[z]
    if _DESPECKLE > 1:
        sl[...] = median_filter(sl, size=_DESPECKLE)
    if _OTSU:
        inside = _slice_mask(z, *sl.shape) if _WALL is not None else (_MASK > 0)
        bg = _otsu_background(sl, inside)
        if bg is None:
            bg = _BG if _BG is not None else 0
        sl -= bg
        np.maximum(sl, 0, out=sl)
        sl *= inside
        return z
    if _BG is not None:
        sl -= _BG
        sl -= _OFFS[z]
        np.maximum(sl, 0, out=sl)
    if _WALL is not None:
        sl *= _slice_mask(z, *sl.shape)
    elif _MASK is not None:
        sl *= _MASK
    return z


def preprocess(vol, mask_radius=0.77, background=True, despeckle=3, workers=8, log=print,
               wall_fit=True, wall_margin=0.03, wall_block=64, cylinder=None, medium_pct=10.0,
               equalize=True, medium="otsu"):
    """In place on the shared volume (medium="otsu": per-slice tissue/medium split, recommended
    when tissue fills most of the cross-section; medium="ring": older low-percentile estimate):
      1. optional 2-D median filter per slice (removes floating particles)
      2. subtract the radial background (medium + cupping) PLUS a per-slice offset
         measured in a ring just inside the mask, so denser medium layers vanish too
      3. zero everything outside `mask_radius` (container walls)
    """
    global _BG, _MASK, _RING, _OFFS, _DESPECKLE, _WALL, _RING_PCT, _OTSU, _RI
    nz, ny, nx = vol.shape
    yy, xx = np.indices((ny, nx))
    r = np.hypot(yy - (ny - 1) / 2, xx - (nx - 1) / 2)
    R0 = min(nx, ny) / 2
    rc = r
    _MASK = (rc < mask_radius * R0).astype(np.float32) if mask_radius > 0 else None
    _WALL = None
    _DESPECKLE = despeckle
    _BG = None
    if background:
        from scipy.ndimage import median_filter
        ri = r.astype(np.int64).ravel()
        counts = np.maximum(np.bincount(ri), 1)
        zs = np.linspace(0, nz - 1, min(nz, 16)).astype(int)
        prof = ring_percentile_profile(vol, zs, ri, medium_pct)
        _BG = median_filter(prof, 31)[ri].reshape(ny, nx).astype(np.float32)
        if cylinder is not None:                         # (cy[], cx[], r[]) per slice, from _run
            _WALL = tuple(np.asarray(a, np.float32) for a in cylinder)
            log(f"geometric cut: r={_WALL[2][0]:.0f}, centre first slice y={_WALL[0][0]:.0f} x={_WALL[1][0]:.0f}"
                f" -> last slice y={_WALL[0][-1]:.0f} x={_WALL[1][-1]:.0f} (level px)")
        elif wall_fit and mask_radius > 0:
            _WALL = fit_walls(vol, _BG, wall_block, wall_margin, mask_radius * R0, log)
        if _WALL is not None:
            _RING = None                     # per-slice ring from the wall fit
        else:
            rr = mask_radius if mask_radius > 0 else 0.77
            _RING = (rc >= (rr - 0.07) * R0) & (rc < (rr - 0.01) * R0)
        _RING_PCT = medium_pct
        with Pool(workers) as pool:
            med = np.array(pool.map(_ring_median, range(nz), chunksize=8))
        _RING_PCT = 50.0
        med_s = median_filter(med, 9, mode="nearest")
        _OFFS = (med_s - float(np.median(med_s))).astype(np.float32)   # deviation from the typical medium
        log(f"medium offset per slice: min={_OFFS.min():.0f} median={np.median(_OFFS):.0f} max={_OFFS.max():.0f}")
    _OTSU = medium == "otsu"
    _RI = r.astype(np.int64)
    if _OTSU:
        log("medium removal: per-slice Otsu split, radial profile from medium pixels only")
    with Pool(workers) as pool:
        pool.map(_prep_slice, range(nz), chunksize=8)
    _OTSU = False
    if equalize:
        global _SCALE, _TISSUE_T
        from scipy.ndimage import median_filter
        from skimage.filters import threshold_otsu
        smp = vol[::4, ::8, ::8]; smp = smp[smp > 0]
        _TISSUE_T = float(threshold_otsu(smp[:: max(1, smp.size // 300000)])) if smp.size > 1000 else 0.0
        with Pool(workers) as pool:
            p99 = np.array(pool.map(_slice_p99, range(nz), chunksize=8))
        good = p99 > 0
        if good.sum() > 10:
            ps = np.interp(np.arange(nz), np.flatnonzero(good), p99[good])
            ps = median_filter(ps, max(3, nz // 20) | 1, mode="nearest")
            target = float(np.median(ps))
            _SCALE = np.clip(target / np.maximum(ps, 1e-6), 0.6, 1.6).astype(np.float32)
            with Pool(workers) as pool:
                pool.map(_scale_slice, range(nz), chunksize=8)
            log(f"equalize along Z: gain {_SCALE.min():.2f}..{_SCALE.max():.2f}")
    log(f"preprocess: despeckle={despeckle} background={'on' if background else 'off'} mask_radius={mask_radius}")
    return vol


def crop_box(imgs, thresh=8, margin=16):
    """Bounding box (rows, cols) of content over a list of uint8 images."""
    acc = np.zeros(imgs[0].shape, bool)
    for im in imgs:
        acc |= im > thresh
    if not acc.any():
        return slice(None), slice(None)
    rows, cols = np.where(acc.any(1))[0], np.where(acc.any(0))[0]
    h, w = acc.shape
    return (slice(max(rows[0] - margin, 0), min(rows[-1] + margin + 1, h)),
            slice(max(cols[0] - margin, 0), min(cols[-1] + margin + 1, w)))


# ---------------------------------------------------------------------------
# rendering (runs in worker processes)
# ---------------------------------------------------------------------------
def _render_frame(args):
    theta, mode, lo, hi, opacity, shade, depth = args
    from scipy import ndimage
    nz, ny, nx = _VOL.shape
    out = np.zeros((nz, nx), np.float32)
    inv = 1.0 / max(hi - lo, 1e-6)
    wdepth = (1.0 - depth * np.linspace(0, 1, ny, dtype=np.float32))[:, None]   # front (y=0) brightest
    for z in range(nz):
        sl = _VOL[z]
        if theta % 360:
            sl = ndimage.rotate(sl, theta, axes=(1, 0), reshape=False, order=1,
                                mode="constant", cval=0.0)
        v = np.clip((sl - lo) * inv, 0, 1)            # rays travel along Y
        if mode == "mip":
            out[z] = (v * wdepth).max(axis=0) if depth else v.max(axis=0)
            continue
        a = v * opacity
        c = v
        if shade:
            gy, gx = np.gradient(v)
            gm = np.hypot(gy, gx)
            # surfaces opaque, flat interiors translucent; lit from the viewer
            a = a * ((1 - shade) + shade * np.clip(gm / 0.05, 0, 1))
            c = v * ((1 - shade * 0.65) + shade * 0.65 * np.abs(gy) / (gm + 1e-6))
        if depth:
            c = c * wdepth
        a = np.clip(a, 0, 1)
        T = np.cumprod(1.0 - a, axis=0)                 # front-to-back compositing
        T[1:] = T[:-1].copy()
        T[0] = 1.0
        out[z] = (a * c * T).sum(axis=0)
    return out


def render_turntable(n_frames, mode, lo, hi, opacity, workers, log=print, shade=0.0, depth=0.0):
    angles = [360.0 * i / n_frames for i in range(n_frames)]
    t0 = time.time()
    with Pool(workers) as pool:
        frames = []
        for i, f in enumerate(pool.imap(_render_frame, [(a, mode, lo, hi, opacity, shade, depth) for a in angles])):
            frames.append(f)
            if (i + 1) % 20 == 0 or i + 1 == n_frames:
                log(f"  {mode} turntable: {i + 1}/{n_frames}  ({time.time() - t0:.0f}s)")
    return frames


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------
def to_uint8(frames, gamma=1.0):
    """One global scale for all frames (no flicker)."""
    sample = np.concatenate([f[::4, ::4].ravel() for f in frames[:: max(1, len(frames) // 12)]])
    nz = sample[sample > 0]
    top = np.percentile(nz, 99.8) if nz.size else 1.0
    out = []
    for f in frames:
        g = np.clip(f / max(top, 1e-6), 0, 1) ** gamma
        out.append((g * 255).astype(np.uint8))
    return out


def colorize(imgs, cmap):
    """uint8 gray -> uint8 RGB via a matplotlib colormap ('gray' returns input)."""
    if cmap in ("gray", "grey"):
        return imgs
    import matplotlib
    lut = (matplotlib.colormaps[cmap](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    return [lut[im] for im in imgs]


def write_frames(imgs, folder, prefix):
    import imageio.v3 as iio
    folder.mkdir(parents=True, exist_ok=True)
    for i, im in enumerate(imgs):
        iio.imwrite(folder / f"{prefix}_{i:04d}.png", im)


def write_mp4(imgs, path, fps):
    import imageio.v2 as iio2
    h, w = imgs[0].shape[:2]
    ph, pw = (-h) % 16, (-w) % 16                     # pad to codec-friendly size
    with iio2.get_writer(str(path), fps=fps, codec="libx264", quality=8,
                         pixelformat="yuv420p", macro_block_size=1) as wr:
        for im in imgs:
            pad = ((0, ph), (0, pw)) + (((0, 0),) if im.ndim == 3 else ())
            wr.append_data(np.pad(im, pad, mode="constant"))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("zarr_path", type=Path)
    p.add_argument("--out", type=Path, default=Path("renders"))
    p.add_argument("--level", default=None, help="pyramid level to render (default: finest within --budget-gb)")
    p.add_argument("--budget-gb", type=float, default=24, help="max float32 size of the level in RAM (default 24)")
    p.add_argument("--frames", type=int, default=180, help="turntable frames (default 180 = 6 s at 30 fps)")
    p.add_argument("--modes", default="mip,comp", help="comma list of: mip, comp (default both)")
    p.add_argument("--slice-sweep", type=int, default=240, help="slice fly-through frames (0 = skip)")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--mask-radius", type=float, default=0.77,
                   help="fallback mask radius as a fraction of the half-width, used if the wall fit fails "
                        "or with --no-wall-fit (0 = no mask at all; default 0.77)")
    p.add_argument("--no-background", action="store_true", help="don't subtract the radial background")
    p.add_argument("--despeckle", type=int, default=3, help="2-D median filter size per slice (0/1 = off; default 3)")
    p.add_argument("--no-crop", action="store_true", help="don't crop frames to the sample's bounding box")
    p.add_argument("--no-wall-fit", action="store_true",
                   help="use a fixed circle (--mask-radius, image-centred) instead of fitting the container wall")
    p.add_argument("--wall-margin", type=float, default=0.03,
                   help="mask edge this far inside the fitted wall, as a fraction of the half-width (default 0.03)")
    p.add_argument("--no-equalize", action="store_true",
                   help="don't equalize tissue brightness along Z (stitched scans often drift)")
    p.add_argument("--medium-pct", type=float, default=10.0,
                   help="percentile used to estimate the embedding medium per radius/slice (default 10; "
                        "lower if tissue fills almost everything)")
    p.add_argument("--cylinder", type=float, nargs="+", default=None, metavar="V",
                   help="explicit geometric cut, FULL-RESOLUTION pixels (overrides the automatic wall fit): "
                        "'CY CX R' = vertical cylinder, or 'CY_TOP CX_TOP CY_BOTTOM CX_BOTTOM R' = tilted "
                        "cylinder whose centre moves linearly from the first to the last slice")
    p.add_argument("--no-z-trim", action="store_true",
                   help="don't auto-trim air/meniscus slices (darker than the medium) at the top/bottom")
    p.add_argument("--z-range", type=int, nargs=2, metavar=("Z0", "Z1"), default=None,
                   help="render only full-resolution slices Z0..Z1 (overrides auto trim)")
    p.add_argument("--lo-pct", type=float, default=90, help="transfer-function low percentile (default 90)")
    p.add_argument("--hi-pct", type=float, default=99.9, help="transfer-function high percentile (default 99.9)")
    p.add_argument("--opacity", type=float, default=0.08, help="comp mode opacity per voxel (default 0.08)")
    p.add_argument("--shade", type=float, default=0.0,
                   help="comp mode surface shading 0..1: surfaces opaque + lit, interiors translucent (default 0)")
    p.add_argument("--depth-cue", type=float, default=0.0,
                   help="darken with distance from the viewer, 0..1 (default 0)")
    p.add_argument("--cmaps", default="gray",
                   help="comma list of matplotlib colormaps for the videos/stills, e.g. gray,bone,magma (default gray)")
    p.add_argument("--factor", type=int, default=1,
                   help="for .npy input: downsampling factor vs full resolution (for --z-range/--cylinder units)")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    args = p.parse_args(argv)

    def log(*a):
        print(*a, flush=True)

    global _VOL
    if str(args.zarr_path).endswith(".npy"):
        arr = np.load(args.zarr_path, mmap_mode="r")
        factor = args.factor
        log(f"rendering {args.zarr_path}: {arr.shape} {arr.dtype} (factor {factor})")
    else:
        root, keys = open_levels(args.zarr_path)
        for k in keys:
            a = root[k]
            log(f"  level {k}: shape={a.shape} {np.prod(a.shape) * 2 / 1024**3:.2f} GB (uint16)")
        lvl = args.level if args.level is not None else pick_level(root, keys, args.budget_gb)
        log(f"rendering level {lvl}: {root[lvl].shape}")
        arr = root[lvl]
        factor = 2 ** int(lvl)

    t0 = time.time()
    from multiprocessing import shared_memory
    shm = shared_memory.SharedMemory(create=True, size=int(np.prod(arr.shape)) * 4)
    try:
        _VOL = np.ndarray(arr.shape, np.float32, buffer=shm.buf)
        step = max(1, arr.chunks[0] if hasattr(arr, "chunks") and arr.chunks else 64)
        for z0 in range(0, arr.shape[0], step):
            _VOL[z0:z0 + step] = arr[z0:z0 + step]
        log(f"loaded in {time.time() - t0:.0f}s")
        _run(args, _VOL, lo_hi_log=log, t0=t0, factor=factor)
    finally:
        _VOL = None
        shm.close()
        shm.unlink()


def auto_z_trim(vol, mask_radius, workers, tol=0.02, margin=8, log=print):
    """Trim slices at the top/bottom ends that are darker than the medium (air,
    meniscus) or much brighter (container base). Only contiguous runs from the
    ends are trimmed, so interior tissue is never removed."""
    global _RING
    nz, ny, nx = vol.shape
    yy, xx = np.indices((ny, nx))
    r = np.hypot(yy - (ny - 1) / 2, xx - (nx - 1) / 2)
    _RING = r < (mask_radius if mask_radius > 0 else 0.77) * min(nx, ny) / 2
    with Pool(workers) as pool:
        med = np.array(pool.map(_ring_median, range(nz), chunksize=8))
    ref = float(np.median(med))
    dark = (med < ref * (1 - tol)) | (med > ref * (1 + 2.5 * tol))   # air/meniscus or container base
    z0 = 0
    while z0 < nz and dark[z0]:
        z0 += 1
    z1 = nz
    while z1 > z0 and dark[z1 - 1]:
        z1 -= 1
    if z0 > 0:
        z0 = min(z0 + margin, z1)
    if z1 < nz:
        z1 = max(z1 - margin, z0)
    return z0, z1


def _run(args, vol, lo_hi_log, t0, factor=1):
    global _VOL
    log = lo_hi_log
    nz = vol.shape[0]
    if args.z_range:
        z0, z1 = (max(0, args.z_range[0] // factor), min(nz, -(-args.z_range[1] // factor)))
    elif not args.no_z_trim:
        z0, z1 = auto_z_trim(vol, args.mask_radius, args.workers, log=log)
    else:
        z0, z1 = 0, nz
    cyl = None
    if args.cylinder is not None:
        c = list(args.cylinder)
        if len(c) == 3:
            c = [c[0], c[1], c[0], c[1], c[2]]
        if len(c) != 5:
            raise SystemExit("--cylinder takes 3 values (CY CX R) or 5 (CY_TOP CX_TOP CY_BOTTOM CX_BOTTOM R)")
        t = np.arange(z0, z1) / max(nz - 1, 1)           # position in the WHOLE volume, 0 = top
        cyl = ((c[0] + (c[2] - c[0]) * t) / factor, (c[1] + (c[3] - c[1]) * t) / factor,
               np.full(z1 - z0, c[4] / factor))
    if (z0, z1) != (0, nz):
        log(f"Z crop: level slices {z0}..{z1} of {nz}  (= full-res {z0 * factor}..{z1 * factor})")
        _VOL = vol = vol[z0:z1]
    raw_for_slices = None
    if args.slice_sweep:
        step = max(1, vol.shape[0] // args.slice_sweep)
        raw_for_slices = vol[::step].copy()
    preprocess(vol, args.mask_radius, not args.no_background, args.despeckle, args.workers, log,
               wall_fit=not args.no_wall_fit, wall_margin=args.wall_margin,
               wall_block=max(8, 512 // factor),
               cylinder=cyl, medium_pct=args.medium_pct, equalize=not args.no_equalize)

    s = vol[::2, ::4, ::4]
    s = s[s > 0]
    lo, hi = (np.percentile(s, [args.lo_pct, args.hi_pct]) if s.size else (0.0, 1.0))
    log(f"transfer function: lo={lo:.0f} hi={hi:.0f} (p{args.lo_pct}/p{args.hi_pct} of foreground)")

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "views").mkdir(exist_ok=True)
    import imageio.v3 as iio

    workers = max(1, min(args.workers, args.frames))
    box = None                     # crop box from the first mode, reused so all videos match
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        log(f"\n[{mode} turntable, {args.frames} frames, {workers} workers]")
        frames = render_turntable(args.frames, mode, lo, hi, args.opacity, workers, log,
                                  shade=args.shade if mode == "comp" else 0.0, depth=args.depth_cue)
        gray = to_uint8(frames, gamma=0.8 if mode == "comp" else 1.0)
        del frames
        if not args.no_crop:
            if box is None:
                box = crop_box(gray)
            gray = [im[box] for im in gray]
        for k, cmap in enumerate([c.strip() for c in args.cmaps.split(",") if c.strip()]):
            imgs = colorize(gray, cmap)
            tag = mode if cmap in ("gray", "grey") else f"{mode}_{cmap}"
            if k == 0:
                write_frames(imgs, out / f"turntable_{tag}", "frame")
            write_mp4(imgs, out / f"turntable_{tag}.mp4", args.fps)
            log(f"  -> {out / f'turntable_{tag}.mp4'}")
            for deg in (0, 90, 180, 270):
                i = int(round(deg / 360 * args.frames)) % args.frames
                iio.imwrite(out / "views" / f"{tag}_{deg:03d}deg.png", imgs[i])

    # orthogonal MIP composite of the preprocessed volume (Z, Y, X projections)
    from matplotlib.figure import Figure
    fig = Figure(figsize=(15, 5))
    for ax, axis, title in zip(fig.subplots(1, 3), range(3), ["MIP along Z", "MIP along Y", "MIP along X"]):
        m = np.clip((vol.max(axis=axis) - lo) / max(hi - lo, 1e-6), 0, 1)
        ax.imshow(m, cmap="gray", aspect="equal"); ax.set_title(title); ax.axis("off")
    fig.tight_layout(); fig.savefig(out / "ortho_mips.png", dpi=150, bbox_inches="tight")

    # top-down MIP (looking along Z)
    top = vol.max(axis=0)
    top = (np.clip((top - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
    if not args.no_crop:
        top = top[crop_box([top])]
    iio.imwrite(out / "views" / "mip_top.png", top)
    log(f"  -> {out / 'views'}/")

    if raw_for_slices is not None:
        log(f"\n[slice sweep, {raw_for_slices.shape[0]} frames]")
        nzv = raw_for_slices[raw_for_slices > 0]
        a, b = np.percentile(nzv[:: max(1, nzv.size // 2_000_000)], [1, 99.8]) if nzv.size else (0, 1)
        imgs = [(np.clip((sl - a) / max(b - a, 1e-6), 0, 1) * 255).astype(np.uint8) for sl in raw_for_slices]
        write_frames(imgs, out / "slices", "slice")
        write_mp4(imgs, out / "slice_sweep.mp4", args.fps)
        log(f"  -> {out / 'slice_sweep.mp4'}")

    log(f"\nAll outputs in: {out}/  ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
