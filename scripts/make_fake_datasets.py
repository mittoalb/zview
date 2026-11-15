#!/usr/bin/env python3
"""Generate a suite of fake datasets that exercise every zview feature.

The content is a procedural "tomography sample" phantom, defined in physical
coordinates so it looks the same at any resolution or voxel anisotropy:

  * a capillary-like container cylinder (bright wall, textured matrix with a
    z gradient), air outside;
  * ~160 spherical inclusions — dense grains and dark pores;
  * curvy vessel-like tubes and a helix (great in 3D MIP / isosurface);
  * a bar-pattern resolution target (periods 2–24 voxels) that blurs out at
    coarse pyramid levels — makes level switches easy to see;
  * three asymmetric marker cubes for reading orientation in 3D;
  * Gaussian noise.

Blocks are rendered independently (process pool) and written chunk-aligned,
so even multi-GB volumes are generated with bounded memory.

Datasets written to OUT_DIR (default /tmp/zview_fake):

  phantom_iso.zarr            Zarr v3, uint16 512³, 6 levels, OME-NGFF 0.4.
                              Main showcase: 2D panes, pyramid, 3D render.
  phantom_aniso_5d.zarr       Zarr v3, (t,c,z,y,x) = (1,2,128,768,768) uint16,
                              z voxel 4 µm vs 1 µm in-plane, z NOT downsampled
                              in the first levels, OME-NGFF 0.5 ("ome" key).
                              Channel 1 = vessels only. Tests 5D, aspect ratio,
                              per-axis level selection.
  phantom_ngff_centered.zarr  Same as a 256³ phantom_iso but with standard
                              OME-NGFF pixel-centre per-level translations.
                              Tests the translation handling (levels must not
                              jump when zooming across a level change).
  phantom_v2_float32.zarr     Zarr v2, float32 192×256×256 with negative values
                              and a NaN slab, 4 levels.
  phantom.n5                  N5 (via z5py), uint8 256³, numbered levels and no
                              multiscales metadata (fallback discovery).
  numbered_no_metadata.zarr   Zarr v2 group with arrays 0..3, no attributes.
  image_2d.zarr               2D uint8 4096², 5 levels.
  big_single_level.zarr       Zarr v3 uint8, default 768×2048×2048 (3.2 GB raw),
                              ONE level (no pyramid). Stress test for the
                              strided 3D path, progress and cancel.
                              Skip with --no-big; --big-pyramid adds levels.

Usage:
    python scripts/make_fake_datasets.py [OUT_DIR] [--quick] [--no-big]
                                         [--big-shape Z Y X] [--big-pyramid]
"""

import argparse
import math
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import zarr
except ImportError:
    sys.exit("zarr>=3 is required:  pip install 'zarr>=3'")

WORKERS = max(1, min(32, os.cpu_count() or 4))


# ============================================================== phantom

def _smooth(dist_inside, width):
    """Anti-aliased occupancy from a signed distance (positive inside)."""
    return np.clip(dist_inside / width + 0.5, 0.0, 1.0)


@dataclass
class Phantom:
    extent: tuple                 # physical (Lz, Ly, Lx)
    seed: int = 0
    noise: float = 500.0
    vessels_only: bool = False
    spheres: list = field(default_factory=list)
    curves: list = field(default_factory=list)   # (points[N,3], radius, value)
    cubes: list = field(default_factory=list)    # (center[3], half, value)

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        Lz, Ly, Lx = self.extent
        m = min(Lx, Ly)
        self.cx, self.cy = Lx / 2, Ly / 2
        self.R = 0.44 * m
        self.wall = 0.02 * m
        self.z_lo, self.z_hi = 0.04 * Lz, 0.96 * Lz

        def inside_point(margin):
            while True:
                x, y = rng.uniform(self.cx - self.R, self.cx + self.R, 2) + [0, self.cy - self.cx]
                if math.hypot(x - self.cx, y - self.cy) < self.R - self.wall - margin:
                    z = rng.uniform(self.z_lo + margin, self.z_hi - margin)
                    return np.array([z, y, x])

        for _ in range(160):
            r = min(rng.uniform(0.012, 0.06) * m, 0.35 * (self.z_hi - self.z_lo))
            c = inside_point(r)
            val = rng.uniform(26000, 42000) if rng.random() < 0.65 else 1500.0
            self.spheres.append((c, r, val))

        # Vessel-like tubes: straight-ish runs with two sine wiggles.
        for _ in range(7):
            p0 = inside_point(0.0)
            d = rng.normal(size=3); d /= np.linalg.norm(d)
            e1 = np.cross(d, [1.0, 0, 0]); e1 /= np.linalg.norm(e1) + 1e-9
            e2 = np.cross(d, e1)
            r = rng.uniform(0.006, 0.014) * m
            L = rng.uniform(0.6, 1.2) * m
            n = max(32, int(L / (r / 2)))
            t = np.linspace(-0.5, 0.5, n)[:, None]
            f1, f2, ph1, ph2 = rng.uniform(1, 3), rng.uniform(1, 3), *rng.uniform(0, 6.3, 2)
            amp = 0.08 * m
            pts = (p0 + t * L * d + amp * np.sin(2 * np.pi * f1 * t + ph1) * e1
                   + amp * np.sin(2 * np.pi * f2 * t + ph2) * e2)
            self.curves.append((self._clip_to_container(pts, r), r, 52000.0))

        # Helix around the axis.
        r = 0.01 * m
        turns = 3.0
        n = int(turns * 2 * np.pi * 0.25 * m / (r / 2)) + 64
        t = np.linspace(0, 1, n)
        pts = np.stack([self.z_lo + 0.05 * Lz + t * (self.z_hi - self.z_lo - 0.1 * Lz),
                        self.cy + 0.25 * m * np.sin(2 * np.pi * turns * t),
                        self.cx + 0.25 * m * np.cos(2 * np.pi * turns * t)], axis=1)
        self.curves.append((pts, r, 47000.0))

        # Orientation markers: three cubes of different sizes, asymmetric.
        zc = self.z_lo + 0.12 * (self.z_hi - self.z_lo)
        self.cubes = [
            (np.array([zc, self.cy - 0.20 * m, self.cx - 0.20 * m]), 0.015 * m, 60000.0),
            (np.array([zc, self.cy - 0.20 * m, self.cx + 0.10 * m]), 0.025 * m, 60000.0),
            (np.array([zc, self.cy + 0.05 * m, self.cx - 0.20 * m]), 0.035 * m, 60000.0),
        ]

    def _clip_to_container(self, pts, r):
        rad = np.hypot(pts[:, 2] - self.cx, pts[:, 1] - self.cy)
        keep = (rad < self.R - self.wall - r) & (pts[:, 0] > self.z_lo + r) & (pts[:, 0] < self.z_hi - r)
        return pts[keep]

    # ---------------------------------------------------------- render

    LABELS = {3: "matrix", 5: "dense grain", 7: "pore", 9: "marker cube", 13: "resolution target",
              21: "container wall", 50: "helix", 70: "vessel lumen", 255: "void (outside the sample)"}

    def render_labels(self, origin, shape, voxel):
        """uint8 class IDs for the block (see LABELS); 255 = outside."""
        self._lab = np.full(shape, 255, np.uint8)
        self._labelling = True
        try:
            self.render(origin, shape, voxel)
        finally:
            self._labelling = False
        return self._lab

    def _mark(self, sl, alpha, cls):
        if getattr(self, "_labelling", False):
            view = self._lab[sl] if sl is not None else self._lab
            view[np.broadcast_to(alpha, view.shape) > 0.5] = cls

    def render(self, origin, shape, voxel):
        """Float32 block for level-0 voxel indices origin..origin+shape."""
        (z0, y0, x0), (nz, ny, nx), (vz, vy, vx) = origin, shape, voxel
        Z = ((np.arange(nz) + z0 + 0.5) * vz)[:, None, None].astype(np.float32)
        Y = ((np.arange(ny) + y0 + 0.5) * vy)[None, :, None].astype(np.float32)
        X = ((np.arange(nx) + x0 + 0.5) * vx)[None, None, :].astype(np.float32)
        w = float(max(vz, vy, vx))
        Lz, Ly, Lx = self.extent

        rad = np.sqrt((X - self.cx) ** 2 + (Y - self.cy) ** 2)            # (1, ny, nx)
        zin = np.minimum(Z - self.z_lo, self.z_hi - Z)                     # (nz, 1, 1)
        inside = _smooth(np.minimum(self.R - rad, zin), w)                 # (nz, ny, nx)
        vol = np.full((nz, ny, nx), 800.0, np.float32)
        if not self.vessels_only:
            matrix = (11000 + 3000 * Z / Lz
                      + 2500 * np.sin(6 * np.pi * X / Lx) * np.sin(4 * np.pi * Y / Ly)
                      * np.cos(2 * np.pi * Z / Lz))
            vol = vol * (1 - inside) + matrix * inside
            self._mark(None, inside, 3)
            shell = _smooth(np.minimum(self.wall / 2 - np.abs(rad - self.R + self.wall / 2), zin), w)
            vol = vol * (1 - shell) + 20000 * shell
            self._mark(None, shell, 21)
        else:
            vol = vol * (1 - inside) + 3000 * inside

        def subbox(center, half):
            lo = [max(0, int(math.floor((c - half) / v)) - o) for c, v, o in zip(center, voxel, origin)]
            hi = [min(n, int(math.ceil((c + half) / v)) - o + 1) for c, v, o, n in zip(center, voxel, origin, shape)]
            if any(h <= l for l, h in zip(lo, hi)):
                return None
            return tuple(slice(l, h) for l, h in zip(lo, hi))

        def blend(sl, alpha, value):
            vol[sl] = vol[sl] * (1 - alpha) + value * alpha

        if not self.vessels_only:
            for c, r, val in self.spheres:
                sl = subbox(c, r + w)
                if sl is None:
                    continue
                d = np.sqrt((Z[sl[0]] - c[0]) ** 2 + (Y[:, sl[1]] - c[1]) ** 2 + (X[..., sl[2]] - c[2]) ** 2)
                a = _smooth(r - d, w)
                blend(sl, a, val)
                self._mark(sl, a, 7 if val < 5000 else 5)

            # Resolution target: x bars with periods given in level-0 voxels.
            m = min(Lx, Ly)
            tz = (self.z_lo + 0.15 * (self.z_hi - self.z_lo), self.z_lo + 0.35 * (self.z_hi - self.z_lo))
            ty = (self.cy + 0.12 * m, self.cy + 0.22 * m)
            xs = self.cx - 0.25 * m
            x_vox = np.arange(nx) + x0                                    # level-0 x index
            bars = np.zeros(nx, np.float32)
            pos = xs / vx
            for p in (2, 3, 4, 6, 8, 12, 16, 24):
                start, end = pos, pos + 4 * p
                sel = (x_vox >= start) & (x_vox < end)
                bars[sel] = ((x_vox[sel] - start) % p) < p / 2
                pos = end + 8
            if bars.any():
                zm = ((Z[:, 0, 0] > tz[0]) & (Z[:, 0, 0] < tz[1]))
                ym = ((Y[0, :, 0] > ty[0]) & (Y[0, :, 0] < ty[1]))
                a = zm[:, None, None] * ym[None, :, None] * bars[None, None, :]
                a = a * _smooth(self.R - self.wall - rad, w)          # stay inside the container
                vol = vol * (1 - a) + 40000 * a
                self._mark(None, a, 13)

        for pts, r, val in self.curves:
            lo = np.array([z0 * vz, y0 * vy, x0 * vx]) - r - w
            hi = np.array([(z0 + nz) * vz, (y0 + ny) * vy, (x0 + nx) * vx]) + r + w
            near = pts[np.all((pts >= lo) & (pts <= hi), axis=1)]
            if len(near) == 0:
                continue
            alpha = np.zeros_like(vol)
            for c in near:
                sl = subbox(c, r + w)
                if sl is None:
                    continue
                d = np.sqrt((Z[sl[0]] - c[0]) ** 2 + (Y[:, sl[1]] - c[1]) ** 2 + (X[..., sl[2]] - c[2]) ** 2)
                np.maximum(alpha[sl], _smooth(r - d, w), out=alpha[sl])
            vol = vol * (1 - alpha) + val * alpha
            self._mark(None, alpha, 50 if val == 47000.0 else 70)

        if not self.vessels_only:
            for c, half, val in self.cubes:
                sl = subbox(c, half + w)
                if sl is None:
                    continue
                d = np.minimum.reduce([half - np.abs(Z[sl[0]] - c[0]) + 0 * Y[:, sl[1]] + 0 * X[..., sl[2]],
                                       half - np.abs(Y[:, sl[1]] - c[1]) + 0 * Z[sl[0]] + 0 * X[..., sl[2]],
                                       half - np.abs(X[..., sl[2]] - c[2]) + 0 * Z[sl[0]] + 0 * Y[:, sl[1]]])
                a = _smooth(d, w)
                blend(sl, a, val)
                self._mark(sl, a, 9)

        if getattr(self, "_labelling", False):
            return vol
        if self.noise > 0:
            rng = np.random.default_rng((self.seed, z0, y0, x0, int(self.vessels_only)))
            vol += rng.standard_normal(vol.shape, dtype=np.float32) * self.noise
        return vol


def _convert(vol, dtype):
    dtype = np.dtype(dtype)
    if dtype == np.uint16:
        return np.clip(vol, 0, 65535).astype(np.uint16)
    if dtype == np.uint8:
        return np.clip(vol / 256.0, 0, 255).astype(np.uint8)
    if dtype == np.float32:
        return (vol / 65535.0 - 0.1).astype(np.float32)   # includes negatives
    raise ValueError(dtype)


# ============================================================== workers

_ARR_CACHE = {}


def _open(path):
    arr = _ARR_CACHE.get(path)
    if arr is None:
        arr = _ARR_CACHE[path] = zarr.open_array(path, mode="r+")
    return arr


def _render_task(args):
    path, lead, phantom, origin, shape, voxel, dtype = args
    if dtype == "labels":
        data = phantom.render_labels(origin, shape, voxel)
    else:
        data = _convert(phantom.render(origin, shape, voxel), dtype)
    arr = _open(path)
    if arr.ndim == 2:                      # 2D image: single slice
        arr[origin[1]:origin[1] + shape[1], origin[2]:origin[2] + shape[2]] = data[0]
        return data.nbytes
    region = tuple(slice(o, o + s) for o, s in zip(origin, shape))
    arr[lead + region] = data
    return data.nbytes


def _down_task(args):
    src_path, dst_path, lead, out_origin, out_shape, factors = args
    src, dst = _open(src_path), _open(dst_path)
    if dst.dtype == np.uint8 and getattr(dst, "attrs", {}).get("zview_stride_labels"):
        # Label pyramid: stride-2 subsampling (like many segmentation tools).
        lo = [o * f for o, f in zip(out_origin, factors)]
        blk = np.asarray(src[lead + tuple(slice(a, a + s * f, f) for a, s, f in zip(lo, out_shape, factors))])
        dst[lead + tuple(slice(o, o + b) for o, b in zip(out_origin, blk.shape))] = blk
        return blk.nbytes
    nd = len(factors)
    src_sp = src.shape[-nd:]
    lo = [o * f for o, f in zip(out_origin, factors)]
    hi = [min(n, (o + s) * f) for o, s, f, n in zip(out_origin, out_shape, factors, src_sp)]
    block = np.asarray(src[lead + tuple(slice(a, b) for a, b in zip(lo, hi))], dtype=np.float32)
    # Edge-pad to a multiple of the factor, then block-mean (NaN-aware).
    pad = [(0, max(0, s * f - n)) for s, f, n in zip(out_shape, factors, block.shape)]
    if any(p[1] for p in pad):
        block = np.pad(block, pad, mode="edge")
    shp = []
    for s, f in zip(out_shape, factors):
        shp += [s, f]
    block = block.reshape(shp)
    axes = tuple(range(1, 2 * nd, 2))
    with np.errstate(invalid="ignore"):
        out = np.nanmean(block, axis=axes) if np.isnan(block).any() else block.mean(axis=axes)
    dst[lead + tuple(slice(o, o + s) for o, s in zip(out_origin, out_shape))] = out.astype(dst.dtype)
    return out.nbytes


def _blocks(shape, chunks, group=(1, 1, 1)):
    """Chunk-aligned blocks (origin, shape) covering `shape`."""
    steps = [c * g for c, g in zip(chunks, group)]
    for z in range(0, shape[0], steps[0]):
        for y in range(0, shape[1], steps[1]):
            for x in range(0, shape[2], steps[2]):
                o = (z, y, x)
                yield o, tuple(min(s, n - a) for s, n, a in zip(steps, shape, o))


def _run(pool, fn, tasks, label):
    t0 = time.perf_counter()
    total = len(tasks)
    done = 0
    nbytes = 0
    for nb in pool.map(fn, tasks, chunksize=max(1, total // (WORKERS * 8))):
        done += 1
        nbytes += nb
        if (done == total) or (sys.stdout.isatty() and done % max(1, total // 20) == 0):
            print(f"\r    {label}: {done}/{total} blocks  {nbytes / 2**20:8.0f} MB  "
                  f"{time.perf_counter() - t0:6.1f}s", end="", flush=True)
    print()


# ============================================================== writers

def _ngff_axes(names, units):
    kinds = {"t": "time", "c": "channel"}
    out = []
    for n, u in zip(names, units):
        a = {"name": n, "type": kinds.get(n, "space")}
        if u:
            a["unit"] = u
        out.append(a)
    return out


def write_pyramid(pool, path, *, shape, chunks, dtype, phantom, voxel, levels,
                  factors_per_level=None, lead_shape=(), lead_phantoms=None,
                  zarr_format=3, ngff="0.4", centered_translations=False,
                  axis_names=("z", "y", "x"), units=("micrometer",) * 3, fill_value=0,
                  name=None, metadata=True, group_blocks=(1, 1, 1)):
    """Render level 0 block-wise, then build each level by block-mean."""
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    root = zarr.open_group(str(path), mode="w", zarr_format=zarr_format)
    nd = len(shape)
    factors_per_level = factors_per_level or [(2,) * nd] * (levels - 1)
    lead_phantoms = lead_phantoms or [phantom]

    shapes = [tuple(shape)]
    for f in factors_per_level[:levels - 1]:
        shapes.append(tuple(max(1, math.ceil(s / k)) for s, k in zip(shapes[-1], f)))

    arrays = []
    store_dtype = "uint8" if dtype == "labels" else dtype
    for i, shp in enumerate(shapes):
        ch = tuple(min(c, s) for c, s in zip(chunks, shp))
        arr = root.create_array(str(i), shape=tuple(lead_shape) + shp,
                                chunks=(1,) * len(lead_shape) + ch, dtype=store_dtype,
                                fill_value=fill_value)
        if dtype == "labels":
            arr.attrs["zview_stride_labels"] = True
        arrays.append(str(path / str(i)))
        _ = arr

    leads = list(np.ndindex(*lead_shape)) if lead_shape else [()]
    print(f"  {path.name}: level 0 {tuple(lead_shape) + shapes[0]} {dtype}")
    sp3 = shapes[0] if nd == 3 else (1,) + shapes[0]
    ch3 = chunks if nd == 3 else (1,) + tuple(chunks)
    tasks = []
    for li, lead in enumerate(leads):
        ph = lead_phantoms[li % len(lead_phantoms)]
        for o, s in _blocks(sp3, ch3, group_blocks):
            if nd == 2:
                # 2D image = the phantom's middle z slice (z=0 is outside it).
                o = (int(ph.extent[0] / voxel[0]) // 2,) + tuple(o[1:])
            tasks.append((arrays[0], lead, ph, o, s, voxel, dtype))
    _run(pool, _render_task, tasks, "render L0")

    for i in range(1, len(shapes)):
        f = factors_per_level[i - 1]
        ch = tuple(min(c, s) for c, s in zip(chunks, shapes[i]))
        tasks = []
        for lead in leads:
            if nd == 3:
                for o, s in _blocks(shapes[i], ch):
                    tasks.append((arrays[i - 1], arrays[i], lead, o, s, f))
            else:
                for o, s in _blocks((1,) + shapes[i], (1,) + ch):
                    tasks.append((arrays[i - 1], arrays[i], lead, o[1:], s[1:], f))
        _run(pool, _down_task, tasks, f"build L{i} {shapes[i]}")

    if metadata:
        datasets = []
        cum = np.ones(nd)
        for i in range(len(shapes)):
            if i > 0:
                cum = cum * np.array(factors_per_level[i - 1], dtype=float)
            scale = list(np.array(voxel[-nd:]) * cum)
            if centered_translations:
                trans = list((cum - 1) / 2 * np.array(voxel[-nd:]))
            else:
                trans = [0.0] * nd
            lead_one = [1.0] * len(lead_shape)
            lead_zero = [0.0] * len(lead_shape)
            datasets.append({"path": str(i), "coordinateTransformations": [
                {"type": "scale", "scale": lead_one + [float(v) for v in scale]},
                {"type": "translation", "translation": lead_zero + [float(v) for v in trans]},
            ]})
        lead_names = ("t", "c")[-len(lead_shape):] if lead_shape else ()
        ms = [{
            "version": ngff if ngff != "0.5" else None,
            "name": name or path.stem,
            "axes": _ngff_axes(tuple(lead_names) + tuple(axis_names),
                               (None,) * len(lead_shape) + tuple(units)),
            "datasets": datasets,
        }]
        if ngff == "0.5":
            del ms[0]["version"]
            root.attrs["ome"] = {"version": "0.5", "multiscales": ms}
        else:
            root.attrs["multiscales"] = ms
    return shapes


def write_n5(path, phantom, shape, voxel, levels):
    try:
        import z5py
    except ImportError:
        print("  (z5py not installed — skipping N5 dataset)")
        return
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    f = z5py.File(str(path), use_zarr_format=False)
    vol = _convert(phantom.render((0, 0, 0), shape, voxel), np.uint8)
    for i in range(levels):
        ds = f.create_dataset(str(i), shape=vol.shape, chunks=tuple(min(64, s) for s in vol.shape),
                              dtype="uint8", compression="gzip")
        ds[:] = vol
        z, y, x = (s - s % 2 for s in vol.shape)
        vol = vol[:z, :y, :x].reshape(z // 2, 2, y // 2, 2, x // 2, 2).mean(axis=(1, 3, 5)).astype(np.uint8)
    print(f"  {path.name}: N5, {levels} numbered levels, no multiscales metadata")


# ============================================================== main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", nargs="?", default="/tmp/zview_fake")
    ap.add_argument("--quick", action="store_true", help="smaller sizes, no big dataset")
    ap.add_argument("--no-big", action="store_true", help="skip big_single_level.zarr")
    ap.add_argument("--big-shape", type=int, nargs=3, default=(768, 2048, 2048), metavar=("Z", "Y", "X"))
    ap.add_argument("--big-pyramid", action="store_true", help="give the big dataset a pyramid too")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    q = args.quick
    t_all = time.perf_counter()
    print(f"Writing fake datasets to {out}  ({WORKERS} workers)")

    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        # 1. Isotropic showcase.
        n = 256 if q else 512
        write_pyramid(pool, out / "phantom_iso.zarr", shape=(n, n, n), chunks=(64, 64, 64),
                      dtype="uint16", phantom=Phantom((n, n, n), seed=1), voxel=(1.0, 1.0, 1.0),
                      levels=5 if q else 6)

        # 2. Anisotropic 5D: z voxel 4x larger, z kept until roughly isotropic.
        zs, xy = (64, 384) if q else (128, 768)
        vox = (4.0, 1.0, 1.0)
        ext = (zs * vox[0], xy * vox[1], xy * vox[2])
        write_pyramid(pool, out / "phantom_aniso_5d.zarr", shape=(zs, xy, xy), chunks=(32, 128, 128),
                      dtype="uint16", phantom=None, voxel=vox, levels=5,
                      factors_per_level=[(1, 2, 2), (1, 2, 2), (2, 2, 2), (2, 2, 2)],
                      lead_shape=(1, 2),
                      lead_phantoms=[Phantom(ext, seed=2), Phantom(ext, seed=2, vessels_only=True, noise=300)],
                      ngff="0.5")

        # 3. Standard pixel-centre translations.
        write_pyramid(pool, out / "phantom_ngff_centered.zarr", shape=(256, 256, 256), chunks=(64, 64, 64),
                      dtype="uint16", phantom=Phantom((256, 256, 256), seed=1), voxel=(1.0, 1.0, 1.0),
                      levels=4, centered_translations=True)

        # 4. Zarr v2 float32 with negatives and a NaN slab.
        shp = (192, 256, 256)
        p = out / "phantom_v2_float32.zarr"
        write_pyramid(pool, p, shape=shp, chunks=(64, 64, 64), dtype="float32",
                      phantom=Phantom(shp, seed=3), voxel=(1.0, 1.0, 1.0), levels=4,
                      zarr_format=2, fill_value=float("nan"))
        a0 = zarr.open_array(str(p / "0"), mode="r+")
        a0[90:100, 40:200, 40:200] = np.nan
        print("    + NaN slab at z=90..99 (level 0 only)")

        # 4b. Label map (segmentation): class IDs with a lookup table, 255 = void,
        # stride-subsampled pyramid — the layout many segmentation tools write.
        n = 256 if q else 512
        p = out / "phantom_labels.zarr"
        write_pyramid(pool, p, shape=(n, n, n), chunks=(64, 64, 64), dtype="labels",
                      phantom=Phantom((n, n, n), seed=1), voxel=(1.0, 1.0, 1.0), levels=4 if q else 5,
                      fill_value=255)
        zarr.open_group(str(p), mode="r+").attrs["lookup"] = {str(k): v for k, v in Phantom.LABELS.items()}
        print("    + lookup table (class names)")

        # 5. Numbered levels, no metadata (Zarr v2).
        write_pyramid(pool, out / "numbered_no_metadata.zarr", shape=(128, 128, 128), chunks=(64, 64, 64),
                      dtype="uint8", phantom=Phantom((128, 128, 128), seed=4), voxel=(1.0, 1.0, 1.0),
                      levels=4, zarr_format=2, metadata=False)

        # 6. 2D image.
        n2 = 1024 if q else 4096
        write_pyramid(pool, out / "image_2d.zarr", shape=(n2, n2), chunks=(512, 512), dtype="uint8",
                      phantom=Phantom((n2, n2, n2), seed=5, noise=300), voxel=(1.0, 1.0, 1.0),
                      levels=5, axis_names=("y", "x"), units=("micrometer",) * 2)

        # 7. Big single-level volume.
        if not (q or args.no_big):
            bz, by, bx = args.big_shape
            write_pyramid(pool, out / "big_single_level.zarr", shape=(bz, by, bx), chunks=(128, 256, 256),
                          dtype="uint8", phantom=Phantom((bz, by, bx), seed=6, noise=400),
                          voxel=(1.0, 1.0, 1.0), levels=5 if args.big_pyramid else 1,
                          group_blocks=(1, 1, 2))

    # 8. N5 (z5py writes in-process; small).
    write_n5(out / "phantom.n5", Phantom((192, 192, 192), seed=7), (192, 192, 192), (1.0, 1.0, 1.0), 3)

    du = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"\nDone in {time.perf_counter() - t_all:.0f}s — {du / 2**30:.2f} GB on disk in {out}")
    for d in sorted(out.iterdir()):
        print(f"  zview {d}")


if __name__ == "__main__":
    main()
