"""Prepare any volume for rendering: a render-sized copy + a config.

    zview-render prepare SOURCE --out DIR [--preset P] [--max-axis 768 | --step 8]

SOURCE is a Zarr (v2/v3) or N5 store, an HDF5 file or a folder of TIFF
slices. Writes into DIR:

  volume_max<N>.npy / volume_step<S>.npy   the cached volume used for rendering
  side_views_grid.png    XZ + YZ sections with a full-resolution pixel grid
                         (read z_range / cylinder / tour regions off it)
  render.json            the render config; edit it, then
                         zview-render run DIR/render.json

Zarr/N5: the finest pyramid level that fits --max-axis plus a stride, read
in parallel (label maps with label-aware pooling). HDF5: every S-th slice and
row (averaged along X), so even multi-TB files take minutes. TIFF: every
S-th slice, averaged S x S in-plane.
"""

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

_G = {}


def is_zarr(path):
    p = Path(path)
    return p.suffix == ".zarr" or (p.is_dir() and (p / "zarr.json").exists())


# ---------------------------------------------------------------------------
# HDF5
# ---------------------------------------------------------------------------
def find_dataset(f, name):
    import h5py
    if name:
        return name
    for c in ("/exchange/data", "/data", "/volume", "/entry/data/data"):
        if c in f and isinstance(f[c], h5py.Dataset) and f[c].ndim == 3:
            return c
    best = [None, 0]

    def visit(n, o):
        if isinstance(o, h5py.Dataset) and o.ndim == 3 and o.size > best[1]:
            best[:] = ["/" + n, o.size]
    f.visititems(visit)
    if best[0] is None:
        raise SystemExit("no 3-D dataset in the HDF5; use --dataset")
    return best[0]


def _ds():
    if "ds" not in _G:
        import h5py
        _G["ds"] = h5py.File(_G["h5"], "r")[_G["dset"]]
    return _G["ds"]


def _row_sampled(z):
    ds, S = _ds(), _G["S"]
    ny, nx = ds.shape[1] // S, ds.shape[2] // S
    a = ds[z, ::S, :][:ny, : nx * S].astype(np.float32)
    return a.reshape(ny, nx, S).mean(axis=2)


def _sample(z):
    return _row_sampled(z)[::4, ::4].ravel()


def _extract(i):
    lo, hi = _G["lohi"]
    v = (np.clip((_row_sampled(_G["zs"][i]) - lo) / (hi - lo), 0, 1) * 65535).astype(np.uint16)
    m = np.load(_G["tmp"], mmap_mode="r+"); m[i] = v; m.flush()
    return i


def prepare_h5(src, dset, S, out_npy, workers):
    import h5py
    with h5py.File(src, "r") as f:
        dset = find_dataset(f, dset)
        nz, ny, nx = f[dset].shape
        print(f"{src}:{dset} shape={(nz, ny, nx)} dtype={f[dset].dtype}", flush=True)
    zs = list(range(0, nz, S))
    _G.update(h5=str(src), dset=dset, S=S, zs=zs)
    t = time.time()
    with Pool(workers) as p:
        smp = np.concatenate(p.map(_sample, zs[:: max(1, len(zs) // 64)]))
    lo, hi = np.percentile(smp, [0.5, 99.5])
    _G["lohi"] = (lo, hi)
    print(f"contrast p0.5/p99.5: {lo:.5g} .. {hi:.5g} ({time.time() - t:.0f}s)", flush=True)
    shape = (len(zs), ny // S, nx // S)
    tmp = str(out_npy) + ".partial.npy"
    np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint16, shape=shape).flush()
    _G["tmp"] = tmp
    t = time.time()
    with Pool(workers) as p:
        for k, _ in enumerate(p.imap_unordered(_extract, range(len(zs)), chunksize=2)):
            if (k + 1) % 100 == 0 or k + 1 == len(zs):
                el = time.time() - t
                print(f"  {k + 1}/{len(zs)} slices  {el:.0f}s  ETA {el / (k + 1) * (len(zs) - k - 1):.0f}s",
                      flush=True)
    os.replace(tmp, out_npy)
    return dict(full_shape=[nz, ny, nx], dataset=dset)


# ---------------------------------------------------------------------------
# Zarr / N5 (any zview-readable store) and TIFF folders
# ---------------------------------------------------------------------------
def _pick_zarr_plan(mi, max_axis=None, step=None):
    """(level, stride, factor_zyx) for the cached volume.

    max_axis: finest pyramid level whose largest side fits, plus a per-axis
    stride if even the coarsest is too big (zview.core.sysinfo.plan_volume).
    step: the old fixed-step behaviour — level whose downsample is closest to
    `step`, plus an integer stride to make up the rest.
    """
    from zview.core.sysinfo import VolumeBudget, plan_volume
    shapes = [lvl.shape for lvl in mi.levels]
    if step:
        lvl = min(range(len(shapes)), key=lambda i: abs(mi.level_downsample(i)[2] - step))
        st = max(1, int(round(step / mi.level_downsample(lvl)[2])))
        stride = (st, st, st)
    else:
        plan = plan_volume(shapes, VolumeBudget(max_axis=int(max_axis), max_voxels=int(max_axis) ** 3))
        lvl, stride = plan.level, plan.stride
    ds = mi.level_downsample(lvl)
    factor = [round(d * s, 4) for d, s in zip(ds, stride)]
    return lvl, tuple(stride), factor


def prepare_zarr(src, out_npy, max_axis=None, step=None, workers=None):
    from zview.render.sources import open_pyramid
    mi = open_pyramid(src)
    lvl, stride, factor = _pick_zarr_plan(mi, max_axis, step)
    nbytes = mi.estimate_read_bytes(lvl, stride)
    print(f"level {lvl} {mi.levels[lvl].shape} stride {stride} -> factor {factor} "
          f"(reads ~{nbytes / 2**30:.2f} GB)", flush=True)
    t = time.time()
    last = [0.0]

    def prog(done, total):
        if time.time() - last[0] > 5 or done == total:
            last[0] = time.time()
            print(f"  {done}/{total} blocks  {time.time() - t:.0f}s", flush=True)

    table = mi.labels
    if table is not None and tuple(stride) != (1, 1, 1):
        print("label map: label-aware downsampling (rare classes kept)", flush=True)
        vol = mi.read_volume_pooled(lvl, stride, table.priority_lut(), progress=prog)
    else:
        vol = mi.read_volume(lvl, stride, progress=prog)
    tmp = str(out_npy) + ".partial.npy"
    np.save(tmp, vol)
    os.replace(tmp, out_npy)
    vs = mi.voxel_size()
    info = dict(full_shape=list(mi.levels[0].shape), factor=factor, voxel_size=list(vs),
                pixel_size=vs[2], level=lvl, stride=list(stride))
    if table is not None:
        info["labels"] = table.to_config()
    return info


def prepare_tiff(src, out_npy, S, workers):
    """Every S-th slice of a TIFF folder, block-averaged S x S in-plane."""
    from concurrent.futures import ThreadPoolExecutor
    import tifffile
    from zview.tools.convert import TiffStack
    st = TiffStack(src)
    nz, ny, nx = st.shape
    zs = list(range(0, nz, S))
    oy, ox = ny // S, nx // S
    print(f"{src}: {nz} slices {ny}x{nx} {st.dtype}; step {S} -> {(len(zs), oy, ox)}", flush=True)
    m = np.lib.format.open_memmap(str(out_npy) + ".partial.npy", mode="w+",
                                  dtype=np.float32 if st.dtype.kind == "f" else st.dtype, shape=(len(zs), oy, ox))

    def one(i):
        a = tifffile.imread(st.files[zs[i]])[:oy * S, :ox * S].astype(np.float32)
        m[i] = a.reshape(oy, S, ox, S).mean(axis=(1, 3))
        return i

    t = time.time()
    with ThreadPoolExecutor(min(32, workers or 8)) as ex:
        for k, _ in enumerate(ex.map(one, range(len(zs)))):
            if (k + 1) % 100 == 0 or k + 1 == len(zs):
                print(f"  {k + 1}/{len(zs)} slices  {time.time() - t:.0f}s", flush=True)
    m.flush()
    m = None              # close the memmap before renaming
    os.replace(str(out_npy) + ".partial.npy", out_npy)
    return dict(full_shape=[nz, ny, nx], factor=S)


# ---------------------------------------------------------------------------
def side_views(npy, factor, path):
    """XZ / YZ sections through the centre with a full-resolution pixel grid."""
    try:
        from matplotlib.figure import Figure
    except ImportError:
        print("(matplotlib not installed: no side_views_grid.png)")
        return
    fz, fy, fx = (list(factor) * 3)[:3] if not isinstance(factor, (int, float)) else (factor,) * 3
    v = np.load(npy, mmap_mode="r")
    nz, ny, nx = v.shape

    def norm(a):
        a = np.nan_to_num(a.astype(np.float32)); nzv = a[a > 0]
        lo, hi = np.percentile(nzv, [1, 99.5]) if nzv.size else (0, 1)
        return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)

    def ticks(n):
        step = 10 ** max(0, int(np.log10(max(n, 10))) - 1) * 2
        while n / step > 14:
            step *= 2.5 if str(step)[0] == "2" else 2
        return range(0, int(n) + 1, int(step))

    fig = Figure(figsize=(16, 10))
    for ax, img, fh, title, xl in zip(fig.subplots(1, 2), (norm(v[:, ny // 2, :]), norm(v[:, :, nx // 2])),
                                      (fx, fy),
                                      ("XZ section (through centre Y)", "YZ section (through centre X)"),
                                      ("X (full-res px)", "Y (full-res px)")):
        W, Hh = img.shape[1] * fh, nz * fz
        ax.imshow(img, cmap="gray", extent=[0, W, Hh, 0], aspect="equal")
        ax.set_title(title); ax.set_xlabel(xl); ax.set_ylabel("Z (full-res slice)")
        ax.set_xticks(list(ticks(W))); ax.set_yticks(list(ticks(Hh)))
        ax.grid(color="yellow", alpha=0.35, lw=0.6); ax.tick_params(labelsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=110)


def prepare(source, out, preset=None, max_axis=None, step=None, dataset=None, pixel_size=None,
            workers=None, force=False, config_name="render.json"):
    """Cache a render-sized volume of `source` in `out` and write a config.

    Returns the config path. Re-running keeps an existing config (and the
    cached volume unless force=True).
    """
    from zview.render import config as rc
    from zview.render.sources import kind
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    src = Path(source).resolve()
    workers = workers or os.cpu_count()
    base = rc.effective({"preset": preset or "generic"})
    if max_axis is None and step is None:
        max_axis = base.get("max_axis")
        if not max_axis:
            step = int(base.get("factor") or 8)
    k = kind(src)
    tag = f"step{step}" if step else f"max{max_axis}"
    npy = out / f"volume_{tag}.npy"
    info_path = out / f"volume_{tag}.json"
    info = {}
    if npy.exists() and info_path.exists() and not force:
        print(f"cached volume exists: {npy}")
        info = json.loads(info_path.read_text())
    else:
        if k == "zarr":
            info = prepare_zarr(src, npy, max_axis=max_axis, step=step, workers=workers)
        else:
            if not step:
                shape = _source_shape(src, k, dataset)
                step = max(1, int(np.ceil(max(shape) / max_axis)))
                print(f"step {step} for {shape} (max side {max_axis})")
            if k == "hdf5":
                info = prepare_h5(src, dataset, step, npy, workers)
            else:
                info = prepare_tiff(src, npy, step, workers)
            info.setdefault("factor", step)
        info_path.write_text(json.dumps(info))
    factor = info.get("factor", step)
    side_views(npy, factor, out / "side_views_grid.png")
    print(f"side views -> {out / 'side_views_grid.png'}")

    cfg_path = out / config_name
    if cfg_path.exists():
        print(f"{cfg_path} exists, left unchanged")
        return cfg_path
    fac = factor
    if isinstance(fac, list) and len(set(fac)) == 1:
        fac = fac[0]
    vs = info.get("voxel_size")
    cfg = {
        "preset": preset or "generic",
        "source": str(src),
        "hdf5_dataset": info.get("dataset", dataset),
        "volume": str(npy.resolve()),
        "factor": fac,
        "pixel_size_um": pixel_size or info.get("pixel_size") or 1.0,
        "voxel_size_um": vs if vs and len(set(vs)) > 1 else None,
        "outputs": None,
        **({"preset": "labels", "preprocess": "labels", "labels": info["labels"]} if info.get("labels") else {}),
        "title": "", "subtitle": "", "credit": "",
        "z_range": None, "cylinder": None, "regions": [],
        "look": None, "lighting": None, "size": None,
        "_help": {
            "preset": "name(s) of presets: " + ", ".join(rc.available_presets()) +
                      "; a list combines them, e.g. [\"generic\", \"presentation-4k\"]",
            "null": "null = use the preset's value; any DEFAULTS key may be set here",
            "outputs": "[\"stills\", \"turntable\", \"tour\", \"surface\"]",
            "z_range": "[Z0, Z1] full-res slices to keep; null = all",
            "cylinder": "[CY, CX, R] or [CY_TOP, CX_TOP, CY_BOTTOM, CX_BOTTOM, R] full-res px: keep only "
                        "inside the container; null = no cut (tomo preprocessing fits the wall itself)",
            "regions": "[{\"name\": \"Cerebellum\", \"z\": 3800, \"y\": 6144, \"x\": 4200}, ...] "
                       "full-res centres of the tour close-ups",
            "look": "bone, organic, tissue, ivory, gold, copper, jade, ice, ember, coolwarm, xray, grays, "
                    "viridis, inferno, magma, plasma, cividis, turbo, or any matplotlib colormap",
            "lighting": "studio, soft, dramatic, top, flat",
            "read": "positions can be read off side_views_grid.png",
        },
    }
    cfg_path.write_text(json.dumps(cfg, indent=2))
    print(f"settings -> {cfg_path}\nEdit it, then:  zview-render run {cfg_path}")
    return cfg_path


def _source_shape(src, k, dataset):
    from zview.render.sources import open_fullres
    return tuple(open_fullres(src, dataset).shape[-3:])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", type=Path, help="Zarr/N5 store, HDF5 file or folder of TIFF slices")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--preset", default=None, help="preset written into the config (default generic)")
    p.add_argument("--max-axis", type=int, default=None, help="largest side of the cached volume (default 768)")
    p.add_argument("--step", type=int, default=None, help="fixed downsampling step per axis (old behaviour)")
    p.add_argument("--dataset", default=None, help="HDF5 dataset path (default: auto)")
    p.add_argument("--pixel-size", type=float, default=None, help="voxel size in um (default: from metadata, else 1)")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--force", action="store_true", help="re-extract even if the cached volume exists")
    p.add_argument("--config-name", default="render.json")
    a = p.parse_args(argv)
    prepare(a.source, a.out, a.preset, a.max_axis, a.step, a.dataset, a.pixel_size, a.workers, a.force,
            a.config_name)


if __name__ == "__main__":
    main()
