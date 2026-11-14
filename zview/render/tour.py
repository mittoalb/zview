# Moved from zarr-auto-render/render_tour.py.
"""Guided 3D tour video of a tomography volume (VTK, GPU, volumetric shadows).

Storyboard:
  1. overview       - the sample turns half way round
  2. sections       - a cutting plane sweeps through, exposing the interior
  3. per region     - horizontal cut at the region, the camera tilts down until it
                      looks straight at the cut face, which cross-fades into the
                      REAL full-resolution slices from the HDF5 (native grey
                      values); scroll through neighbouring slices with a slow
                      zoom and a scale bar, then back out
  4. outro          - the rest of the turn

Needs a display (DISPLAY set) and VTK. The overview uses the cached .npy from
prepare3d.py; the close-ups read full-resolution slices from the HDF5 or Zarr.
Usually run through pipeline3d.py with a dataset.json.

    python render_tour.py volume_step8.npy --factor 8 --h5 rec.h5 --out tour \\
        --z-range 1700 13800 --cylinder 6186 5931 6270 6267 5270 --pixel-size 1.88 \\
        --region cerebellum 3800 6144 4200 --region hippocampus 6600 2500 6144
"""

import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from zview.render import vtk_render as rv                                            # noqa: E402
from zview.render.vtk_render import log                                         # noqa: E402

_H5 = {}


# ---------------------------------------------------------------------------
# real full-resolution slices from the HDF5
# ---------------------------------------------------------------------------
def _is_zarr(path):
    pth = Path(path)
    return pth.suffix == ".zarr" or (pth.is_dir() and (pth / "zarr.json").exists())


def _open_full(path, dset):
    """Full-resolution (z, y, x) array: Zarr v2/v3/N5, HDF5 or TIFF folder."""
    from zview.render.sources import open_fullres
    return open_fullres(path, dset)


def _read_slice(args):
    z, y0, y1, x0, x1 = args
    if "ds" not in _H5:
        _H5["ds"] = _open_full(_H5["path"], _H5["dset"])
    return np.asarray(_H5["ds"][z, y0:y1, x0:x1], np.float32)


def read_slices(h5, dset, zs, yc, xc, hh, ww, workers):
    """Native slices zs, crop hh x ww centred on (yc, xc), straight from the HDF5 or Zarr level 0."""
    _, ny, nx = _open_full(h5, dset).shape
    y0 = int(np.clip(yc - hh // 2, 0, ny - hh)); x0 = int(np.clip(xc - ww // 2, 0, nx - ww))
    _H5.update(path=str(h5), dset=dset)
    with Pool(min(workers, len(zs))) as p:
        sl = p.map(_read_slice, [(int(z), y0, y0 + hh, x0, x0 + ww) for z in zs], chunksize=2)
    _H5.clear()
    return np.stack(sl), (y0, x0)


def real_frame(sl, lo, hi, W, H, zoom):
    """A native slice as a grey RGB frame (no smoothing / colouring), centre zoom."""
    from PIL import Image
    hh, ww = sl.shape
    ch, cw = int(hh / zoom), int(ww / zoom)
    oy, ox = (hh - ch) // 2, (ww - cw) // 2
    g = np.clip((sl[oy:oy + ch, ox:ox + cw] - lo) / max(hi - lo, 1e-12), 0, 1)
    pil = Image.fromarray((g * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS).convert("RGB")
    return np.asarray(pil), H / ch                                       # px per full-res voxel


# ---------------------------------------------------------------------------
# 2-D overlays
# ---------------------------------------------------------------------------
def _font(size):
    from PIL import ImageFont
    import matplotlib
    try:
        return ImageFont.truetype(os.path.join(matplotlib.get_data_path(), "fonts/ttf/DejaVuSans.ttf"), size)
    except Exception:
        return ImageFont.load_default()


def overlay(im, title=None, sub=None, bar_px=None, bar_label=None, alpha=1.0):
    from PIL import Image, ImageDraw
    if alpha <= 0 or (title is None and bar_px is None):
        return im
    H, W = im.shape[:2]
    pil = Image.fromarray(im); d = ImageDraw.Draw(pil, "RGBA")
    a = int(255 * alpha)
    def text(xy, t, f, fill):                                          # soft dark shadow for legibility
        for dx, dy in ((2, 2), (1, 1), (-1, 1), (1, -1)):
            d.text((xy[0] + dx, xy[1] + dy), t, font=f, fill=(0, 0, 0, int(a * 0.6)))
        d.text(xy, t, font=f, fill=fill)
    if title:
        text((int(W * 0.04), int(H * 0.05)), title, _font(int(H * 0.045)), (235, 228, 220, a))
    if sub:
        text((int(W * 0.04), int(H * 0.05 + H * 0.06)), sub, _font(int(H * 0.028)), (215, 208, 200, a))
    if bar_px:
        x1 = int(W * 0.95); x0 = int(x1 - bar_px); y = int(H * 0.92)
        d.rectangle([x0 + 2, y + 2, x1 + 2, y + 2 + max(3, H // 180)], fill=(0, 0, 0, int(a * 0.6)))
        d.rectangle([x0, y, x1, y + max(3, H // 180)], fill=(235, 228, 220, a))
        f = _font(int(H * 0.028))
        tw = d.textlength(bar_label, font=f)
        text(((x0 + x1) / 2 - tw / 2, y - int(H * 0.045)), bar_label, f, (235, 228, 220, a))
    return np.asarray(pil)


def credit_layer(text, W, H):
    """Pre-rendered RGBA credit line for the bottom-right corner (below the scale bar)."""
    from PIL import Image, ImageDraw
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    f = _font(int(H * 0.024))
    tw = d.textlength(text, font=f)
    x, y = W - int(W * 0.025) - tw, H - int(H * 0.022) - int(H * 0.024)
    for dx, dy in ((2, 2), (1, 1)):
        d.text((x + dx, y + dy), text, font=f, fill=(0, 0, 0, 150))
    d.text((x, y), text, font=f, fill=(225, 220, 212, 230))
    return np.asarray(lay).astype(np.float32) / 255


def apply_credit(im, lay):
    if lay is None:
        return im
    a = lay[..., 3:4]
    return (im.astype(np.float32) * (1 - a) + lay[..., :3] * 255 * a).astype(np.uint8)


def nice_bar(um_per_px_screen, target_px):
    """A round scale-bar length (um) close to target_px on screen."""
    want = um_per_px_screen * target_px
    for v in (10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 5000):
        if v >= want * 0.6:
            return v, v / um_per_px_screen
    return want, target_px


def ease(t):
    return t * t * (3 - 2 * t)


# ---------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    rv.add_common_args(p)
    p.add_argument("--h5", "--fullres", dest="h5", type=Path, required=True,
                   help="full-resolution source for the close-ups: the original HDF5 or the .zarr (level 0)")
    p.add_argument("--title", default="", help="title shown on the overview (default none)")
    p.add_argument("--subtitle", default="", help="subtitle under the title")
    p.add_argument("--dataset", default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--region", nargs=4, action="append", metavar=("NAME", "Z", "Y", "X"), default=None,
                   help="close-up region: name and centre in full-res pixels (repeatable)")
    p.add_argument("--region-size", type=int, default=800,
                   help="height of the full-res close-up in voxels (default 800 = 1.5 mm at 1.88 um)")
    p.add_argument("--scroll", type=int, default=150, help="native slices scrolled through per region (default 150)")
    p.add_argument("--zoom", type=float, default=1.8, help="zoom reached at the end of each close-up (default 1.8)")
    p.add_argument("--pixel-size", type=float, default=1.0, help="voxel size in um (for scale bars)")
    p.add_argument("--style", choices=["volume", "surface"], default="volume",
                   help="volume: ray-cast volume with shadows; surface: lit surface with ambient occlusion")
    p.add_argument("--credit", default=None, help="credit line at the bottom right of every frame")
    p.add_argument("--speed", type=float, default=1.0, help="<1 = shorter video (fewer frames per segment)")
    p.set_defaults(size="1920x1080")
    args = p.parse_args(argv)
    args.region = args.region or []                                      # no regions -> no close-ups
    import imageio.v2 as iio2
    import imageio.v3 as iio
    t0 = time.time()
    W, H = (int(v) for v in args.size.lower().split("x"))
    n = lambda k: max(2, int(round(k * args.speed)))                        # frames per segment
    out = args.out; out.mkdir(parents=True, exist_ok=True)
    keyd = out / "keyframes"; keyd.mkdir(exist_ok=True)

    vol = rv.prepare_volume(args)
    F, Z0 = vol["F"], vol["z0"]
    if args.style == "surface":
        from zview.render import surface as rs
        sc = rs.SurfaceScene(vol["U8"], args, W, H)
    else:
        sc = rv.Scene(vol["U8"], args, W, H)
    del vol
    dist0 = sc.default_dist()
    c0 = sc.center
    EL = 22.0

    wr = iio2.get_writer(str(out / "tour.mp4"), fps=args.fps, codec="libx264", quality=9,
                         pixelformat="yuv420p", macro_block_size=8)
    count = [0]
    last = [None]

    cred = credit_layer(args.credit, W, H) if args.credit else None

    def emit(im, key=None):
        last[0] = im                                                     # keep the clean frame for fades
        im = apply_credit(im, cred)
        wr.append_data(im); count[0] += 1
        if key:
            iio.imwrite(keyd / f"{count[0]:05d}_{key}.png", im)
        if count[0] % 50 == 0:
            log(f"  frame {count[0]}  ({time.time() - t0:.0f}s)")

    def fade(a, b, k):
        for i in range(k):
            t = ease((i + 1) / (k + 1))
            emit((a.astype(np.float32) * (1 - t) + b.astype(np.float32) * t).astype(np.uint8))

    # ---- 1. overview: half turn
    log("[1] overview")
    k = n(150)
    for i in range(k):
        sc.look(-60 + 180 * ease(i / (k - 1)), EL, dist0)
        im = sc.grab()
        emit(overlay(im, args.title or None, args.subtitle or None,
                     alpha=min(1, i / 20, (k - 1 - i) / 20)), key="overview" if i == k // 2 else None)
    AZ = 120.0                                                               # end of overview

    # ---- 2. sections: plane sweeps through, facing the camera
    log("[2] sections")
    a = np.radians(AZ)
    toward_cam = np.array([np.cos(a), np.sin(a), 0.0])                      # horizontal, towards camera
    ext = abs(toward_cam[0]) * sc.shape[2] / 2 + abs(toward_cam[1]) * sc.shape[1] / 2
    k = n(210)
    for i in range(k):
        s = 0.85 - 1.5 * ease(i / (k - 1))                                  # from front (+) to back (-)
        sc.clip(c0 + toward_cam * ext * s, -toward_cam)                     # keep the far side
        sc.section(c0 + toward_cam * ext * s, -toward_cam)
        sc.look(AZ, EL, dist0)
        emit(overlay(sc.grab(), "Virtual sections", "a cutting plane sweeps through the tissue",
                     alpha=min(1, i / 20, (k - 1 - i) / 20)), key="section" if i == k // 2 else None)
    for i in range(n(40)):                                                  # plane back to the middle
        s = -0.65 + 0.65 * ease(i / (n(40) - 1))
        sc.clip(c0 + toward_cam * ext * s, -toward_cam); sc.section(c0 + toward_cam * ext * s, -toward_cam)
        sc.look(AZ, EL, dist0); emit(sc.grab())

    # ---- 3. regions: horizontal cut at the region, look straight down, cross-fade to real slices
    FOV = 30.0
    for name, zc, yc, xc in args.region:
        zc, yc, xc = int(float(zc)), int(float(yc)), int(float(xc))
        log(f"[3] region {name} at z={zc} y={yc} x={xc}")
        target = np.array([xc / F, yc / F, (zc - Z0) / F])                  # in overview voxels
        hn = np.array([0.0, 0.0, 1.0])                                      # keep tissue below the cut
        ho = np.array([c0[0], c0[1], target[2]])

        # A. switch to a horizontal cut at the region's height (seen from above)
        sc.clip(ho, hn); sc.section(ho, hn)
        p_start = sc.look(AZ, 50, dist0)
        fade(last[0], overlay(sc.grab(), name, alpha=0.0), n(25))

        # B. fly in until the region fills the frame, camera straight down
        hh = args.region_size; ww = int(round(hh * W / H / 2)) * 2         # full-res crop matching the frame
        dist1 = (hh / F / 2) / np.tan(np.radians(FOV / 2))
        p_end = target + np.array([0.0, 0.0, -dist1])
        k = n(110)
        for i in range(k):
            t = ease(i / (k - 1))
            up = (1 - t) * np.array([0.0, 0.0, -1.0]) + t * np.array([0.0, -1.0, 0.0])
            sc.pose(p_start * (1 - t) + p_end * t, c0 * (1 - t) + target * t, up / np.linalg.norm(up), FOV)
            emit(overlay(sc.grab(), name, alpha=t))
        top_down = last[0]

        # C. real full-resolution slices from the HDF5 (native grey values)
        k = n(150)
        zs = zc + np.round(np.linspace(0, args.scroll, k)).astype(int)
        tb = time.time()
        stack, (y0, x0) = read_slices(args.h5, args.dataset, np.unique(zs), yc, xc, hh, ww, args.workers)
        zmap = {z: i for i, z in enumerate(np.unique(zs))}
        lo, hi = np.percentile(stack[len(stack) // 2], [0.5, 99.7])
        log(f"  {len(stack)} native slices {hh}x{ww} from z{zs[0]} y{y0} x{x0} ({time.time() - tb:.0f}s)")
        um = args.pixel_size
        frames = []
        for i, z in enumerate(zs):
            im, ppv = real_frame(stack[zmap[z]], lo, hi, W, H, zoom=1.0 + (args.zoom - 1) * ease(i / (k - 1)))
            bar_um, bar_px = nice_bar(um / ppv, W * 0.15)
            frames.append(overlay(im, name, f"native slice z={z}, {um:g} \u00b5m pixels", bar_px,
                                  f"{bar_um:g} \u00b5m"))
        del stack
        fade(top_down, frames[0], n(30))                                    # blurry 3D cut -> real data
        for _ in range(n(20)):
            emit(frames[0])                                                 # hold
        for i, im in enumerate(frames):
            emit(im, key=f"{name}_real" if i in (0, len(frames) - 1) else None)
        for _ in range(n(20)):
            emit(frames[-1])
        fade(frames[-1], top_down, n(30))
        del frames

        # D. fly back out, then close the cut
        k = n(90)
        for i in range(k):
            t = ease(i / (k - 1))
            up = (1 - t) * np.array([0.0, -1.0, 0.0]) + t * np.array([0.0, 0.0, -1.0])
            sc.pose(p_end * (1 - t) + p_start * t, target * (1 - t) + c0 * t, up / np.linalg.norm(up), FOV)
            emit(sc.grab())
        sc.clip(); sc.section(); sc.look(AZ, 50, dist0)
        fade(last[0], sc.grab(), n(20))
        k = n(30)
        for i in range(k):
            sc.look(AZ, 50 + (EL - 50) * ease(i / (k - 1)), dist0); emit(sc.grab())

    # ---- 4. outro: rest of the turn
    log("[4] outro")
    sc.clip(); sc.section()
    k = n(150)
    for i in range(k):
        sc.look(AZ + 180 * ease(i / (k - 1)), EL, dist0)
        emit(overlay(sc.grab(), args.title or None, alpha=min(1, (k - 1 - i) / 25) if i > k - 30 else 0))
    wr.close(); sc.close()
    log(f"-> {out / 'tour.mp4'}  ({count[0]} frames, {count[0] / args.fps:.0f} s)  keyframes in {keyd}/  "
        f"({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
