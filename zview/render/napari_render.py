# Moved from zarr-auto-render/render_zarr_3d.py.
"""Render 3D views of a multiscale OME-Zarr with napari.

Modes:
  (default)   open the viewer interactively so you can tune camera / transfer
              function / colormap by hand.
  --turntable N   render N-frame 360-deg rotation and (optionally) encode MP4
  --slice-sweep N render N frames marching the slice cursor through the volume
  --ortho         save orthogonal MIPs (XY, XZ, YZ) as one composite PNG
  --views         save stills from 6 canonical viewpoints (front/back/top/bottom/left/right)
                  plus an isometric
  --batch         run everything headless: ortho + views + turntable + slice-sweep
                  and encode the two animations to MP4.

For an 8 TB dataset you render from a coarser mip level that fits in GPU
memory. The script picks the finest level that fits in --budget-gb by default.

Install (into your env):
    pip install 'napari[all]' zarr dask imageio imageio-ffmpeg ome-zarr matplotlib

Headless use (no display / over ssh):
    QT_QPA_PLATFORM=offscreen python3 render_zarr_3d.py /path/dataset.zarr --batch

    # or with a virtual display:
    xvfb-run -a python3 render_zarr_3d.py /path/dataset.zarr --batch

Run it in the background and come back later:
    nohup bash -c "QT_QPA_PLATFORM=offscreen python3 render_zarr_3d.py \\
        /path/to/dataset.zarr --batch --out ./renders_run1" \\
        > render.log 2>&1 &

Examples:
    # interactive, auto mip pick, auto contrast
    python3 render_zarr_3d.py /path/to/dataset.zarr

    # headless batch: everything to ./renders
    QT_QPA_PLATFORM=offscreen python3 render_zarr_3d.py \\
        /path/to/dataset.zarr --batch --out ./renders

    # just a 240-frame turntable at mip 3
    QT_QPA_PLATFORM=offscreen python3 render_zarr_3d.py \\
        /path/to/dataset.zarr --mip 3 --turntable 240 --mp4
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def load_multiscale(zarr_path: Path):
    """Return list of dask arrays, finest first (OME-Zarr convention: '0'=finest)."""
    import dask.array as da
    import zarr

    root = zarr.open_group(str(zarr_path), mode="r")
    try:
        keys = sorted(root.array_keys(), key=lambda k: int(k))
    except (ValueError, AttributeError):
        keys = sorted(root)
    if not keys:
        raise SystemExit(f"No arrays found under {zarr_path}")
    arrays = [da.from_zarr(root[k]) for k in keys]
    return arrays, keys


def pick_mip(arrays, budget_bytes: int) -> int:
    for i, a in enumerate(arrays):
        size = int(np.prod(a.shape)) * a.dtype.itemsize
        if size <= budget_bytes:
            return i
    return len(arrays) - 1


def compute_contrast(vol, lo_pct=1.0, hi_pct=99.0, max_voxels=10_000_000):
    stride = max(1, int((int(np.prod(vol.shape)) / max_voxels) ** (1 / vol.ndim)))
    slicer = tuple(slice(None, None, stride) for _ in range(vol.ndim))
    sample = np.asarray(vol[slicer])
    return float(np.percentile(sample, lo_pct)), float(np.percentile(sample, hi_pct))


def _screenshot(viewer, path: Path):
    viewer.window._qt_viewer.canvas.native.repaint()
    viewer.screenshot(str(path), canvas_only=True, flash=False)


def render_turntable(viewer, out_dir: Path, n_frames: int, axis: int = 1):
    out_dir.mkdir(parents=True, exist_ok=True)
    base = list(viewer.camera.angles)
    for i in range(n_frames):
        angles = base.copy()
        angles[axis] = (base[axis] + 360.0 * i / n_frames) % 360.0
        viewer.camera.angles = tuple(angles)
        _screenshot(viewer, out_dir / f"frame_{i:04d}.png")
        if (i + 1) % 20 == 0 or i == n_frames - 1:
            print(f"  turntable: {i + 1}/{n_frames}")


def render_slice_sweep(viewer, vol_shape, out_dir: Path, n_frames: int, axis: int = 0):
    """Switch to 2D and march the slice cursor through the volume along `axis`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    orig_ndisplay = viewer.dims.ndisplay
    viewer.dims.ndisplay = 2
    n_slices = vol_shape[axis]
    for i in range(n_frames):
        idx = int(round(i * (n_slices - 1) / max(1, n_frames - 1)))
        viewer.dims.set_point(axis, idx)
        _screenshot(viewer, out_dir / f"slice_{i:04d}.png")
        if (i + 1) % 20 == 0 or i == n_frames - 1:
            print(f"  slice-sweep: {i + 1}/{n_frames} (slice {idx}/{n_slices - 1})")
    viewer.dims.ndisplay = orig_ndisplay


def render_canonical_views(viewer, out_dir: Path):
    """Save stills from 7 canonical viewpoints."""
    out_dir.mkdir(parents=True, exist_ok=True)
    views = {
        "front":    (0,   0,   0),
        "back":     (0, 180,   0),
        "left":     (0,  90,   0),
        "right":    (0, -90,   0),
        "top":      (-90, 0,   0),
        "bottom":   (90,  0,   0),
        "iso":      (-30, 45,  0),
    }
    for name, angles in views.items():
        viewer.camera.angles = angles
        _screenshot(viewer, out_dir / f"view_{name}.png")
        print(f"  view: {name}")


def render_ortho(vol, out_path: Path, colormap: str = "gray_r"):
    # Figure without pyplot: renders straight to file, never touches a GUI backend
    from matplotlib.figure import Figure

    arr = np.asarray(vol)
    mips = [arr.max(axis=i) for i in range(3)]
    fig = Figure(figsize=(15, 5))
    axes = fig.subplots(1, 3)
    titles = ["MIP along Z (XY view)", "MIP along Y (XZ view)", "MIP along X (YZ view)"]
    for ax, img, title in zip(axes, mips, titles):
        ax.imshow(img, cmap=colormap, aspect="equal")
        ax.set_title(title)
        ax.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  ortho MIP composite -> {out_path}")


def frames_to_mp4(frames_dir: Path, pattern: str, out_mp4: Path, fps: int = 30):
    """Try imageio.ffmpeg; fall back to a shell ffmpeg command printout."""
    try:
        import imageio.v3 as iio
        import imageio_ffmpeg  # noqa: F401 — presence check
        frames = sorted(frames_dir.glob(pattern))
        if not frames:
            print(f"  no frames matching {pattern} in {frames_dir}")
            return
        imgs = [iio.imread(f) for f in frames]
        iio.imwrite(out_mp4, imgs, fps=fps, codec="libx264", pixelformat="yuv420p")
        print(f"  encoded {len(imgs)} frames -> {out_mp4}")
    except Exception as e:
        print(f"  (couldn't encode MP4 directly: {e})")
        print(f"  encode manually with:")
        print(f"    ffmpeg -framerate {fps} -i {frames_dir}/{pattern.replace('*', '%04d').replace('frame_', 'frame_').replace('slice_', 'slice_')} "
              f"-c:v libx264 -pix_fmt yuv420p {out_mp4}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("zarr_path", type=Path)
    p.add_argument("--mip", type=int, default=None)
    p.add_argument("--budget-gb", type=float, default=8.0)
    p.add_argument("--rendering", default="attenuated_mip",
                   choices=["mip", "attenuated_mip", "additive", "translucent", "iso", "minip", "average"])
    p.add_argument("--colormap", default="gray_r")
    p.add_argument("--contrast", type=str, default=None,
                   help='"lo,hi" (default: auto 1-99 pct subsample)')
    p.add_argument("--out", type=Path, default=Path("./renders"))
    # modes
    p.add_argument("--turntable", type=int, default=0, metavar="N",
                   help="Render N-frame rotation animation (headless)")
    p.add_argument("--slice-sweep", type=int, default=0, metavar="N",
                   help="Render N-frame slice fly-through along --slice-axis")
    p.add_argument("--slice-axis", type=int, default=0, choices=[0, 1, 2])
    p.add_argument("--rotation-axis", type=int, default=1, choices=[0, 1, 2])
    p.add_argument("--ortho", action="store_true", help="Write orthogonal MIP composite")
    p.add_argument("--views", action="store_true", help="Write 7 canonical viewpoint stills")
    p.add_argument("--batch", action="store_true",
                   help="Run everything headless: ortho + views + turntable (180) + slice-sweep (120), encode MP4s")
    p.add_argument("--mp4", action="store_true", help="Encode frames to MP4 (needs imageio-ffmpeg)")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--gpu", action="store_true",
                   help="With --batch: use napari/OpenGL instead of the CPU renderer. Needs a working "
                        "OpenGL context; Qt 'offscreen' on headless servers gives black frames.")
    args = p.parse_args(argv)

    # Headless batch -> CPU renderer (render_cpu.py): no OpenGL needed, works on any server.
    if args.batch and not args.gpu:
        from zview.render import cpu as render_cpu
        argv = [str(args.zarr_path), "--out", str(args.out),
                "--budget-gb", str(args.budget_gb),
                "--frames", str(args.turntable or 180),
                "--slice-sweep", str(args.slice_sweep if args.slice_sweep else 240),
                "--fps", str(args.fps)]
        if args.mip is not None:
            argv += ["--level", str(args.mip)]
        print("[--batch] using CPU renderer: render_cpu.py " + " ".join(argv), flush=True)
        render_cpu.main(argv)
        return

    # --batch sets reasonable defaults for a complete headless run
    if args.batch:
        args.ortho = True
        args.views = True
        if args.turntable == 0:
            args.turntable = 180
        if args.slice_sweep == 0:
            args.slice_sweep = 120
        args.mp4 = True

    headless = any([args.turntable, args.slice_sweep, args.ortho, args.views])

    print(f"Loading {args.zarr_path}")
    arrays, keys = load_multiscale(args.zarr_path)
    print("Multiscale pyramid:")
    for i, a in enumerate(arrays):
        gb = int(np.prod(a.shape)) * a.dtype.itemsize / 1024**3
        print(f"  [{i}] key='{keys[i]}' shape={a.shape} dtype={a.dtype} size={gb:.2f} GB")

    mip = args.mip if args.mip is not None else pick_mip(arrays, int(args.budget_gb * 1024**3))
    vol = arrays[mip]
    gb = int(np.prod(vol.shape)) * vol.dtype.itemsize / 1024**3
    print(f"\nRendering mip {mip}: shape={vol.shape} ({gb:.2f} GB)")

    print("Materializing into memory...")
    vol_np = np.asarray(vol)

    if args.contrast is None:
        lo, hi = compute_contrast(vol_np)
        print(f"Auto contrast [1-99 pct]: {lo:.3g} - {hi:.3g}")
    else:
        lo, hi = (float(v) for v in args.contrast.split(","))

    # Ortho doesn't need napari; render it first and cheaply
    args.out.mkdir(parents=True, exist_ok=True)
    if args.ortho:
        print("\n[ortho MIPs]")
        render_ortho(vol_np, args.out / "ortho_mips.png", colormap=args.colormap)

    # Everything else needs a napari viewer
    needs_viewer = args.turntable or args.slice_sweep or args.views or not headless
    if not needs_viewer:
        return

    if headless and not os.environ.get("QT_QPA_PLATFORM"):
        print("\nHint: for true headless, run with QT_QPA_PLATFORM=offscreen or under xvfb-run.")

    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)  # napari camera API noise, once per frame
    import napari
    viewer = napari.Viewer(ndisplay=3, show=not headless)
    viewer.add_image(
        vol_np,
        rendering=args.rendering,
        colormap=args.colormap,
        contrast_limits=[lo, hi],
        name=f"mip{mip}",
    )
    viewer.dims.ndisplay = 3
    viewer.reset_view()

    if args.views:
        print("\n[canonical views]")
        render_canonical_views(viewer, args.out / "views")

    if args.turntable > 0:
        print(f"\n[turntable, {args.turntable} frames]")
        viewer.dims.ndisplay = 3
        viewer.reset_view()
        render_turntable(viewer, args.out / "turntable", args.turntable, axis=args.rotation_axis)
        if args.mp4:
            frames_to_mp4(args.out / "turntable", "frame_*.png",
                          args.out / "turntable.mp4", fps=args.fps)

    if args.slice_sweep > 0:
        print(f"\n[slice sweep, {args.slice_sweep} frames along axis {args.slice_axis}]")
        render_slice_sweep(viewer, vol_np.shape, args.out / "slices",
                           args.slice_sweep, axis=args.slice_axis)
        if args.mp4:
            frames_to_mp4(args.out / "slices", "slice_*.png",
                          args.out / "slice_sweep.mp4", fps=args.fps)

    if headless:
        print(f"\nAll outputs in: {args.out}/")
        return

    print("\nInteractive viewer ready. Close the window to exit.")
    napari.run()


if __name__ == "__main__":
    main()
