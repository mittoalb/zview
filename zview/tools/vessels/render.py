"""3D video of a vessel mask: red vessels inside a translucent brain (VTK).

    zview-vessels render VESSELS.zarr --image INPUT --out DIR [--title ...] [--credit ...]
    zview-vessels render VESSELS.zarr --config render.json --out DIR      # full-res masks

VESSELS.zarr comes from zview-vessels segment or fullres. The level rendered is the
finest whose size is <= --max-voxels (default 4e9; a 15 um whole brain fits at level 0).
The translucent brain comes from
  --image   the pre-masked volume the vessels were segmented from (voxels > 0), same grid
  --config  render.json from zview-render prepare (brain mask of the render pipeline; use this for
            zview-vessels fullres masks of an unmasked scan)
Title / subtitle / credit / size can be given directly or are taken from --config.

Outputs in DIR: turntable_vessels.mp4, stills/. Needs a display (DISPLAY set) and VTK.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
def log(*a):
    print(*a, flush=True)


def mesh_from_mask(vtk, ns, M, spacing, smooth_sigma, decimate, max_tris=2_500_000):
    """Smooth surface of a binary mask (uint8, cropped to its bounding box)."""
    from scipy import ndimage as ndi
    M = np.asarray(M)
    nzs = [np.flatnonzero(M.any(axis=tuple(a for a in range(3) if a != k))) for k in range(3)]
    if any(len(n) == 0 for n in nzs):
        return vtk.vtkPolyData()
    sl = tuple(slice(max(0, n[0] - 3), n[-1] + 4) for n in nzs)
    F = ndi.gaussian_filter((M[sl] > 0).astype(np.uint8) * 255, smooth_sigma, output=np.uint8)
    img = vtk.vtkImageData(); img.SetDimensions(F.shape[2], F.shape[1], F.shape[0])
    img.SetSpacing(spacing, spacing, spacing)
    img.SetOrigin(sl[2].start * spacing, sl[1].start * spacing, sl[0].start * spacing)
    img.GetPointData().SetScalars(ns.numpy_to_vtk(F.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR))
    del F
    fe = vtk.vtkFlyingEdges3D(); fe.SetInputData(img); fe.SetValue(0, 127.5); fe.ComputeNormalsOff(); fe.Update()
    poly = fe.GetOutput()
    del img
    if decimate and poly.GetNumberOfPolys() > max_tris:
        dec = vtk.vtkQuadricDecimation(); dec.SetInputData(poly)
        dec.SetTargetReduction(1 - max_tris / poly.GetNumberOfPolys()); dec.Update(); poly = dec.GetOutput()
    sm = vtk.vtkWindowedSincPolyDataFilter(); sm.SetInputData(poly); sm.SetNumberOfIterations(15)
    sm.SetPassBand(0.1); sm.NormalizeCoordinatesOn(); sm.Update()
    nm = vtk.vtkPolyDataNormals(); nm.SetInputData(sm.GetOutput()); nm.SplittingOff(); nm.Update()
    return nm.GetOutput()


def render(out, V, brain2, size, frames, elev, title, sub, credit):
    """V: vessel mask (z,y,x); brain2: brain mask on a 2x coarser grid (or None)."""
    import vtk
    from vtk.util import numpy_support as ns
    import imageio.v2 as iio2
    import imageio.v3 as iio
    from zview.render.tour import credit_layer, apply_credit, overlay
    t0 = time.time()
    W, H = (int(v) for v in size.lower().split("x"))
    vpoly = mesh_from_mask(vtk, ns, V, 1.0, 0.7, True)
    log(f"vessels: {vpoly.GetNumberOfPolys():,} triangles ({time.time() - t0:.0f}s)")
    ren = vtk.vtkRenderer()
    ren.SetBackground(0.0, 0.0, 0.0); ren.SetBackground2(0.12, 0.12, 0.15); ren.GradientBackgroundOn()
    vm = vtk.vtkPolyDataMapper(); vm.SetInputData(vpoly); vm.ScalarVisibilityOff()
    va = vtk.vtkActor(); va.SetMapper(vm)
    pr = va.GetProperty(); pr.SetInterpolationToPBR(); pr.SetColor(0.72, 0.12, 0.10)
    pr.SetRoughness(0.45); pr.SetMetallic(0.0); pr.SetAmbient(0.25)
    ren.AddActor(va)
    if brain2 is not None:
        bpoly = mesh_from_mask(vtk, ns, brain2, 2.0, 1.5, True, max_tris=1_000_000)
        log(f"brain: {bpoly.GetNumberOfPolys():,} triangles ({time.time() - t0:.0f}s)")
        bm = vtk.vtkPolyDataMapper(); bm.SetInputData(bpoly); bm.ScalarVisibilityOff()
        ba = vtk.vtkActor(); ba.SetMapper(bm)
        q = ba.GetProperty(); q.SetColor(0.92, 0.84, 0.78); q.SetOpacity(0.10); q.SetAmbient(0.3)
        q.SetDiffuse(0.6); q.SetSpecular(0.0); q.BackfaceCullingOn()
        ren.AddActor(ba)
    ren.AutomaticLightCreationOff()
    for pos, inten in (((1.0, 1.3, 0.9), 1.0), ((-1.2, 0.0, 0.8), 0.3), ((0.0, 0.9, -1.0), 0.45)):
        lt = vtk.vtkLight(); lt.SetLightTypeToCameraLight(); lt.SetPositional(False)
        lt.SetPosition(*pos); lt.SetFocalPoint(0, 0, 0); lt.SetIntensity(inten); ren.AddLight(lt)
    rw = vtk.vtkRenderWindow(); rw.SetOffScreenRendering(1); rw.AddRenderer(ren); rw.SetSize(W, H)
    rw.SetAlphaBitPlanes(1); rw.SetMultiSamples(0)
    ren.SetUseDepthPeeling(1); ren.SetMaximumNumberOfPeels(8); ren.SetOcclusionRatio(0.05)
    ren.SetUseFXAA(True)

    nz, ny, nx = V.shape
    c = np.array([(nx - 1) / 2, (ny - 1) / 2, (nz - 1) / 2])
    dist = 0.5 * np.sqrt(nx ** 2 + ny ** 2 + nz ** 2) / np.tan(np.radians(15)) * 0.8
    cam = ren.GetActiveCamera()

    def look(az, el):
        a, e = np.radians(az), np.radians(el)
        pos = c + dist * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), -np.sin(e)])
        cam.SetFocalPoint(*c); cam.SetPosition(*pos); cam.SetViewUp(0, 0, -1); cam.SetViewAngle(30)
        ren.ResetCameraClippingRange()

    def grab():
        rw.Render()
        w2i = vtk.vtkWindowToImageFilter(); w2i.SetInput(rw); w2i.ReadFrontBufferOff(); w2i.Update()
        im = ns.vtk_to_numpy(w2i.GetOutput().GetPointData().GetScalars()).reshape(H, W, -1)[::-1, :, :3]
        return np.clip(im.astype(np.float32) * 1.15, 0, 255).astype(np.uint8)

    cred = credit_layer(credit, W, H) if credit else None
    st = out / "stills"; st.mkdir(parents=True, exist_ok=True)
    for el in (elev, 45, 0):
        for az in range(0, 360, 45):
            look(az, el); iio.imwrite(st / f"az{az:03d}_el{int(el):+03d}.png", apply_credit(grab(), cred))
    log(f"stills -> {st}/")
    with iio2.get_writer(str(out / "turntable_vessels.mp4"), fps=30, codec="libx264", quality=9,
                         pixelformat="yuv420p", macro_block_size=8) as wr:
        for i in range(frames):
            look(360.0 * i / frames, elev)
            im = overlay(grab(), title, sub, alpha=min(1, i / 20, (frames - 1 - i) / 20))
            wr.append_data(apply_credit(im, cred))
            if (i + 1) % 60 == 0:
                log(f"  turntable {i + 1}/{frames}")
    log(f"-> {out / 'turntable_vessels.mp4'}  ({time.time() - t0:.0f}s)")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mask", type=Path, help="vessel-mask Zarr (zview-vessels segment / fullres)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--image", type=Path, default=None, help="pre-masked volume (TIFF folder / Zarr / HDF5) for the brain")
    p.add_argument("--image-level", default="0", help="Zarr image: level matching mask level 0 (default 0)")
    p.add_argument("--config", type=Path, default=None, help="dataset.json (brain mask + title/credit defaults)")
    p.add_argument("--level", default=None, help="mask level to render (default: finest <= --max-voxels)")
    p.add_argument("--max-voxels", type=float, default=4e9)
    p.add_argument("--title", default=None)
    p.add_argument("--subtitle", default=None)
    p.add_argument("--credit", default=None)
    p.add_argument("--size", default=None, help="video size WxH (default 1920x1080)")
    p.add_argument("--frames", type=int, default=360, help="turntable frames (default 360 = 12 s)")
    p.add_argument("--elev", type=float, default=20)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    args = p.parse_args(argv)
    import zarr
    cfg = json.loads(args.config.read_text()) if args.config else {}
    root = zarr.open_group(str(args.mask), mode="r")
    keys = sorted(root.array_keys(), key=int)
    lvl = args.level or next((k for k in keys if np.prod(root[k].shape) <= args.max_voxels), keys[-1])
    f = 2 ** int(lvl)
    px = None
    try:
        px = root.attrs["multiscales"][0]["datasets"][int(lvl)]["coordinateTransformations"][0]["scale"][0]
    except Exception:
        pass

    if args.image is not None:
        from zview.tools.vessels.segment import open_volume
        V = np.asarray(root[lvl])
        img = open_volume(args.image, args.image_level)
        nz, ny, nx = V.shape
        b = np.zeros((-(-nz // 2), -(-ny // 2), -(-nx // 2)), bool)
        s = 2 * f                                                  # brain at 2x coarser than the mask level
        for k, z in enumerate(range(0, img.shape[0], s)):
            if k < b.shape[0]:
                sl = np.asarray(img[z])[::s, ::s] > 0
                b[k, :sl.shape[0], :sl.shape[1]] = sl[:b.shape[1], :b.shape[2]]
        brain2 = b
    elif args.config is not None:
        from zview.render import scene as rv
        from scipy import ndimage as ndi
        pa = argparse.Namespace(volume=Path(cfg["volume"]), factor=cfg["factor"], down=2, z_range=cfg.get("z_range"),
                                cylinder=cfg.get("cylinder"), lo_pct=60, hi_pct=99.7, keep_largest=True, open_iter=6,
                                mask_level=20, workers=args.workers)
        vol = rv.prepare_volume(pa)
        m = ndi.gaussian_filter(vol["U8"].astype(np.float32), 1.0) > 20
        step = cfg["factor"] * 2                                   # brain mask step vs full res
        if f * 2 != step:
            raise SystemExit(f"with --config the mask level must be {int(np.log2(step // 2))} "
                             f"(matching the cached volume); use --level")
        z0 = vol["z0"] // f
        V = np.asarray(root[lvl][z0:z0 + 2 * m.shape[0], :2 * m.shape[1], :2 * m.shape[2]])
        brain2 = m
    else:
        V, brain2 = np.asarray(root[lvl]), None
    log(f"mask level {lvl}: {V.shape}, {V.mean() * 100:.2f}% vessel voxels")
    title = args.title if args.title is not None else ((cfg.get("title", "") + " – vasculature").strip(" –")
                                                       if cfg.get("title") else "Vasculature")
    sub = args.subtitle if args.subtitle is not None else (f"vessel segmentation, rendered at {px:g} µm" if px else None)
    credit = args.credit if args.credit is not None else cfg.get("credit")
    args.out.mkdir(parents=True, exist_ok=True)
    render(args.out, V, brain2, args.size or cfg.get("size", "1920x1080"), args.frames, args.elev, title, sub, credit)


if __name__ == "__main__":
    main()
