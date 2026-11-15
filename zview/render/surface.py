"""Surface rendering with ambient occlusion + shadows ("Dragonfly-style" look), from a dataset.json.

    zview-render run DIR/render.json surface [--frames 360] [--size 1920x1080]

  * brain surface = smoothed isosurface of the tissue mask (container cut, medium
    removed, debris dropped - same preprocessing as the volume renders)
  * surface colour = tissue intensity sampled just beneath the surface, organic palette
  * lighting: key light with shadow map, fill + rim lights, screen-space ambient
    occlusion (SSAO) for contact shadows in folds, matte physically-based material

Outputs in DIR/surface/: stills/, turntable_surface.mp4 (title / credit from dataset.json)
Needs a display (DISPLAY set) and VTK.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
from zview.render import scene as rv                                    # noqa: E402
from zview.render.scene import log                                 # noqa: E402
from zview.render import looks as lk                                     # noqa: E402


def brain_surface(vtk, ns, U8, mask_level, smooth_sigma=1.2, max_tris=3_000_000, spacing=(1, 1, 1)):
    from scipy import ndimage as ndi
    m = np.pad((U8 > mask_level).astype(np.float32), 3)                     # closed surface: caps at cuts
    m = ndi.gaussian_filter(m, smooth_sigma)
    img = vtk.vtkImageData(); img.SetDimensions(m.shape[2], m.shape[1], m.shape[0]); img.SetSpacing(*spacing)
    img.SetOrigin(*(-3 * v for v in spacing))
    img.GetPointData().SetScalars(ns.numpy_to_vtk(m.ravel(), deep=True, array_type=vtk.VTK_FLOAT))
    del m
    fe = vtk.vtkFlyingEdges3D(); fe.SetInputData(img); fe.SetValue(0, 0.5); fe.ComputeNormalsOff(); fe.Update()
    poly = fe.GetOutput()
    if poly.GetNumberOfPolys() > max_tris:
        dec = vtk.vtkQuadricDecimation(); dec.SetInputData(poly)
        dec.SetTargetReduction(1 - max_tris / poly.GetNumberOfPolys()); dec.Update(); poly = dec.GetOutput()
    sm = vtk.vtkWindowedSincPolyDataFilter(); sm.SetInputData(poly); sm.SetNumberOfIterations(20)
    sm.SetPassBand(0.08); sm.NormalizeCoordinatesOn(); sm.FeatureEdgeSmoothingOff(); sm.BoundarySmoothingOn()
    sm.Update()
    nm = vtk.vtkPolyDataNormals(); nm.SetInputData(sm.GetOutput()); nm.SplittingOff()
    nm.ConsistencyOn(); nm.AutoOrientNormalsOn(); nm.Update()
    return nm.GetOutput()


def color_by_tissue(vtk, ns, poly, U8, depth=2.0, look=None, spacing=(1, 1, 1)):
    """Vertex colours = tissue intensity a little beneath the surface, mapped
    through the look's palette (default: the original warm tissue tone)."""
    from scipy import ndimage as ndi
    P = ns.vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float32)          # x, y, z
    N = ns.vtk_to_numpy(poly.GetPointData().GetNormals()).astype(np.float32)
    inner = (P - depth * N * np.asarray(spacing, np.float32)) / np.asarray(spacing, np.float32)
    S = ndi.gaussian_filter(U8.astype(np.float32), 1.0)
    v = ndi.map_coordinates(S, [inner[:, 2], inner[:, 1], inner[:, 0]], order=1, mode="nearest")
    lo, hi = np.percentile(v, [2, 99.5])
    t = np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1) * 255
    if look is None or look is lk.LOOKS["organic"]:
        base = np.array([0.90, 0.70, 0.62])                                    # warm, light tissue tone
        shade = 0.72 + 0.38 * (t[:, None] / 255.0)                            # intensity -> subtle variation
        rgb = base[None, :] * shade
    else:
        rgb = lk.color_lut(look)[np.clip(t, 0, 255).astype(int)]
    arr = ns.numpy_to_vtk((np.clip(rgb, 0, 1) * 255).astype(np.uint8), deep=True,
                          array_type=vtk.VTK_UNSIGNED_CHAR)
    arr.SetName("tissue_rgb")
    poly.GetPointData().SetScalars(arr)
    return poly


def build_scene(vtk, poly, W, H, args, look=None, light=None, supersample=1):
    mapper = vtk.vtkPolyDataMapper(); mapper.SetInputData(poly)
    mapper.ScalarVisibilityOn(); mapper.SetColorModeToDirectScalars()
    actor = vtk.vtkActor(); actor.SetMapper(mapper)
    pr = actor.GetProperty()
    look = look or lk.LOOKS["organic"]
    light = light or lk.LIGHTING["studio"]
    if args.pbr:
        pr.SetInterpolationToPBR(); pr.SetRoughness(args.roughness)
        pr.SetMetallic(look["material"].get("metallic", 0.0))
    else:
        pr.SetInterpolationToPhong(); pr.SetSpecular(0.08); pr.SetSpecularPower(10)
    pr.SetAmbient(0.30); pr.SetDiffuse(0.85)

    ren = vtk.vtkRenderer(); ren.AddActor(actor)
    rv.apply_background(ren, look, getattr(args, "background", "look"))
    if light is lk.LIGHTING["studio"]:
        # The original surface rig (slightly brighter than the volume one).
        light = dict(light, lights=[((1.0, 1.4, 1.2), (1.0, 0.96, 0.9), 1.0),
                                    ((-1.3, 0.2, 0.9), (0.85, 0.9, 1.0), 0.30),
                                    ((0.0, 1.0, -1.2), (1.0, 1.0, 1.0), 0.45)])
    rv.add_lights(vtk, ren, light)

    # render passes: (shadow map) -> camera -> SSAO -> FXAA
    b = poly.GetBounds()
    size = max(b[1] - b[0], b[3] - b[2], b[5] - b[4])
    scene = vtk.vtkRenderStepsPass()
    delegate = scene
    if args.shadows:
        shadows = vtk.vtkShadowMapPass()
        seq = vtk.vtkSequencePass(); coll = vtk.vtkRenderPassCollection()
        coll.AddItem(shadows.GetShadowMapBakerPass()); coll.AddItem(shadows); seq.SetPasses(coll)
        cam_pass = vtk.vtkCameraPass(); cam_pass.SetDelegatePass(seq)
        delegate = cam_pass
    if args.ao > 0:
        ssao = vtk.vtkSSAOPass()
        ssao.SetRadius(args.ao * size); ssao.SetBias(0.002 * size); ssao.SetKernelSize(128); ssao.BlurOn()
        ssao.SetDelegatePass(delegate)
        delegate = ssao
    ren.SetPass(delegate)
    ren.SetUseFXAA(True)

    rw = rv.offscreen_render_window(); rw.AddRenderer(ren)
    rw.SetSize(W * supersample, H * supersample)
    rw.SetMultiSamples(0)
    return ren, rw, actor


class SurfaceScene:
    """Same interface as scene.Scene (look / pose / clip / section / grab / ...), but the
    brain is a lit surface with ambient occlusion; cuts are filled with the tissue section."""

    def __init__(self, U8, args, W, H, ao=0.03, roughness=0.65, pbr=True):
        import vtk
        from vtk.util import numpy_support as ns
        self.vtk, self.ns, self.W, self.H, self.args = vtk, ns, W, H, args
        self.look_style, self.light_style = rv.resolve_style(args)
        self.ss = max(1, int(args.supersample))
        nz, ny, nx = U8.shape
        self.shape = (nz, ny, nx)
        poly = brain_surface(vtk, ns, U8, args.mask_level)
        poly = color_by_tissue(vtk, ns, poly, U8, look=self.look_style)
        log(f"surface: {poly.GetNumberOfPolys():,} triangles")
        sargs = argparse.Namespace(pbr=pbr, roughness=roughness, shadows=False, ao=ao,
                                   background=getattr(args, "background", "look"))
        self.ren, self.rw, actor = build_scene(vtk, poly, W, H, sargs, self.look_style, self.light_style, self.ss)
        self.mapper = actor.GetMapper()
        self.cam = self.ren.GetActiveCamera()
        img = vtk.vtkImageData(); img.SetDimensions(nx, ny, nz); img.SetSpacing(1, 1, 1)
        img.GetPointData().SetScalars(ns.numpy_to_vtk(np.ascontiguousarray(U8).ravel(), deep=True,
                                                      array_type=vtk.VTK_UNSIGNED_CHAR))
        self.img = img

    center = rv.Scene.center
    radius = rv.Scene.radius
    look = rv.Scene.look
    pose = rv.Scene.pose
    default_dist = rv.Scene.default_dist
    pixels_per_unit = rv.Scene.pixels_per_unit
    section = rv.Scene.section

    def clip(self, origin=None, normal=None):
        self.mapper.RemoveAllClippingPlanes()
        if origin is None:
            return
        pl = self.vtk.vtkPlane(); pl.SetOrigin(*origin); pl.SetNormal(*normal)
        self.mapper.AddClippingPlane(pl)

    def grab(self):
        vtk, ns = self.vtk, self.ns
        self.rw.Render()
        w2i = vtk.vtkWindowToImageFilter(); w2i.SetInput(self.rw); w2i.ReadFrontBufferOff(); w2i.Update()
        ss = self.ss
        im = ns.vtk_to_numpy(w2i.GetOutput().GetPointData().GetScalars()).reshape(self.H * ss, self.W * ss, -1)
        tm = getattr(self.args, "tone_mapping", "none")
        exp = getattr(self.args, "surface_exposure", None) or (1.0 if tm == "filmic" else 1.15)
        return lk.finish(im[::-1, :, :3], exp, tm, ss)

    def close(self):
        self.rw.Finalize()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", type=Path)
    p.add_argument("--look", default=None, help="colour look (default: config 'look' or 'organic')")
    p.add_argument("--lighting", default=None, help="lighting rig (default: config or studio)")
    p.add_argument("--tone-mapping", default=None, choices=["filmic", "none"])
    p.add_argument("--supersample", type=int, default=None)
    p.add_argument("--background", default=None)
    p.add_argument("--out", type=Path, default=None, help="output folder (default: next to the config)")
    p.add_argument("--size", default=None)
    p.add_argument("--frames", type=int, default=360)
    p.add_argument("--elev", type=float, default=20)
    p.add_argument("--ao", type=float, default=0.03, help="ambient-occlusion radius as fraction of size (0 = off)")
    p.add_argument("--shadows", action="store_true", help="add a shadow map (hard shadows; off by default)")
    p.add_argument("--no-pbr", dest="pbr", action="store_false", help="Phong instead of physically based")
    p.add_argument("--roughness", type=float, default=0.65)
    p.add_argument("--exposure", type=float, default=None,
                   help="brightness (default 1.0 with filmic tone mapping, 1.15 without)")
    p.add_argument("--stills-only", action="store_true")
    p.add_argument("--out-name", default="surface")
    p.add_argument("--workers", type=int, default=32)
    args = p.parse_args(argv)
    cfg = json.loads(args.config.read_text())
    out = args.out or (args.config.parent / args.out_name)
    W, H = (int(v) for v in (args.size or cfg.get("size", "1920x1080")).lower().split("x"))
    t0 = time.time()

    import vtk
    from vtk.util import numpy_support as ns
    import imageio.v2 as iio2
    import imageio.v3 as iio
    from zview.render.tour import credit_layer, apply_credit, overlay

    def c(key, default):
        v = cfg.get(key)
        return default if v is None else v

    pa = argparse.Namespace(volume=Path(cfg["volume"]), factor=cfg["factor"], down=c("down", 2),
                            z_range=cfg.get("z_range"), cylinder=cfg.get("cylinder"),
                            lo_pct=c("lo_pct", 60), hi_pct=c("hi_pct", 99.7),
                            keep_largest=c("keep_largest", True), open_iter=c("open_iter", 6),
                            mask_level=c("mask_level", 20), workers=args.workers,
                            preprocess=c("preprocess", "tomo"), window=cfg.get("window"),
                            voxel_size=cfg.get("voxel_size_um"), smooth=c("smooth", 0.0))
    if pa.preprocess == "labels":
        log("surface style is not available for label maps yet — use stills / turntable "
            "(label maps are rendered with per-class shading there)")
        return
    vol = rv.prepare_volume(pa)
    U8, spacing = vol["U8"], vol.get("spacing", (1, 1, 1))
    sty = argparse.Namespace(look=args.look or cfg.get("look") or "organic",
                             lighting=args.lighting or c("lighting", "studio"),
                             tone_mapping=args.tone_mapping or c("tone_mapping", "filmic"),
                             supersample=args.supersample or c("supersample", 2),
                             background=args.background or c("background", "look"),
                             exposure=args.exposure, curve=None, specular=None, scatter=None)
    look, light = rv.resolve_style(sty)
    if args.exposure is None:
        args.exposure = 1.0 if sty.tone_mapping == "filmic" else 1.15
    args.background = sty.background
    ss = max(1, int(sty.supersample))
    poly = brain_surface(vtk, ns, U8, pa.mask_level, spacing=spacing)
    poly = color_by_tissue(vtk, ns, poly, U8, look=look, spacing=spacing)
    log(f"surface: {poly.GetNumberOfPolys():,} triangles ({time.time() - t0:.0f}s)")
    del U8
    ren, rw, actor = build_scene(vtk, poly, W, H, args, look, light, ss)
    b = poly.GetBounds()
    c = np.array([(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2])
    R = 0.5 * np.sqrt((b[1] - b[0]) ** 2 + (b[3] - b[2]) ** 2 + (b[5] - b[4]) ** 2)
    dist = R / np.tan(np.radians(15)) * 0.85
    cam = ren.GetActiveCamera()

    def look(az, el):
        a, e = np.radians(az), np.radians(el)
        pos = c + dist * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), -np.sin(e)])
        cam.SetFocalPoint(*c); cam.SetPosition(*pos); cam.SetViewUp(0, 0, -1); cam.SetViewAngle(30)
        ren.ResetCameraClippingRange()

    def grab():
        rw.Render()
        w2i = vtk.vtkWindowToImageFilter(); w2i.SetInput(rw); w2i.ReadFrontBufferOff(); w2i.Update()
        im = ns.vtk_to_numpy(w2i.GetOutput().GetPointData().GetScalars()).reshape(H * ss, W * ss, -1)[::-1, :, :3]
        return lk.finish(im, args.exposure, sty.tone_mapping, ss)

    cred = credit_layer(cfg["credit"], W, H) if cfg.get("credit") else None
    st = out / "stills"; st.mkdir(parents=True, exist_ok=True)
    for el in (args.elev, 45, -20, 0):
        for az in range(0, 360, 45):
            look(az, el); iio.imwrite(st / f"az{az:03d}_el{int(el):+03d}.png", apply_credit(grab(), cred))
    log(f"stills -> {st}/  ({time.time() - t0:.0f}s)")
    if args.stills_only:
        return
    title, sub = cfg.get("title") or None, cfg.get("subtitle") or None
    with iio2.get_writer(str(out / "turntable_surface.mp4"), fps=30, codec="libx264", quality=9,
                         pixelformat="yuv420p", macro_block_size=8) as wr:
        for i in range(args.frames):
            look(360.0 * i / args.frames, args.elev)
            im = overlay(grab(), title, sub, alpha=min(1, i / 20, (args.frames - 1 - i) / 20))
            wr.append_data(apply_credit(im, cred))
            if (i + 1) % 60 == 0:
                log(f"  turntable {i + 1}/{args.frames}")
    log(f"-> {out / 'turntable_surface.mp4'}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
