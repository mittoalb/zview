"""VTK scenes for zview-render: volume ray casting with looks and lighting.

`Scene` is an off-screen GPU ray-cast scene (look-driven colour / opacity /
material, lighting rigs, volumetric shadows, jittered sampling, supersampled
filmic finish); label maps use a per-class label-map mask. The module also
holds the volume preparation (`prepare_volume`: windowing, tomography
preprocessing or label codes) and the stills/turntable entry point used by
`zview-render run … stills turntable`. The tour and surface renderers and
the viewer's VTK pane reuse its building blocks.
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np

from zview.render import looks as lk

HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# arguments shared with the tour renderer
# ---------------------------------------------------------------------------
def add_common_args(p):
    p.add_argument("volume", type=Path, help=".npy volume from zview-render prepare")
    p.add_argument("--factor", type=float, nargs="+", default=[8],
                   help="step of the .npy vs full resolution: one value, or Z Y X (default 8)")
    p.add_argument("--down", type=int, default=2, help="extra downsampling on load (default 2)")
    p.add_argument("--z-range", type=int, nargs=2, default=None, metavar=("Z0", "Z1"), help="full-res slices")
    p.add_argument("--cylinder", type=float, nargs="+", default=None,
                   help="full-res: CY CX R, or CY_TOP CX_TOP CY_BOTTOM CX_BOTTOM R (tilted)")
    p.add_argument("--size", default="1600x1200", help="image size WxH")
    p.add_argument("--look", default="organic",
                   help=f"colour/opacity/material preset: {', '.join(sorted(lk.LOOKS))}, "
                        "or any matplotlib colormap (default organic)")
    p.add_argument("--lighting", default="studio", help=f"lighting rig: {', '.join(lk.LIGHTING)} (default studio)")
    p.add_argument("--curve", default=None, help="opacity curve: solid, soft, layered, glass, mip (default: from look)")
    p.add_argument("--tone-mapping", default="filmic", choices=["filmic", "none"],
                   help="filmic = ACES highlight roll-off (default); none = plain exposure multiply")
    p.add_argument("--supersample", type=int, default=2, help="render at N x size and downsample (anti-aliasing; default 2)")
    p.add_argument("--sample-distance", type=float, default=0.5, help="ray step in voxels (smaller = finer, slower)")
    p.add_argument("--background", default="look", help="look | black | white | grey (default: the look's gradient)")
    p.add_argument("--labels-json", default=None,
                   help="label maps: class table as JSON {value: {name, color, opacity, visible}}")
    p.add_argument("--label-smooth", type=float, default=1.0, help="label maps: surface smoothing (voxels)")
    p.add_argument("--preprocess", default="tomo", choices=["tomo", "none", "labels"],
                   help="tomo = medium removal, container fit, Z equalisation (sample in a tube); "
                        "none = just window the intensities (any data); labels = class IDs (segmentations)")
    p.add_argument("--window", default=None, choices=["otsu", "percentile"],
                   help="intensity window: otsu (lower edge = Otsu/2) or percentile (--lo-pct/--hi-pct); "
                        "default otsu for tomo, percentile otherwise")
    p.add_argument("--window-values", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                   help="absolute intensity window (overrides the percentiles; generic preprocessing)")
    p.add_argument("--smooth", type=float, default=0.0,
                   help="Gaussian pre-smoothing sigma in voxels (cleaner surfaces on noisy data; 0 = off)")
    p.add_argument("--voxel-size", type=float, nargs=3, default=None, metavar=("VZ", "VY", "VX"),
                   help="full-resolution voxel size (any unit) for anisotropic data")
    p.add_argument("--lo-pct", type=float, default=60)
    p.add_argument("--hi-pct", type=float, default=99.7)
    p.add_argument("--opacity", type=float, default=0.6, help="max scalar opacity per sample (default 0.6 = solid)")
    p.add_argument("--interior", type=float, default=1.0,
                   help="opacity factor for uniform interiors vs surfaces, 0..1 (default 1 = solid)")
    p.add_argument("--scatter", type=float, default=None,
                   help="volumetric scattering / shadows 0..2 (0 = off, 1 = blend, 2 = full; default: from lighting)")
    p.add_argument("--gi-reach", type=float, default=0.4,
                   help="how far shadow rays travel through the volume, 0..1 (default 0.4)")
    p.add_argument("--anisotropy", type=float, default=0.0, help="scattering anisotropy -1..1 (default 0)")
    p.add_argument("--specular", type=float, default=None,
                   help="specular highlight strength (default: from look; higher looks glossy/plastic)")
    p.add_argument("--exposure", type=float, default=None,
                   help="brightness (default 1.0 with filmic tone mapping, 1.35 without)")
    p.add_argument("--no-keep-largest", dest="keep_largest", action="store_false",
                   help="keep everything, not just the largest connected object")
    p.add_argument("--open-iter", type=int, default=6,
                   help="opening iterations to cut thin bridges before picking the largest object (default 6)")
    p.add_argument("--mask-level", type=float, default=20, help="tissue threshold on the 0-255 volume (default 20)")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--workers", type=int, default=os.cpu_count())


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# volume preparation
# ---------------------------------------------------------------------------
def keep_largest_object(U8, mask_level, open_iter):
    """Keep only the main object: threshold, cut thin bridges (opening), largest connected
    component, grow back to the original boundary. Drops debris, strands, particles."""
    from scipy import ndimage as ndi
    sm = ndi.gaussian_filter(U8.astype(np.float32), 1.5)
    m = sm > mask_level
    del sm
    ball = ndi.generate_binary_structure(3, 1)
    core = ndi.binary_opening(m, structure=ball, iterations=open_iter)
    lab, n = ndi.label(core)
    if n == 0:
        return U8, 0, 0.0
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    keep = lab == int(np.argmax(sizes))
    del lab, core
    keep = ndi.binary_dilation(keep, structure=ball, iterations=open_iter + 3) & m
    keep = ndi.binary_dilation(keep, structure=ball, iterations=1)
    dropped = 1 - (keep & m).sum() / max(m.sum(), 1)
    U8 *= keep
    return U8, n, dropped


def prepare_volume(args):
    """Load the cached .npy (z-range, extra downsampling), remove the medium, cut the
    container, equalise along Z, scale to uint8, keep the main object.
    Returns dict(U8=(nz,ny,nx) uint8, F=step vs full res, z0=first full-res slice)."""
    t0 = time.time()
    if getattr(args, "preprocess", "tomo") == "labels":
        return prepare_labels(args)
    src = np.load(args.volume, mmap_mode="r")
    Sz, Sy, Sx = factors(args.factor)
    S = Sz
    nz_all = src.shape[0]
    z0, z1 = (0, nz_all) if not args.z_range else (max(0, int(args.z_range[0] // Sz)),
                                                   min(nz_all, int(-(-args.z_range[1] // Sz))))
    d = args.down
    nzr, ny, nx = (z1 - z0) // d, src.shape[1] // d, src.shape[2] // d
    V = np.empty((nzr, ny, nx), np.float32)
    for a in range(0, nzr, 16):
        b = min(a + 16, nzr)
        blk = np.asarray(src[z0 + a * d:z0 + b * d, :ny * d, :nx * d], np.float32)
        V[a:b] = blk.reshape(b - a, d, ny, d, nx, d).mean(axis=(1, 3, 5))
    F = S * d
    Fz, Fy, Fx = Sz * d, Sy * d, Sx * d
    log(f"loaded {V.shape} (full-res z {z0 * S:.0f}..{z1 * S:.0f}, step {Fz:g}/{Fy:g}/{Fx:g}) "
        f"in {time.time() - t0:.0f}s")
    cyl = None
    if args.cylinder:
        c = list(args.cylinder)
        if len(c) == 3:
            c = [c[0], c[1], c[0], c[1], c[2]]
        tt = (z0 + np.arange(nzr) * d) / max(nz_all - 1, 1)
        cyl = ((c[0] + (c[2] - c[0]) * tt) / Fy, (c[1] + (c[3] - c[1]) * tt) / Fx, np.full(nzr, c[4] / Fx))
    spacing = physical_spacing((Fz, Fy, Fx), getattr(args, "voxel_size", None))
    if getattr(args, "preprocess", "tomo") == "none":
        U8 = window_volume(V, args, cyl)
        del V
        if args.keep_largest:
            t1 = time.time()
            U8, n, dropped = keep_largest_object(U8, args.mask_level, args.open_iter)
            log(f"keep-largest: {n} pieces, kept the main one ({dropped * 100:.1f}% dropped) "
                f"({time.time() - t1:.0f}s)")
        return dict(U8=U8, F=F, z0=int(z0 * S), Fzyx=(Fz, Fy, Fx), spacing=spacing)
    from zview.render import cpu as rc
    from multiprocessing import shared_memory
    shm = shared_memory.SharedMemory(create=True, size=V.nbytes)
    try:
        Vs = np.ndarray(V.shape, np.float32, buffer=shm.buf); Vs[...] = V; del V
        rc._VOL = Vs
        rc.preprocess(Vs, 0.77, True, 3, min(args.workers, 32), log, wall_fit=True,
                      wall_block=max(8, int(512 // Fy)),
                      cylinder=cyl, medium_pct=10.0, equalize=True)
        smp = Vs[::2, ::4, ::4]; smp = smp[smp > 0]
        if (getattr(args, "window", None) or "otsu") == "otsu":
            from skimage.filters import threshold_otsu
            t = threshold_otsu(smp[:: max(1, smp.size // 500000)])
            lo = 0.5 * t                                            # just above the medium noise
        else:
            lo = np.percentile(smp, args.lo_pct)
        hi = np.percentile(smp, args.hi_pct)
        U8 = (np.clip((Vs - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    finally:
        rc._VOL = None
        shm.close(); shm.unlink()
    log(f"transfer: {lo:.0f}..{hi:.0f} (Otsu/2 .. p{args.hi_pct}) ({time.time() - t0:.0f}s)")
    if args.keep_largest:
        t1 = time.time()
        U8, n, dropped = keep_largest_object(U8, args.mask_level, args.open_iter)
        log(f"keep-largest: {n} pieces, kept the main one ({dropped * 100:.1f}% of tissue dropped) "
            f"({time.time() - t1:.0f}s)")
    return dict(U8=U8, F=F, z0=int(z0 * S), Fzyx=(Fz, Fy, Fx), spacing=spacing)


def prepare_labels(args):
    """Label volume: crop, label-aware downsampling, compact render codes."""
    import json as _json
    from zview.core.labels import LabelTable, build_table, pool_labels, render_codes
    t0 = time.time()
    src = np.load(args.volume, mmap_mode="r")
    Sz, Sy, Sx = factors(args.factor)
    nz_all = src.shape[0]
    z0, z1 = (0, nz_all) if not args.z_range else (max(0, int(args.z_range[0] // Sz)),
                                                   min(nz_all, int(-(-args.z_range[1] // Sz))))
    data = np.asarray(src[z0:z1])
    lj = getattr(args, "labels_json", None)
    if lj:
        table = LabelTable.from_config(_json.loads(lj))
    else:
        v, c = np.unique(data[::2, ::2, ::2], return_counts=True)
        table = build_table(v.tolist(), dict(zip(v.tolist(), (c / c.sum()).tolist())))
    d = max(1, int(args.down))
    if d > 1:
        data = pool_labels(data, (d, d, d), table.priority_lut(max(table.values) + 1))
    if args.cylinder:
        c = list(args.cylinder)
        if len(c) == 3:
            c = [c[0], c[1], c[0], c[1], c[2]]
        bg = next(iter(table.background), None)
        if bg is not None:
            yy = np.arange(data.shape[1])[:, None]
            xx = np.arange(data.shape[2])[None, :]
            for i in range(data.shape[0]):
                tt = (z0 + i * d) / max(nz_all - 1, 1)
                cy = (c[0] + (c[2] - c[0]) * tt) / (Sy * d)
                cx = (c[1] + (c[3] - c[1]) * tt) / (Sx * d)
                data[i][(yy - cy) ** 2 + (xx - cx) ** 2 > (c[4] / (Sx * d)) ** 2] = bg
    codes, code_values = render_codes(data, table)
    layers = label_layers(table, code_values)
    import vtk as _vtk
    if not label_masks_supported(_vtk):
        log(f"NOTE: VTK {_vtk.vtkVersion.GetVTKVersion()} — label maps are drawn unshaded; "
            "VTK >= 9.4 gives smooth lit per-class surfaces (pip install -U vtk)")
    Fz, Fy, Fx = Sz * d, Sy * d, Sx * d
    log(f"labels: {codes.shape}, {len(layers)} drawn classes "
        f"({', '.join(name for name in (table.classes[l[1]].name for l in layers[:6]))}"
        f"{' …' if len(layers) > 6 else ''}) ({time.time() - t0:.0f}s)")
    return dict(U8=codes, F=Sz * d, z0=int(z0 * Sz), Fzyx=(Fz, Fy, Fx),
                spacing=physical_spacing((Fz, Fy, Fx), getattr(args, "voxel_size", None)),
                labels=layers, label_table=table)


def factors(f):
    """--factor as (z, y, x) floats: accepts a number or a 1/3-element list."""
    f = list(f) if isinstance(f, (list, tuple)) else [f]
    if len(f) == 1:
        f = f * 3
    if len(f) != 3:
        raise ValueError(f"factor must have 1 or 3 values, got {f}")
    return tuple(float(v) for v in f)


def physical_spacing(Fzyx, voxel_size=None):
    """VTK (x, y, z) spacing of the prepared volume, normalised to min 1.

    Each prepared voxel spans F full-res voxels along its axis; times the
    full-res voxel size gives its physical extent. Isotropic data -> (1,1,1).
    """
    vz, vy, vx = voxel_size if voxel_size else (1.0, 1.0, 1.0)
    sx, sy, sz = Fzyx[2] * vx, Fzyx[1] * vy, Fzyx[0] * vz
    m = min(sx, sy, sz)
    sp = (sx / m, sy / m, sz / m)
    return (1.0, 1.0, 1.0) if np.allclose(sp, 1.0, rtol=0.02) else sp


def window_volume(V, args, cyl=None):
    """Generic preprocessing: window intensities to uint8, optional cylinder cut.

    No tomography assumptions: NaN/inf -> min, window from percentiles of the
    whole volume (or Otsu/2 .. p_hi), optional container cut.
    """
    t0 = time.time()
    finite = np.isfinite(V)
    if not finite.all():
        V[~finite] = np.nanmin(np.where(finite, V, np.nan)) if finite.any() else 0.0
    sigma = float(getattr(args, "smooth", 0.0) or 0.0)
    if sigma > 0:
        from scipy import ndimage as ndi
        ndi.gaussian_filter(V, sigma, output=V)      # in place: noise -> smooth normals
    outside = None
    if cyl is not None:
        cy, cx, r = cyl
        yy = np.arange(V.shape[1])[:, None]
        xx = np.arange(V.shape[2])[None, :]
        outside = np.stack([(yy - cy[i]) ** 2 + (xx - cx[i]) ** 2 > r[i] ** 2 for i in range(V.shape[0])])
    smp = V[::2, ::2, ::2] if outside is None else V[::2, ::2, ::2][~outside[::2, ::2, ::2]]
    smp = smp.ravel()[:: max(1, smp.size // 2_000_000)]
    wv = getattr(args, "window_values", None)
    if wv:
        lo, hi = float(wv[0]), float(wv[1])
    elif (getattr(args, "window", None) or "percentile") == "otsu":
        from skimage.filters import threshold_otsu
        lo = 0.5 * threshold_otsu(smp)
        hi = float(np.percentile(smp, args.hi_pct))
    else:
        lo = float(np.percentile(smp, args.lo_pct))
        hi = float(np.percentile(smp, args.hi_pct))
    if hi <= lo:
        hi = lo + 1.0
    U8 = np.empty(V.shape, np.uint8)
    for a in range(0, V.shape[0], 32):
        U8[a:a + 32] = (np.clip((V[a:a + 32] - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    if outside is not None:
        U8[outside] = 0
    log(f"window: {lo:.4g}..{hi:.4g} ({args.window or 'percentile'}, p{args.lo_pct}/p{args.hi_pct}) "
        f"({time.time() - t0:.0f}s)")
    return U8


def brain_mask_from_config(cfg, workers):
    """Step-(2*factor) brain mask for a dataset.json: prepared volume, filled, eroded.
    Returns (mask, U8, first full-res slice)."""
    from scipy import ndimage as ndi
    args = argparse.Namespace(volume=Path(cfg["volume"]), factor=cfg["factor"], down=2,
                              z_range=cfg.get("z_range"), cylinder=cfg.get("cylinder"),
                              lo_pct=60, hi_pct=99.7, keep_largest=True, open_iter=6, mask_level=20,
                              workers=workers)
    vol = prepare_volume(args)
    U8 = vol["U8"]
    m = ndi.gaussian_filter(U8.astype(np.float32), 1.0) > 20
    for z in range(m.shape[0]):                                # lumens are holes in each slice
        m[z] = ndi.binary_fill_holes(m[z])
    m = ndi.binary_erosion(m, iterations=2)                    # keep fissure edges / surface out
    return m, U8, vol["z0"]


# ---------------------------------------------------------------------------
# VTK scene
# ---------------------------------------------------------------------------
def resolve_style(args):
    """Fill look/lighting-dependent defaults on `args` (in place) and return
    (look, lighting) dicts."""
    look = lk.get_look(getattr(args, "look", None) or "organic")
    light = lk.get_lighting(getattr(args, "lighting", "studio") or "studio")
    if getattr(args, "curve", None) is None:
        args.curve = look["curve"]
    if getattr(args, "specular", None) is None:
        args.specular = look["material"]["specular"]
    if getattr(args, "scatter", None) is None:
        args.scatter = light["scatter"]
    if getattr(args, "exposure", None) is None:
        args.exposure = 1.0 if getattr(args, "tone_mapping", "none") == "filmic" else 1.35
    for k, v in (("tone_mapping", "none"), ("supersample", 1), ("background", "look"),
                 ("gi_reach", 0.4), ("anisotropy", 0.0), ("interior", 1.0)):
        if not hasattr(args, k) or getattr(args, k) is None:
            setattr(args, k, v)
    return look, light


def apply_background(ren, look, mode):
    bg = {"black": ((0, 0, 0), (0, 0, 0)), "white": ((1, 1, 1), (0.92, 0.92, 0.94)),
          "grey": ((0.16, 0.16, 0.17), (0.32, 0.32, 0.34))}.get(mode, look["background"])
    ren.SetBackground(*bg[0]); ren.SetBackground2(*bg[1]); ren.GradientBackgroundOn()


def add_lights(vtk, ren, light):
    ren.AutomaticLightCreationOff(); ren.RemoveAllLights()
    for pos, rgb, inten in light["lights"]:
        lt = vtk.vtkLight(); lt.SetLightTypeToCameraLight(); lt.SetPositional(False)
        lt.SetPosition(*pos); lt.SetFocalPoint(0, 0, 0); lt.SetColor(*rgb); lt.SetIntensity(inten)
        ren.AddLight(lt)


def transfer_functions(vtk, look, args, window=(0.0, 255.0)):
    """(color, scalar-opacity, gradient-opacity) functions for a look.

    The look's curves are defined on a 0..255 windowed scale; `window` maps
    that scale onto the actual data values, so an interactive window change is
    just new transfer functions (no data re-upload).
    """
    lo, hi = float(window[0]), float(window[1])
    k = max(hi - lo, 1e-6) / 255.0
    ctf = vtk.vtkColorTransferFunction()
    for x, (r, g, b) in lk.color_stops(look):
        ctf.AddRGBPoint(lo + x * k, r, g, b)
    otf = vtk.vtkPiecewiseFunction()
    for x, a in lk.opacity_points(args.curve, args.mask_level, args.opacity):
        otf.AddPoint(lo + x * k, a)
    gof = vtk.vtkPiecewiseFunction()
    for x, a in lk.gradient_points(args.curve, args.interior):
        gof.AddPoint(x, a)
    return ctf, otf, gof


def volume_property(vtk, look, light, args, window=(0.0, 255.0)):
    """vtkVolumeProperty with the look's transfer functions and material."""
    ctf, otf, gof = transfer_functions(vtk, look, args, window)
    mat = look["material"]
    prop = vtk.vtkVolumeProperty()
    prop.SetColor(ctf); prop.SetScalarOpacity(otf); prop.SetGradientOpacity(gof)
    prop.SetInterpolationTypeToLinear()
    if light["shade"]:
        prop.ShadeOn()
    else:
        prop.ShadeOff()
    prop.SetAmbient(max(0.0, mat["ambient"] + light["ambient_boost"]))
    prop.SetDiffuse(mat["diffuse"])
    prop.SetSpecular(args.specular); prop.SetSpecularPower(mat["specular_power"])
    prop.SetScatteringAnisotropy(args.anisotropy)
    return prop


def volume_mapper(vtk, args, light, sample_distance=None):
    """GPU ray-cast mapper: jittered sampling + volumetric shadows."""
    m = vtk.vtkGPUVolumeRayCastMapper()
    m.SetSampleDistance(sample_distance or getattr(args, "sample_distance", 0.5) or 0.5)
    m.AutoAdjustSampleDistancesOff()
    if hasattr(m, "UseJitteringOn"):
        m.UseJitteringOn()                                          # no wood-grain banding
    set_scatter(m, args.scatter if light["shade"] else 0.0, args.gi_reach)
    return m


def set_scatter(mapper, scatter, gi_reach=0.4):
    """Volumetric scattering ("cinematic" shadows) on/off."""
    if hasattr(mapper, "SetVolumetricScatteringBlending"):
        mapper.SetVolumetricScatteringBlending(max(0.0, float(scatter)))
        if scatter > 0:
            mapper.SetGlobalIlluminationReach(gi_reach)


_OFFSCREEN_KIND = None


def offscreen_render_window():
    """Off-screen vtkRenderWindow, preferring EGL (GPU, no X display needed).

    VTK's default window tries GLX first; inside a Qt application (or with
    mixed GL libraries) that can fail before VTK falls back. The working kind
    is probed once per process. ZVIEW_VTK_OFFSCREEN=egl|default forces one.
    """
    global _OFFSCREEN_KIND
    import vtk
    forced = os.environ.get("ZVIEW_VTK_OFFSCREEN", "").strip().lower()
    # EGL is only tried on VTK >= 9.7: VTK 9.4's EGL window segfaults (which
    # can't be caught), its default GLX window works.
    egl_ok = hasattr(vtk, "vtkEGLRenderWindow") and \
        (vtk.vtkVersion.GetVTKMajorVersion(), vtk.vtkVersion.GetVTKMinorVersion()) >= (9, 7)
    kinds = [forced] if forced in ("egl", "default") else (
        [_OFFSCREEN_KIND] if _OFFSCREEN_KIND else (["egl", "default"] if egl_ok else ["default"]))

    def make(kind):
        rw = vtk.vtkEGLRenderWindow() if kind == "egl" else vtk.vtkRenderWindow()
        rw.SetOffScreenRendering(1)
        return rw

    if len(kinds) == 1 and (forced or _OFFSCREEN_KIND):
        return make(kinds[0])
    for kind in kinds:
        rw = make(kind)
        warn = vtk.vtkObject.GetGlobalWarningDisplay()
        vtk.vtkObject.GlobalWarningDisplayOff()
        try:
            rw.SetSize(8, 8)
            rw.Render()
            ok = "renderer string" in (rw.ReportCapabilities() or "").lower()
        except Exception:
            ok = False
        finally:
            vtk.vtkObject.SetGlobalWarningDisplay(warn)
        if ok:
            _OFFSCREEN_KIND = kind
            return rw
        try:
            rw.Finalize()
        except Exception:
            pass
    _OFFSCREEN_KIND = "default"
    return make("default")


def image_data(vtk, ns, U8, spacing=(1.0, 1.0, 1.0)):
    nz, ny, nx = U8.shape
    img = vtk.vtkImageData(); img.SetDimensions(nx, ny, nz); img.SetSpacing(*spacing)
    img.GetPointData().SetScalars(ns.numpy_to_vtk(np.ascontiguousarray(U8).ravel(), deep=True,
                                                  array_type=vtk.VTK_UNSIGNED_CHAR))
    return img


# ---------------------------------------------------------------------------
# label maps
# ---------------------------------------------------------------------------
# Labels are rendered with a VTK label-map mask: the *mask* holds the class
# codes (nearest sampling — no invented in-between classes) and selects a
# per-class colour/opacity; the *data* is a smooth "shading field" whose
# gradients give every visible class lit, smooth surfaces. Opaque classes sit
# at 255, glass (bulk) classes at GLASS_LEVEL, hidden classes at 0, so class
# boundaries inside the volume are shaded too.
GLASS_LEVEL = 110


def label_layers(table, code_values=None):
    """[(code, value, rgb, opacity, kind)] for the codes in a loaded volume.

    kind: "opaque", "glass" (opacity < 0.35) or "hidden".
    """
    out = []
    codes = range(256) if code_values is None else range(len(code_values))
    for code in codes:
        v = code if code_values is None else code_values[code]
        c = table.classes.get(v) if v is not None else None
        if c is None or code == 0:
            continue          # code 0 = background, never drawn
        if not c.visible or c.opacity <= 0:
            kind = "hidden"
        elif c.opacity < 0.35:
            kind = "glass"
        else:
            kind = "opaque"
        out.append((code, v, tuple(c.color), float(c.opacity), kind))
    return out


def label_shading_field(codes, layers, sigma=1.0):
    """uint8 shading field: opaque 255, glass GLASS_LEVEL, hidden 0, smoothed."""
    level = np.zeros(256, np.uint8)
    for code, _v, _rgb, _op, kind in layers:
        level[code] = 255 if kind == "opaque" else (GLASS_LEVEL if kind == "glass" else 0)
    f = level[codes]
    if sigma > 0:
        try:
            from scipy import ndimage as ndi
            g = ndi.gaussian_filter(f.astype(np.float32), sigma)
            f = np.clip(g + 0.5, 0, 255).astype(np.uint8)
        except ImportError:
            pass
    return f


def label_volume_property(vtk, layers, light, material=None):
    """vtkVolumeProperty with one colour / opacity function per class."""
    if not label_masks_supported(vtk):
        return label_volume_property_flat(vtk, layers)
    mat = material or {"ambient": 0.22, "diffuse": 0.80, "specular": 0.15, "specular_power": 16}
    prop = vtk.vtkVolumeProperty()
    base_c = vtk.vtkColorTransferFunction(); base_c.AddRGBPoint(0, 0, 0, 0); base_c.AddRGBPoint(255, 0, 0, 0)
    base_o = vtk.vtkPiecewiseFunction(); base_o.AddPoint(0, 0.0); base_o.AddPoint(255, 0.0)
    prop.SetColor(base_c); prop.SetScalarOpacity(base_o)
    prop.SetInterpolationTypeToLinear()
    for code, _v, rgb, op, kind in layers:
        c = vtk.vtkColorTransferFunction(); c.AddRGBPoint(0, *rgb); c.AddRGBPoint(255, *rgb)
        o = vtk.vtkPiecewiseFunction()
        if kind == "opaque":
            for x, a in ((0, 0), (150, 0), (200, op), (255, op)):
                o.AddPoint(x, a)
        elif kind == "glass":
            for x, a in ((0, 0), (40, 0), (GLASS_LEVEL - 10, op), (255, op)):
                o.AddPoint(x, a)
        else:
            o.AddPoint(0, 0); o.AddPoint(255, 0)
        prop.SetLabelColor(code, c)
        prop.SetLabelScalarOpacity(code, o)
    if light["shade"]:
        prop.ShadeOn()
    else:
        prop.ShadeOff()
    prop.SetAmbient(max(0.0, mat["ambient"] + light["ambient_boost"]))
    prop.SetDiffuse(mat["diffuse"]); prop.SetSpecular(mat["specular"]); prop.SetSpecularPower(mat["specular_power"])
    return prop


def label_masks_supported(vtk):
    """Label-map masks (smooth per-class shading); verified on VTK 9.4 and 9.7.

    ZVIEW_VTK_LABELMASK=1/0 forces the choice.
    """
    forced = os.environ.get("ZVIEW_VTK_LABELMASK", "").strip()
    if forced in ("0", "1"):
        return forced == "1"
    return (vtk.vtkVersion.GetVTKMajorVersion(), vtk.vtkVersion.GetVTKMinorVersion()) >= (9, 4)


def label_volume_property_flat(vtk, layers):
    """Fallback for older VTK: class colours on nearest-sampled codes, unshaded."""
    prop = vtk.vtkVolumeProperty()
    ctf, otf = vtk.vtkColorTransferFunction(), vtk.vtkPiecewiseFunction()
    ctf.AddRGBPoint(0, 0, 0, 0)
    otf.AddPoint(0, 0.0)
    for code, _v, rgb, op, kind in layers:
        a = 0.0 if kind == "hidden" else op
        for x in (code - 0.49, code + 0.49):
            ctf.AddRGBPoint(x, *rgb)
            otf.AddPoint(x, a)
    otf.AddPoint(255, 0.0)
    prop.SetColor(ctf); prop.SetScalarOpacity(otf)
    prop.SetInterpolationTypeToNearest()
    prop.ShadeOff()
    prop.SetAmbient(1.0); prop.SetDiffuse(0.0)
    return prop


def set_label_inputs(vtk, ns, mapper, codes, layers, spacing, sigma=1.0):
    """Feed a label volume (codes) to a GPU mapper.

    VTK >= 9.7: smooth shading field as data + codes as a label-map mask.
    Older VTK: the codes themselves (see label_volume_property_flat).
    """
    if not label_masks_supported(vtk):
        mapper.SetInputData(image_data(vtk, ns, codes.astype(np.uint8, copy=False), spacing))
        return None
    field = label_shading_field(codes, layers, sigma)
    mapper.SetInputData(image_data(vtk, ns, field, spacing))
    mapper.SetMaskInput(image_data(vtk, ns, codes.astype(np.uint8, copy=False), spacing))
    mapper.SetMaskTypeToLabelMap()
    return field


class Scene:
    """Off-screen VTK volume scene: look (colour/opacity/material), lighting rig,
    volumetric shadows, jittered ray casting, supersampling + filmic finish."""

    def __init__(self, U8, args, W, H, sample_distance=None, spacing=None, window=None, labels=None):
        import vtk
        from vtk.util import numpy_support as ns
        self.vtk, self.ns, self.W, self.H, self.args = vtk, ns, W, H, args
        self.look_style, self.light_style = resolve_style(args)
        self.ss = max(1, int(args.supersample))
        self.spacing = tuple(spacing) if spacing is not None else (1.0, 1.0, 1.0)
        self.mapper = volume_mapper(vtk, args, self.light_style, sample_distance)
        self.label_layers = labels
        if labels is not None:
            # labels = list of label_layers(); U8 = class codes.
            self.shape = tuple(U8.shape)
            self.img = image_data(vtk, ns, U8.astype(np.uint8, copy=False), self.spacing)   # codes (cut faces)
            set_label_inputs(vtk, ns, self.mapper, U8, labels, self.spacing, getattr(args, "label_smooth", 1.0))
            prop = label_volume_property(vtk, labels, self.light_style, self.look_style["material"])
        else:
            self.set_data(U8)
            prop = volume_property(vtk, self.look_style, self.light_style, args, window or (0.0, 255.0))
        self.prop = prop

        self.volume = vtk.vtkVolume(); self.volume.SetMapper(self.mapper); self.volume.SetProperty(prop)
        ren = vtk.vtkRenderer(); ren.AddVolume(self.volume)
        add_lights(vtk, ren, self.light_style)
        apply_background(ren, self.look_style, args.background)
        self.ren = ren
        self.rw = offscreen_render_window(); self.rw.AddRenderer(ren)
        self.rw.SetSize(W * self.ss, H * self.ss); self.rw.SetMultiSamples(0)
        self.cam = ren.GetActiveCamera()
        self.planes = []

    def set_data(self, U8):
        img = image_data(self.vtk, self.ns, U8, getattr(self, "spacing", (1, 1, 1)))
        self.img, self.shape = img, tuple(U8.shape)
        self.mapper.SetInputData(img)

    @property
    def center(self):
        nz, ny, nx = self.shape
        sx, sy, sz = getattr(self, "spacing", (1.0, 1.0, 1.0))
        return np.array([(nx - 1) / 2 * sx, (ny - 1) / 2 * sy, (nz - 1) / 2 * sz])

    def radius(self):
        nz, ny, nx = self.shape
        sx, sy, sz = getattr(self, "spacing", (1.0, 1.0, 1.0))
        return 0.5 * np.sqrt((nx * sx) ** 2 + (ny * sy) ** 2 + (nz * sz) ** 2)

    def look(self, az, el, dist, focal=None, view_angle=30):
        """Camera on a sphere around `focal` (VTK x,y,z). Sample 'up' = towards slice 0 (-z)."""
        f = self.center if focal is None else np.asarray(focal, float)
        a, e = np.radians(az), np.radians(el)
        pos = f + dist * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), -np.sin(e)])
        self.cam.SetFocalPoint(*f); self.cam.SetPosition(*pos); self.cam.SetViewUp(0, 0, -1)
        self.cam.SetViewAngle(view_angle); self.ren.ResetCameraClippingRange()
        return pos

    def pose(self, pos, focal, up, view_angle=30):
        """Free camera pose (VTK x,y,z)."""
        self.cam.SetFocalPoint(*focal); self.cam.SetPosition(*pos); self.cam.SetViewUp(*up)
        self.cam.SetViewAngle(view_angle); self.ren.ResetCameraClippingRange()

    def default_dist(self, view_angle=30, fill=0.85):
        return self.radius() / np.tan(np.radians(view_angle) / 2) * fill

    def clip(self, origin=None, normal=None):
        """Set one clipping plane (keeps the side the normal points to); clip() removes it."""
        self.mapper.RemoveAllClippingPlanes()
        if origin is None:
            return
        pl = self.vtk.vtkPlane(); pl.SetOrigin(*origin); pl.SetNormal(*normal)
        self.mapper.AddClippingPlane(pl)

    def section(self, origin=None, normal=None, U8=None):
        """Show a textured slice of the volume on a plane (organic LUT, empty space transparent).
        section() hides it. Use together with clip() on the same plane for a crisp cut face."""
        vtk = self.vtk
        if not hasattr(self, "_slice"):
            lut = vtk.vtkLookupTable(); lut.SetNumberOfTableValues(256); lut.SetRange(0, 255)
            look = getattr(self, "look_style", None) or lk.LOOKS["organic"]
            layers = getattr(self, "label_layers", None)
            if layers is not None:
                # Label map: the cut face shows each class in its colour.
                for i in range(256):
                    lut.SetTableValue(i, 0, 0, 0, 0)
                for code, _v, rgb, _op, kind in layers:
                    if kind != "hidden":
                        lut.SetTableValue(code, *(0.85 * c for c in rgb), 1.0)
            else:
                cols = lk.color_lut(look)
                ml = float(self.args.mask_level)
                for i in range(256):
                    br = (0.45 + 0.40 * (i / 255) ** 0.8) / self.args.exposure   # match the lit surface brightness
                    lut.SetTableValue(i, *[min(1.0, c * br) for c in cols[i]], 0.0 if i < ml * 0.5 else 1.0)
            lut.Build()
            self._splane = vtk.vtkPlane()
            m = vtk.vtkImageResliceMapper(); m.SetInputData(self.img); m.SetSlicePlane(self._splane)
            m.SliceFacesCameraOff(); m.SliceAtFocalPointOff()
            pr = vtk.vtkImageProperty(); pr.SetLookupTable(lut); pr.UseLookupTableScalarRangeOn()
            if layers is not None:
                pr.SetInterpolationTypeToNearest()      # class codes must not be blended
            else:
                pr.SetInterpolationTypeToLinear()
            pr.SetAmbient(1.0); pr.SetDiffuse(0.0)
            self._slice = vtk.vtkImageSlice(); self._slice.SetMapper(m); self._slice.SetProperty(pr)
            self.ren.AddViewProp(self._slice)
        if origin is None:
            self._slice.VisibilityOff(); return
        n = np.asarray(normal, float); n /= np.linalg.norm(n)
        self._splane.SetOrigin(*(np.asarray(origin) - 0.6 * n)); self._splane.SetNormal(*n)   # just in front
        self._slice.VisibilityOn()

    def grab(self):
        vtk, ns = self.vtk, self.ns
        self.rw.Render()
        w2i = vtk.vtkWindowToImageFilter(); w2i.SetInput(self.rw); w2i.ReadFrontBufferOff(); w2i.Update()
        ss = getattr(self, "ss", 1)
        im = ns.vtk_to_numpy(w2i.GetOutput().GetPointData().GetScalars()).reshape(self.H * ss, self.W * ss, -1)
        return lk.finish(im[::-1, :, :3], self.args.exposure, getattr(self.args, "tone_mapping", "none"), ss)

    def pixels_per_unit(self, dist, view_angle=30):
        """Screen pixels per voxel at the focal distance (for scale bars)."""
        return self.H / (2 * dist * np.tan(np.radians(view_angle) / 2))

    def close(self):
        self.rw.Finalize()


# ---------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--frames", type=int, default=240, help="turntable frames (0 = no turntable)")
    p.add_argument("--elev", type=float, default=20)
    p.add_argument("--elevations", type=float, nargs="+", default=None,
                   help="still elevations (default: --elev 45 -20 0)")
    p.add_argument("--azimuths", type=int, default=8, help="stills per elevation (default 8)")
    p.add_argument("--azimuth", type=float, default=0.0, help="start azimuth of stills and turntable (degrees)")
    p.add_argument("--zoom", type=float, default=1.0, help="camera zoom (>1 = closer; default 1 = whole volume)")
    p.add_argument("--no-stills", dest="stills", action="store_false")
    p.add_argument("--keep-frames", action="store_true", help="also write every turntable frame as PNG")
    args = p.parse_args(argv)
    import imageio.v2 as iio2
    import imageio.v3 as iio
    t0 = time.time()

    vol = prepare_volume(args)
    W, H = (int(v) for v in args.size.lower().split("x"))
    sc = Scene(vol["U8"], args, W, H, spacing=vol.get("spacing"), labels=vol.get("labels"))
    del vol
    dist = sc.default_dist() / max(0.05, args.zoom)
    tag = args.look
    if tag.lstrip().startswith("{"):
        tag = "custom"
    light_tag = "custom" if str(args.lighting).lstrip().startswith("{") else args.lighting
    log(f"look {tag} · curve {args.curve} · lighting {light_tag} · tone {args.tone_mapping} · "
        f"supersample {sc.ss}x · spacing {tuple(round(v, 3) for v in sc.spacing)}")

    args.out.mkdir(parents=True, exist_ok=True)
    if args.stills:
        st = args.out / f"stills_{tag}"; st.mkdir(exist_ok=True)
        for el in (args.elevations or (args.elev, 45, -20, 0)):
            for az in args.azimuth + np.linspace(0, 360, args.azimuths, endpoint=False):
                sc.look(az, el, dist)
                iio.imwrite(st / f"az{int(az) % 360:03d}_el{int(el):+03d}.png", sc.grab())
        log(f"stills -> {st}/  ({time.time() - t0:.0f}s)")

    if args.frames > 0:
        fr = args.out / f"frames_{tag}"
        if args.keep_frames:
            fr.mkdir(exist_ok=True)
        mp4 = args.out / f"turntable_{tag}.mp4"
        with iio2.get_writer(str(mp4), fps=args.fps, codec="libx264", quality=9, pixelformat="yuv420p",
                             macro_block_size=8) as wr:
            for i in range(args.frames):
                sc.look(args.azimuth + 360.0 * i / args.frames, args.elev, dist)
                im = sc.grab()
                if args.keep_frames:
                    iio.imwrite(fr / f"frame_{i:04d}.png", im)
                wr.append_data(im)
                if (i + 1) % 40 == 0:
                    log(f"  turntable {i + 1}/{args.frames}")
        log(f"-> {mp4}")
    sc.close()
    log(f"All outputs in {args.out}/  ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
