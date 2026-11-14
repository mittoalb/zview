"""`zview-render`: presentation-quality 3D renders of any volume.

    zview-render quick  SOURCE --out DIR [--preset P] [what ...] [--set k=v ...]
    zview-render prepare SOURCE --out DIR [--preset P] [--max-axis 768 | --step 8]
    zview-render run    DIR/render.json [what ...] [--preset P] [--set k=v ...] [-- extra args]
    zview-render presets [NAME]              list presets, or show one
    zview-render presets --save NAME --from DIR/render.json   save a config as a user preset
    zview-render looks                       list looks, opacity curves, lighting rigs
    zview-render cpu ...                     CPU renderer (no GPU / display needed)
    zview-render napari ...                  napari viewer / quick renders

`what` is any of: stills, turntable, tour, surface, all (default: the
config's "outputs"). SOURCE is a Zarr (v2/v3) or N5 store, an HDF5 file, or a
folder of TIFF slices, of any size: `prepare` reads a downsampled copy sized
by --max-axis (finest pyramid level that fits, plus a stride if needed).

GPU renders (stills/turntable/tour/surface) use VTK and need an X display
(DISPLAY set; a VNC session works) — nothing appears on screen.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from zview.render import config as rc

WHAT = ("stills", "turntable", "tour", "surface")


def _log(*a):
    print(*a, flush=True)


def _check_display():
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        _log("WARNING: DISPLAY is not set — VTK needs an X display (VNC/X session) for GPU renders.")


def _expand_what(items, c):
    items = list(items or []) or list(c.get("outputs") or ["stills", "turntable"])
    out = []
    for w in items:
        if w == "all":
            out += ["stills", "turntable", "tour"] + (["surface"] if c.get("style") == "surface" else [])
        elif w in WHAT:
            out.append(w)
        else:
            raise SystemExit(f"unknown output {w!r}; use {', '.join(WHAT)} or all")
    return list(dict.fromkeys(out))


def ensure_volume(cfg_path, cfg, c):
    """Prepare the cached volume if the config doesn't point at one yet."""
    vol = c.get("volume")
    if vol and Path(vol).exists():
        return c
    if not c.get("source"):
        raise SystemExit(f"{cfg_path}: no 'volume' and no 'source' to prepare one from")
    from zview.render import prepare as rp
    _log(f"no cached volume — preparing from {c['source']}")
    tmp = rp.prepare(c["source"], Path(cfg_path).parent, preset=c["preset"],
                     max_axis=c.get("max_axis") if c.get("max_axis") else None,
                     step=None if c.get("max_axis") else int(rc.DEFAULTS["factor"] if isinstance(c["factor"], list)
                                                             else c["factor"]),
                     dataset=c.get("hdf5_dataset"), config_name="render.prepared.json")
    prepared = json.loads(Path(tmp).read_text())
    for k in ("volume", "factor", "voxel_size_um", "pixel_size_um"):
        if prepared.get(k) is not None and cfg.get(k) is None or k in ("volume", "factor"):
            cfg[k] = prepared.get(k)
    Path(cfg_path).write_text(json.dumps(cfg, indent=2))
    Path(tmp).unlink(missing_ok=True)
    return rc.effective(cfg)


def run(cfg_path, what=None, overrides=None, out=None, extra=()):
    """Render the outputs of a config. Returns the output folder."""
    cfg_path = Path(cfg_path).resolve()
    cfg = rc.load(cfg_path)
    c = rc.effective(cfg, overrides)
    c = ensure_volume(cfg_path, cfg, c) if not (c.get("volume") and Path(c["volume"]).exists()) else c
    if overrides:
        c = rc.effective(rc.load(cfg_path), overrides)
    todo = _expand_what(what, c)
    base = Path(out).resolve() if out else cfg_path.parent
    base.mkdir(parents=True, exist_ok=True)
    eff = base / "render.effective.json"
    eff.write_text(json.dumps(c, indent=2, default=str))
    _log(f"preset {c['preset']} · look {c['look']} · lighting {c['lighting']} · {c['size']} · "
         f"outputs {', '.join(todo)}  (effective settings: {eff})")
    _check_display()
    t0 = time.time()
    extra = list(extra)

    if "stills" in todo or "turntable" in todo:
        from zview.render import vtk_render
        argv = rc.turntable_argv(c, base / "renders", stills="stills" in todo,
                                 turntable="turntable" in todo) + extra
        _log("vtk_render " + " ".join(argv))
        vtk_render.main(argv)
    if "tour" in todo:
        if not c.get("source"):
            raise SystemExit("tour needs 'source' (the full-resolution data) in the config")
        from zview.render import tour
        argv = rc.tour_argv(c, base / "tour") + extra
        _log("tour " + " ".join(argv))
        tour.main(argv)
    if "surface" in todo:
        from zview.render import surface
        argv = rc.surface_argv(c, eff, base / "surface") + extra
        _log("surface " + " ".join(argv))
        surface.main(argv)
    _log(f"done in {time.time() - t0:.0f}s — outputs in {base}")
    return base


# ------------------------------------------------------------------ commands

def _cmd_presets(a):
    if a.save:
        if not a.from_config:
            raise SystemExit("--save needs --from CONFIG")
        cfg = rc.load(a.from_config)
        keep = {k: v for k, v in cfg.items()
                if k not in ("source", "volume", "hdf5_dataset", "factor", "pixel_size_um", "voxel_size_um",
                             "regions", "z_range", "cylinder", "_help") and v is not None}
        keep["_description"] = a.description or f"saved from {a.from_config}"
        f = rc.save_user_preset(a.save, keep)
        _log(f"saved preset {a.save!r} -> {f}")
        return
    if a.name:
        print(json.dumps(rc.load_preset(a.name), indent=2))
        return
    print("Presets (combine several: --preset generic --preset presentation-4k):")
    for name, desc in rc.available_presets().items():
        print(f"  {name:18s} {desc}")
    print(f"\nUser presets live in {rc.user_preset_dir()}/NAME.json")


def _cmd_looks(_a):
    from zview.render import looks as lk
    print("Looks (--set look=NAME, or any matplotlib colormap):")
    for name, lo in lk.LOOKS.items():
        m = lo["material"]
        print(f"  {name:10s} curve={lo['curve']:8s} specular={m['specular']:.2f}")
    print("\nOpacity curves (--set curve=NAME): solid, soft, layered, glass, mip")
    print("Lighting rigs (--set lighting=NAME): " + ", ".join(lk.LIGHTING))
    print("Finishing: --set tone_mapping=filmic|none  supersample=2  exposure=1.0  background=look|black|white|grey")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return
    cmd, rest = argv[0], argv[1:]

    if cmd == "cpu":
        from zview.render import cpu
        return cpu.main(rest)
    if cmd == "napari":
        from zview.render import napari_render
        return napari_render.main(rest)
    if cmd == "prepare":
        from zview.render import prepare
        return prepare.main(rest)

    extra = []
    if "--" in rest:
        i = rest.index("--")
        rest, extra = rest[:i], rest[i + 1:]

    if cmd in ("run", "quick"):
        p = argparse.ArgumentParser(prog=f"zview-render {cmd}")
        p.add_argument("target", type=Path, help="render.json" if cmd == "run" else "source data")
        p.add_argument("what", nargs="*", help=f"{', '.join(WHAT)}, all")
        p.add_argument("--preset", action="append", default=None, help="preset name/file (repeatable)")
        p.add_argument("--set", dest="sets", action="append", default=[], metavar="KEY=VALUE",
                       help="override any setting, e.g. --set look=gold --set tour.speed=0.5")
        p.add_argument("--out", type=Path, default=None, help="output folder (default: next to the config)")
        if cmd == "quick":
            p.add_argument("--max-axis", type=int, default=None)
            p.add_argument("--step", type=int, default=None)
            p.add_argument("--dataset", default=None)
        a = p.parse_intermixed_args(rest)
        overrides = rc.parse_set(a.sets)
        if a.preset:
            overrides["preset"] = a.preset if len(a.preset) > 1 else a.preset[0]
        if cmd == "quick":
            from zview.render import prepare as rp
            if a.out is None:
                raise SystemExit("quick needs --out DIR")
            preset = overrides.get("preset") or "generic"
            cfg_path = rp.prepare(a.target, a.out, preset=preset, max_axis=a.max_axis, step=a.step,
                                  dataset=a.dataset)
            return run(cfg_path, a.what, overrides, None, extra)
        return run(a.target, a.what, overrides, a.out, extra)

    if cmd == "presets":
        p = argparse.ArgumentParser(prog="zview-render presets")
        p.add_argument("name", nargs="?")
        p.add_argument("--save", metavar="NAME")
        p.add_argument("--from", dest="from_config", type=Path)
        p.add_argument("--description", default=None)
        return _cmd_presets(p.parse_args(rest))
    if cmd == "looks":
        return _cmd_looks(None)

    # Compatibility with zarr-auto-render's pipeline3d.py: CONFIG [what].
    if cmd.endswith(".json"):
        return run(cmd, [w for w in rest if not w.startswith("-")], None, None,
                   [w for w in rest if w.startswith("-")] + extra)
    raise SystemExit(f"unknown command {cmd!r}\n\n{__doc__}")


if __name__ == "__main__":
    main()
