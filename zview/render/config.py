"""Render configuration: defaults, presets, merging, and argv translation.

A render config is a JSON file (`render.json`, written by `zview-render
prepare`; the old `dataset.json` from zarr-auto-render works too). The values
actually used are, lowest priority first::

    DEFAULTS  <  preset(s) named in the config  <  config values  <  --set k=v

`null` in the config means "not set" (the preset decides). `preset` may be a
single name or a list, applied left to right, e.g.
``"preset": ["generic", "presentation-4k"]``. Names are built-ins (`PRESETS`), files in ~/.config/zview/render_presets/
(`NAME.json`), or a path to a JSON file.
"""

import copy
import json
import os
from pathlib import Path

DEFAULTS = {
    # --- dataset (filled in by `prepare`)
    "source": None,            # original data: Zarr/N5 store, HDF5 file or TIFF folder
    "hdf5_dataset": None,
    "volume": None,            # cached downsampled volume (.npy)
    "volume_source": None,     # source the cached volume was made from (set by zview)
    "volume_max_axis": None,   # max_axis it was made with (set by zview)
    "factor": 8,               # .npy step vs full resolution: number or [z, y, x]
    "pixel_size_um": 1.0,      # x voxel size, for scale bars
    "voxel_size_um": None,     # [z, y, x] full-res voxel size (anisotropic data)

    # --- what to make
    "preset": "generic",
    "outputs": ["stills", "turntable"],   # stills, turntable, tour, surface

    # --- preprocessing
    "preprocess": "none",      # none (any data) | tomo (sample in a tube) | labels (segmentation / class IDs)
    "labels": None,            # label maps: {value: {name, color, opacity, visible}} (default: from metadata)
    "label_smooth": 1.0,       # label maps: shading smoothness (voxels)
    "window": None,            # percentile | otsu (default: percentile for none, otsu for tomo)
    "window_values": None,     # [lo, hi] absolute window (overrides lo_pct/hi_pct; preprocess none)
    "lo_pct": 1.0,
    "hi_pct": 99.8,
    "z_range": None,           # [Z0, Z1] full-res slices to keep
    "cylinder": None,          # [CY, CX, R] or [CY_TOP, CX_TOP, CY_BOT, CX_BOT, R], full-res px
    "keep_largest": False,     # keep only the largest connected object (drops debris)
    "open_iter": 6,
    "mask_level": 20,          # material threshold on the 0..255 windowed volume
    "smooth": 0.0,             # Gaussian pre-smoothing (voxels) before windowing; cleaner normals
    "down": 1,                 # extra downsampling of the cached volume when rendering

    # --- look (see zview.render.looks)
    "look": "bone",            # colour / opacity / material preset or matplotlib colormap
    "lighting": "studio",      # studio | soft | dramatic | top | flat
    "curve": None,             # opacity curve (default: from the look)
    "opacity": 0.6,
    "interior": 1.0,
    "scatter": None,           # volumetric shadows (default: from the lighting rig)
    "gi_reach": 0.4,
    "anisotropy": 0.0,
    "specular": None,          # default: from the look
    "exposure": None,          # default: 1.0 (filmic) / 1.35 (none)
    "tone_mapping": "filmic",  # filmic | none
    "supersample": 2,
    "sample_distance": 0.5,
    "background": "look",      # look | black | white | grey

    # --- camera / output
    "size": "1920x1080",
    "fps": 30,
    "frames": 360,             # turntable frames
    "elev": 20,
    "zoom": 1.0,               # camera zoom for stills/turntable (>1 = closer)
    "elevations": None,        # still elevations (default: elev, 45, -20, 0)
    "azimuths": 8,
    "azimuth": 0.0,            # start azimuth of stills / turntable (degrees)

    # --- text overlays (tour / surface)
    "title": "",
    "subtitle": "",
    "credit": "",

    # --- tour
    "regions": [],             # [{"name": .., "z": .., "y": .., "x": ..}] full-res centres
    "style": "volume",         # volume | surface
    "tour": {"speed": 1.0, "scroll": 150, "zoom": 1.8, "region_size": 800},

    # --- surface
    "surface": {"ao": 0.03, "shadows": False, "pbr": True, "roughness": 0.65},

    # --- prepare
    "max_axis": 768,           # prepare: largest cached-volume side (pyramid level + stride)
}

PRESETS = {
    # Any volume, no assumptions: windowed by percentiles, the material's outer
    # surface rendered solid in bone tones under studio light.
    "generic": dict(preprocess="none", smooth=0.8, window="percentile", lo_pct=1.0, hi_pct=99.8,
                    look="bone", lighting="studio", keep_largest=False, down=1),
    # See dense structures inside a less dense matrix (inclusions, grains, vessels).
    "interior": dict(preprocess="none", smooth=0.8, window="percentile", lo_pct=70.0, hi_pct=99.9,
                     look="ember", curve="soft", lighting="soft", opacity=0.8),
    # Two phases: translucent low-density phase, opaque dense phase.
    "multiphase": dict(preprocess="none", smooth=0.8, window="percentile", lo_pct=20.0, hi_pct=99.9,
                       look="coolwarm", curve="layered", lighting="soft", opacity=0.9),
    # Metallic / mineral grains.
    "metal": dict(preprocess="none", smooth=0.8, window="percentile", lo_pct=93.0, hi_pct=99.95,
                  look="gold", lighting="studio"),
    # Label maps (segmentations): class colours from the metadata, background
    # hidden, bulk classes as faint glass, smooth per-class shading.
    "labels": dict(preprocess="labels", look="bone", lighting="soft", label_smooth=1.0),
    # X-ray-like projection look.
    "xray": dict(preprocess="none", look="xray", curve="glass", lighting="flat", scatter=0.0,
                 tone_mapping="none", exposure=1.6),
    # The original zarr-auto-render pipeline: brain (or other sample) in a
    # container — medium removal, wall fit, Z equalisation, debris removal.
    "tomo-brain": dict(preprocess="tomo", window="otsu", lo_pct=60.0, hi_pct=99.7, look="organic",
                       lighting="studio", keep_largest=True, down=2, factor=8, max_axis=None,
                       outputs=["stills", "turntable", "tour"]),
    "tomo-surface": dict(preprocess="tomo", window="otsu", hi_pct=99.7, look="organic", lighting="studio",
                         keep_largest=True, down=2, style="surface", outputs=["surface"]),
    # Output-quality modifiers (combine with one of the above).
    "draft": dict(size="960x540", supersample=1, frames=120, scatter=0.0, sample_distance=1.0, azimuths=4),
    "presentation-4k": dict(size="3840x2160", supersample=2, frames=720, sample_distance=0.35),
    "publication": dict(size="2400x2400", supersample=3, background="white", lighting="soft",
                        outputs=["stills"], elevations=[25], azimuths=4),
}

PRESET_HELP = {
    "generic": "any data; outer surface, bone tones, studio light",
    "interior": "dense structures inside a matrix, glowing, soft light",
    "multiphase": "translucent low phase + opaque dense phase",
    "metal": "metallic grains, dramatic light",
    "xray": "flat X-ray-like look",
    "labels": "label maps / segmentations: per-class colours, background hidden",
    "tomo-brain": "original pipeline: sample in a container (medium removal, wall fit)",
    "tomo-surface": "tomo-brain as a lit surface with ambient occlusion",
    "draft": "modifier: small and fast",
    "presentation-4k": "modifier: 4K, smooth 24 s turntable",
    "publication": "modifier: square high-res stills on white",
}


def user_preset_dir():
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "zview" / "render_presets"


def available_presets():
    """{name: description} for built-ins and user preset files."""
    out = dict(PRESET_HELP)
    d = user_preset_dir()
    if d.is_dir():
        for f in sorted(d.glob("*.json")):
            try:
                desc = json.loads(f.read_text()).get("_description", "user preset")
            except Exception:
                desc = "user preset (unreadable)"
            out[f.stem] = desc
    return out


def load_preset(name):
    if isinstance(name, dict):
        return dict(name)
    p = Path(str(name)).expanduser()
    if p.suffix == ".json" and p.is_file():
        return json.loads(p.read_text())
    f = user_preset_dir() / f"{name}.json"
    if f.is_file():
        return json.loads(f.read_text())
    if name in PRESETS:
        return copy.deepcopy(PRESETS[name])
    raise KeyError(f"unknown preset {name!r}; available: {', '.join(available_presets())}")


def save_user_preset(name, values):
    d = user_preset_dir()
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{name}.json"
    f.write_text(json.dumps(values, indent=2))
    return f


def _merge(dst, src):
    for k, v in src.items():
        if k.startswith("_") or v is None:
            continue
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            dst[k] = _merge(dict(dst[k]), v)
        else:
            dst[k] = copy.deepcopy(v)
    return dst


def effective(cfg, overrides=None):
    """Merged config: DEFAULTS < presets < cfg < overrides."""
    out = copy.deepcopy(DEFAULTS)
    presets = (overrides or {}).get("preset") or cfg.get("preset") or "generic"
    if isinstance(presets, (str, dict)):
        presets = [presets]
    for name in presets:
        _merge(out, load_preset(name))
    _merge(out, cfg)
    _merge(out, overrides or {})
    out["preset"] = presets if len(presets) > 1 else presets[0]
    return out


def load(path):
    path = Path(path)
    cfg = json.loads(path.read_text())
    if "cmap" in cfg and cfg.get("look") is None:     # zarr-auto-render dataset.json
        cfg["look"] = cfg.pop("cmap")
    # Relative paths in a config are relative to the config file.
    for key in ("volume", "source"):
        v = cfg.get(key)
        if v and not Path(v).is_absolute() and not str(v).startswith(("http:", "https:", "s3:")):
            cfg[key] = str((path.parent / v).resolve())
    return cfg


def parse_set(items):
    """['look=gold', 'frames=120', 'tour.speed=0.5', 'z_range=[100,900]'] -> nested dict."""
    out = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        try:
            val = json.loads(raw)
        except json.JSONDecodeError:
            val = raw
        d = out
        parts = key.strip().split(".")
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = val
    return out


# ------------------------------------------------------------ argv building

def _style_arg(v):
    """Look / lighting as a CLI value: a name, or a JSON object for custom ones."""
    return json.dumps(v) if isinstance(v, dict) else str(v)


def _factor_args(f):
    return [str(v) for v in (f if isinstance(f, (list, tuple)) else [f])]


def common_argv(c):
    """Arguments shared by vtk_render and tour (see vtk_render.add_common_args)."""
    a = [str(c["volume"]), "--factor", *_factor_args(c["factor"]), "--size", c["size"],
         "--look", _style_arg(c["look"]), "--lighting", _style_arg(c["lighting"]),
         "--tone-mapping", c["tone_mapping"],
         "--supersample", str(c["supersample"]), "--sample-distance", str(c["sample_distance"]),
         "--background", c["background"], "--preprocess", c["preprocess"],
         "--lo-pct", str(c["lo_pct"]), "--hi-pct", str(c["hi_pct"]),
         "--opacity", str(c["opacity"]), "--interior", str(c["interior"]),
         "--gi-reach", str(c["gi_reach"]), "--anisotropy", str(c["anisotropy"]),
         "--mask-level", str(c["mask_level"]), "--open-iter", str(c["open_iter"]),
         "--down", str(c["down"]), "--fps", str(c["fps"]), "--smooth", str(c.get("smooth") or 0)]
    for key, flag in (("curve", "--curve"), ("scatter", "--scatter"), ("specular", "--specular"),
                      ("exposure", "--exposure"), ("window", "--window")):
        if c.get(key) is not None:
            a += [flag, str(c[key])]
    if c.get("z_range"):
        a += ["--z-range", *[str(int(v)) for v in c["z_range"]]]
    if c.get("cylinder"):
        a += ["--cylinder", *[str(v) for v in c["cylinder"]]]
    if c.get("preprocess") == "labels":
        a += ["--label-smooth", str(c.get("label_smooth", 1.0))]
        if c.get("labels"):
            a += ["--labels-json", json.dumps(c["labels"])]
    if c.get("window_values"):
        a += ["--window-values", *[str(v) for v in c["window_values"]]]
    if c.get("voxel_size_um"):
        a += ["--voxel-size", *[str(v) for v in c["voxel_size_um"]]]
    if not c.get("keep_largest"):
        a += ["--no-keep-largest"]
    return a


def turntable_argv(c, out, stills=True, turntable=True):
    a = common_argv(c) + ["--out", str(out), "--frames", str(c["frames"] if turntable else 0),
                          "--elev", str(c["elev"]), "--azimuths", str(c["azimuths"]),
                          "--zoom", str(c.get("zoom") or 1.0), "--azimuth", str(c.get("azimuth") or 0.0)]
    if c.get("elevations"):
        a += ["--elevations", *[str(v) for v in c["elevations"]]]
    if not stills:
        a += ["--no-stills"]
    return a


def tour_argv(c, out):
    t = c.get("tour") or {}
    a = common_argv(c) + ["--h5", str(c["source"]), "--out", str(out),
                          "--pixel-size", str(c.get("pixel_size_um") or 1.0),
                          "--speed", str(t.get("speed", 1.0)), "--scroll", str(t.get("scroll", 150)),
                          "--zoom", str(t.get("zoom", 1.8)), "--region-size", str(t.get("region_size", 800)),
                          "--style", c.get("style", "volume")]
    if c.get("hdf5_dataset"):
        a += ["--dataset", c["hdf5_dataset"]]
    for key, flag in (("title", "--title"), ("subtitle", "--subtitle"), ("credit", "--credit")):
        if c.get(key):
            a += [flag, c[key]]
    for r in c.get("regions") or []:
        a += ["--region", str(r["name"]), str(int(r["z"])), str(int(r["y"])), str(int(r["x"]))]
    return a


def surface_argv(c, cfg_path, out):
    s = c.get("surface") or {}
    a = [str(cfg_path), "--out", str(out), "--frames", str(c["frames"]), "--elev", str(c["elev"]),
         "--ao", str(s.get("ao", 0.03)), "--roughness", str(s.get("roughness", 0.65)),
         "--look", _style_arg(c["look"]), "--lighting", _style_arg(c["lighting"]),
         "--tone-mapping", c["tone_mapping"],
         "--supersample", str(c["supersample"]), "--background", c["background"], "--size", c["size"]]
    if s.get("shadows"):
        a += ["--shadows"]
    if not s.get("pbr", True):
        a += ["--no-pbr"]
    if c.get("exposure") is not None:
        a += ["--exposure", str(c["exposure"])]
    return a


# ------------------------------------------------------------ validation

_OUTPUTS = {"stills", "turntable", "tour", "surface"}
_TOUR_KEYS = {"speed", "scroll", "zoom", "region_size"}
_SURFACE_KEYS = {"ao", "shadows", "pbr", "roughness"}
_META_KEYS = {"_help", "_description", "_comment"}


def validate(cfg, overrides=None):
    """Problems in a render config, as a list of human-readable strings.

    Checks the raw config (unknown or misspelled keys, which would otherwise
    be ignored silently) and the merged settings (names of presets, looks,
    lighting rigs and curves, value shapes, what each output needs).
    """
    from zview.render import looks as lk
    problems = []

    def close(key, pool):
        import difflib
        m = difflib.get_close_matches(key, pool, n=1)
        return f" (did you mean {m[0]!r}?)" if m else ""

    for src in (cfg, overrides or {}):
        for k, v in src.items():
            if k in _META_KEYS or k.startswith("_"):
                continue
            if k not in DEFAULTS:
                problems.append(f"unknown key {k!r}{close(k, DEFAULTS)} — it is ignored")
            elif k == "tour" and isinstance(v, dict):
                problems += [f"unknown key tour.{t!r}{close(t, _TOUR_KEYS)}" for t in v if t not in _TOUR_KEYS]
            elif k == "surface" and isinstance(v, dict):
                problems += [f"unknown key surface.{t!r}{close(t, _SURFACE_KEYS)}" for t in v if t not in _SURFACE_KEYS]
    try:
        c = effective(cfg, overrides)
    except KeyError as e:
        return problems + [str(e).strip('"')]

    def check(cond, msg):
        if not cond:
            problems.append(msg)

    look = c.get("look")
    if isinstance(look, str):
        try:
            lk.get_look(look)
        except ValueError as e:
            problems.append(str(e))
    light = c.get("lighting")
    if isinstance(light, str) and light not in lk.LIGHTING:
        problems.append(f"unknown lighting {light!r}; choose from {', '.join(lk.LIGHTING)}")
    check(c.get("curve") in (None, "solid", "soft", "layered", "glass", "mip"),
          f"unknown curve {c.get('curve')!r}; use solid, soft, layered, glass or mip")
    check(c.get("preprocess") in ("none", "tomo", "labels"),
          f"preprocess must be none, tomo or labels, not {c.get('preprocess')!r}")
    check(c.get("tone_mapping") in ("filmic", "none"), "tone_mapping must be filmic or none")
    check(c.get("background") in ("look", "black", "white", "grey"), "background must be look, black, white or grey")
    outs = c.get("outputs") or []
    check(isinstance(outs, list) and all(o in _OUTPUTS for o in outs),
          f"outputs must be a list of {sorted(_OUTPUTS)}, got {outs!r}")
    size = str(c.get("size", ""))
    check(len(size.lower().split("x")) == 2 and all(x.isdigit() for x in size.lower().split("x")),
          f"size must look like '1920x1080', got {size!r}")
    zr = c.get("z_range")
    check(zr is None or (isinstance(zr, list) and len(zr) == 2 and zr[0] < zr[1]),
          f"z_range must be [Z0, Z1] with Z0 < Z1, got {zr!r}")
    cyl = c.get("cylinder")
    check(cyl is None or (isinstance(cyl, list) and len(cyl) in (3, 5)),
          f"cylinder must be [CY, CX, R] or [CY_TOP, CX_TOP, CY_BOTTOM, CX_BOTTOM, R], got {cyl!r}")
    wv = c.get("window_values")
    check(wv is None or (isinstance(wv, list) and len(wv) == 2 and wv[0] < wv[1]),
          f"window_values must be [lo, hi] with lo < hi, got {wv!r}")
    for i, r in enumerate(c.get("regions") or []):
        check(isinstance(r, dict) and all(k in r for k in ("name", "z", "y", "x")),
              f"regions[{i}] needs name, z, y, x (full-resolution voxels), got {r!r}")
    if "tour" in outs:
        check(bool(c.get("source")), "the tour needs 'source' (the full-resolution data)")
    if not c.get("volume") and not c.get("source"):
        problems.append("set 'source' (the data) — or 'volume' (a prepared .npy)")
    labels = c.get("labels")
    if labels is not None:
        check(isinstance(labels, dict) and all(str(k).lstrip("-").isdigit() and isinstance(v, dict)
                                               for k, v in labels.items()),
              "labels must map class values (as strings) to {name, color, opacity, visible}")
        if c.get("preprocess") != "labels":
            problems.append("'labels' is set but preprocess is not 'labels' (use the labels preset)")
    return problems
