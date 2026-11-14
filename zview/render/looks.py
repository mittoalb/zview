"""Looks (colour + opacity + material), lighting rigs and image finishing.

A *look* describes how a scalar volume (already scaled to 0..255, with the
"material starts here" threshold at `mask_level`) is turned into colour and
opacity, plus the surface material used by the shading:

    colors     list of (value 0..255, (r, g, b)) stops, or a matplotlib name
    curve      opacity curve: solid | soft | layered | glass | mip
    material   ambient / diffuse / specular / specular_power (Phong terms)
    background two gradient colours (bottom, top)

A *lighting rig* is a set of camera-relative lights plus how strong the
volumetric shadows (VTK scattering) are. Together with filmic tone mapping
and supersampled anti-aliasing (`finish`) they give the soft, studio-lit look
of commercial renderers (Dragonfly, Amira, Imaris).

All tables are plain data, so presets and user JSON files can override any
part of them.
"""

import numpy as np

# --------------------------------------------------------------------- looks

LOOKS = {
    # Muted, light tissue tones (the original zarr-auto-render palette).
    "organic": dict(
        colors=[(0, (0.40, 0.27, 0.24)), (60, (0.66, 0.50, 0.45)), (120, (0.80, 0.66, 0.60)),
                (180, (0.88, 0.77, 0.71)), (255, (0.94, 0.87, 0.81))],
        curve="solid", material=dict(ambient=0.20, diffuse=0.85, specular=0.06, specular_power=8),
        background=((0.0, 0.0, 0.0), (0.13, 0.13, 0.16))),
    # Warmer, slightly saturated flesh — soft biological samples.
    "tissue": dict(
        colors=[(0, (0.45, 0.16, 0.14)), (70, (0.78, 0.42, 0.36)), (140, (0.93, 0.66, 0.58)),
                (210, (0.98, 0.84, 0.76)), (255, (1.00, 0.94, 0.88))],
        curve="solid", material=dict(ambient=0.18, diffuse=0.85, specular=0.10, specular_power=12),
        background=((0.02, 0.02, 0.03), (0.16, 0.15, 0.18))),
    # Bone / ivory — the classic CT look; good default for unknown data.
    "bone": dict(
        colors=[(0, (0.30, 0.24, 0.20)), (60, (0.62, 0.55, 0.47)), (130, (0.86, 0.80, 0.70)),
                (200, (0.96, 0.93, 0.86)), (255, (1.00, 0.99, 0.96))],
        curve="solid", material=dict(ambient=0.20, diffuse=0.85, specular=0.18, specular_power=20),
        background=((0.0, 0.0, 0.0), (0.12, 0.13, 0.16))),
    "ivory": dict(
        colors=[(0, (0.55, 0.50, 0.42)), (120, (0.92, 0.89, 0.80)), (255, (1.00, 1.00, 0.97))],
        curve="solid", material=dict(ambient=0.25, diffuse=0.80, specular=0.25, specular_power=30),
        background=((0.05, 0.05, 0.06), (0.22, 0.23, 0.26))),
    # Metals: darker base, strong tight highlights.
    "gold": dict(
        colors=[(0, (0.25, 0.14, 0.03)), (80, (0.62, 0.42, 0.10)), (160, (0.88, 0.70, 0.28)),
                (255, (1.00, 0.93, 0.65))],
        curve="solid", material=dict(ambient=0.12, diffuse=0.70, specular=0.55, specular_power=40, metallic=0.7),
        background=((0.0, 0.0, 0.0), (0.10, 0.09, 0.08))),
    "copper": dict(
        colors=[(0, (0.22, 0.07, 0.03)), (90, (0.60, 0.27, 0.13)), (180, (0.86, 0.52, 0.34)),
                (255, (0.98, 0.80, 0.66))],
        curve="solid", material=dict(ambient=0.12, diffuse=0.72, specular=0.45, specular_power=32, metallic=0.6),
        background=((0.0, 0.0, 0.0), (0.10, 0.09, 0.09))),
    "jade": dict(
        colors=[(0, (0.02, 0.18, 0.16)), (90, (0.10, 0.48, 0.42)), (180, (0.45, 0.80, 0.70)),
                (255, (0.85, 0.98, 0.92))],
        curve="soft", material=dict(ambient=0.22, diffuse=0.80, specular=0.30, specular_power=24),
        background=((0.0, 0.01, 0.01), (0.08, 0.12, 0.13))),
    # Translucent glassy blue — interiors and porous materials.
    "ice": dict(
        colors=[(0, (0.03, 0.08, 0.22)), (80, (0.15, 0.38, 0.70)), (170, (0.55, 0.80, 0.95)),
                (255, (0.95, 0.99, 1.00))],
        curve="glass", material=dict(ambient=0.30, diffuse=0.70, specular=0.40, specular_power=40),
        background=((0.0, 0.0, 0.02), (0.06, 0.08, 0.14))),
    # Black-body "hot iron" — dense inclusions glow.
    "ember": dict(
        colors=[(0, (0.10, 0.00, 0.00)), (70, (0.60, 0.05, 0.02)), (140, (0.95, 0.40, 0.05)),
                (210, (1.00, 0.78, 0.30)), (255, (1.00, 0.97, 0.85))],
        curve="soft", material=dict(ambient=0.30, diffuse=0.75, specular=0.10, specular_power=10),
        background=((0.0, 0.0, 0.0), (0.06, 0.05, 0.06))),
    # Two materials: low phase translucent cool, dense phase opaque warm.
    "coolwarm": dict(
        colors=[(0, (0.15, 0.30, 0.65)), (90, (0.45, 0.65, 0.90)), (130, (0.85, 0.85, 0.85)),
                (175, (0.95, 0.60, 0.35)), (255, (0.80, 0.18, 0.10))],
        curve="layered", material=dict(ambient=0.22, diffuse=0.80, specular=0.20, specular_power=20),
        background=((0.0, 0.0, 0.0), (0.12, 0.12, 0.14))),
    # X-ray-like grey, best with the "flat" rig and glass/mip curve.
    "xray": dict(
        colors=[(0, (0.0, 0.0, 0.0)), (255, (1.0, 1.0, 1.0))],
        curve="glass", material=dict(ambient=1.0, diffuse=0.0, specular=0.0, specular_power=1),
        background=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))),
    "grays": dict(
        colors=[(0, (0.08, 0.08, 0.08)), (255, (1.0, 1.0, 1.0))],
        curve="solid", material=dict(ambient=0.20, diffuse=0.85, specular=0.10, specular_power=12),
        background=((0.0, 0.0, 0.0), (0.13, 0.13, 0.16))),
}

# Perceptual maps sampled from matplotlib at render time (upper 85 % so the
# darkest end doesn't vanish into the background).
for _name in ("viridis", "inferno", "magma", "plasma", "cividis", "turbo"):
    LOOKS[_name] = dict(colors=_name, curve="soft",
                        material=dict(ambient=0.22, diffuse=0.80, specular=0.15, specular_power=16),
                        background=((0.0, 0.0, 0.0), (0.10, 0.10, 0.12)))


def _maybe_json(v):
    if isinstance(v, str) and v.lstrip().startswith("{"):
        import json
        return json.loads(v)
    return v


def get_look(name_or_dict):
    """A look by name, a dict / JSON object (merged over 'bone'), or any matplotlib map name."""
    name_or_dict = _maybe_json(name_or_dict)
    if isinstance(name_or_dict, dict):
        base = dict(LOOKS["bone"])
        base.update(name_or_dict)
        return base
    if name_or_dict in LOOKS:
        return LOOKS[name_or_dict]
    try:
        import matplotlib
        matplotlib.colormaps[name_or_dict]
    except Exception:
        raise ValueError(f"unknown look / colormap {name_or_dict!r}; "
                         f"choose from {', '.join(sorted(LOOKS))} or a matplotlib name") from None
    look = dict(LOOKS["viridis"])
    look["colors"] = name_or_dict
    return look


def color_stops(look, n=16):
    """[(value, (r, g, b)), ...] for a look (samples matplotlib maps)."""
    cols = look["colors"]
    if isinstance(cols, str):
        import matplotlib
        cm = matplotlib.colormaps[cols]
        return [(float(x), tuple(cm(0.15 + 0.85 * x / 255)[:3])) for x in np.linspace(0, 255, n)]
    return [(float(x), tuple(c)) for x, c in cols]


def color_lut(look, n=256):
    """(n, 3) float array: colour of each 0..255 value."""
    stops = color_stops(look)
    xs = np.array([s[0] for s in stops])
    cs = np.array([s[1] for s in stops])
    v = np.linspace(0, 255, n)
    return np.stack([np.interp(v, xs, cs[:, k]) for k in range(3)], axis=1)


def opacity_points(curve, mask_level, opacity):
    """Scalar-opacity control points (value, alpha) on the 0..255 volume."""
    ml = float(mask_level)
    op = float(opacity)
    if curve == "solid":      # crisp material boundary, opaque interior
        return [(0, 0), (ml * 0.5, 0), (ml * 2, 0.5 * op), (ml * 4, op), (255, op)]
    if curve == "soft":       # gentle ramp: see into the material
        return [(0, 0), (ml, 0), (128, 0.35 * op), (255, op)]
    if curve == "layered":    # translucent low phase, opaque dense phase
        # Opacity is per voxel and accumulates along the ray: a phase that is
        # hundreds of voxels thick needs per-sample alpha of ~0.005 to stay see-through.
        return [(0, 0), (ml, 0), (ml * 2, 0.004 * op), (140, 0.008 * op), (165, 0.5 * op),
                (190, op), (255, op)]
    if curve == "glass":      # faint everywhere; surfaces come from gradient opacity
        return [(0, 0), (ml, 0), (255, 0.25 * op)]
    if curve == "mip":        # emphasises the brightest structures
        return [(0, 0), (160, 0), (255, op)]
    raise ValueError(f"unknown opacity curve {curve!r} (solid, soft, layered, glass, mip)")


def gradient_points(curve, interior):
    """Gradient-opacity points: surfaces (high gradient) vs uniform interiors."""
    if curve == "glass":
        return [(0, 0.05), (10, 0.4), (40, 1.0), (255, 1.0)]
    return [(0, float(interior)), (6, 1.0), (255, 1.0)]


# ------------------------------------------------------------------ lighting

# Each light: (camera-relative direction, rgb, intensity). "scatter" scales the
# VTK volumetric-shadow blending; "ambient_boost" is added to the material.
LIGHTING = {
    # Three-point studio: warm key above right, cool weak fill, white rim.
    "studio": dict(lights=[((1.0, 1.3, 0.9), (1.0, 0.95, 0.88), 0.90),
                           ((-1.2, 0.0, 0.8), (0.85, 0.90, 1.00), 0.18),
                           ((0.0, 0.9, -1.0), (1.0, 1.0, 1.0), 0.35)],
                   scatter=1.0, ambient_boost=0.0, shade=True),
    # Big soft box: open shadows, gentle — fine detail, publications.
    "soft": dict(lights=[((0.6, 1.0, 1.0), (1.0, 0.98, 0.95), 0.65),
                         ((-1.0, 0.3, 0.9), (0.95, 0.97, 1.00), 0.45),
                         ((0.0, -0.6, 1.0), (1.0, 1.0, 1.0), 0.20),
                         ((0.0, 0.9, -1.0), (1.0, 1.0, 1.0), 0.25)],
                 scatter=0.6, ambient_boost=0.10, shade=True),
    # Low side key, almost no fill, strong cool rim: sculptural, deep shadows.
    "dramatic": dict(lights=[((1.4, 0.6, 0.5), (1.0, 0.92, 0.80), 1.15),
                             ((-1.0, 0.0, 0.6), (0.80, 0.85, 1.00), 0.14),
                             ((-0.4, 0.8, -1.0), (0.75, 0.85, 1.00), 0.70)],
                     scatter=1.6, ambient_boost=-0.08, shade=True),
    # Light from above, like a microscope ring light.
    "top": dict(lights=[((0.0, 1.6, 0.4), (1.0, 1.0, 1.0), 0.95),
                        ((0.0, -0.5, 1.0), (0.95, 0.95, 1.00), 0.25)],
                scatter=1.2, ambient_boost=0.0, shade=True),
    # No shading: pure emission/absorption (X-ray, MIP-like).
    "flat": dict(lights=[((0.0, 0.0, 1.0), (1.0, 1.0, 1.0), 1.0)],
                 scatter=0.0, ambient_boost=0.0, shade=False),
}


def get_lighting(name_or_dict):
    """A rig by name, or a dict / JSON object merged over 'studio'."""
    name_or_dict = _maybe_json(name_or_dict)
    if isinstance(name_or_dict, dict):
        base = dict(LIGHTING["studio"])
        base.update(name_or_dict)
        return base
    if name_or_dict not in LIGHTING:
        raise ValueError(f"unknown lighting {name_or_dict!r}; choose from {', '.join(LIGHTING)}")
    return LIGHTING[name_or_dict]


# ------------------------------------------------------------- image finish

def _aces(x):
    """Narkowicz ACES filmic fit on linear values."""
    return np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0)


def finish(rgb, exposure=1.0, tone_mapping="filmic", supersample=1):
    """Downsample a supersampled frame and apply exposure + tone mapping.

    rgb: (H*s, W*s, 3) uint8/float display values from the renderer.
    tone_mapping: "filmic" (ACES curve on linearised values — highlights roll
    off instead of clipping, midtones gain contrast), or "none" (plain
    exposure multiply, the original behaviour).
    """
    im = rgb.astype(np.float32)
    s = int(supersample)
    if s > 1:
        h, w = im.shape[0] // s, im.shape[1] // s
        im = im[:h * s, :w * s].reshape(h, s, w, s, 3).mean(axis=(1, 3))
    if tone_mapping == "filmic":
        lin = (im / 255.0) ** 2.2 * float(exposure) * 1.6   # 1.6: keep mid-grey at ~ its input level
        im = _aces(lin) ** (1 / 2.2) * 255.0
    else:
        im = im * float(exposure)
    return np.clip(im + 0.5, 0, 255).astype(np.uint8)
