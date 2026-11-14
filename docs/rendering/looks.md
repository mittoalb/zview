# Looks, lighting and finishing

Render quality comes from four independent choices, each one a config key:

| Key | Chooses | Values |
|---|---|---|
| `look` | colour stops, default opacity curve, material, background | see below, or any matplotlib colormap |
| `curve` | how opacity rises with intensity | `solid`, `soft`, `layered`, `glass`, `mip` |
| `lighting` | light directions and colours, shadow strength, shading | `studio`, `soft`, `dramatic`, `top`, `flat` |
| finishing | exposure, tone mapping, supersampling | `tone_mapping`, `exposure`, `supersample` |

All images on this page are renders of `phantom_iso.zarr` from the
[test data](../test-data.md). Each one shows a look in the context it suits.

## Looks

### Solid surfaces

For the outer surface of a sample, with the `generic` preset: the window
starts just above the background and the `solid` curve gives a crisp,
opaque boundary.

::::{grid} 2 3 3 3
:gutter: 2

:::{grid-item-card} organic
:img-top: ../_static/img/looks/organic.jpg
Muted tissue tones (the original pipeline's palette).
:::
:::{grid-item-card} tissue
:img-top: ../_static/img/looks/tissue.jpg
Warmer, slightly saturated flesh.
:::
:::{grid-item-card} bone
:img-top: ../_static/img/looks/bone.jpg
Classic CT ivory. The default.
:::
:::{grid-item-card} ivory
:img-top: ../_static/img/looks/ivory.jpg
Lighter, glossier.
:::
:::{grid-item-card} grays
:img-top: ../_static/img/looks/grays.jpg
Neutral grey.
:::
::::

### Metals

Dark base colours and tight, strong highlights. On surfaces they use a
metallic PBR material.

::::{grid} 2 3 3 3
:gutter: 2

:::{grid-item-card} gold
:img-top: ../_static/img/looks/gold.jpg
:::
:::{grid-item-card} copper
:img-top: ../_static/img/looks/copper.jpg
:::
::::

### Interiors and translucency

Shown with the `interior` preset, where the window starts at the 70th
percentile, so the matrix becomes see-through and dense structures stand out.

::::{grid} 2 3 3 3
:gutter: 2

:::{grid-item-card} ember
:img-top: ../_static/img/looks/ember.jpg
Black-body glow.
:::
:::{grid-item-card} jade
:img-top: ../_static/img/looks/jade.jpg
:::
:::{grid-item-card} ice (glass curve)
:img-top: ../_static/img/looks/ice.jpg
Glassy blue. Surfaces come from gradient opacity.
:::
:::{grid-item-card} coolwarm (layered)
:img-top: ../_static/img/looks/coolwarm.jpg
Two materials: a translucent cool phase and an opaque warm phase.
:::
:::{grid-item-card} xray (flat)
:img-top: ../_static/img/looks/xray.jpg
Emission/absorption only.
:::
:::{grid-item-card} viridis
:img-top: ../_static/img/looks/viridis.jpg
:::
:::{grid-item-card} inferno
:img-top: ../_static/img/looks/inferno.jpg
:::
:::{grid-item-card} magma
:img-top: ../_static/img/looks/magma.jpg
:::
:::{grid-item-card} plasma
:img-top: ../_static/img/looks/plasma.jpg
:::
:::{grid-item-card} cividis
:img-top: ../_static/img/looks/cividis.jpg
:::
:::{grid-item-card} turbo
:img-top: ../_static/img/looks/turbo.jpg
:::
::::

The perceptual maps (viridis, inferno, magma, plasma, cividis, turbo) are
sampled from matplotlib over their upper 85 %, so the darkest end doesn't
disappear into the background. Any other matplotlib colormap name also works
as a look.

## Opacity curves

The curve maps the windowed 0–255 volume to per-voxel opacity, relative to
`mask_level` (*ml*) and `opacity` (*op*):

| Curve | Shape | Use |
|---|---|---|
| `solid` | 0 below ml/2, ramps to *op* by 4·ml | Opaque material with a crisp boundary |
| `soft` | gentle ramp from ml to 255 | Seeing into a material |
| `layered` | ≈0.005·*op* plateau, then a step to *op* above ~165 | Two phases: a translucent matrix and opaque inclusions |
| `glass` | faint everywhere; gradient opacity makes surfaces | Glassy, porous materials |
| `mip` | only the brightest values | Bright-structure emphasis |

:::{tip}
Opacity accumulates along each ray, voxel by voxel. A phase hundreds of
voxels thick needs a per-voxel alpha of about 0.005 to stay see-through.
That is why `layered` uses such small values for its low phase.
:::

## Lighting rigs

Each rig is a set of camera-relative lights plus a volumetric-shadow
strength (`scatter`, which you can override). The same ivory grains under each
rig:

::::{grid} 2 3 3 3
:gutter: 2

:::{grid-item-card} studio
:img-top: ../_static/img/lighting/studio.jpg
Three-point: warm key above right, cool weak fill, white rim. The default.
:::
:::{grid-item-card} soft
:img-top: ../_static/img/lighting/soft.jpg
A big soft box with open shadows: fine detail, publications.
:::
:::{grid-item-card} dramatic
:img-top: ../_static/img/lighting/dramatic.jpg
Low side key, little fill, strong cool rim, deep shadows.
:::
:::{grid-item-card} top
:img-top: ../_static/img/lighting/top.jpg
From above, like a microscope ring light.
:::
:::{grid-item-card} flat
:img-top: ../_static/img/lighting/flat.jpg
No shading: X-ray or MIP-like.
:::
::::

## Finishing: tone mapping and anti-aliasing

The renderer draws at `supersample`× the output size and averages the result
down. It then applies `exposure` and, with `tone_mapping: filmic`, an ACES
curve on linearised values: midtones gain contrast and highlights roll off
smoothly instead of clipping. Crops of the same view:

::::{grid} 3
:gutter: 2

:::{grid-item-card} raw
:img-top: ../_static/img/finish/raw.jpg
`tone_mapping: none`, `supersample: 1`
:::
:::{grid-item-card} filmic
:img-top: ../_static/img/finish/filmic.jpg
`tone_mapping: filmic`
:::
:::{grid-item-card} filmic + 2× supersampling
:img-top: ../_static/img/finish/filmic_ss2.jpg
The defaults.
:::
::::

Other knobs: `sample_distance` (ray step in voxels; smaller is finer and
slower), `specular` (default: from the look), `gi_reach` and `anisotropy`
(shadow rays), `background` (`look`, `black`, `white`, `grey`).

## Custom looks and rigs

`look` and `lighting` also accept JSON objects in a config, merged over
`bone` and `studio` respectively:

```json
{
  "look": {
    "colors": [[0, [0.05, 0.10, 0.25]], [128, [0.30, 0.65, 0.85]], [255, [1.0, 1.0, 1.0]]],
    "curve": "soft",
    "material": {"ambient": 0.25, "diffuse": 0.8, "specular": 0.3, "specular_power": 30},
    "background": [[0, 0, 0], [0.05, 0.07, 0.12]]
  },
  "lighting": {
    "lights": [[[1.0, 1.2, 0.8], [1.0, 0.95, 0.9], 1.0], [[-1.0, 0.0, 0.7], [0.8, 0.85, 1.0], 0.3]],
    "scatter": 1.2
  }
}
```

Save a combination you like as a reusable preset with
`zview-render presets --save NAME --from render.json`, or **Save as preset…**
in the [render dialog](dialog.md).
