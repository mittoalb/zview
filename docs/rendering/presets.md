# Presets

A preset is a partial config: it sets only the keys it cares about. Combine
presets from left to right, typically one **base** preset for what to show plus
one **modifier** for output quality:

```console
$ zview-render run render.json --preset interior --preset presentation-4k
```

In `render.json` the same thing is written `"preset": ["interior", "presentation-4k"]`.
`zview-render presets` lists them all, and `zview-render presets NAME` shows
one.

## Base presets

::::{grid} 2 3 3 3
:gutter: 2

:::{grid-item-card} generic
:img-top: ../_static/img/presets/generic.jpg
Any data, no assumptions. Percentile window 1–99.8 %, light smoothing, the
outer surface solid, `bone` look, studio light.
:::
:::{grid-item-card} interior
:img-top: ../_static/img/presets/interior.jpg
Dense structures inside a matrix. Window 70–99.9 %, `ember` look, soft
curve and soft light.
:::
:::{grid-item-card} multiphase
:img-top: ../_static/img/presets/multiphase.jpg
A translucent low-density phase over an opaque dense phase. `coolwarm` look,
`layered` curve.
:::
:::{grid-item-card} metal
:img-top: ../_static/img/presets/metal.jpg
Only the densest ~7 % (window 93–99.95 %), `gold` look, studio light.
:::
:::{grid-item-card} xray
:img-top: ../_static/img/presets/xray.jpg
A flat X-ray-like projection: `xray` look, `glass` curve, no shading or
shadows.
:::
::::

`labels`
: [Label maps](../labels.md): class colours from the metadata, background
  hidden, bulk classes as faint glass, smooth per-class shading. `prepare`
  selects it automatically for label data.

`tomo-brain`
: The original zarr-auto-render pipeline for a sample (such as a brain) in a
  container: `tomo` preprocessing (medium removal, wall fit, Z equalisation),
  keep the largest object, Otsu window, `organic` look. It caches the volume
  at a fixed step of 8 and renders it 2× downsampled. Outputs: stills,
  turntable, tour.

`tomo-surface`
: `tomo-brain` as a lit surface with ambient occlusion.

## Quality modifiers

| Modifier | Sets |
|---|---|
| `draft` | 960×540, no supersampling, no shadows, coarser rays, 120 frames, 4 stills per elevation: fast checks |
| `presentation-4k` | 3840×2160, 2× supersampling, finer rays, 720-frame (24 s) turntable |
| `publication` | 2400×2400 square stills on white, 3× supersampling, soft light, 4 views at 25° elevation |

```{image} ../_static/img/presets/publication.jpg
:width: 360px
:alt: metal + publication: gold grains on a white background
```

*`metal` + `publication`.*

## Your own presets

User presets are JSON files in `~/.config/zview/render_presets/NAME.json`
(this follows `$XDG_CONFIG_HOME`). They take precedence over built-ins with
the same name. A preset may hold any [config key](config.md), plus
`_description` for the listings:

```json
{
  "_description": "our beamline house style",
  "look": "tissue",
  "lighting": "soft",
  "background": "white",
  "credit": "Imaging group, Example Institute",
  "size": "2560x1440"
}
```

Create one from a config you're happy with:

```console
$ zview-render presets --save house --from render.json --description "our house style"
```

or with **Save as preset…** in the [render dialog](dialog.md). A path to a JSON
file also works anywhere a preset name does.
