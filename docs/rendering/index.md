# Rendering overview

```{image} ../_static/img/hero_metal.jpg
:class: hero
:alt: Gold-look render of dense grains and vessels
```

`zview-render` makes presentation-quality images and videos of any volume.
It runs offline (no interaction), so it can afford effects that the
interactive view can't:

- **GPU ray casting** with VTK, using jittered sampling so there is no
  wood-grain banding.
- **Volumetric shadows** (VTK scattering) and multi-light **rigs**.
- **Looks**: colour stops, an opacity curve and a matching material
  (ambient, diffuse, specular).
- **Filmic tone mapping** (an ACES curve on linearised values) so highlights
  roll off instead of clipping, and **supersampling** for clean edges.
- **Lit surfaces** with screen-space ambient occlusion and physically based
  materials.
- **Guided tour videos** that cross-fade into the real full-resolution
  slices.

## Workflow

The renderer works in three steps:

1. **prepare** reads a render-sized copy of the data: the finest pyramid
   level whose longest side fits `max_axis` (768 by default), plus a stride
   if needed. It writes `render.json`, the cached volume and
   `side_views_grid.png`.
2. **edit** the config, or pass `--set key=value`: presets, look, lighting,
   window, crop, title, tour regions…
3. **run** renders the requested outputs: stills, turntable, tour, surface.

```console
$ zview-render prepare DATA --out WORK --preset generic
$ $EDITOR WORK/render.json
$ zview-render run WORK/render.json stills turntable
```

or all at once:

```console
$ zview-render quick DATA --out WORK --preset interior stills turntable
```

`DATA` can be a Zarr (v2/v3) or N5 store, an HDF5 file or a folder of TIFF
slices, of any size. HDF5 files are row-sampled, so even multi-TB files
prepare in minutes.

## How settings combine

Lowest priority first:

```text
DEFAULTS  <  preset(s)  <  values in render.json  <  --set key=value
```

`null` in `render.json` means "not set": the preset decides. Presets combine
from left to right, so a base preset plus a quality modifier is common:

```console
$ zview-render run WORK/render.json --preset metal --preset presentation-4k
```

The exact settings used are saved next to the outputs as
`render.effective.json`.

## Preprocessing

There are two preprocessing modes:

`none` (default; any data)
: Windows the intensities between percentiles (or between Otsu/2 and a high
  percentile), after an optional Gaussian `smooth` that gives noisy data clean
  surface normals. NaNs are tolerated, and `cylinder` and `z_range` crop.

`tomo` (a sample in a container)
: The original tomography pipeline: per-slice medium removal (Otsu), radial
  cupping correction, automatic container-wall fit (or `cylinder`), Z
  brightness equalisation, and optionally keeping only the largest connected
  object to drop debris.

After windowing, the volume is 8-bit, and `mask_level` (default 20 of 255)
marks where material starts for the opacity curves.

## Where to go next

- [](looks.md): the looks, lighting rigs, opacity curves and finishing, with
  a gallery.
- [](presets.md): ready-made combinations.
- [](outputs.md): stills, turntables, tour videos and surfaces.
- [](dialog.md): rendering from the viewer.
- [](config.md): every config key.
- [](cli.md): command reference.
