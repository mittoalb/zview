# Outputs

Choose outputs on the command line (`zview-render run cfg.json stills turntable`)
or with the config's `outputs` list. `all` means stills, turntable and tour
(plus surface when `style` is `surface`).

## Stills — `renders/stills_<look>/`

`azimuths` views (8 by default) evenly spaced around the sample, at each of
`elevations` (default: `elev`, 45°, −20°, 0°). Files are named
`az<deg>_el<deg>.png`. `zoom` moves the camera closer (1 frames the whole
volume).

## Turntable — `renders/turntable_<look>.mp4`

A 360° rotation in `frames` frames at `fps` (H.264). Add `--keep-frames` (pass
it after `--`) to also keep every frame as a PNG.

```{image} ../_static/img/turntable.gif
:alt: Turntable of the interior preset
```

## Tour — `tour/tour.mp4`

A guided video, made for tomography:

1. **overview**: the sample turns half way round, with the title and
   subtitle;
2. **sections**: a cutting plane sweeps through, showing the interior with
   the look's palette;
3. **for each region**: a horizontal cut at the region; the camera tilts down
   to face the cut, which cross-fades into the **real full-resolution slices**
   read from the source. It scrolls through neighbouring slices with a slow
   zoom and a scale bar, then backs out;
4. **outro**: the rest of the turn.

Regions are full-resolution centres in the config. In the viewer, add them
from the crosshair:

```json
"regions": [{"name": "Helix", "z": 256, "y": 256, "x": 384},
            {"name": "Resolution target", "z": 120, "y": 350, "x": 200}]
```

::::{grid} 2 2 2 2
:gutter: 2

:::{grid-item-card} Overview
:img-top: ../_static/img/tour/overview.jpg
:::
:::{grid-item-card} Virtual sections
:img-top: ../_static/img/tour/section.jpg
:::
:::{grid-item-card} Close-up: native slices
:img-top: ../_static/img/tour/helix_real.jpg
:::
:::{grid-item-card} Close-up: resolution target
:img-top: ../_static/img/tour/resolution_target_real.jpg
:::
::::

Tour settings live under `tour`: `speed` (below 1 gives a shorter video),
`scroll` (native slices per close-up), `zoom` (zoom reached in a close-up) and
`region_size` (full-res pixels shown). `style: surface` gives the tour a lit
surface. `credit` is drawn bottom right on every frame. The tour reads full
resolution from any [source format](index.md#workflow) and needs `source` in
the config.

## Surface — `surface/`

A smoothed isosurface of the material at `mask_level`, coloured by the
intensity just beneath the surface (through the look's palette). It's lit by
the rig with screen-space ambient occlusion and a physically based material
(metallic for `gold`/`copper`). Writes `stills/` and `turntable_surface.mp4`.

::::{grid} 2
:gutter: 2

:::{grid-item-card} metal, gold
:img-top: ../_static/img/surface/metal.jpg
:::
:::{grid-item-card} generic, bone
:img-top: ../_static/img/surface/generic.jpg
:::
::::

Settings live under `surface`: `ao` (occlusion radius as a fraction of the
size; 0 turns it off), `shadows` (hard shadow map), `pbr`, `roughness`.

## Physical proportions

Anisotropic voxels are rendered with their real proportions. The voxel size
comes from the metadata (`voxel_size_um`), combined with the per-axis step of
the cached volume:

```{image} ../_static/img/aniso.jpg
:width: 420px
:alt: Anisotropic dataset rendered with the correct aspect ratio
```

*`phantom_aniso_5d.zarr`, 4 µm z versus 1 µm in-plane.*

## Side views — `side_views_grid.png`

`prepare` also writes XZ and YZ sections through the centre with a
full-resolution pixel grid. Read `z_range`, `cylinder` and tour regions off
it:

```{image} ../_static/img/side_views_grid.jpg
:alt: XZ and YZ sections with a full-resolution grid
```
