# The viewer

```console
$ zview [PATH] [--layout single|dual|quad|3d] [--cache-mb N]
```

`PATH` is a Zarr (v2 or v3) or N5 store. You can also open one with
**File › Open** (Ctrl+O), from **File › Open Recent**, or by dropping a folder
onto the window. zview remembers the window geometry, the panel layout and the
last layout mode.

## Supported data

- **OME-NGFF 0.4 and 0.5** `multiscales` (0.5 nests it under an `ome` key),
  including per-level `scale` and `translation`.
- **Numbered levels** (`0/`, `1/`, …) with no metadata. Downsample factors
  are inferred from the shapes.
- **2D to 5D arrays.** For `(t, c, z, y, x)` and `(c, z, y, x)` the leading
  axes are pinned to index 0, and a 2D image becomes a one-slice volume.
- **Any dtype**: unsigned and signed integers, and float (NaNs are
  tolerated).
- **Anisotropic voxels.** Panes and the 3D view use the physical voxel size
  from the metadata (toggle with **View › Physical voxel aspect**).

## Layouts

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} 4 views (Ctrl+3)
```{image} _static/img/ui/quad.jpg
```
XY | 3D over XZ | YZ, arranged so neighbours share an axis: XY and XZ share
x, and XZ and YZ share z (z runs down in both). Double-click a pane header
to maximise it.
:::

:::{grid-item-card} 2 views (Ctrl+2)
```{image} _static/img/ui/dual.jpg
```
Two panes. Each header has a plane selector (XY, XZ or YZ).
:::

:::{grid-item-card} 1 view (Ctrl+1)
```{image} _static/img/ui/single_zoom.jpg
```
One large pane with any plane. Zooming in loads finer pyramid levels; the
header shows the level and its downsample factor.
:::

:::{grid-item-card} 3D (Ctrl+4)
```{image} _static/img/ui/view3d.jpg
```
The 3D pane full-window with all its settings: render mode, colormap,
contrast, gamma, iso level, step size, box and slice planes.
:::

::::

Each plane has one colour throughout the program: XY red, XZ green, YZ blue.
It's used for the pane's header accent, for the crosshair line that marks the
plane in the other panes, and for its outline in 3D.

## Controls panel

The dock on the right (Ctrl+D toggles it) is shared by all layouts:

Dataset
: Name, shape, dtype, chunks, number of levels, voxel size.

Position
: One `(z, y, x)` position for every layout, as sliders and spin boxes.
  Switching layouts keeps it.

Display
: One histogram and lookup table for all panes. Drag the handles to set the
  contrast; right-click the colour bar to pick a colormap. **Auto** (or the
  **A** key) sets the contrast from the pixels currently visible, using the
  chosen percentile preset. Right after opening a dataset the contrast is
  refined automatically once the panes have loaded real resolution, unless
  you've already changed it.

Pyramid
: Every level, with the ones in use highlighted and the panes that use them.

System
: CPU count, free RAM, GPU, I/O threads, chunk cache and the automatic 3D
  budget.

The status bar shows the `(z, y, x)` coordinate under the cursor, the pixel
value, the level in use and the cache statistics.

## How loading works

Each pane computes which part of the plane is visible and the pyramid level
that samples it at about one voxel per screen pixel. It then fetches only the
chunks that cover that region, in parallel. The fetch runs on a background
thread, so panning and slicing never block. A newer request supersedes an
older one. On open, each pane shows the coarsest level at once and then
refines.

Level selection uses the pane's own axes (XZ uses z and x), so pyramids that
keep z at full resolution don't pick an over-coarse level. Small pans don't
flip between levels: there is a one-octave dead zone around the current level.

## The 3D pane

The 3D pane renders a low-resolution copy of the volume with VisPy. Its size
is chosen for the machine (see [](performance.md#3d-budget)): on an 11 GB GPU
that is up to 1024³, and 256³ on a software renderer. If even the coarsest
pyramid level is too big, the level is read with a stride. A small preview
appears first when that is much cheaper to read. The quality menu (Auto,
Preview, Low, Medium, High, Max) overrides the budget, **✕** cancels a load,
and the tooltip on the status line explains which limit applied.

```{image} _static/img/ui/quad_aniso.jpg
:class: shot
:alt: Anisotropic 5D dataset: XZ and YZ panes are drawn with physical proportions and the 3D box is elongated accordingly
```

*An anisotropic `(t, c, z, y, x)` dataset with 4 µm z voxels and 1 µm in-plane.
The XZ and YZ panes and the 3D box use physical proportions.*

## Keyboard and mouse

| Key | Action |
|---|---|
| Ctrl+O / Ctrl+W | Open / close dataset |
| Ctrl+Shift+R | Render 3D… (offline renderer) |
| Ctrl+1 … Ctrl+4 | 1 view / 2 views / 4 views / 3D |
| PgUp / PgDn | Previous / next slice of the pane under the mouse (Shift: ×10) |
| Home / End | First / last slice |
| R / Shift+R | Reset the pane under the mouse / all panes |
| A | Auto contrast from what is visible |
| Ctrl+D | Show / hide the controls panel |
| F11 / Esc | Full screen / leave full screen |

| Mouse | Action |
|---|---|
| Wheel | Zoom (2D and 3D) |
| Ctrl+wheel | Scroll through slices |
| Drag | Pan (2D); rotate (3D; Shift+drag pans) |
| Click | Move the crosshair |
| Double-click a pane header | Maximise / restore (4 views) |

## Plugins

The second toolbar holds plugin actions. Plugins call
`viewer.add_plugin_button(text, slot, icon=None, tooltip="", checkable=False)`,
and remove buttons with `remove_plugin_button(action)` or
`clear_plugin_buttons()`.
