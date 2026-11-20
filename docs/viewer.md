# The viewer

```console
$ zview [PATH] [--layout single|dual|quad|3d] [--cache-mb N]
```

`PATH` is a Zarr (v2 or v3) or N5 store, or an HDF5 file (`scan.h5`, or
`scan.h5:/exchange/data` to name the dataset). You can also open a store
with **File › Open** (Ctrl+O), an HDF5 file with **File › Open HDF5 file**
(Ctrl+Shift+O), either from **File › Open Recent**, or by dropping it onto
the window. zview remembers the window geometry, the panel layout and the
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
- **HDF5 files** with a single resolution (see below).

## HDF5 files

An HDF5 file usually holds one full-resolution dataset and no pyramid. zview
builds the pyramid on the fly: each coarser level takes **one voxel every 2,
4, 8, …** along each axis, read straight from the file with a strided
selection. Nothing is averaged and nothing is written, so a file of any size
opens at once and only the sampled planes and rows are read. Levels stop
when the longest side is 256 or less. The *Dataset* panel shows
"1 stored + strided", and each virtual level's tooltip in the *Pyramid* panel
gives its stride.

Which dataset
: `/exchange/data`, `/data`, `/volume` or `/entry/data/data` if present,
  otherwise the largest 3D dataset. If the file holds several volumes of the
  same rank (for example `data`, `data_white` and `data_dark`), zview asks,
  with the default preselected. Name it on the command line to skip the
  question: `zview scan.h5:/exchange/data_white`. **Open Recent** remembers
  the dataset.

Voxel size
: Read from the attributes `element_size_um`, `voxel_size_um`, `voxel_size`,
  `pixel_size_um`, `pixel_size`, `resolution` or `spacing` (dataset first,
  then file; one value or `[z, y, x]`, in µm), or from the DXchange
  `/measurement/instrument/detection_system/objective/resolution`. Without
  them, voxels are 1 × 1 × 1. Anisotropic data keeps its coarse axis: with
  4 µm z and 1 µm y/x, only y and x are strided until they reach 4 µm.

Speed and file layout
: **Uncompressed contiguous** datasets (the usual reconstruction output) are
  read directly, not through the HDF5 library: zview computes each sampled
  row's position in the file and reads the rows in parallel. On network
  storage this is about 10× faster than HDF5's own strided reads, which fetch
  one row at a time. A strided level still has to read whole rows, so it reads
  *file size / (z stride × y stride)*: for a 100 GB file, 0.4 GB at stride
  16 and 6 GB at stride 4. The level's tooltip in the *Pyramid* panel shows the
  figure. Set `ZVIEW_H5_DIRECT=0` to go through h5py instead.
: The 3D pane's automatic level therefore also caps what it reads from an
  HDF5 file: it picks the finest level that fits the GPU **and** reads at
  most 2 GB (`ZVIEW_3D_MAX_READ_GB`; the status-line tooltip says when the
  cap applied). A click on a level in the *Pyramid* panel renders it anyway.
: Files with small chunks are also fast at every level. With
  chunks that hold **whole slices**, often gzip-compressed (a common
  tomography layout), any read from a slice decodes all of it, so the first
  XZ or YZ view at a level decodes one slice per row. zview keeps those slices,
  already strided, in memory, so later XZ/YZ views at any position are
  instant. Expect a few seconds on open for such files. For repeated work on
  very large files, convert them to OME-Zarr once ([](conversion.md)).

Label maps, the 3D pane, the level picker and **Render 3D…** all work as with
Zarr. A render job gets the file as `source` and the dataset as
`hdf5_dataset`.

```{image} _static/img/ui/quad_h5.jpg
:class: shot
:alt: A contiguous HDF5 file open in 4 views: the Dataset panel reads "2 (1 stored + strided)" and level L1 is chosen for 3D in the Pyramid panel
```

*`phantom_scan.h5` from the [test data](test-data.md): one stored level plus a
strided one. L1 was clicked in the Pyramid panel (●), so the 3D pane renders
it.*

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
: Every level, with the panes that use it. **Click a level to render it in
  3D** at full resolution of that level, instead of the automatic choice. The
  choice is marked with ● and kept across reloads; **Auto** (or changing the
  quality menu of the 3D pane) goes back to the automatic level. Levels too
  large for this machine are greyed out: the GPU texture limit, half the free
  VRAM and a quarter of free RAM are checked. Each level's tooltip shows its
  GPU memory, the RAM needed while loading, how much data it reads and, if it
  doesn't fit, why.

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

The 3D pane renders a copy of the volume sized for the machine (see
[](performance.md#3d-budget)): on an 11 GB GPU up to 1024³, and 256³ on a
software renderer. If even the coarsest pyramid level is too big, the level is
read with a stride (with [label-aware pooling](labels.md) for label maps). A
small preview appears first when that is much cheaper to read. The quality
menu (Auto, Preview, Low, Medium, High, Max) overrides the budget, a click
on a level in the *Pyramid* panel renders exactly that level (if it fits),
**✕** cancels a load, and the tooltip on the status line explains which limit
applied.

There are two engines (**View › 3D engine**, applied on restart):

VTK, high quality (the default when VTK is installed)
: The same rendering as `zview-render`: [looks, lighting rigs](rendering/looks.md),
  volumetric shadows, opacity curves and the filmic finish. The toolbar holds
  the **preset**, **look** and **lighting** menus. **⚙** opens the intensity
  window, opacity, shadow strength, ray step, smoothing, box, slice planes and
  background. Rendering is progressive: a draft frame while you drag, then
  full quality, then 2× supersampling once you pause.
  **Shadows** (toolbar) switches volumetric shadows on and off. They are off by
  default: shadows give far more realistic depth, but on dense data each
  full-quality frame takes a second or more (without them, tens of
  milliseconds). The choice is remembered, the strength is set in **⚙ ›
  Shadows**, and offline jobs (**▶ job**) always render with shadows.
  Drag rotates, Shift+drag pans, Ctrl+drag rolls, the wheel zooms and a
  double-click resets. **📷** saves a still of the current view at any size,
  and **▶ job** opens [Render 3D…](rendering/dialog.md) with this pane's look,
  window and camera.

VisPy, fast and minimal
: MIP / attenuated MIP / MinIP / translucent / additive / average /
  isosurface with a colormap, contrast and gamma. It needs no VTK.

VTK renders off-screen (preferring EGL on the GPU), and the frames are drawn in
the pane, so it works the same on any driver, without a GL window embedded in
Qt.

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
| Ctrl+Shift+O | Open an HDF5 file |
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
