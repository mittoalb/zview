# Label maps

A **label map** (a segmentation or annotation volume) stores *class IDs*,
not intensities: 3 = grey matter, 70 = vessel lumen, 255 = outside, and so
on. zview treats such data as a separate type, because the usual intensity
tools give misleading results on it:

- **Windowing and colormaps** assume "higher = denser". For IDs the order is
  arbitrary; a background coded 255 would render as the most solid thing.
- **Interpolation** between neighbouring IDs invents classes that aren't
  there: between 3 and 70 a ray samples 5, 7, 9… and draws fake shells.
- **Stride downsampling** keeps one voxel in N, so thin or small classes
  (vessels, cells) turn into speckle or vanish at coarse scales.

```{image} _static/img/ui/quad_labels.jpg
:class: shot
:alt: The label phantom: class colours in the 2D panes, per-class 3D rendering
```

## Detection

A dataset is treated as a label map when:

1. its metadata names the classes: OME-NGFF `image-label`
   (`colors` / `properties`), or a `lookup`, `labels`, `label_names`
   or `class_names` table in the attributes; or
2. it's an integer volume with few distinct values (≤ 32 in a sample)
   **and** it's piecewise constant (neighbouring voxels nearly always equal).
   Low-dynamic-range 8-bit images fail the second test because of their noise.

**View › Treat as label map** overrides the detection either way.

Each class gets a name (from the metadata), a colour, an opacity and a
visibility. The defaults follow the names and sizes:

| Class | Default |
|---|---|
| background / void / outside / unlabelled | hidden |
| bulk classes (white matter, matrix, tissue…, or > 15 % of the volume) | faint glass (6 % opacity) |
| everything else | opaque; colour from the name (vessel → red, nucleus → blue, soma → yellow…) or a distinct colour |

## In the viewer

**2D panes** show class colours (a categorical colormap), the status bar names
the class under the cursor, and the contrast controls are disabled (they have
no meaning for IDs).

**3D (VTK engine)** renders labels with a label-map mask. Classes are sampled
nearest-neighbour, so no in-between classes appear. A smooth shading field gives
every visible class lit surfaces, including the boundaries *between* classes.
The **Classes** panel lists every class with its share of the volume:

- tick to show or hide a class (**All**, **None**, **Defaults** for all of them);
- click the swatch to change the colour, and set the opacity (below 35 % the
  class is drawn as glass);
- **Smooth surfaces** switches the shading field's smoothing on or off.

```{image} _static/img/ui/view3d_labels.jpg
:class: shot
:alt: 3D label rendering with the class panel; matrix and wall hidden
```

**Label-aware downsampling.** When the 3D volume has to be read at a stride,
each output voxel takes the *highest-priority* class present in its cell.
Priority is rarity, with background last, so thin vessels and small cells
survive. They're slightly dilated rather than lost, as they would be with
stride sampling, or outvoted, as they would be with a majority vote. The same
pooling is used when preparing offline renders.

## Offline renders

`zview-render prepare` detects label maps, reads them with label-aware pooling,
and writes `"preset": "labels"`, `"preprocess": "labels"` and the class table
into `render.json`. Edit the table to change what's shown:

```json
"labels": {
  "3":   {"name": "matrix",         "color": [0.85, 0.70, 0.62], "opacity": 0.06, "visible": true, "role": "bulk"},
  "21":  {"name": "container wall", "color": [0.75, 0.55, 0.35], "opacity": 1.0,  "visible": false},
  "70":  {"name": "vessel lumen",   "color": [0.90, 0.12, 0.10], "opacity": 1.0,  "visible": true},
  "255": {"name": "void",           "color": [0, 0, 0],          "opacity": 0.0,  "visible": false, "role": "background"}
}
```

**▶ job** in the 3D pane carries the class table, together with the look,
lighting and camera, into the [render dialog](rendering/dialog.md) and job
scripts.

::::{grid} 3
:gutter: 2

:::{grid-item-card} Default classes
:img-top: _static/img/labels/default.jpg
The opaque container wall encloses everything.
:::
:::{grid-item-card} Wall and matrix hidden
:img-top: _static/img/labels/inside.jpg
Grains, pores, vessels, helix, markers.
:::
:::{grid-item-card} Tour cut face
:img-top: _static/img/labels/tour_section.jpg
Sections are coloured by class; tour close-ups show the native slices in
class colours.
:::
::::

Stills, turntables and tours support label maps. The `surface` output (meshes)
doesn't yet.

## Authoring label data

- **Build the pyramid with pooling, not stride subsampling**: keep rare classes
  (max over priorities) or use a majority vote, otherwise small structures
  disappear from the coarse levels used by the viewer and by 3D.
- **Add enough levels** that the coarsest is a few hundred voxels per side, so
  3D loads read a few MB instead of decoding a whole level.
- **Shard (Zarr v3).** Millions of small chunk files in one directory make
  listing and opening slow on network filesystems.
- **Name the classes** in the metadata (`image-label` or a `lookup` table) so
  zview can pick sensible colours and hide the background.
- **Translations:** with stride subsampling, level *L* voxel *i* is exactly
  level-0 voxel *i·2^L*, so the per-level translation should be 0. The
  half-voxel offsets of mean-downsampled pyramids would shift the levels.
