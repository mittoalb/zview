# zview

```{image} _static/img/hero.jpg
:class: hero
:alt: Volume render of a procedural phantom: glowing inclusions and vessels inside a translucent container
```

**zview** is a viewer and renderer for large 3D image volumes: multi-resolution
**Zarr (v2 and v3)** and **N5** stores, single-resolution HDF5 files (viewed
through a strided virtual pyramid), and TIFF stacks for rendering. It is built for tomography and microscopy data that runs from gigabytes
to many terabytes.

It has two halves:

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Interactive viewer — `zview`
:link: viewer
:link-type: doc

Orthogonal slicing with automatic pyramid-level selection, asynchronous
chunk-aligned loading, linked crosshairs, shared contrast, label maps, and an
interactive VTK 3D view with the renderer's looks and lighting, sized to your
machine's GPU and RAM.
:::

:::{grid-item-card} Offline renderer — `zview-render`
:link: rendering/index
:link-type: doc

Presentation-quality stills, turntables and guided tour videos with
VTK: curated looks, studio lighting rigs, volumetric shadows, filmic tone
mapping and anti-aliasing. Driven by presets and JSON configs, from the
command line or from the viewer.
:::

::::

```{image} _static/img/ui/quad.jpg
:class: shot
:alt: The zview viewer in the 4-view layout with XY, XZ, YZ slices, a 3D pane and the controls panel
```

## Highlights

- **Any size.** Only the chunks you can see are read, in parallel and off the
  UI thread. 3D views and renders pick the finest pyramid level that fits,
  and fall back to strided reads when there is no pyramid.
- **Adapts to the machine.** I/O threads, cache size and the 3D texture
  budget come from the CPU count, free RAM and the GPU.
- **Many data layouts.** OME-NGFF 0.4 and 0.5 metadata, 2D to 5D arrays
  (`t`/`c` axes), anisotropic voxels, and numbered levels without metadata.
- **Label maps.** Segmentations are detected and shown as classes: class
  colours in 2D, per-class shaded 3D with a class panel, and label-aware
  downsampling that keeps vessels and cells. [More](labels.md).
- **Render quality.** Looks pair colour, opacity and material, and lighting
  rigs go with them. Supersampling and an ACES filmic curve finish the image.
  [See the gallery](rendering/looks.md).
- **Configs you can reproduce.** Every render is a JSON config made of
  presets plus overrides, and it runs the same on a headless GPU node.
- **Tomography tools.** HDF5/TIFF to sharded OME-Zarr v3 conversion, and
  blood-vessel segmentation and rendering.

```{toctree}
:maxdepth: 2
:hidden:
:caption: Getting started

installation
quickstart
test-data
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Viewer

viewer
labels
performance
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Rendering

rendering/index
rendering/looks
rendering/presets
rendering/outputs
rendering/dialog
rendering/batch
rendering/config
rendering/cli
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Tomography tools

conversion
vessels
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Reference

troubleshooting
api
citing
```
