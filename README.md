# zview

A viewer and renderer for large 3D volumes: multi-resolution Zarr (v2 and v3)
and N5, single-resolution HDF5 files (viewed through a strided virtual
pyramid), plus TIFF stacks for rendering.

- **`zview`**: an interactive viewer (PyQt5 + pyqtgraph) with chunk-aligned,
  parallel, asynchronous loading, automatic pyramid-level selection, linked
  orthogonal views, label-map support (segmentations shown as classes), and a
  3D view sized to the machine: VTK with the renderer's looks and lighting, or
  a fast VisPy fallback.
- **`zview-render`**: an offline, preset-driven 3D renderer (VTK) for
  presentation-quality stills, turntables, guided tour videos and lit
  surfaces, with curated looks, lighting rigs, volumetric shadows, filmic tone
  mapping and anti-aliasing. It runs from the command line or from the viewer
  (**File › Render 3D…**).
- **`zview-convert`** (HDF5/TIFF to sharded OME-Zarr v3) and **`zview-vessels`**
  (blood-vessel segmentation and rendering), which came from zarr-auto-render.

Full documentation is in [`docs/`](docs/) (Sphinx; built by Read the Docs from
`.readthedocs.yaml`). Build it locally with
`pip install -e '.[docs]' && sphinx-build docs docs/_build/html`.

```bash
pip install -e .                 # viewer
pip install -e '.[render]'       # + renderer, conversion, vessels
zview /path/to/data.zarr --layout quad
zview-render quick /path/to/data.zarr --out /tmp/render --preset interior stills turntable
```

## Features

- **Multi-resolution pyramid support** — reads OME-NGFF v0.4/v0.5 `multiscales`
  metadata from Zarr v2, Zarr v3, or N5; also falls back to numbered
  subgroups (`0/`, `1/`, …) when no metadata is present. 2D–5D arrays are
  accepted (`t`/`c` axes are pinned to index 0).
- **HDF5 files without a pyramid** — coarser levels take one voxel every
  2, 4, 8, … straight from the file; uncompressed contiguous datasets are read
  with parallel direct reads (fast on network storage). `zview scan.h5` or
  `zview scan.h5:/exchange/data`.
- **Chunk-aligned, parallel, asynchronous loading** — only the chunks that
  cover the visible region are fetched, in parallel, on background threads,
  so panning and slicing never block the UI. Each pane shows the coarsest
  level instantly and then refines.
- **Automatic level selection with hysteresis** — per pane and per axis
  (XY uses Y/X, XZ uses Z/X, YZ uses Z/Y), with a 1-octave dead zone.
- **Four layouts**: 1 view, 2 views, 4 views (XY | 3D over XZ | YZ, so
  neighbouring panes share an axis), and full-window 3D. Single and dual
  panes can show any plane.
- **Low-resolution 3D volume rendering** (VisPy) that works for datasets of
  any size: it picks the finest pyramid level that fits this machine's GPU
  (3D-texture limit, free VRAM) and RAM, adds a stride if even the coarsest
  level is too large, loads a quick preview first, and shows the current
  slice planes. MIP, attenuated MIP, MinIP, translucent, additive, average
  and isosurface modes.
- **Adapts to the machine** — I/O thread count, chunk-cache size and 3D
  budget are derived from CPU count, free RAM and the GPU (see
  *Help › System info*).
- **Shared controls** — one position (z, y, x), one histogram/LUT and one
  colormap drive every pane; physical voxel aspect from metadata.
- **Status bar** with cursor coordinates and the pixel value under it.

## Install

```bash
conda env create -f environment.yml
conda activate zview
pip install -e .
```

`z5py` ships via conda-forge; everything else is handled by `pyproject.toml`.

## Usage

```bash
zview /path/to/dataset.zarr
zview /path/to/dataset.n5 --layout quad
zview /path/to/scan.h5     # HDF5; or scan.h5:/exchange/data to name the dataset
zview                      # then File › Open (Ctrl+O), Open HDF5 file (Ctrl+Shift+O), or drop it on the window
```

The last layout, window geometry and recent files are remembered.

### Keyboard

| Key                    | Action                                             |
|------------------------|----------------------------------------------------|
| **Ctrl+O / Ctrl+W**    | Open / close dataset                               |
| **Ctrl+1 … Ctrl+4**    | 1 view / 2 views / 4 views / 3D                    |
| **PgUp / PgDn**        | Previous / next slice of the pane under the mouse  |
| **Shift+PgUp/PgDn**    | Step ±10 slices                                    |
| **Home / End**         | First / last slice                                 |
| **R / Shift+R**        | Reset the pane under the mouse / all panes         |
| **A**                  | Auto contrast from what is currently visible       |
| **Ctrl+D**             | Show / hide the controls panel                     |
| **F11 / Esc**          | Full screen / leave full screen                    |

### Mouse

- **Wheel** — zoom (2D and 3D). **Ctrl+wheel** — scroll through slices.
- **Drag** — pan (2D); rotate (3D, Shift+drag pans).
- **Click** in a 2D pane — move the crosshair; the other panes and the 3D
  slice outlines follow. Crosshair lines are coloured like the plane they
  mark (XY red, XZ green, YZ blue).
- **Double-click a pane header** in the 4-view layout — maximize / restore.

## Environment variables

| Variable                 | Default            | Purpose |
|--------------------------|--------------------|---------|
| `ZVIEW_IO_THREADS`       | CPU count (4–32)   | Chunk-fetch concurrency. Raise on high-latency network storage. |
| `ZVIEW_CACHE_MB`         | 15% of free RAM (256–4096) | 2D chunk-cache size. Also settable in *View › Chunk cache size*. |
| `ZVIEW_3D_MAX_AXIS`      | from GPU           | Override the per-axis 3D texture cap. |
| `ZVIEW_3D_MAX_VOXELS`    | from GPU/RAM       | Override the total 3D texel budget (RAM/VRAM limits still apply). |
| `ZVIEW_NO_TRANSLATION`   | *unset*            | Set to `1` to ignore OME-NGFF per-level `translation` metadata if your dataset writes it in a non-standard way and levels visibly jump. |
| `ZVIEW_NO_GL_COMPAT`     | *unset*            | Set to `1` to disable the PyOpenGL library fix-up described below. |

### How the 3D budget is chosen

The texture is single-channel `uint8`. Its size is the tightest of: the
quality preset (Auto picks 1024³ for ≥ 8 GB VRAM, 768³ for ≥ 4 GB, 512³ for
≥ 2 GB, 256³ on software renderers), `GL_MAX_3D_TEXTURE_SIZE`, half the free
VRAM (when the driver reports it — NVIDIA/AMD) and a quarter of free RAM.
The finest pyramid level that fits is read; if none fits, the coarsest level
is read with a per-axis stride, skipping chunks that contain no sampled
voxel. The quality menu in the 3D pane overrides the preset.

### PyOpenGL in conda environments

VisPy uploads 3D textures through PyOpenGL. In conda environments PyOpenGL
can load the system `libOpenGL.so` while Qt uses the environment's copy,
which makes volumes render black or garbled depending on import order.
zview binds PyOpenGL to the same versioned libraries as Qt on import; set
`ZVIEW_NO_GL_COMPAT=1` to turn this off.

## Test dataset

A generator for a clean 6-level OME-NGFF test pyramid lives in
[scripts/make_test_zarr.py](scripts/make_test_zarr.py). It produces a 512³
uint8 volume with features chosen so sub-pixel misalignment is immediately
visible (grid planes, centre crosshairs, nested axis-aligned cube shells,
asymmetric marker cubes, 2-voxel-thick diagonal line):

```bash
python scripts/make_test_zarr.py /tmp/zview_test.zarr
zview /tmp/zview_test.zarr
```

### Full fake-data suite

[scripts/make_fake_datasets.py](scripts/make_fake_datasets.py) writes a set of
procedural "tomography sample" phantoms (container, grains, pores, vessels,
helix, bar-pattern resolution target, orientation cubes, noise) covering every
feature: isotropic pyramid, anisotropic 5D OME-NGFF 0.5, pixel-centre
translations, Zarr v2 float32 with NaNs, N5, numbered levels without
metadata, a 2D image, an HDF5 file (single resolution), and a large
single-level volume (no pyramid) for the 3D stress path. About 30 s and 2 GB with defaults; `--quick` is 3 s.

```bash
python scripts/make_fake_datasets.py /tmp/zview_fake          # --quick, --no-big, --big-shape Z Y X
zview /tmp/zview_fake/phantom_iso.zarr --layout quad
```

## Project layout

```
zview/
  core/            data access + machine capabilities (no Qt)
    pyramid.py     MultiResolutionImage, Volume3D adapter, ChunkCache, parallel/pooled reads
    labels.py      label maps: detection, class tables, label-aware pooling
    io.py          open Zarr v2/v3 (zarr-python) and N5 (z5py)
    hdf5.py        HDF5: dataset discovery, voxel size, parallel direct reads
    sysinfo.py     CPU/RAM/GPU probing, cache/thread defaults, 3D budget + planner
    log.py         small coloured logger
  viewer/          the Qt viewer (zview)
    app.py         main window, layouts, controls dock, menus, `main`
    panes.py       SlicePane: XY/XZ/YZ with async lazy loading, crosshair, picking
    loader.py      VolumeLoader: budgeted background 3D reads (shared by both 3D engines)
    volume_vtk.py  high-quality 3D pane (VTK, off-screen, progressive)
    volume_vispy.py fast fallback 3D pane (VisPy)
    render_dialog.py File > Render 3D…, job-script export
    metadata.py, theme.py
  render/          offline renderer (zview-render), no Qt
    cli.py config.py looks.py scene.py prepare.py sources.py surface.py tour.py cpu.py
  tools/           zview-convert (convert.py), zview-vessels (vessels/)
scripts/
  make_fake_datasets.py   fake-data suite (incl. a label map)
  make_test_zarr.py       alignment test pyramid
  pipeline/               unattended convert + render launchers
tests/             pytest suite (no display needed)
docs/              Sphinx documentation (Read the Docs)
```

## Citing

If you use zview in your work, please cite the software repository. GitHub's
**Cite this repository** button (from [`CITATION.cff`](CITATION.cff)) gives
APA and BibTeX, or use:

```bibtex
@software{zview,
  author  = {Mittone, Alberto},
  title   = {zview: viewer and renderer for large 3D image volumes},
  url     = {https://github.com/mittoalb/zview},
  version = {0.1.0},
  year    = {2026}
}
```

## Current known limitations

- **Shallow pyramids** still work everywhere (including 3D, via strided
  reads), but the first frame and the 3D volume load much faster when the
  dataset has 5–6 levels.
- **Non-integer downsample ratios between levels** (e.g. L2 Z being 4.0195×
  of L0 Z instead of exactly 4×) cause genuine per-axis drift in XZ/YZ
  ortho panes — this is a data authoring issue, not a viewer bug.
- The 3D texture is 8-bit, mapped from the 0.05–99.95 percentile range of
  the loaded volume; 3D contrast sliders work within that range.
