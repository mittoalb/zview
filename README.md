# zview

A fast viewer for large multi-resolution Zarr (v2 and v3) and N5 image
volumes. Built on PyQt5 + pyqtgraph, with chunk-aligned lazy loading, a
parallel I/O thread pool, and automatic pyramid-level selection.

## Features

- **Multi-resolution pyramid support** — reads OME-NGFF v0.4 `multiscales`
  metadata from Zarr v2, Zarr v3, or N5; also falls back to numbered
  subgroups (`0/`, `1/`, …) when no metadata is present.
- **Chunk-aligned, parallel loading** — only the chunks that cover the
  visible region are fetched, in parallel across a configurable thread pool.
- **Automatic level selection with hysteresis** — picks the finest pyramid
  level that samples at ≥ 1 screen pixel per voxel, with a 1-octave dead
  zone around the current level so small pans don't flip-flop between
  levels.
- **Axis-aware level selection for ortho panes** — XY uses Y/X downsample,
  XZ uses Z/X, YZ uses Z/Y, so anisotropic volumes don't pick an
  over-coarse level in Z.
- **Four view modes**: single XY, dual-plane, 4-quadrant orthogonal, (3D
  volume rendering via VisPy — currently disabled while the ortho view is
  being iterated on).
- **Status bar** showing cursor world coord `(z, y, x)` for whichever pane
  the mouse is in, plus volume dimensions.
- **Keyboard shortcuts** for slice navigation, view reset, full screen.

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
# or
zview /path/to/dataset.n5
```

Both OME-Zarr v0.4 pyramids and plain numbered-subgroup pyramids
(`<path>/0`, `<path>/1`, …) are supported.

### Keyboard shortcuts

| Key                  | Action                                      |
|----------------------|---------------------------------------------|
| **F11**              | Toggle application full-screen              |
| **R**                | Reset active pane's zoom (fit to image)     |
| **PgUp / PgDn**      | Step Z slice ±1                             |
| **Shift+PgUp/PgDn**  | Step Z slice ±10                            |
| **Home / End**       | Jump to first / last Z slice                |

### Mouse

- **Scroll wheel** — zoom (centered on cursor).
- **Click + drag** — pan.
- **Click in an ortho pane** — move the crosshair; the other two panes
  update their slices to pass through that point.
- **Double-click a pane header** in the 4-quadrant layout — maximize that
  pane. Double-click again to restore.

## Environment variables

| Variable                  | Default | Purpose                                                                 |
|---------------------------|---------|-------------------------------------------------------------------------|
| `ZVIEW_IO_THREADS`        | `8`     | Chunk-fetch concurrency. Raise on fast storage (16, 32).                |
| `ZVIEW_NO_TRANSLATION`    | *unset* | Set to `1` to ignore OME-NGFF per-level `translation` metadata if your dataset writes it in a non-standard way and levels visibly jump. |

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

## Project layout

```
zview/
  sview.py      DynamicImageView — XY pane with chunk-aligned lazy loading
  oview.py      OrthoImageView   — XY/XZ/YZ pane for the 4-quadrant layout
  msres.py      MultiResolutionImage, ChunkCache, parallel chunk I/O
  meta.py       Zarr metadata extractor + viewer widget
  view3d.py     Optional VisPy volume renderer (currently stubbed out)
  zview2.py     Main app (UnifiedZarrViewer)
scripts/
  make_test_zarr.py   Generate a 6-level OME-NGFF test pyramid
pyproject.toml, environment.yml
```

## Current known limitations

- **3D VisPy pane is disabled** while the ortho view is being iterated on.
  Re-enable by restoring the `from zview.view3d import …` import at the top
  of [zview/zview2.py](zview/zview2.py).
- **Only 3 pyramid levels in your dataset?** Even with parallel I/O, the
  coarsest level's chunk count dominates initial load time. For responsive
  startup on datasets > 10k voxels per axis, author with 5–6 levels.
- **Non-integer downsample ratios between levels** (e.g. L2 Z being 4.0195×
  of L0 Z instead of exactly 4×) cause genuine per-axis drift in XZ/YZ
  ortho panes — this is a data authoring issue, not a viewer bug. Can be
  masked with a planned `ZVIEW_SNAP_INTEGER=1` flag.
