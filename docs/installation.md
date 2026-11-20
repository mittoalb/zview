# Installation

zview needs Python 3.10 or newer.

## Viewer only

```console
$ conda env create -f environment.yml
$ conda activate zview
$ pip install -e .
```

This installs the `zview` command, with PyQt5, pyqtgraph, VisPy,
zarr-python 3 and h5py (for HDF5 files). N5 support comes from `z5py` (in `environment.yml`, from
conda-forge).

## With the renderer and tomography tools

The offline renderer, the high-quality 3D pane, the converter and the vessel tools need VTK, SciPy,
scikit-image, imageio with ffmpeg, matplotlib, Pillow, h5py and tifffile:

```console
$ pip install -e '.[render]'
```

That adds the `zview-render`, `zview-convert` and `zview-vessels` commands, and
the high-quality VTK 3D pane in the viewer.

### Using a separate render environment

The rendering code doesn't import Qt, so it can run in a different
environment from the viewer: for example, one built around a specific VTK
version on a GPU node.

```console
$ conda create -n zview-render -c conda-forge python=3.11 \
    'zarr>=3.1' numpy scipy scikit-image h5py tifffile imageio imageio-ffmpeg matplotlib pillow
$ conda run -n zview-render pip install vtk
$ conda run -n zview-render pip install -e /path/to/zview --no-deps
```

The viewer's **File › Render 3D…** dialog finds an environment that has VTK
automatically, or you can pick the interpreter by hand. It runs the renderer
there with this copy of zview on `PYTHONPATH`, so you don't even need to
install zview into that environment.

## Requirements for 3D

| What | Needs |
|---|---|
| Viewer 3D pane | OpenGL 3.x; any GPU or a software renderer (the budget adapts) |
| `zview-render` stills, turntables, tour, surface | VTK and an X display (`DISPLAY` set; a VNC session works). The render window is off-screen, so nothing appears |
| `zview-render cpu` | Nothing graphical: a pure NumPy/SciPy renderer for headless servers |
| Conversion, vessel segmentation | Many cores and plenty of RAM; no display |

## Building these docs

```console
$ pip install -e '.[docs]'
$ sphinx-build -b html docs docs/_build/html
```

The images come from `docs/gen_images.py` (renders, which need a GPU) and
`docs/gen_screenshots.py` (viewer screenshots, run under `xvfb-run`).
