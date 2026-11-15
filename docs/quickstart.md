# Quick start

## 1. Get some data

To try things out without real data, generate the fake-data suite (about
30 s; 2 GB, or 0.15 GB with `--quick`):

```console
$ python scripts/make_fake_datasets.py /tmp/zview_demo/data
```

See [](test-data.md) for what each dataset exercises.

## 2. Look at it

```console
$ zview /tmp/zview_demo/data/phantom_iso.zarr --layout quad
```

```{image} _static/img/ui/quad.jpg
:class: shot
```

- **Click** in any slice to move the crosshair; the other panes and the 3D
  slice outlines follow.
- **Scroll** to zoom, **drag** to pan. Panes sharpen to finer pyramid levels
  by themselves.
- **Ctrl+wheel** or **PgUp/PgDn** steps through slices. **A** sets the
  contrast automatically. **Ctrl+1…4** switches layouts.

More in [](viewer.md).

## 3. Make a presentation render

From the viewer: **File › Render 3D…**, pick a preset, then press **Render**.

From the command line, `quick` prepares a render-sized copy of the data and
renders it in one go:

```console
$ zview-render quick /tmp/zview_demo/data/phantom_iso.zarr --out /tmp/phantom_render \
    --preset interior stills turntable
```

::::{grid} 2
:gutter: 2

:::{grid-item}
```{image} _static/img/presets/interior.jpg
```
:::

:::{grid-item}
```{image} _static/img/turntable.gif
```
:::

::::

The outputs land in `/tmp/phantom_render/renders/`, next to
`render.json`. Edit that config (or pass `--set key=value`) and re-run it:

```console
$ zview-render run /tmp/phantom_render/render.json stills --set look=gold --set lighting=dramatic
```

Continue with [](rendering/index.md).
