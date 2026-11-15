# Command reference

All commands also run as `python -m zview.render …`, which is handy in an
environment where zview isn't installed: set `PYTHONPATH` to the repository.

## `zview-render quick`

Prepare and render in one step.

```text
zview-render quick SOURCE --out DIR [what ...] [--preset P]... [--set KEY=VALUE]...
                   [--max-axis N | --step S] [--dataset HDF5_PATH] [-- extra args]
```

## `zview-render prepare`

```text
zview-render prepare SOURCE --out DIR [--preset P] [--max-axis 768 | --step 8]
                     [--dataset PATH] [--pixel-size UM] [--workers N] [--force]
                     [--config-name render.json]
```

Writes the cached volume (`volume_max<N>.npy` or `volume_step<S>.npy`),
`side_views_grid.png` and the config. An existing config is left unchanged,
and the cached volume is reused unless you pass `--force`.

| Source | How it is read |
|---|---|
| Zarr v2/v3, N5 | The finest pyramid level that fits `--max-axis`, plus a stride. Reads are parallel and memory-bounded, and chunks without a sampled voxel are skipped |
| HDF5 | Every S-th slice and every S-th row, averaged along x, so multi-TB files take minutes. The dataset is auto-detected (`/exchange/data`, …) |
| TIFF folder | Every S-th slice, averaged S×S in-plane |

`--step` gives the old fixed-step behaviour (the `tomo-brain` preset uses 8).

## `zview-render run`

```text
zview-render run CONFIG [what ...] [--preset P]... [--set KEY=VALUE]... [--out DIR] [-- extra args]
```

`what`: `stills`, `turntable`, `tour`, `surface`, `all` (default: the
config's `outputs`). `--preset` replaces the config's presets and can be
repeated. Arguments after `--` go straight to the underlying renderer, for
example `-- --keep-frames`.

## `zview-render presets`, `looks`

```text
zview-render presets                 # list
zview-render presets NAME            # show one as JSON
zview-render presets --save NAME --from CONFIG [--description TEXT]
zview-render looks                   # looks, curves, lighting rigs
```

## `zview-render cpu`

A pure NumPy/SciPy renderer for machines with no GPU and no display. It does
MIP and compositing turntables, stills from 4 sides, and a slice fly-through,
with the tomography preprocessing:

```console
$ zview-render cpu /data/sample/rec.zarr --out renders --budget-gb 24 --modes mip,comp
```

`zview-render cpu --help` lists all options.

## Launchers for unattended runs

`scripts/pipeline/run_pipeline_tomo.sh` (read an HDF5 in place) and
`run_pipeline.sh` (wait for a copy to finish) convert a reconstruction to
OME-Zarr and then render it, with no supervision. Put site paths in
`scripts/pipeline/config.local.sh` (copy `config.example.sh`; it's never
committed):

```console
$ nohup bash scripts/pipeline/run_pipeline_tomo.sh > pipeline.log 2>&1 &
```

`scripts/pipeline/gridftp_copy.sh` does fast transfers with the Globus CLI or
globus-url-copy.
