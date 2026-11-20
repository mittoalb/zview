# Python API

The modules below are the stable entry points for scripting. The viewer and
renderer classes can be imported too, but their interfaces may change.

## Reading data

```{eval-rst}
.. automodule:: zview.core.pyramid
   :members: MultiResolutionImage, Volume3D, ChunkCache, StridedLevel, strided_factors
```

```{eval-rst}
.. autofunction:: zview.core.io.open_dataset
```

```{eval-rst}
.. automodule:: zview.render.sources
   :members:
```

### HDF5 (single resolution)

```{eval-rst}
.. automodule:: zview.core.hdf5
   :members: open_image, split_spec, is_hdf5, find_dataset, candidates, voxel_size, direct_reader, ContiguousDataset
```

Example: open an HDF5 file as a pyramid and read a strided level.

```python
from zview.core import hdf5

mi, f, dset = hdf5.open_image("/data/scan.h5")      # or "scan.h5:/exchange/data"
print(dset, [lvl.shape for lvl in mi.levels])        # level k = one voxel every 2**k
vol = mi.read_volume(mi.get_num_levels() - 1)        # numpy (z, y, x)
mi.close()
f.close()
```

Any array with numpy slicing works the same way through
`MultiResolutionImage.from_array(array, voxel_size=(z, y, x))`.

Example: read a render-sized volume of any store.

```python
from zview.render.sources import open_pyramid
from zview.core import sysinfo

mi = open_pyramid("/data/sample/rec.zarr")
plan = sysinfo.plan_volume([l.shape for l in mi.levels], sysinfo.volume_budget(quality="medium"))
vol = mi.read_volume(plan.level, plan.stride)          # numpy (z, y, x)
```

## Label maps

```{eval-rst}
.. automodule:: zview.core.labels
   :members: LabelTable, LabelClass, detect_labels, build_table, pool_labels, render_codes
```

## Machine capabilities

```{eval-rst}
.. automodule:: zview.core.sysinfo
   :members: volume_budget, plan_volume, VolumeBudget, VolumePlan, GLCaps, system_memory, cpu_count,
             default_io_workers, default_cache_mb, apply_gl_compat
```

## Rendering

```{eval-rst}
.. automodule:: zview.render.config
   :members: effective, load, load_preset, available_presets, save_user_preset, parse_set
```

```{eval-rst}
.. autofunction:: zview.render.cli.run
```

```{eval-rst}
.. autofunction:: zview.render.prepare.prepare
```

```{eval-rst}
.. automodule:: zview.render.looks
   :members: get_look, get_lighting, opacity_points, finish
```

Example: render stills of a dataset from a script.

```python
from zview.render.prepare import prepare
from zview.render.cli import run

cfg = prepare("/data/sample/rec.zarr", "/work/sample", preset="interior")
run(cfg, ["stills"], {"look": "jade", "lighting": "soft", "size": "2560x1440"})
```
