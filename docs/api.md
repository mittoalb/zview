# Python API

The modules below are the stable entry points for scripting. The viewer and
renderer classes can be imported too, but their interfaces may change.

## Reading data

```{eval-rst}
.. automodule:: zview.msres
   :members: MultiResolutionImage, Volume3D, ChunkCache
```

```{eval-rst}
.. autofunction:: zview.io.open_dataset
```

```{eval-rst}
.. automodule:: zview.render.sources
   :members:
```

Example: read a render-sized volume of any store.

```python
from zview.render.sources import open_pyramid
from zview import sysinfo

mi = open_pyramid("/data/sample/rec.zarr")
plan = sysinfo.plan_volume([l.shape for l in mi.levels], sysinfo.volume_budget(quality="medium"))
vol = mi.read_volume(plan.level, plan.stride)          # numpy (z, y, x)
```

## Machine capabilities

```{eval-rst}
.. automodule:: zview.sysinfo
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
