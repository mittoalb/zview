# Blood vessels

Tools for segmenting the blood vessels in tomography volumes and making 3D
videos of them. Here, vessels are **fine dark tubes** (empty lumens).

```console
$ zview-vessels segment | fullres | render  ...   # each has --help
```

## Pre-masked volume — `segment`

Use this for a volume where everything that isn't tissue has already been set
to 0: for example a cut, masked and binned 8-bit TIFF stack. It takes minutes
to an hour and needs no display.

```console
$ zview-vessels segment /data/sample/rec_bin8_8bit \
    --out /work/sample/vessels_bin8.zarr --pixel-size 15.04 \
    --tiff-out /work/sample/vessels_bin8_tiff          # optional 0/255 TIFF copy
```

The input can also be a Zarr (`--level`) or an HDF5 file. For each tile:

1. **Darkness** is measured against the *local* tissue level and the darkest
   tissue voxels (the lumen level).
2. **Thick dark regions** are excluded: anything thicker than about
   `2 × --thick` voxels (ventricles, clefts).
3. **Hysteresis** keeps seeds above `--high`, plus voxels above `--low` that
   connect to a seed. A 1-voxel closing then bridges small gaps.
4. **Cleanup** removes pieces smaller than `--min-size`, and only the largest
   connected body is used.

| Option | Default | Meaning |
|---|---|---|
| `--low` / `--high` | 0.35 / 0.55 | Hysteresis thresholds on darkness (lower = more, fainter vessels) |
| `--thick` | 2 | Exclude dark regions thicker than about 2× this many voxels |
| `--tissue-sigma` | 8 | Size of the local tissue neighbourhood (voxels) |
| `--min-size` | 30 | Smallest piece kept (voxels) |
| `--erode` | 2 | Stay this many voxels inside the tissue mask |

The defaults are tuned for voxels of about 15 µm. For finer voxels, scale
`--tissue-sigma`, `--thick` and `--min-size` up with the vessel size in
voxels. Debris that touches the tissue is not removed: mask it in the input.
Check `vessels_bin8_preview.png` after each run.

## Whole sample at full resolution — `fullres`

Use this for an unmasked full-resolution scan, using the tissue mask of the
render pipeline (a `render.json` prepared with the `tomo-brain` preset). It
takes hours on a 64-core machine and needs no display.

```console
$ zview-render prepare /data/sample/rec.zarr --out /work/sample --preset tomo-brain
$ nohup zview-vessels fullres /work/sample/render.json \
    --zarr /data/sample/rec.zarr --out /data/sample/vessels.zarr > vessels.log 2>&1 &
```

The defaults are `--low 0.40 --high 0.55 --min-size 500`. Test them first on a
small window with `--z-window Z0 Z1 --yx-window Y0 Y1 X0 X1`.

- **Progress file.** Finished tiles are listed in `vessels.zarr.progress`;
  after a crash or a stop, re-run with `--resume`.
- **Write checks.** The job stops if a shard it wrote isn't on disk, or if
  the store's `zarr.json` disappears. Write `--out` to a folder nothing else
  cleans up.

## Rendering — `render`

```console
$ zview-vessels render /work/sample/vessels_bin8.zarr --image /data/sample/rec_bin8_8bit \
    --out /work/sample/vessels_video --title "Sample — vasculature" --credit "Lab, Institute"

$ zview-vessels render /data/sample/vessels.zarr --config /work/sample/render.json \
    --level 3 --out /work/sample/vessels_video
```

It writes `turntable_vessels.mp4` (12 s) and `stills/`: red vessels inside a
translucent sample. The translucent body comes from the image (`--image`,
voxels > 0) or from the render config. The finest mask level up to
`--max-voxels` is rendered.

## Using the masks

A vessel mask is `uint8` with 1 = vessel. Its level 0 is on **the same grid as
its input**, voxel for voxel; coarser levels are max-pooled.

```python
import zarr
img = zarr.open_group("/data/sample/rec.zarr", mode="r")
ves = zarr.open_group("/data/sample/vessels.zarr", mode="r")
block = img["0"][7000:7512, 6000:7024, 6000:7024]
mask = ves["0"][7000:7512, 6000:7024, 6000:7024]       # same coordinates
```

Masks also open in zview like any other multiscale Zarr.
