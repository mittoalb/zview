# Test data

`scripts/make_fake_datasets.py` writes a set of synthetic datasets that
exercise every part of zview. Each one is a procedural "tomography sample"
defined in physical coordinates, so it looks the same at any resolution or
voxel anisotropy:

- a container cylinder with a bright wall, a textured matrix with a z
  gradient, and air outside;
- about 160 spherical inclusions: dense grains and dark pores;
- vessel-like tubes and a helix;
- a bar-pattern **resolution target** with periods of 2–24 voxels, which blurs
  out at coarse pyramid levels and makes level switches easy to see;
- three asymmetric marker cubes for reading orientation in 3D;
- Gaussian noise.

```console
$ python scripts/make_fake_datasets.py OUT_DIR [--quick] [--no-big] [--big-shape Z Y X] [--big-pyramid]
```

Blocks are rendered in parallel and written chunk-aligned, so even the
multi-GB volume uses little memory. Defaults: about 30 s and 2 GB on disk;
`--quick` takes 3 s.

| Dataset | Format | Exercises |
|---|---|---|
| `phantom_iso.zarr` | Zarr v3, uint16 512³, 6 levels, OME-NGFF 0.4 | The main demo: slicing, level switching, 3D, every render preset |
| `phantom_aniso_5d.zarr` | Zarr v3, `(1, 2, 128, 768, 768)` uint16, OME-NGFF 0.5 | `t`/`c` axes, 4:1 voxel anisotropy, z kept at full resolution in the first levels; channel 1 holds only the vessels |
| `phantom_ngff_centered.zarr` | Zarr v3, 256³ | Standard pixel-centre per-level translations; levels must not jump when you zoom across a change |
| `phantom_v2_float32.zarr` | Zarr v2, float32 | Negative values and a slab of NaNs |
| `phantom_labels.zarr` | Zarr v3, uint8 512³, 5 levels | A **label map** of the same phantom (matrix, grains, pores, wall, vessels, helix, markers; 255 = void) with a `lookup` table and a stride-subsampled pyramid |
| `phantom.n5` | N5 (z5py) | The N5 backend; numbered levels without metadata |
| `numbered_no_metadata.zarr` | Zarr v2 | Level discovery without attributes |
| `image_2d.zarr` | Zarr v3, 4096² uint8 | A 2D image as a one-slice volume |
| `big_single_level.zarr` | Zarr v3, 768×2048×2048 uint8 (3.2 GB raw), **one level** | The large-data path: strided 3D reads, progress, cancel |

There's also `scripts/make_test_zarr.py`. It writes a 6-level pyramid of
1-voxel grid planes, cube shells and markers, designed to make sub-pixel
misalignment between levels obvious.
