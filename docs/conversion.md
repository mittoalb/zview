# Converting to OME-Zarr

`zview-convert` turns a multi-TB HDF5 reconstruction or a folder of TIFF
slices into a **multiscale, sharded OME-Zarr v3** store. The result has few
files, transfers quickly, and opens in zview, napari, Neuroglancer and
zarr-python.

```console
$ zview-convert INPUT OUTPUT.zarr --pixel-size 1.88 [options]
```

`INPUT` is an HDF5 file (the 3-D dataset is found automatically, or set
`--dataset`) or a folder of TIFF slices, sorted by name.

| Option | Default | Meaning |
|---|---|---|
| `--dtype` | `uint16` | Output type. Float input is scaled between percentiles; with `uint8` input and `--dtype uint8`, values are copied unchanged |
| `--min-val`/`--max-val`, `--min-pct`/`--max-pct` | auto, 0.5/99.5 | Scaling range |
| `--chunk-size`, `--shard-size` | 128³, 512³ | Inner chunk size and shard size |
| `--slab` | 128 | Z slices per step; sets the RAM use (rounded to the shard size) |
| `--max-levels` | 6 | Number of 2× pyramid levels |
| `--pixel-size` | 1.0 | Voxel size (µm), written to the OME metadata |
| `--overwrite` | off | Replace an existing output |

Examples:

```console
# multi-TB HDF5 reconstruction (big machine)
$ nohup zview-convert /data/sample/rec.h5 /data/sample/rec.zarr \
    --pixel-size 1.88 --slab 1024 > convert.log 2>&1 &

# binned, cropped, masked 8-bit TIFF stack (0 = removed): copied unchanged
$ zview-convert /data/sample/rec_bin8_8bit /data/sample/rec_bin8.zarr \
    --dtype uint8 --pixel-size 15.04 --slab 512 --max-levels 4
```

## How it works

- **Streaming.** The input is read once, in Z slabs. Each slab is scaled,
  written to level 0, and mean-downsampled into levels 1…N. A background
  thread reads the next slab while the current one is written. Slab height
  is aligned to the shard size and to 2^levels, so all levels line up, odd
  sizes included.
- **Sharding.** 512³ shard files, each holding 128³ chunks. That's 64× fewer
  files than plain chunking, which keeps transfers, `du` and `rm` fast, while
  viewers still fetch individual chunks.

## Sizing

| | Rule of thumb |
|---|---|
| RAM | ≈ (2.4 × SLAB + 200) × Y × X × 2 bytes (12288² slices: ~430 GB at SLAB 512, ~800 GB at 1024) |
| Time | Limited by how fast the HDF5 can be read: 8.5 TB at 300 MB/s ≈ 8 h |
| Disk | ≤ 1.14 × level-0 size, usually less after compression |

If the process is `Killed`, lower `--slab`. A conversion can't resume, so a
restart overwrites the output.
