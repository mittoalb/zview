# Performance and sizing

zview sizes itself to the machine it runs on. **Help › System info** shows
the numbers it chose and why.

## What adapts

| Setting | Default | Override |
|---|---|---|
| I/O threads (2D chunk fetches) | CPU count, clamped to 4–32 | `ZVIEW_IO_THREADS` |
| 2D chunk cache | 15 % of free RAM, 256 MB – 4 GB | **View › Chunk cache size**, `--cache-mb`, `ZVIEW_CACHE_MB` |
| 3D texture budget (viewer) | from GPU, VRAM and RAM (below) | quality menu in the 3D pane, `ZVIEW_3D_MAX_AXIS`, `ZVIEW_3D_MAX_VOXELS` |
| 3D loader threads | half the I/O threads | — |

(3d-budget)=
## 3D budget

The viewer's 3D texture is single-channel `uint8`, so it costs 1 byte per
voxel on the GPU. While loading it also holds the native dtype in RAM
(`itemsize + 1` bytes per voxel). The texture size is the tightest of:

1. **Quality preset.** *Auto* picks a per-axis cap from the VRAM: 1024 for
   ≥ 8 GB, 768 for ≥ 4 GB, 512 for ≥ 2 GB, otherwise 384, and 256 for
   software renderers such as llvmpipe.
2. `GL_MAX_3D_TEXTURE_SIZE`.
3. **Half the free VRAM**, when the driver reports it (NVIDIA
   `NVX_gpu_memory_info`, AMD `ATI_meminfo`).
4. **A quarter of free RAM.**

The planner then takes the **finest pyramid level that fits**. If none fits,
it reads the coarsest level with a per-axis stride that grows on the longest
axis first, which keeps voxels close to cubic.

To override the automatic choice, click a level in the *Pyramid* panel. That
level is rendered whole if the machine can hold it: the GPU texture limit,
half the free VRAM and a quarter of the free RAM are checked, and levels
over the limit are greyed out.

For **HDF5 files** the automatic level also has a read limit. A strided level
of a contiguous file reads whole rows, *file size / (z stride × y stride)*,
so the level that fits the GPU could take minutes to read from network
storage. The automatic level reads at most 2 GB (`ZVIEW_3D_MAX_READ_GB`). A
click in the *Pyramid* panel still renders a finer level.

Rendering speed is a separate matter. Rotating is fast with **Shadows** off
(the default): tens of milliseconds per frame. With volumetric shadows on, a
full-quality frame of dense data can take a second or more.

Reads go block by block, in parallel, with a bounded amount of decoded data
in flight (at most 1 GB or 10 % of free RAM). Peak memory is therefore the
output plus a few blocks, whatever the size of the dataset. On a 537 GB test
array, peak RSS stayed under 1 GB.

:::{note}
Without a deep pyramid, a strided read still decodes **every chunk** along
any axis where the stride is smaller than the chunk size. Loading time then
grows with the dataset. The status line shows how much will be read, and ✕
cancels. Datasets with 5–6 pyramid levels load in seconds.
:::

## Environment variables

| Variable | Purpose |
|---|---|
| `ZVIEW_IO_THREADS` | Chunk-fetch concurrency. Raise it on high-latency network storage. |
| `ZVIEW_CACHE_MB` | 2D chunk-cache size. |
| `ZVIEW_3D_MAX_AXIS` | Per-axis cap for the viewer's 3D texture. |
| `ZVIEW_3D_MAX_VOXELS` | Total texel budget for the viewer's 3D texture (RAM and VRAM limits still apply). |
| `ZVIEW_3D_MAX_READ_GB` | HDF5 only: most data the 3D pane's automatic level may read from the file (default 2; `0` = no cap). |
| `ZVIEW_H5_DIRECT` | `0` reads uncompressed contiguous HDF5 datasets through h5py instead of zview's parallel direct reads. |
| `ZVIEW_PREAD_THREADS` | Parallel reads for direct HDF5 access (default 32, 8–64). |
| `ZVIEW_NO_TRANSLATION` | `1` ignores OME-NGFF per-level translations (for datasets that write them in a non-standard way). |
| `ZVIEW_NO_GL_COMPAT` | `1` disables the PyOpenGL library fix-up (see [](troubleshooting.md)). |
| `ZVIEW_VTK_OFFSCREEN` | `egl` or `default`: off-screen VTK window kind (default: EGL on VTK ≥ 9.7, else GLX). |
| `ZVIEW_VTK_LABELMASK` | `1`/`0`: force per-class shaded label rendering on or off. |

## Authoring data for speed

- **Build a pyramid** of 5–6 levels with 2× steps. Both the first frame and the
  3D load come from coarse levels.
- **Chunks of 64–128 voxels** per side work well for slicing. Sharding (Zarr
  v3) keeps file counts low without making chunks bigger;
  [`zview-convert`](conversion.md) writes 128³ chunks in 512³ shards.
- **Keep downsample ratios integer.** Non-integer ratios between levels (for
  example 4.0195× instead of 4×) cause real per-axis drift between levels in
  the XZ and YZ panes.
