#!/usr/bin/env python3
"""Generate a 6-level OME-NGFF Zarr pyramid with axis-aligned, pixel-precise
features designed so any sub-pixel misalignment between levels is immediately
visible. (Spheres and other round shapes are deliberately avoided — their
edges are gradient, so small shifts are hard to see.)

Content:
    * Grid planes every 32 voxels (Z, Y, X). Perpendicular lines at the same
      world coord must land at the same screen pixel across levels.
    * Center crosshairs: three mutually perpendicular 2-voxel-thick lines
      through (256, 256, 256). Each ortho pane shows two of them crossing at
      the pane's centre — a shift at a level change jumps the cross.
    * Nested axis-aligned cube SHELLS centred at (256, 256, 256) at half-sizes
      16, 32, 64, 128, 192. Walls only (1 voxel thick). Concentric squares
      that stay concentric only if alignment is correct.
    * Asymmetric position markers: 4-voxel cubes at (64, 128, 192), (192, 64,
      320), (320, 192, 64), (448, 384, 256), (96, 416, 128). Asymmetric so
      you can read orientation at a glance.
    * A 2-voxel-thick diagonal line from (0,0,0) to (N,N,N) for a secondary
      reference across the whole volume.

Usage:
    python scripts/make_test_zarr.py <output.zarr>
"""
import sys
from pathlib import Path

import numpy as np

try:
    import zarr
except ImportError:
    sys.stderr.write(
        "zarr is required (not z5py — need the writer).\n"
        "  conda install -c conda-forge zarr\n"
        "  # or: pip install zarr\n"
    )
    sys.exit(2)


def _place_cube_shell(vol, cz, cy, cx, half, value):
    """Hollow axis-aligned cube: walls only, 1 voxel thick."""
    z0, z1 = cz - half, cz + half
    y0, y1 = cy - half, cy + half
    x0, x1 = cx - half, cx + half
    n = vol.shape[0]
    if min(z0, y0, x0) < 0 or max(z1, y1, x1) > n - 1:
        return
    # Six faces.
    vol[z0, y0:y1 + 1, x0:x1 + 1] = np.maximum(vol[z0, y0:y1 + 1, x0:x1 + 1], value)
    vol[z1, y0:y1 + 1, x0:x1 + 1] = np.maximum(vol[z1, y0:y1 + 1, x0:x1 + 1], value)
    vol[z0:z1 + 1, y0, x0:x1 + 1] = np.maximum(vol[z0:z1 + 1, y0, x0:x1 + 1], value)
    vol[z0:z1 + 1, y1, x0:x1 + 1] = np.maximum(vol[z0:z1 + 1, y1, x0:x1 + 1], value)
    vol[z0:z1 + 1, y0:y1 + 1, x0] = np.maximum(vol[z0:z1 + 1, y0:y1 + 1, x0], value)
    vol[z0:z1 + 1, y0:y1 + 1, x1] = np.maximum(vol[z0:z1 + 1, y0:y1 + 1, x1], value)


def _place_cube_solid(vol, cz, cy, cx, half, value):
    """Solid axis-aligned cube centred at (cz, cy, cx) with half-extent `half`."""
    n = vol.shape[0]
    z0, z1 = max(0, cz - half), min(n, cz + half + 1)
    y0, y1 = max(0, cy - half), min(n, cy + half + 1)
    x0, x1 = max(0, cx - half), min(n, cx + half + 1)
    vol[z0:z1, y0:y1, x0:x1] = np.maximum(vol[z0:z1, y0:y1, x0:x1], value)


def build_level0(n=512):
    """Build the full-resolution volume with alignment-friendly features."""
    vol = np.zeros((n, n, n), dtype=np.uint8)

    # 2-voxel-thick diagonal line (0,0,0)..(n-1,n-1,n-1).
    idx = np.arange(n)
    vol[idx, idx, idx] = 255
    j = np.clip(idx + 1, 0, n - 1)
    vol[idx, idx, j] = 255

    # Grid planes every 32 voxels, 1 voxel thick.
    grid_step = 32
    grid_val = 160
    for i in range(0, n, grid_step):
        vol[i, :, :] = np.maximum(vol[i, :, :], grid_val)
        vol[:, i, :] = np.maximum(vol[:, i, :], grid_val)
        vol[:, :, i] = np.maximum(vol[:, :, i], grid_val)

    # Three centre crosshairs (full span, 2 voxels thick), through (n/2, n/2, n/2).
    c = n // 2
    cross_val = 255
    vol[c, c, :] = cross_val; vol[c, c + 1, :] = cross_val
    vol[c, :, c] = cross_val; vol[c, :, c + 1] = cross_val
    vol[:, c, c] = cross_val; vol[:, c, c + 1] = cross_val

    # Nested axis-aligned cube SHELLS centred on the volume centre.
    for half in (16, 32, 64, 128, 192):
        _place_cube_shell(vol, c, c, c, half, 230)

    # Asymmetric position marker cubes (4-voxel half-extent).
    markers = [
        (64, 128, 192),
        (192, 64, 320),
        (320, 192, 64),
        (448, 384, 256),
        (96, 416, 128),
    ]
    for (mz, my, mx) in markers:
        _place_cube_solid(vol, mz, my, mx, 4, 255)

    return vol


def mean_downsample_2x(vol):
    """2x block-mean downsample along each axis. Trims to even extent."""
    nz, ny, nx = (s - (s % 2) for s in vol.shape)
    v = vol[:nz, :ny, :nx]
    # Reshape and average in blocks of 2 per axis.
    v = v.reshape(nz // 2, 2, ny // 2, 2, nx // 2, 2)
    return v.mean(axis=(1, 3, 5)).astype(vol.dtype)


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    out = Path(sys.argv[1])
    if out.exists():
        print(f"Refusing to overwrite existing path: {out}", file=sys.stderr)
        sys.exit(1)

    N = 512
    NUM_LEVELS = 6
    chunk = (64, 64, 64)

    print(f"Building level 0 ({N}^3)...")
    vol = build_level0(N)

    print(f"Opening {out}")
    # Force Zarr v2 — z5py (what the viewer uses) only speaks v2.
    root = zarr.open_group(str(out), mode="w", zarr_format=2)

    levels = []
    scales_pix = []  # per-level scale as level-0 pixel units per level-N pixel
    current = vol
    for i in range(NUM_LEVELS):
        shape = current.shape
        print(f"  Writing level {i}: shape={shape}, dtype={current.dtype}")
        arr = root.create_array(
            name=str(i),
            shape=shape,
            chunks=chunk,
            dtype=current.dtype,
            overwrite=False,
        )
        arr[...] = current
        levels.append(str(i))
        scales_pix.append(2 ** i)
        if i + 1 < NUM_LEVELS:
            current = mean_downsample_2x(current)

    # OME-NGFF v0.4 multiscales metadata. Scales are in level-0 pixel units so
    # the viewer's "world = level-0 pixels" assumption holds exactly. Translations
    # are 0 because mean 2x downsample with corner origin needs no half-pixel
    # offset for levels to line up.
    multiscales = [{
        "version": "0.4",
        "name": "test-pyramid",
        "axes": [
            {"name": "z", "type": "space", "unit": "pixel"},
            {"name": "y", "type": "space", "unit": "pixel"},
            {"name": "x", "type": "space", "unit": "pixel"},
        ],
        "datasets": [
            {
                "path": path,
                "coordinateTransformations": [
                    {"type": "scale", "scale": [float(s), float(s), float(s)]},
                    {"type": "translation", "translation": [0.0, 0.0, 0.0]},
                ],
            }
            for path, s in zip(levels, scales_pix)
        ],
    }]
    root.attrs["multiscales"] = multiscales

    print(f"Done. Open with: zview {out}")


if __name__ == "__main__":
    main()
