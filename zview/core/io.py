"""Opening datasets (no Qt): Zarr v2/v3 via zarr-python, N5 via z5py."""

from pathlib import Path


def open_dataset(path):
    """Open a Zarr v2, Zarr v3, or N5 store and return (root_group, backend).

    `zarr-python` handles both v2 and v3 natively; `z5py` is kept as a fallback
    for N5 (which the zarr library doesn't open).
    """
    import zarr  # noqa: PLC0415
    try:
        return zarr.open_group(str(path), mode='r'), 'zarr'
    except Exception as zarr_err:
        try:
            import z5py  # noqa: PLC0415
            # z5py: use_zarr_format controls Zarr vs N5. Detect from marker files.
            p = Path(path)
            looks_like_zarr = any(p.glob('.zgroup')) or any(p.glob('.zarray')) or (p / 'zarr.json').exists()
            return (
                z5py.File(str(path), mode='r', use_zarr_format=looks_like_zarr),
                'z5py',
            )
        except Exception as z5_err:
            raise RuntimeError(
                f"Could not open {path!r}:\n  zarr-python: {zarr_err}\n  z5py: {z5_err}"
            ) from zarr_err
