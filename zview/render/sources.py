"""Open any supported input as a (z, y, x) array for the render tools.

    Zarr v2 / v3 / N5 store  -> pyramid via zview.msres (2D–5D arrays, OME-NGFF)
    HDF5 file                -> the 3-D dataset (auto-detected or named)
    folder of TIFF slices    -> lazy slice stack
"""

from pathlib import Path

H5_SUFFIXES = {".h5", ".hdf5", ".hdf", ".he5", ".nxs", ".nx5"}
TIFF_SUFFIXES = {".tif", ".tiff"}


def kind(path):
    p = Path(path)
    if p.is_file() and p.suffix.lower() in H5_SUFFIXES:
        return "hdf5"
    if p.is_file():
        try:
            import h5py
            if h5py.is_hdf5(str(p)):
                return "hdf5"
        except Exception:
            pass
    if p.is_dir():
        markers = (".zgroup", ".zarray", "zarr.json", "attributes.json")
        if not any((p / m).exists() for m in markers) and any(
                f.suffix.lower() in TIFF_SUFFIXES for f in p.iterdir() if f.is_file()):
            return "tiff"
        return "zarr"
    raise FileNotFoundError(f"{path}: not an HDF5 file, TIFF folder or Zarr/N5 store")


def open_pyramid(path):
    """MultiResolutionImage for a Zarr/N5 store."""
    from zview.io import open_dataset
    from zview.msres import MultiResolutionImage
    store, _ = open_dataset(path)
    return MultiResolutionImage(store, cache_size_mb=64)


def open_fullres(path, dset=None):
    """Full-resolution (z, y, x) array-like with numpy slicing."""
    k = kind(path)
    if k == "hdf5":
        import h5py
        from zview.render.prepare import find_dataset
        f = h5py.File(str(path), "r")
        return f[find_dataset(f, dset)]
    if k == "tiff":
        from zview.convert import TiffStack
        return TiffStack(Path(path))
    return open_pyramid(path).levels[0]
