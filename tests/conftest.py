"""Small synthetic datasets shared by the tests (written once per session)."""

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")


@pytest.fixture(scope="session")
def ref5d():
    """Reference (t, c, z, y, x) data; tests compare against [0, 0]."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 60000, (1, 2, 37, 53, 71), dtype=np.uint16)


@pytest.fixture(scope="session")
def store5d(tmp_path_factory, ref5d):
    path = tmp_path_factory.mktemp("data") / "t5d.zarr"
    g = zarr.open_group(str(path), mode="w")
    g.create_array("0", data=ref5d, chunks=(1, 1, 8, 16, 16))
    g.create_array("1", data=ref5d[:, :, ::2, ::2, ::2].copy(), chunks=(1, 1, 8, 16, 16))
    g.attrs["multiscales"] = [{
        "version": "0.4",
        "axes": [{"name": "t"}, {"name": "c"}, {"name": "z", "unit": "micrometer"},
                 {"name": "y", "unit": "micrometer"}, {"name": "x", "unit": "micrometer"}],
        "datasets": [
            {"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [1, 1, 2.0, 0.5, 0.5]}]},
            {"path": "1", "coordinateTransformations": [{"type": "scale", "scale": [1, 1, 4.0, 1.0, 1.0]}]},
        ]}]
    return path


@pytest.fixture(scope="session")
def labels_data():
    rng = np.random.default_rng(1)
    a = np.full((70, 90, 110), 4, np.uint8)
    a[rng.random(a.shape) < 0.002] = 70
    a[:5] = 255
    return a


@pytest.fixture(scope="session")
def label_store(tmp_path_factory, labels_data):
    path = tmp_path_factory.mktemp("data") / "labels.zarr"
    g = zarr.open_group(str(path), mode="w")
    g.create_array("0", data=labels_data, chunks=(32, 32, 32))
    g.attrs["lookup"] = {"4": "white matter", "70": "vessel lumen", "255": "void (outside)"}
    return path


def open_mi(path):
    from zview.core.io import open_dataset
    from zview.core.pyramid import MultiResolutionImage
    store, _ = open_dataset(path)
    return MultiResolutionImage(store, cache_size_mb=64)
