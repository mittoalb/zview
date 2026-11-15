import threading

import numpy as np
import pytest
import zarr

from conftest import open_mi
from zview.core.pyramid import ChunkCache, VolumeReadCancelled


def test_5d_adapter_and_metadata(store5d):
    mi = open_mi(store5d)
    assert mi.get_level_shape(0) == (37, 53, 71)
    assert mi.voxel_size() == (2.0, 0.5, 0.5)
    assert mi.get_num_levels() == 2


@pytest.mark.parametrize("z", [0, 5, 36])
def test_xy_region_reads_z_not_t(store5d, ref5d, z):
    mi = open_mi(store5d)
    got = mi.get_slice_2d_region(0, "xy", z, (3, 40), (7, 70))
    assert np.array_equal(got, ref5d[0, 0, z, 3:40, 7:70])


def test_xz_yz_regions(store5d, ref5d):
    mi = open_mi(store5d)
    ref = ref5d[0, 0]
    for y in (0, 17, 52):
        assert np.array_equal(mi.get_slice_2d_region(0, "xz", y, (2, 33), (5, 66)), ref[2:33, y, 5:66])
    for x in (0, 30, 70):
        assert np.array_equal(mi.get_slice_2d_region(0, "yz", x, (1, 36), (4, 50)), ref[1:36, 4:50, x])
    assert np.array_equal(mi.get_slice_2d(0, "xz", 9), ref[:, 9, :])


@pytest.mark.parametrize("stride", [(1, 1, 1), (2, 2, 2), (3, 5, 7), (9, 1, 4), (40, 60, 80)])
@pytest.mark.parametrize("block", [2 ** 10, 64 * 2 ** 20])
def test_read_volume_strided(store5d, ref5d, stride, block):
    mi = open_mi(store5d)
    got = mi.read_volume(0, stride, target_block_bytes=block)
    exp = ref5d[0, 0, ::stride[0], ::stride[1], ::stride[2]]
    assert got.shape == exp.shape and np.array_equal(got, exp)


def test_read_volume_progress_and_cancel(store5d):
    mi = open_mi(store5d)
    calls = []
    mi.read_volume(0, (1, 1, 1), progress=lambda d, t: calls.append((d, t)), target_block_bytes=2 ** 12)
    assert calls and calls[-1][0] == calls[-1][1]
    ev = threading.Event()
    ev.set()
    with pytest.raises(VolumeReadCancelled):
        mi.read_volume(0, (1, 1, 1), cancel=ev, target_block_bytes=2 ** 12)


def test_estimate_read_bytes(store5d):
    mi = open_mi(store5d)
    full = mi.estimate_read_bytes(0, (1, 1, 1))
    assert full == 37 * 53 * 71 * 2 or full >= 37 * 53 * 71 * 2
    assert mi.estimate_read_bytes(0, (40, 1, 1)) < full     # stride > chunk skips chunks


def test_2d_image_is_one_slice_volume(tmp_path):
    img = np.random.default_rng(3).integers(0, 255, (40, 50), dtype=np.uint8)
    g = zarr.open_group(str(tmp_path / "img.zarr"), mode="w")
    g.create_array("0", data=img, chunks=(16, 16))
    mi = open_mi(tmp_path / "img.zarr")
    assert mi.get_level_shape(0) == (1, 40, 50)
    assert np.array_equal(mi.get_slice_2d(0, "xy", 0), img)
    assert np.array_equal(mi.read_volume(0)[0], img)


def test_chunk_cache_lru_and_resize():
    c = ChunkCache(max_memory_mb=1)
    a = np.zeros(400_000, np.uint8)
    c.put("a", a)
    c.put("b", a)
    assert c.get("a") is a
    c.put("c", a)                       # evicts b (least recently used)
    assert c.get("b") is None and c.get("a") is a
    c.set_max_memory(0.5)
    assert len(c.cache) == 1
