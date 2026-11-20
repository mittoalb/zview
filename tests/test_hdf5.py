"""HDF5 files: dataset discovery and strided virtual pyramid levels."""

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from zview.core import hdf5  # noqa: E402
from zview.core.pyramid import MultiResolutionImage, StridedLevel, Volume3D, strided_factors  # noqa: E402


@pytest.fixture(scope="module")
def ref():
    return np.random.default_rng(3).integers(0, 60000, (45, 133, 97), dtype=np.uint16)


LAYOUTS = {"contiguous": {},
           "chunked": dict(chunks=(4, 32, 32), compression="gzip"),
           "slices": dict(chunks=(1, 133, 97), compression="gzip")}   # whole-plane chunks: plane cache


@pytest.fixture(scope="module", params=list(LAYOUTS))
def h5file(request, tmp_path_factory, ref):
    path = tmp_path_factory.mktemp("h5") / f"{request.param}.h5"
    kw = LAYOUTS[request.param]
    with h5py.File(path, "w") as f:
        d = f.create_dataset("exchange/data", data=ref, **kw)
        d.attrs["voxel_size"] = [2.0, 1.0, 1.0]
        f.create_dataset("exchange/data_white", data=ref[:4])
        f.create_dataset("exchange/theta", data=np.linspace(0, 180, 45))
    return path


def _open(path, coarsest=16):
    f = hdf5.open_file(path)
    mi = MultiResolutionImage.from_array(f["/exchange/data"], voxel_size=hdf5.voxel_size(f, "/exchange/data"),
                                         group=f, coarsest=coarsest)
    return mi, f


def test_strided_levels_equal_numpy(h5file, ref):
    mi, f = _open(h5file)
    try:
        assert mi.virtual_levels and mi.get_num_levels() >= 3
        for lvl in range(mi.get_num_levels()):
            fz, fy, fx = mi.level_factors(lvl)
            exp = ref[::fz, ::fy, ::fx]
            arr = mi.levels[lvl]
            assert arr.shape == exp.shape
            np.testing.assert_array_equal(arr[:, :, :], exp)
            np.testing.assert_array_equal(arr[2, 3:20:3, 1:], exp[2, 3:20:3, 1:])
            np.testing.assert_array_equal(arr[:, -1, :5], exp[:, -1, :5])
            np.testing.assert_array_equal(arr[-1, 2, 3], exp[-1, 2, 3])
            np.testing.assert_array_equal(arr[1:4, :, 2], exp[1:4, :, 2])
            assert arr[0, 4:4, :].shape == (0, exp.shape[2])
            assert arr[3:3, :, 1].shape == (0, exp.shape[1])
            np.testing.assert_array_equal(mi.read_volume(lvl, (2, 1, 3)), exp[::2, :, ::3])
            for axis, idx, plane in (("xy", 1, exp[1]), ("xz", 2, exp[:, 2]), ("yz", 3, exp[:, :, 3])):
                np.testing.assert_array_equal(mi.get_slice_2d(lvl, axis, idx), plane)
    finally:
        f.close()


def test_anisotropic_voxels_keep_coarse_axis(h5file):
    mi, f = _open(h5file)
    try:
        assert mi.voxel_size() == (2.0, 1.0, 1.0)
        assert mi.level_factors(1) == (1, 2, 2)      # z (2 µm) waits for y/x to catch up
        assert mi.level_factors(2) == (2, 4, 4)
        assert mi.transforms[2]["scale"] == [4.0, 4.0, 4.0]
    finally:
        f.close()


def test_read_estimate_counts_file_reads(h5file):
    mi, f = _open(h5file)
    try:
        lvl = mi.get_num_levels() - 1
        assert mi.estimate_read_bytes(lvl) >= mi.levels[lvl].nbytes
    finally:
        f.close()


def test_find_dataset_and_spec(h5file):
    with hdf5.open_file(h5file) as f:
        assert hdf5.find_dataset(f) == "/exchange/data"
        cands = [c[0] for c in hdf5.candidates(f)]
        assert cands[:2] == ["/exchange/data", "/exchange/data_white"]
        assert "/exchange/theta" not in cands
        with pytest.raises(KeyError):
            hdf5.find_dataset(f, "/nope")
    assert hdf5.split_spec(f"{h5file}:exchange/data_white") == (str(h5file), "/exchange/data_white")
    assert hdf5.split_spec(str(h5file)) == (str(h5file), None)
    assert hdf5.is_hdf5(f"{h5file}:/exchange/data")
    assert not hdf5.is_hdf5(str(h5file.parent))


def test_open_image(h5file, ref):
    mi, f, dset = hdf5.open_image(f"{h5file}:/exchange/data_white")
    try:
        assert dset == "/exchange/data_white" and mi.levels[0].shape == (4, 133, 97)
        assert mi.axis_units is None and mi.voxel_size() == (1.0, 1.0, 1.0)   # no voxel attrs
    finally:
        f.close()
    mi, f, dset = hdf5.open_image(h5file)
    try:
        assert dset == "/exchange/data" and mi.voxel_size() == (2.0, 1.0, 1.0)
        assert mi.axis_units == ["micrometer"] * 3
    finally:
        f.close()


def test_strided_factors():
    assert strided_factors((64, 64, 64), coarsest=64) == []
    assert strided_factors((2048, 8192, 8192))[-1] == (32, 32, 32)
    # A thin stack: z stops at min_side, y/x keep halving.
    assert all(f[0] <= 2 for f in strided_factors((50, 8000, 8000)))
    assert strided_factors((1, 4000, 3000))[0] == (1, 2, 2)


def test_strided_level_on_numpy_2d():
    img = np.arange(30 * 40).reshape(30, 40)
    lvl = StridedLevel(Volume3D(img), (1, 3, 4))
    assert lvl.shape == (1, 10, 10)
    np.testing.assert_array_equal(lvl[0], img[::3, ::4])


# ---------------------------------------------------------------- direct reads

@pytest.fixture(scope="module")
def contiguous_files(tmp_path_factory):
    rng = np.random.default_rng(5)
    d = tmp_path_factory.mktemp("direct")
    data = {
        "f4le": rng.random((23, 41, 37)).astype("<f4"),
        "u2be": rng.integers(0, 60000, (17, 29, 31)).astype(">u2"),
        "i1_4d": rng.integers(-100, 100, (3, 9, 11, 13)).astype("i1"),
        "u1_2d": rng.integers(0, 255, (50, 70)).astype("u1"),
    }
    path = d / "contig.h5"
    with h5py.File(path, "w") as f:
        for k, v in data.items():
            f.create_dataset(k, data=v)
        f.create_dataset("chunked", data=data["f4le"], chunks=(4, 8, 8))
        f.create_dataset("gz", data=data["f4le"], compression="gzip")
        f.create_dataset("empty", shape=(4, 5, 6), dtype="f4")       # never written
    return path, data


KEYS = [(slice(None),), (3,), (-1, slice(2, 30, 3)), (slice(1, 20, 4), slice(None), slice(5, 36, 7)),
        (slice(None, None, 2), 7, slice(None)), (slice(None), slice(None), 4), (2, 5, 6), (slice(5, 5),),
        (slice(0, 16, 1), slice(None), slice(None))]


def test_direct_reader_matches_numpy(contiguous_files):
    path, data = contiguous_files
    with hdf5.open_file(path) as f:
        for name, ref in data.items():
            raw = hdf5.direct_reader(f, name)
            assert raw is not None, name
            try:
                for key in KEYS:
                    key = key[:ref.ndim]
                    try:
                        exp = ref[key]
                    except IndexError:
                        with pytest.raises(IndexError):
                            raw[key]
                        continue
                    got = raw[key]
                    assert got.shape == exp.shape, (name, key)
                    np.testing.assert_array_equal(got, exp)
                    assert got.dtype.isnative and got.dtype == exp.dtype.newbyteorder("=")
            finally:
                raw.close()
        for name in ("chunked", "gz", "empty"):
            assert hdf5.direct_reader(f, name) is None, name


def test_direct_reader_merged_runs(contiguous_files, monkeypatch):
    path, data = contiguous_files
    monkeypatch.setattr(hdf5, "_RUN_MAX", 3 * 37 * 4)   # force several merged runs per plane
    with hdf5.open_file(path) as f:
        raw = hdf5.direct_reader(f, "f4le")
        np.testing.assert_array_equal(raw[4], data["f4le"][4])
        np.testing.assert_array_equal(raw[2:9], data["f4le"][2:9])
        raw.close()
        with pytest.raises(ValueError):
            raw[0]


def test_open_image_uses_direct_reader(contiguous_files):
    path, data = contiguous_files
    mi, f, dset = hdf5.open_image(f"{path}:/f4le")
    try:
        assert isinstance(mi.levels[0].array, hdf5.ContiguousDataset)
        np.testing.assert_array_equal(mi.levels[0][:, :, :], data["f4le"])
        np.testing.assert_array_equal(mi.read_volume(0, (2, 3, 2)), data["f4le"][::2, ::3, ::2])
    finally:
        mi.close()
        f.close()
    mi, f, _ = hdf5.open_image(f"{path}:/gz")
    try:
        assert not isinstance(mi.levels[0].array, hdf5.ContiguousDataset)
    finally:
        mi.close()
        f.close()
