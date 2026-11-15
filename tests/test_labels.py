import numpy as np

from conftest import open_mi
from zview.core.labels import LabelTable, build_table, pool_labels, render_codes

NAMES = {3: "grey-matter neuropil", 4: "white matter", 5: "neuron soma", 7: "nucleus",
         50: "vessel wall", 70: "vessel lumen", 255: "void (outside)"}
FRAC = {3: .27, 4: .65, 5: .02, 7: .005, 50: .008, 70: .0017, 255: .022}


def test_defaults_from_names_and_sizes():
    t = build_table(NAMES, FRAC, NAMES)
    assert t.classes[255].role == "background" and not t.classes[255].visible
    assert t.classes[4].role == "bulk" and t.classes[4].opacity < 0.2
    assert t.classes[70].visible and t.classes[70].opacity == 1.0
    rank = t.priority_lut()
    assert rank[70] > rank[5] > rank[3] > rank[255]


def test_pool_keeps_rare_classes():
    t = build_table(NAMES, FRAC, NAMES)
    b = np.full((4, 4, 4), 4, np.uint8)
    b[0, 0, 0] = 70
    b[2, 2, 2] = 255
    b[3, 3, 3] = 5
    out = pool_labels(b, (2, 2, 2), t.priority_lut())
    assert out[0, 0, 0] == 70 and out[1, 1, 1] == 5 and out[0, 1, 0] == 4


def test_detect_and_pooled_read(label_store, labels_data):
    mi = open_mi(label_store)
    t = mi.labels
    assert t is not None and t.source == "metadata"
    assert t.classes[255].role == "background"
    rank = t.priority_lut()
    for f in ((2, 2, 2), (3, 5, 7)):
        got = mi.read_volume_pooled(0, f, rank, out_block=8)
        assert np.array_equal(got, pool_labels(labels_data, f, rank))
        assert (got == 70).mean() > (labels_data[::f[0], ::f[1], ::f[2]] == 70).mean()


def test_intensity_data_is_not_labels(store5d):
    assert open_mi(store5d).labels is None


def test_render_codes_compact_with_background_zero():
    t = build_table(NAMES, FRAC, NAMES)
    data = np.array([[[255, 3, 70, 4, 5]]], np.uint8)
    codes, values = render_codes(data, t)
    assert codes.dtype == np.uint8 and codes[0, 0, 0] == 0
    assert values[0] == 255 and [values[c] for c in codes.ravel()[1:]] == [3, 70, 4, 5]


def test_render_codes_wide_values():
    t = build_table([0, 1000, 70000], {0: .5, 1000: .3, 70000: .2})
    codes, values = render_codes(np.array([0, 1000, 70000], np.int64).reshape(1, 1, 3), t)
    assert codes.ravel().tolist() == [0, 1, 2] and values[1:] == [1000, 70000]


def test_config_roundtrip_keeps_roles():
    t = build_table(NAMES, FRAC, NAMES)
    t2 = LabelTable.from_config(t.to_config())
    assert t2.classes[255].role == "background" and t2.classes[4].role == "bulk"
    assert t2.classes[70].color == t.classes[70].color


def test_low_dynamic_range_image_is_not_labels(tmp_path):
    import zarr
    rng = np.random.default_rng(5)
    img = (40 + rng.integers(-2, 3, (1, 256, 256))).astype(np.uint8)   # 5 values, but noisy
    g = zarr.open_group(str(tmp_path / "img.zarr"), mode="w")
    g.create_array("0", data=img, chunks=(1, 64, 64))
    mi = open_mi(tmp_path / "img.zarr")
    assert mi.labels is None
    from zview.core.labels import detect_labels
    assert detect_labels(mi, force=True) is not None      # View › Treat as label map


def test_unnamed_flat_labels_detected(tmp_path):
    import zarr
    a = np.zeros((64, 64, 64), np.uint16)
    a[10:30, 10:30, 10:30] = 1
    a[40:60, 5:25, 30:50] = 2
    g = zarr.open_group(str(tmp_path / "seg.zarr"), mode="w")
    g.create_array("0", data=a, chunks=(32, 32, 32))
    t = open_mi(tmp_path / "seg.zarr").labels
    assert t is not None and t.classes[0].role == "background"
