import json

from zview.render import config as rc


def test_merge_order(tmp_path):
    cfg = {"preset": ["metal", "draft"], "look": "copper", "title": "T", "frames": None}
    c = rc.effective(cfg, {"frames": 42})
    assert c["look"] == "copper"            # config beats preset
    assert c["lo_pct"] == 93.0               # from metal
    assert c["size"] == "960x540"            # from draft (applied after metal)
    assert c["frames"] == 42                 # --set beats everything


def test_parse_set_nested_and_json():
    d = rc.parse_set(["tour.speed=0.5", "z_range=[1,2]", "look=gold"])
    assert d == {"tour": {"speed": 0.5}, "z_range": [1, 2], "look": "gold"}


def test_argv_carries_labels_window_and_camera():
    c = rc.effective({"preset": "labels", "volume": "v.npy", "labels": {"5": {"name": "soma"}}})
    a = rc.turntable_argv(c, "out")
    assert "--labels-json" in a and json.loads(a[a.index("--labels-json") + 1]) == {"5": {"name": "soma"}}
    assert "--preprocess" in a and a[a.index("--preprocess") + 1] == "labels"
    c = rc.effective({"volume": "v.npy", "window_values": [10, 20], "azimuth": 30, "zoom": 1.5})
    a = rc.turntable_argv(c, "out")
    assert a[a.index("--window-values") + 1:a.index("--window-values") + 3] == ["10", "20"]
    assert a[a.index("--azimuth") + 1] == "30" and a[a.index("--zoom") + 1] == "1.5"


def test_custom_look_is_passed_as_json():
    c = rc.effective({"volume": "v.npy", "look": {"colors": [[0, [0, 0, 0]], [255, [1, 1, 1]]]}})
    a = rc.common_argv(c)
    assert json.loads(a[a.index("--look") + 1])["colors"][1][0] == 255


def test_old_dataset_json_cmap(tmp_path):
    p = tmp_path / "dataset.json"
    p.write_text(json.dumps({"volume": "volume_step8.npy", "factor": 8, "cmap": "organic"}))
    cfg = rc.load(p)
    assert cfg["look"] == "organic" and cfg["volume"] == str(tmp_path / "volume_step8.npy")


def test_presets_listed_and_loadable():
    names = rc.available_presets()
    for n in ("generic", "interior", "metal", "labels", "tomo-brain", "draft"):
        assert n in names
        rc.load_preset(n)


def test_validate_catches_typos_and_bad_values():
    probs = rc.validate({"source": "x.zarr", "lok": "gold", "lighting": "sunny", "size": "1920 by 1080",
                         "z_range": [900, 100], "regions": [{"name": "A", "z": 1}],
                         "outputs": ["stills", "tour", "movie"], "tour": {"sped": 0.5}})
    text = "\n".join(probs)
    assert "unknown key 'lok'" in text and "'look'" in text           # did-you-mean
    assert "unknown lighting 'sunny'" in text
    assert "size must look like" in text and "z_range" in text
    assert "regions[0]" in text and "outputs must be" in text and "tour.'sped'" in text


def test_validate_clean_config():
    assert rc.validate({"preset": ["interior", "draft"], "source": "x.zarr", "look": "jade",
                        "regions": [{"name": "A", "z": 1, "y": 2, "x": 3}], "outputs": ["stills", "tour"],
                        "_help": {"anything": 1}}) == []


def test_volume_cache_follows_source_and_size(tmp_path):
    from zview.render.cli import volume_is_current
    vol = tmp_path / "v.npy"
    vol.write_bytes(b"x")
    cfg = {"source": "/data/a.zarr", "volume": str(vol), "volume_source": "/data/a.zarr", "volume_max_axis": 768}
    assert volume_is_current(cfg, rc.effective(cfg))
    assert not volume_is_current(cfg, rc.effective(cfg, {"source": "/data/b.zarr"}))   # another dataset
    assert not volume_is_current(cfg, rc.effective(cfg, {"max_axis": 2048}))            # bigger job
    gone = {**cfg, "volume": str(tmp_path / "missing.npy")}
    assert not volume_is_current(gone, rc.effective(gone))
