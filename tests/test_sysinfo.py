from zview.core import sysinfo


def test_plan_picks_finest_fitting_level():
    caps = sysinfo.GLCaps(vendor="NVIDIA", renderer="GPU", max_3d_texture=16384,
                          vram_total_mb=11000, vram_free_mb=9000)
    b = sysinfo.VolumeBudget(max_axis=1024, max_voxels=1024 ** 3)
    shapes = [(4096, 8192, 8192), (2048, 4096, 4096), (1024, 2048, 2048), (512, 1024, 1024)]
    p = sysinfo.plan_volume(shapes, b)
    assert p.level == 3 and p.stride == (1, 1, 1)
    assert sysinfo.volume_budget(caps, 1, "low").max_axis == 256


def test_plan_strides_when_nothing_fits():
    b = sysinfo.VolumeBudget(max_axis=1024, max_voxels=1024 ** 3)
    p = sysinfo.plan_volume([(100_000, 100_000, 100_000)], b)
    assert max(p.out_shape) <= 1024 and p.voxels <= b.max_voxels
    p = sysinfo.plan_volume([(20, 30000, 30000)], sysinfo.VolumeBudget(256, 256 ** 3))
    assert p.stride[0] == 1


def test_software_renderer_tier():
    soft = sysinfo.GLCaps(renderer="llvmpipe (LLVM 15)", max_3d_texture=2048)
    assert sysinfo.volume_budget(soft, 2).max_axis == 256


def test_env_override(monkeypatch):
    monkeypatch.setenv("ZVIEW_3D_MAX_AXIS", "300")
    assert sysinfo.volume_budget(None, 1, "auto").max_axis == 300
