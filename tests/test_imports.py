"""Every module imports (GUI modules only when PyQt5 is available)."""

import importlib

import pytest

CORE = ["zview.core.io", "zview.core.pyramid", "zview.core.labels", "zview.core.sysinfo", "zview.core.log",
        "zview.render.config", "zview.render.looks", "zview.render.sources", "zview.render.prepare",
        "zview.render.cli"]
GUI = ["zview.viewer.app", "zview.viewer.panes", "zview.viewer.loader", "zview.viewer.volume_vispy",
       "zview.viewer.volume_vtk", "zview.viewer.render_dialog", "zview.viewer.metadata"]


@pytest.mark.parametrize("mod", CORE)
def test_core_imports(mod):
    importlib.import_module(mod)


@pytest.mark.parametrize("mod", GUI)
def test_gui_imports(mod):
    pytest.importorskip("PyQt5")
    importlib.import_module(mod)
