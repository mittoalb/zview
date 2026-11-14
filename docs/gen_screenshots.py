#!/usr/bin/env python3
"""Regenerate the viewer screenshots used by the documentation.

Run under a virtual X server so no desktop decorations or overlays end up in
the images, with a scratch config dir so no personal recent files show:

    XDG_CONFIG_HOME=$(mktemp -d) xvfb-run -a -s "-screen 0 1700x1000x24" \\
        python docs/gen_screenshots.py [--data /tmp/zview_demo/data] [--render-python PATH]

--render-python is shown in the Render 3D dialog; pass a neutral path (e.g. a
symlink) so no user name appears in the screenshot.
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "docs" / "_static" / "img" / "ui"
sys.path.insert(0, str(ROOT))

from PyQt5 import QtCore, QtWidgets  # noqa: E402

from zview.zview2 import UnifiedZarrViewer, apply_dark_theme  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/tmp/zview_demo/data"))
    ap.add_argument("--render-python", default=None)
    a = ap.parse_args()
    IMG.mkdir(parents=True, exist_ok=True)

    app = QtWidgets.QApplication(sys.argv[:1])
    apply_dark_theme(app)
    v = UnifiedZarrViewer()
    v.resize(1600, 950)
    v.show()

    def pump(sec):
        end = time.time() + sec
        while time.time() < end:
            app.processEvents()
            time.sleep(0.02)

    def wait_3d(sec=120):
        end = time.time() + sec
        while time.time() < end:
            app.processEvents()
            r = v.loader.result
            if r is not None and r.final:
                return
            time.sleep(0.05)

    def shot(widget, name):
        pump(0.5)
        pm = app.primaryScreen().grabWindow(widget.winId())
        pm.save(str(IMG / f"{name}.jpg"), "JPG", 92)
        print("  ->", (IMG / f"{name}.jpg").relative_to(ROOT), flush=True)

    def run():
        v.set_layout("quad")
        v.open_path(str(a.data / "phantom_iso.zarr"))
        wait_3d()
        v.set_position(300, 330, 180)
        pump(3)
        shot(v, "quad")

        v.set_layout("dual")
        pump(1)
        v._frame_of(v.dual_views[1]).plane_combo.setCurrentIndex(2)
        pump(3)
        shot(v, "dual")

        v.set_layout("single")
        v._frame_of(v.single_view).plane_combo.setCurrentIndex(0)
        pump(1)
        v.single_view.view.scaleBy((0.35, 0.35))
        pump(3)
        shot(v, "single_zoom")

        v.set_layout("3d")
        pump(2)
        w = v.view_3d_only
        w.mode_combo.setCurrentIndex(w.mode_combo.findData("translucent"))
        w.cmap_combo.setCurrentIndex(w.cmap_combo.findText("viridis"))
        pump(3)
        shot(v, "view3d")

        v.set_layout("quad")
        v.open_path(str(a.data / "phantom_aniso_5d.zarr"))
        wait_3d()
        pump(3)
        shot(v, "quad_aniso")

        v.open_path(str(a.data / "phantom_iso.zarr"))
        pump(2)
        v.set_position(256, 256, 384)
        if a.render_python:
            v.settings.setValue("render/python", a.render_python)
        v._open_render_dialog()
        d = v._render_dialog
        d.preset_combo.setCurrentIndex(d.preset_combo.findData("interior"))
        d._add_region()
        d.title_edit.setText("Phantom")
        d.subtitle_edit.setText("procedural test sample")
        d.credit_edit.setText("Lab name, Institute")
        end = time.time() + 60
        while time.time() < end and not d._python_ok:
            pump(0.2)
        d.resize(1100, 860)
        pump(1)
        shot(d, "render_dialog")
        d.close()
        app.quit()

    QtCore.QTimer.singleShot(300, run)
    app.exec_()


if __name__ == "__main__":
    main()
