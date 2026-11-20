"""Fast, minimal 3D pane (VisPy) — the fallback 3D engine.

Renders the volume from the shared `VolumeLoader` with VisPy: MIP /
attenuated MIP / MinIP / translucent / additive / average / isosurface,
colormap, contrast, gamma, plus the bounding box and the three slice planes.
The high-quality engine is `zview.viewer.volume_vtk`.
"""


import traceback

import numpy as np
from PyQt5 import QtCore, QtWidgets

from zview.core import log as _log
from zview.core import sysinfo
from zview.viewer.loader import VolumeLoader, _fmt_bytes

try:
    sysinfo.apply_gl_compat()
    from vispy import app as _vapp
    _vapp.use_app("pyqt5")
    from vispy import scene
    from vispy.color import get_colormaps
    VISPY_AVAILABLE = True
    _VISPY_ERROR = ""
except Exception as _e:  # ImportError, or no usable Qt backend
    VISPY_AVAILABLE = False
    _VISPY_ERROR = str(_e)


from zview.viewer.theme import PLANE_COLORS  # noqa: E402

_QUALITY_ITEMS = [
    ("Auto", "auto"),
    ("Preview · 128³", "preview"),
    ("Low · 256³", "low"),
    ("Medium · 512³", "medium"),
    ("High · 1024³", "high"),
    ("Max · 2048³", "max"),
]

_MODE_ITEMS = [
    ("MIP", "mip"),
    ("Attenuated MIP", "attenuated_mip"),
    ("MinIP", "minip"),
    ("Translucent", "translucent"),
    ("Additive", "additive"),
    ("Average", "average"),
    ("Isosurface", "iso"),
]

_PREFERRED_CMAPS = ["grays", "viridis", "hot", "fire", "ice", "cubehelix",
                    "coolwarm", "reds", "greens", "blues", "orange"]



class VispyVolumeView(QtWidgets.QWidget):
    """Compact VisPy volume pane with a toolbar and collapsible settings."""

    def __init__(self, loader=None, compact=False, parent=None):
        super().__init__(parent)
        self._own_loader = loader is None
        if loader is None:
            loader = VolumeLoader(self)
        self.loader = loader
        self.compact = compact
        self.auto_load = True

        self.canvas = None
        self.view = None
        self.volume = None
        self._box_line = None
        self._plane_line = None
        self._shown_result = None
        self._position = None
        self._caps_queried = False
        self._needs_camera_reset = True

        self._build_ui()

        loader.loadStarted.connect(self._on_load_started)
        loader.progress.connect(self._on_progress)
        loader.volumeReady.connect(self._on_volume_ready)
        loader.loadFailed.connect(self._on_failed)
        loader.loadCancelled.connect(self._on_cancelled)
        loader.busyChanged.connect(self._on_busy_changed)
        loader.capsChanged.connect(self._update_tooltip)

    # ------------------------------------------------------------ UI

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        if not VISPY_AVAILABLE:
            msg = QtWidgets.QLabel(
                "3D rendering unavailable\n\n"
                "Install VisPy + PyOpenGL:\n  pip install vispy PyOpenGL\n\n"
                f"{_VISPY_ERROR}")
            msg.setAlignment(QtCore.Qt.AlignCenter)
            msg.setWordWrap(True)
            msg.setStyleSheet("color:#c88; padding:20px;")
            root.addWidget(msg)
            return

        bar = QtWidgets.QWidget()
        bar.setObjectName("vol3dBar")
        bar.setStyleSheet("#vol3dBar { background:#262626; border-bottom:1px solid #3a3a3a; }")
        hb = QtWidgets.QHBoxLayout(bar)
        hb.setContentsMargins(4, 2, 4, 2)
        hb.setSpacing(4)

        self.render_btn = QtWidgets.QToolButton()
        self.render_btn.setText("Render")
        self.render_btn.setToolTip("Load the volume at the selected quality")
        self.render_btn.clicked.connect(self._on_render_clicked)
        hb.addWidget(self.render_btn)

        self.quality_combo = QtWidgets.QComboBox()
        for label, key in _QUALITY_ITEMS:
            self.quality_combo.addItem(label, key)
        self.quality_combo.setToolTip("Texture size budget. Auto sizes it to this machine's GPU/RAM.")
        self.quality_combo.activated.connect(self._on_quality_changed)
        hb.addWidget(self.quality_combo)

        self.mode_combo = QtWidgets.QComboBox()
        for label, key in _MODE_ITEMS:
            self.mode_combo.addItem(label, key)
        self.mode_combo.currentIndexChanged.connect(self._apply_render_settings)
        hb.addWidget(self.mode_combo)

        self.cmap_combo = QtWidgets.QComboBox()
        available = set(get_colormaps().keys())
        for name in _PREFERRED_CMAPS:
            if name in available:
                self.cmap_combo.addItem(name)
        self.cmap_combo.currentIndexChanged.connect(self._apply_render_settings)
        hb.addWidget(self.cmap_combo)

        hb.addStretch(1)

        self.settings_btn = QtWidgets.QToolButton()
        self.settings_btn.setText("⚙")
        self.settings_btn.setCheckable(True)
        self.settings_btn.setToolTip("Show rendering settings")
        self.settings_btn.toggled.connect(lambda on: self.settings_panel.setVisible(on))
        hb.addWidget(self.settings_btn)

        reset_btn = QtWidgets.QToolButton()
        reset_btn.setText("⟲")
        reset_btn.setToolTip("Reset camera")
        reset_btn.clicked.connect(self.reset_camera)
        hb.addWidget(reset_btn)

        for w in (self.quality_combo, self.mode_combo, self.cmap_combo):
            w.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
            w.setMinimumContentsLength(4)
        root.addWidget(bar)

        self.settings_panel = self._build_settings_panel()
        self.settings_panel.setVisible(not self.compact)
        self.settings_btn.setChecked(not self.compact)
        root.addWidget(self.settings_panel)

        self.canvas_container = QtWidgets.QWidget()
        self.canvas_container.setStyleSheet("background:black;")
        self.canvas_layout = QtWidgets.QVBoxLayout(self.canvas_container)
        self.canvas_layout.setContentsMargins(0, 0, 0, 0)
        self.placeholder = QtWidgets.QLabel("Open a dataset to render it in 3D")
        self.placeholder.setAlignment(QtCore.Qt.AlignCenter)
        self.placeholder.setStyleSheet("color:#777;")
        self.canvas_layout.addWidget(self.placeholder)
        root.addWidget(self.canvas_container, 1)

        status = QtWidgets.QWidget()
        status.setObjectName("vol3dStatus")
        status.setStyleSheet("#vol3dStatus { background:#1e1e1e; }")
        sb = QtWidgets.QHBoxLayout(status)
        sb.setContentsMargins(6, 1, 4, 1)
        sb.setSpacing(6)
        self.status_label = QtWidgets.QLabel("")
        self.status_label.setStyleSheet("color:#aaa; font-size:11px;")
        sb.addWidget(self.status_label, 1)
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setMaximumHeight(12)
        self.progress_bar.setMaximumWidth(140)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.hide()
        sb.addWidget(self.progress_bar)
        self.cancel_btn = QtWidgets.QToolButton()
        self.cancel_btn.setText("✕")
        self.cancel_btn.setToolTip("Cancel loading")
        self.cancel_btn.setAutoRaise(True)
        self.cancel_btn.clicked.connect(self.loader.cancel)
        self.cancel_btn.hide()
        sb.addWidget(self.cancel_btn)
        root.addWidget(status)
        self._update_tooltip()

    def _build_settings_panel(self):
        panel = QtWidgets.QWidget()
        panel.setObjectName("vol3dSettings")
        panel.setStyleSheet("#vol3dSettings { background:#2b2b2b; }")
        outer = QtWidgets.QHBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        inner = QtWidgets.QWidget()
        inner.setMaximumWidth(620)
        outer.addWidget(inner)
        outer.addStretch(1)
        grid = QtWidgets.QGridLayout(inner)
        grid.setContentsMargins(8, 4, 8, 4)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(2)

        def slider(lo, hi, val, tip):
            s = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            s.setRange(lo, hi)
            s.setValue(val)
            s.setToolTip(tip)
            s.valueChanged.connect(self._apply_render_settings)
            return s

        self.clim_lo = slider(0, 254, 0, "Lower contrast limit")
        self.clim_hi = slider(1, 255, 255, "Upper contrast limit")
        self.gamma_slider = slider(10, 300, 100, "Gamma (×0.01)")
        self.iso_slider = slider(0, 255, 128, "Isosurface threshold")
        self.atten_slider = slider(1, 200, 20, "Attenuation (attenuated MIP)")
        self.step_slider = slider(25, 400, 80, "Ray step size (×0.01 texel); larger = faster")

        rows = [("Contrast", self.clim_lo, self.clim_hi),
                ("Gamma", self.gamma_slider, None),
                ("Iso level", self.iso_slider, None),
                ("Attenuation", self.atten_slider, None),
                ("Step size", self.step_slider, None)]
        for r, (label, a, b) in enumerate(rows):
            grid.addWidget(QtWidgets.QLabel(label), r, 0)
            if b is None:
                grid.addWidget(a, r, 1, 1, 2)
            else:
                grid.addWidget(a, r, 1)
                grid.addWidget(b, r, 2)

        opts = QtWidgets.QHBoxLayout()
        self.box_check = QtWidgets.QCheckBox("Box")
        self.box_check.setChecked(True)
        self.planes_check = QtWidgets.QCheckBox("Slice planes")
        self.planes_check.setChecked(True)
        self.interp_check = QtWidgets.QCheckBox("Smooth")
        self.interp_check.setChecked(True)
        for c in (self.box_check, self.planes_check, self.interp_check):
            c.toggled.connect(self._apply_render_settings)
            opts.addWidget(c)
        opts.addWidget(QtWidgets.QLabel("Background"))
        self.bg_combo = QtWidgets.QComboBox()
        self.bg_combo.addItems(["Black", "Dark grey", "White"])
        self.bg_combo.currentIndexChanged.connect(self._apply_render_settings)
        opts.addWidget(self.bg_combo)
        auto_btn = QtWidgets.QPushButton("Auto contrast")
        auto_btn.clicked.connect(self.auto_contrast)
        opts.addWidget(auto_btn)
        opts.addStretch(1)
        grid.addLayout(opts, len(rows), 0, 1, 3)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)
        return panel

    def _update_tooltip(self):
        if not VISPY_AVAILABLE:
            return
        lines = [sysinfo.describe_machine(self.loader.gl_caps)]
        if self.loader.multires is not None:
            b = self.loader.budget()
            p = self.loader.plan()
            lines.append(f"Budget: {b.summary()}")
            lines.append("  limited by: " + "; ".join(b.reasons))
            if p is not None:
                lines.append(f"Plan: level {p.level} {p.level_shape} stride {p.stride} → {p.out_shape}")
                lines.append("  reads ~" + _fmt_bytes(
                    self.loader.multires.estimate_read_bytes(p.level, p.stride)))
        self.status_label.setToolTip("\n".join(lines))

    # ------------------------------------------------------------ canvas

    def _ensure_canvas(self):
        if self.canvas is not None or not VISPY_AVAILABLE:
            return self.canvas is not None
        try:
            self.canvas = scene.SceneCanvas(keys=None, show=False, bgcolor="black")
            self.canvas.create_native()
            self.canvas.native.setParent(self.canvas_container)
            self.placeholder.hide()
            self.canvas_layout.addWidget(self.canvas.native)
            self.view = self.canvas.central_widget.add_view()
            self.view.camera = scene.cameras.TurntableCamera(fov=45, elevation=25, azimuth=-60, up="-z")
            self.canvas.events.draw.connect(self._drain_gl_errors, position="first")
            self.canvas.events.draw.connect(self._on_first_draw, position="first")
        except Exception as e:
            traceback.print_exc()
            self.canvas = None
            self.placeholder.setText(f"Could not create OpenGL canvas:\n{e}")
            self.placeholder.show()
            return False
        return True

    @staticmethod
    def _drain_gl_errors(_event=None):
        """Clear GL errors left by Qt before VisPy draws.

        QOpenGLWidget re-creates its framebuffer in this context on resize,
        and some drivers (Mesa) leave a GL_INVALID_ENUM behind. VisPy checks
        the error flag before each draw and would blame — and skip — the
        volume for that frame.
        """
        try:
            from vispy.gloo import gl  # noqa: PLC0415
            for _ in range(16):
                if gl.glGetError() == 0:
                    break
        except Exception:
            pass

    def _on_first_draw(self, _event):
        if self._caps_queried:
            return
        self._caps_queried = True
        try:
            self.canvas.events.draw.disconnect(self._on_first_draw)
        except Exception:
            pass
        caps = sysinfo.query_gl_caps()
        self.loader.set_gl_caps(caps)
        self._update_tooltip()
        # Start loading once we know the GPU's limits.
        QtCore.QTimer.singleShot(0, self._maybe_autoload)

    def _maybe_autoload(self):
        if (self.auto_load and self.isVisible() and self.loader.multires is not None
                and self.loader.result is None and not self.loader.busy
                and not self.loader.user_cancelled):
            self.loader.request(self.quality_combo.currentData())

    def showEvent(self, event):
        super().showEvent(event)
        if not VISPY_AVAILABLE or self.loader.multires is None:
            return
        if self._ensure_canvas():
            if self.loader.result is not None and self._shown_result is not self.loader.result:
                self._show_result(self.loader.result)
            elif self._caps_queried:
                QtCore.QTimer.singleShot(0, self._maybe_autoload)

    def hideEvent(self, event):
        super().hideEvent(event)
        # Free the texture while hidden; the loader keeps the numpy copy and
        # re-uploading on show is fast. Keeps two 3D panes from doubling VRAM.
        self._release_volume()

    def _release_volume(self):
        if self.volume is not None:
            self.volume.parent = None
            self.volume = None
            self._shown_result = None

    # ------------------------------------------------------------ data

    def set_dataset(self, multires):
        """New dataset (or None). The view's own loader is reset if it owns one."""
        if self._own_loader:
            self.loader.set_dataset(multires)
        self._release_volume()
        for line in (self._box_line, self._plane_line):
            if line is not None:
                line.parent = None
        self._box_line = self._plane_line = None
        self._needs_camera_reset = True
        self._position = None
        if not VISPY_AVAILABLE:
            return
        self.status_label.setText("")
        self.placeholder.setText("Open a dataset to render it in 3D" if multires is None
                                 else "Waiting for GPU…")
        self._update_tooltip()
        if multires is not None and self.isVisible() and self._ensure_canvas():
            if self._caps_queried:
                QtCore.QTimer.singleShot(0, self._maybe_autoload)
        if self.canvas is not None:
            self.canvas.update()

    def set_position(self, z, y, x):
        self._position = (int(z), int(y), int(x))
        self._update_planes()

    def _on_volume_ready(self, res):
        if self.isVisible() and self._ensure_canvas():
            self._show_result(res)
        self._update_tooltip()

    def _aspect(self):
        mi = self.loader.multires
        if mi is None:
            return (1.0, 1.0, 1.0)
        vs = mi.voxel_size()
        m = min(vs)
        return tuple(min(100.0, max(0.01, v / m)) for v in vs)

    def _show_result(self, res):
        if self.canvas is None:
            return
        az, ay, ax = self._aspect()
        wz, wy, wx = res.world_scale
        oz, oy, ox = res.world_offset
        # Texel i sits at visual coord i; its centre in level-0 voxel space is
        # offset + i*ws + (ws-1)/2. Scene units = level-0 voxels × aspect.
        scale = (wx * ax, wy * ay, wz * az)
        translate = ((ox + (wx - 1) / 2) * ax, (oy + (wy - 1) / 2) * ay, (oz + (wz - 1) / 2) * az)
        try:
            if self.volume is None:
                self.volume = scene.visuals.Volume(
                    res.data, parent=self.view.scene, method=self.mode_combo.currentData(),
                    cmap=self.cmap_combo.currentText() or "grays",
                    clim=(0, 255), texture_format="auto", relative_step_size=0.8,
                )
            else:
                self.volume.set_data(res.data, clim=(0, 255))
            self.volume.transform = scene.STTransform(scale=scale, translate=translate)
        except Exception as e:
            traceback.print_exc()
            self._on_failed(f"GPU upload failed: {e}")
            return
        self._shown_result = res
        if res.labels is not None:
            self.volume.interpolation = "nearest"
        self._build_box()
        self._update_planes()
        self._apply_render_settings()
        if self._needs_camera_reset:
            self.reset_camera()
            self._needs_camera_reset = False
        self.status_label.setText(res.describe() + (" · label map: switch View › 3D engine to VTK for class "
                                                    "colours" if res.labels is not None else ""))
        self.canvas.update()

    def _extent(self):
        mi = self.loader.multires
        if mi is None:
            return None
        return mi.get_level_shape(0)

    def _scene_pt(self, z, y, x):
        az, ay, ax = self._aspect()
        return (x * ax, y * ay, z * az)

    def _build_box(self):
        shape = self._extent()
        if shape is None or self.view is None:
            return
        Z, Y, X = shape
        lo = (-0.5, -0.5, -0.5)
        hi = (Z - 0.5, Y - 0.5, X - 0.5)
        corners = [self._scene_pt(z, y, x) for z in (lo[0], hi[0])
                   for y in (lo[1], hi[1]) for x in (lo[2], hi[2])]
        edges = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7),
                 (0, 4), (1, 5), (2, 6), (3, 7)]
        pos = np.array([corners[i] for e in edges for i in e], dtype=np.float32)
        if self._box_line is None:
            self._box_line = scene.visuals.Line(pos=pos, connect="segments",
                                                color=(0.6, 0.6, 0.6, 0.6), width=1,
                                                parent=self.view.scene)
            self._box_line.set_gl_state("translucent", depth_test=False)
            self._box_line.order = 10
        else:
            self._box_line.set_data(pos=pos)

    def _update_planes(self):
        shape = self._extent()
        if shape is None or self.view is None or self._position is None:
            return
        Z, Y, X = shape
        z, y, x = self._position
        z0, y0, x0 = -0.5, -0.5, -0.5
        z1, y1, x1 = Z - 0.5, Y - 0.5, X - 0.5
        rects = {
            'XY': [(z, y0, x0), (z, y0, x1), (z, y1, x1), (z, y1, x0)],
            'XZ': [(z0, y, x0), (z0, y, x1), (z1, y, x1), (z1, y, x0)],
            'YZ': [(z0, y0, x), (z0, y1, x), (z1, y1, x), (z1, y0, x)],
        }
        pos, col = [], []
        for name, pts in rects.items():
            c = PLANE_COLORS[name] + (0.9,)
            for i in range(4):
                pos.append(self._scene_pt(*pts[i]))
                pos.append(self._scene_pt(*pts[(i + 1) % 4]))
                col.extend([c, c])
        pos = np.array(pos, dtype=np.float32)
        col = np.array(col, dtype=np.float32)
        if self._plane_line is None:
            self._plane_line = scene.visuals.Line(pos=pos, color=col, connect="segments",
                                                  width=1.5, parent=self.view.scene)
            self._plane_line.set_gl_state("translucent", depth_test=False)
            self._plane_line.order = 11
        else:
            self._plane_line.set_data(pos=pos, color=col)
        self._plane_line.visible = self.planes_check.isChecked()
        if self.canvas is not None:
            self.canvas.update()

    # ------------------------------------------------------------ settings

    def _apply_render_settings(self, *_):
        if not VISPY_AVAILABLE:
            return
        lo, hi = self.clim_lo.value(), self.clim_hi.value()
        if lo >= hi:
            # Keep at least one step between the handles.
            if self.sender() is self.clim_lo:
                self.clim_lo.blockSignals(True)
                self.clim_lo.setValue(hi - 1)
                self.clim_lo.blockSignals(False)
                lo = hi - 1
            else:
                self.clim_hi.blockSignals(True)
                self.clim_hi.setValue(lo + 1)
                self.clim_hi.blockSignals(False)
                hi = lo + 1
        mode = self.mode_combo.currentData()
        self.iso_slider.setEnabled(mode == "iso")
        self.atten_slider.setEnabled(mode == "attenuated_mip")
        if self.canvas is not None:
            self.canvas.bgcolor = {0: "black", 1: (0.16, 0.16, 0.16, 1), 2: "white"}[
                self.bg_combo.currentIndex()]
        if self._box_line is not None:
            self._box_line.visible = self.box_check.isChecked()
        if self._plane_line is not None:
            self._plane_line.visible = self.planes_check.isChecked()
        v = self.volume
        if v is None:
            if self.canvas is not None:
                self.canvas.update()
            return
        try:
            v.method = mode
            v.cmap = self.cmap_combo.currentText() or "grays"
            v.clim = (float(lo), float(hi))
            v.gamma = self.gamma_slider.value() / 100.0
            # VisPy compares the iso threshold with the clim-normalised sample
            # (0..1), not with raw texel values.
            v.threshold = float(np.clip((self.iso_slider.value() - lo) / max(1, hi - lo), 0.0, 1.0))
            v.attenuation = self.atten_slider.value() / 100.0
            v.relative_step_size = self.step_slider.value() / 100.0
            v.interpolation = "linear" if self.interp_check.isChecked() else "nearest"
        except Exception as e:
            _log.warn(f"[3D] could not apply settings: {e}")
        self.canvas.update()

    def auto_contrast(self):
        res = self._shown_result
        if res is None:
            return
        flat = res.data.reshape(-1)
        sample = flat[::max(1, flat.size // 1_000_000)]
        lo, hi = (int(v) for v in np.percentile(sample, [1, 99.9]))
        hi = max(hi, lo + 1)
        for s, val in ((self.clim_lo, lo), (self.clim_hi, hi)):
            s.blockSignals(True)
            s.setValue(val)
            s.blockSignals(False)
        self._apply_render_settings()

    def reset_camera(self):
        if self.view is None:
            return
        shape = self._extent()
        if shape is None:
            return
        Z, Y, X = shape
        p0 = self._scene_pt(-0.5, -0.5, -0.5)
        p1 = self._scene_pt(Z - 0.5, Y - 0.5, X - 0.5)
        cam = self.view.camera
        cam.set_range(x=(p0[0], p1[0]), y=(p0[1], p1[1]), z=(p0[2], p1[2]), margin=0.08)
        cam.center = tuple((a + b) / 2 for a, b in zip(p0, p1))
        if self.canvas is not None:
            self.canvas.update()

    # ------------------------------------------------------------ loader UI

    def _on_render_clicked(self):
        if self.loader.multires is None:
            return
        if not self._ensure_canvas():
            return
        self.loader.request(self.quality_combo.currentData())

    def _on_quality_changed(self, _idx):
        if self.loader.multires is not None and self.canvas is not None:
            self.loader.request(self.quality_combo.currentData(), progressive=False, auto=True)
        self._update_tooltip()

    def _on_load_started(self, desc):
        if not VISPY_AVAILABLE:
            return
        self.status_label.setText(desc + " …")
        self.progress_bar.setRange(0, 0)
        self.progress_bar.show()
        self.cancel_btn.show()

    def _on_progress(self, done, total):
        if not VISPY_AVAILABLE:
            return
        self.progress_bar.setRange(0, max(1, total))
        self.progress_bar.setValue(done)

    def _on_busy_changed(self, busy):
        if not VISPY_AVAILABLE:
            return
        self.progress_bar.setVisible(busy)
        self.cancel_btn.setVisible(busy)
        self.render_btn.setText("Loading…" if busy else ("Reload" if self.loader.result else "Render"))

    def _on_failed(self, msg):
        if not VISPY_AVAILABLE:
            return
        _log.error(f"[3D] {msg}")
        self.status_label.setText(f"⚠ {msg}")

    def _on_cancelled(self):
        if not VISPY_AVAILABLE:
            return
        shown = self._shown_result
        self.status_label.setText((shown.describe() + " · " if shown else "") + "load cancelled")

    def closeEvent(self, event):
        if self._own_loader:
            self.loader._abort()
        if self.canvas is not None:
            try:
                self.canvas.close()
            except Exception:
                pass
        super().closeEvent(event)

