"""Interactive high-quality 3D pane (VTK) for the viewer.

VTK renders *off-screen* and the frames are drawn in a plain Qt widget, with
the camera driven from Qt mouse events. Embedding VTK's own GL window in a Qt
application is fragile (X visual / GLX config mismatches depending on driver
and toolkit); off-screen rendering works wherever VTK works, and lets the
live view use the same filmic finish as the final renders. Frames are drawn
fast while dragging and refined (supersampled, full shadows) when idle.

Same volume loader as the VisPy pane (budgeted, background, progressive),
but rendered with the offline renderer's pipeline (zview.render.scene):
look-driven colour/opacity/material, lighting rigs, volumetric shadows and
jittered ray casting. Everything you set here can be reproduced offline:
"Still…" renders the current view supersampled with filmic finishing, and
"Render job…" hands look, window and camera to File › Render 3D….
"""

import argparse
import math
import subprocess
import threading

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from zview.core import log as _log, sysinfo
from zview.viewer.theme import PLANE_COLORS

try:
    import vtk
    from vtk.util import numpy_support as ns
    from zview.render import config as rcfg
    from zview.render import looks as lk
    from zview.render import scene as vr
    VTK_AVAILABLE = True
    _VTK_ERROR = ""
except Exception as _e:  # vtk missing
    VTK_AVAILABLE = False
    _VTK_ERROR = str(_e)

_BASE_PRESETS = ("generic", "interior", "multiphase", "metal", "xray")
_QUALITY_ITEMS = [("Auto", "auto"), ("Preview · 128³", "preview"), ("Low · 256³", "low"),
                  ("Medium · 512³", "medium"), ("High · 1024³", "high"), ("Max · 2048³", "max")]
_STILL_SIZES = ["1920x1080", "2560x1440", "3840x2160", "2400x2400", "1280x720"]


def caps_from_vtk(rw):
    """GLCaps from a VTK render window (+ nvidia-smi for VRAM on NVIDIA)."""
    caps = sysinfo.GLCaps()
    try:
        for line in rw.ReportCapabilities().splitlines():
            key, _, val = line.partition(":")
            key = key.strip().lower()
            if key == "opengl vendor string":
                caps.vendor = val.strip()
            elif key == "opengl renderer string":
                caps.renderer = val.strip()
            elif key == "opengl version string":
                caps.version = val.strip()
    except Exception:
        pass
    if "nvidia" in caps.vendor.lower():
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,memory.free",
                                  "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=4).stdout.strip().splitlines()
            tot, free = (int(float(v)) for v in out[0].split(","))
            caps.vram_total_mb, caps.vram_free_mb = tot, free
        except Exception:
            pass
    return caps


class _FrameCanvas(QtWidgets.QWidget):
    """Shows rendered frames and turns mouse input into camera requests."""

    dragged = QtCore.pyqtSignal(float, float, object, object)   # dx, dy, buttons, modifiers
    dragStarted = QtCore.pyqtSignal()
    dragFinished = QtCore.pyqtSignal()
    wheeled = QtCore.pyqtSignal(float)
    resized = QtCore.pyqtSignal()
    doubleClicked = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(False)
        self.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent)
        self.image = None
        self._last = None
        self.message = ""

    def set_frame(self, qimg):
        self.image = qimg
        self.update()

    def paintEvent(self, _event):
        qp = QtGui.QPainter(self)
        qp.fillRect(self.rect(), QtGui.QColor(0, 0, 0))
        if self.image is not None:
            qp.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
            qp.drawImage(self.rect(), self.image)
        elif self.message:
            qp.setPen(QtGui.QColor(130, 130, 130))
            qp.drawText(self.rect(), QtCore.Qt.AlignCenter, self.message)
        qp.end()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.resized.emit()

    def mousePressEvent(self, e):
        self._last = e.pos()
        self.dragStarted.emit()

    def mouseMoveEvent(self, e):
        if self._last is None:
            return
        d = e.pos() - self._last
        self._last = e.pos()
        self.dragged.emit(float(d.x()), float(d.y()), e.buttons(), e.modifiers())

    def mouseReleaseEvent(self, _e):
        self._last = None
        self.dragFinished.emit()

    def mouseDoubleClickEvent(self, _e):
        self.doubleClicked.emit()

    def wheelEvent(self, e):
        self.wheeled.emit(e.angleDelta().y() / 120.0)


class VTKVolumeView(QtWidgets.QWidget):
    """High-quality interactive 3D pane (same interface as VispyVolumeView)."""

    renderJobRequested = QtCore.pyqtSignal(dict)    # view state for the Render 3D dialog
    _smoothed = QtCore.pyqtSignal(object, object)   # (result, smoothed uint8) from a worker
    _field_ready = QtCore.pyqtSignal(object, object, object)   # (result, layers key, field) for labels

    def __init__(self, loader, compact=False, parent=None):
        super().__init__(parent)
        self.loader = loader
        self.compact = compact
        self.auto_load = True
        self.rw = None              # off-screen vtkRenderWindow
        self.ren = None
        self.volume = None
        self.mapper = None
        self._result = None         # VolumeResult on screen
        self._data = None           # uint8 array actually uploaded (maybe smoothed)
        self._position = None
        self._caps_done = False
        self._needs_reset = True
        self._interacting = False
        self._plane_actors = {}
        self._box_actor = None
        self._build_ui()
        loader.loadStarted.connect(self._on_load_started)
        loader.progress.connect(self._on_progress)
        loader.volumeReady.connect(self._on_volume_ready)
        loader.loadFailed.connect(lambda m: self.status_label.setText(f"⚠ {m}"))
        loader.loadCancelled.connect(lambda: self.status_label.setText("load cancelled"))
        loader.busyChanged.connect(self._on_busy)
        self._smoothed.connect(self._on_smoothed)
        self._field_ready.connect(self._on_field_ready)
        self._layers = None
        self._field_key = None
        self._refine_timer = QtCore.QTimer(self)
        self._refine_timer.setSingleShot(True)
        self._next_level = 1
        self._refine_timer.timeout.connect(lambda: self._render(self._next_level))

    # ------------------------------------------------------------ UI

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        if not VTK_AVAILABLE:
            msg = QtWidgets.QLabel(f"VTK 3D unavailable\n\npip install vtk\n\n{_VTK_ERROR}")
            msg.setAlignment(QtCore.Qt.AlignCenter)
            msg.setStyleSheet("color:#c88; padding:20px;")
            root.addWidget(msg)
            return

        bar = QtWidgets.QWidget()
        bar.setObjectName("vtkBar")
        bar.setStyleSheet("#vtkBar { background:#262626; border-bottom:1px solid #3a3a3a; }")
        hb = QtWidgets.QHBoxLayout(bar)
        hb.setContentsMargins(4, 2, 4, 2)
        hb.setSpacing(4)

        self.render_btn = QtWidgets.QToolButton()
        self.render_btn.setText("Load")
        self.render_btn.setToolTip("Load the volume at the selected quality")
        self.render_btn.clicked.connect(self._request_load)
        hb.addWidget(self.render_btn)
        self.quality_combo = QtWidgets.QComboBox()
        for label, key in _QUALITY_ITEMS:
            self.quality_combo.addItem(label, key)
        self.quality_combo.setToolTip("Volume size budget; Auto fits this machine's GPU/RAM")
        self.quality_combo.activated.connect(lambda _: self._request_load(progressive=False))
        hb.addWidget(self.quality_combo)

        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.addItem("preset…", None)
        for name, desc in rcfg.available_presets().items():
            if name in _BASE_PRESETS or name not in rcfg.PRESETS:
                self.preset_combo.addItem(name, name)
                self.preset_combo.setItemData(self.preset_combo.count() - 1, desc, QtCore.Qt.ToolTipRole)
        self.preset_combo.activated.connect(self._on_preset)
        hb.addWidget(self.preset_combo)
        self.look_combo = QtWidgets.QComboBox()
        self.look_combo.setEditable(True)
        self.look_combo.addItems(sorted(lk.LOOKS))
        self.look_combo.setCurrentText("bone")
        self.look_combo.activated.connect(lambda _: self._apply_style())
        self.look_combo.lineEdit().editingFinished.connect(self._apply_style)
        hb.addWidget(self.look_combo)
        self.light_combo = QtWidgets.QComboBox()
        self.light_combo.addItems(list(lk.LIGHTING))
        self.light_combo.activated.connect(lambda _: self._apply_style())
        hb.addWidget(self.light_combo)
        for w in (self.quality_combo, self.preset_combo, self.look_combo, self.light_combo):
            w.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        hb.addStretch(1)

        def tool(text, tip, slot, checkable=False):
            b = QtWidgets.QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.setCheckable(checkable)
            (b.toggled if checkable else b.clicked).connect(slot)
            hb.addWidget(b)
            return b

        self.settings_btn = tool("⚙", "Rendering settings", lambda on: self.settings_panel.setVisible(on), True)
        tool("📷", "Save a still of this view (supersampled, filmic)", self.save_still)
        tool("▶ job", "Offline render of this view (File › Render 3D…)", self._emit_job)
        tool("⟲", "Reset camera", self.reset_camera)
        root.addWidget(bar)

        self.settings_panel = self._build_settings()
        self.settings_panel.setVisible(not self.compact)
        self.settings_btn.setChecked(not self.compact)
        root.addWidget(self.settings_panel)
        self.classes_panel = self._build_classes_panel()
        self.classes_panel.hide()
        root.addWidget(self.classes_panel)

        self.canvas = _FrameCanvas()
        self.canvas.message = "Open a dataset to render it in 3D"
        self.canvas.setToolTip("Drag: rotate · Shift/middle-drag: pan · Ctrl/right-drag: roll · "
                               "wheel: zoom · double-click: reset")
        self.canvas.dragged.connect(self._on_drag)
        self.canvas.dragStarted.connect(self._start_interaction)
        self.canvas.dragFinished.connect(self._end_interaction)
        self.canvas.wheeled.connect(self._on_wheel)
        self.canvas.resized.connect(lambda: self._render(0))
        self.canvas.doubleClicked.connect(self.reset_camera)
        root.addWidget(self.canvas, 1)

        st = QtWidgets.QWidget()
        st.setObjectName("vtkStatus")
        st.setStyleSheet("#vtkStatus { background:#1e1e1e; }")
        sb = QtWidgets.QHBoxLayout(st)
        sb.setContentsMargins(6, 1, 4, 1)
        self.status_label = QtWidgets.QLabel("")
        self.status_label.setStyleSheet("color:#aaa; font-size:11px;")
        sb.addWidget(self.status_label, 1)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setMaximumHeight(12)
        self.progress.setMaximumWidth(140)
        self.progress.setTextVisible(False)
        self.progress.hide()
        sb.addWidget(self.progress)
        self.cancel_btn = QtWidgets.QToolButton()
        self.cancel_btn.setText("✕")
        self.cancel_btn.setAutoRaise(True)
        self.cancel_btn.clicked.connect(self.loader.cancel)
        self.cancel_btn.hide()
        sb.addWidget(self.cancel_btn)
        root.addWidget(st)

    def _build_classes_panel(self):
        panel = QtWidgets.QWidget()
        panel.setObjectName("vtkClasses")
        panel.setStyleSheet("#vtkClasses { background:#2b2b2b; }")
        v = QtWidgets.QVBoxLayout(panel)
        v.setContentsMargins(8, 2, 8, 4)
        v.setSpacing(2)
        head = QtWidgets.QHBoxLayout()
        self.classes_title = QtWidgets.QLabel("Classes")
        self.classes_title.setStyleSheet("font-weight:bold;")
        head.addWidget(self.classes_title)
        head.addStretch(1)
        for text, fn in (("All", lambda: self._set_all_visible(True)), ("None", lambda: self._set_all_visible(False)),
                         ("Defaults", self._reset_classes)):
            b = QtWidgets.QToolButton()
            b.setText(text)
            b.clicked.connect(fn)
            head.addWidget(b)
        self.classes_toggle = QtWidgets.QToolButton()
        self.classes_toggle.setText("▾")
        self.classes_toggle.setCheckable(True)
        self.classes_toggle.setChecked(not self.compact)   # compact (quad) pane: list starts folded
        self.classes_toggle.setToolTip("Show / hide the class list")
        head.addWidget(self.classes_toggle)
        v.addLayout(head)
        t = QtWidgets.QTableWidget(0, 4)
        t.setHorizontalHeaderLabels(["Class", "%", "Colour", "Opacity"])
        t.verticalHeader().hide()
        t.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        for col in (1, 2, 3):
            t.horizontalHeader().setSectionResizeMode(col, QtWidgets.QHeaderView.ResizeToContents)
        t.setMaximumHeight(170 if self.compact else 260)
        t.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        t.itemChanged.connect(self._on_class_item_changed)
        self.classes_table = t
        self.classes_toggle.toggled.connect(t.setVisible)
        t.setVisible(self.classes_toggle.isChecked())
        v.addWidget(t)
        return panel

    def _build_settings(self):
        panel = QtWidgets.QWidget()
        panel.setObjectName("vtkSettings")
        panel.setStyleSheet("#vtkSettings { background:#2b2b2b; }")
        outer = QtWidgets.QHBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        inner = QtWidgets.QWidget()
        inner.setMaximumWidth(640)
        outer.addWidget(inner)
        outer.addStretch(1)
        g = QtWidgets.QGridLayout(inner)
        g.setContentsMargins(8, 4, 8, 4)
        g.setVerticalSpacing(2)

        def slider(lo, hi, val, tip):
            s = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            s.setRange(lo, hi)
            s.setValue(val)
            s.setToolTip(tip)
            s.valueChanged.connect(self._apply_style)
            return s

        self.win_lo = slider(0, 254, 3, "Window low (material starts above this)")
        self.win_hi = slider(1, 255, 255, "Window high")
        self.opacity_slider = slider(1, 100, 60, "Opacity per voxel (%)")
        self.scatter_slider = slider(0, 200, 100, "Volumetric shadows (0 = off, 100 = rig default, 200 = max)")
        self.step_slider = slider(25, 300, 50, "Ray step (×0.01 voxel): smaller = finer, slower")
        self.curve_combo = QtWidgets.QComboBox()
        self.curve_combo.addItems(["(from look)", "solid", "soft", "layered", "glass", "mip"])
        self.curve_combo.currentIndexChanged.connect(self._apply_style)
        rows = [("Window", self.win_lo, self.win_hi), ("Opacity", self.opacity_slider, None),
                ("Shadows", self.scatter_slider, None), ("Ray step", self.step_slider, None)]
        for r, (label, a, b) in enumerate(rows):
            g.addWidget(QtWidgets.QLabel(label), r, 0)
            if b is None:
                g.addWidget(a, r, 1, 1, 2)
            else:
                g.addWidget(a, r, 1)
                g.addWidget(b, r, 2)
        opts = QtWidgets.QHBoxLayout()
        opts.addWidget(QtWidgets.QLabel("Curve"))
        opts.addWidget(self.curve_combo)
        self.smooth_check = QtWidgets.QCheckBox("Smooth")
        self.smooth_check.setToolTip("Gaussian pre-smoothing (σ 0.8 voxel): cleaner surfaces on noisy data")
        self.smooth_check.toggled.connect(self._on_smooth_toggled)
        self.box_check = QtWidgets.QCheckBox("Box")
        self.box_check.setChecked(True)
        self.planes_check = QtWidgets.QCheckBox("Planes")
        self.planes_check.setChecked(True)
        self.lod_check = QtWidgets.QCheckBox("Fast drag")
        self.lod_check.setChecked(True)
        self.lod_check.setToolTip("Coarser rays and no shadows while rotating")
        for c in (self.smooth_check, self.box_check, self.planes_check, self.lod_check):
            opts.addWidget(c)
        for c in (self.box_check, self.planes_check):
            c.toggled.connect(self._update_overlays)
        self.bg_combo = QtWidgets.QComboBox()
        self.bg_combo.addItems(["look", "black", "grey", "white"])
        self.bg_combo.currentIndexChanged.connect(self._apply_style)
        opts.addWidget(self.bg_combo)
        opts.addStretch(1)
        g.addLayout(opts, len(rows), 0, 1, 3)
        g.setColumnStretch(1, 1)
        g.setColumnStretch(2, 1)
        return panel

    # ------------------------------------------------------------ canvas

    def _ensure_canvas(self):
        if self.rw is not None or not VTK_AVAILABLE:
            return self.rw is not None
        try:
            self.rw = vr.offscreen_render_window()
            self.rw.SetMultiSamples(0)
            self.ren = vtk.vtkRenderer()
            self.rw.AddRenderer(self.ren)
            self.rw.SetSize(max(16, self.canvas.width()), max(16, self.canvas.height()))
            self.rw.Render()
            if not self._caps_done:
                self._caps_done = True
                self.loader.set_gl_caps(caps_from_vtk(self.rw))
        except Exception as e:
            _log.error(f"[VTK 3D] could not create an off-screen render window: {e}")
            self.rw = None
            self.canvas.message = f"Could not start VTK rendering:\n{e}"
            self.canvas.update()
            return False
        return True

    def showEvent(self, event):
        super().showEvent(event)
        if not VTK_AVAILABLE or self.loader.multires is None:
            return
        if self._ensure_canvas():
            res = self.loader.result
            if res is not None and res is not self._result:
                self._show_result(res)
            elif self.volume is not None and not self.ren.HasViewProp(self.volume):
                self.ren.AddVolume(self.volume)
                self._render(refine=True)
            QtCore.QTimer.singleShot(0, self._maybe_autoload)

    def hideEvent(self, event):
        super().hideEvent(event)
        self._refine_timer.stop()
        # Free the 3D texture while hidden (the loader keeps the data).
        if self.volume is not None and self.ren is not None and self.ren.HasViewProp(self.volume):
            self.ren.RemoveVolume(self.volume)
            try:
                self.mapper.ReleaseGraphicsResources(self.rw)
            except Exception:
                pass

    def _set_lod(self, on):
        """Draft quality (coarse rays, no shadows) on/off."""
        if self.mapper is None:
            return
        if on:
            self.mapper.SetSampleDistance(max(1.0, self.step_slider.value() / 50.0))
            vr.set_scatter(self.mapper, 0.0)
        else:
            args = self._style_args()
            light = lk.get_lighting(args.lighting) if not str(args.lighting).lstrip().startswith("{") \
                else vr.resolve_style(args)[1]
            self.mapper.SetSampleDistance(args.sample_distance)
            vr.set_scatter(self.mapper, args.scatter if light["shade"] else 0.0, args.gi_reach)

    def _render(self, level=0, refine=True):
        """Progressive rendering.

        level 0: draft (coarse rays, no shadows) — instant feedback;
        level 1: full quality at screen resolution;
        level 2: full quality, 2x supersampled.
        Unless `refine` is False, the next level is scheduled after an idle
        pause, so the image sharpens by itself once you stop interacting.
        """
        self._refine_timer.stop()
        if self.mapper is not None and self.mapper.GetInput() is None:
            return            # data not attached yet (e.g. a resize during upload)
        if self.rw is None or not self.isVisible() or self.volume is None:
            if self.rw is not None and self.isVisible():
                self.canvas.image = None
                self.canvas.update()
            return
        draft = level == 0 and self.lod_check.isChecked()
        self._set_lod(draft)
        w, h = max(16, self.canvas.width()), max(16, self.canvas.height())
        ss = 2 if level >= 2 else 1
        if list(self.rw.GetSize()) != [w * ss, h * ss]:
            self.rw.SetSize(w * ss, h * ss)
        self.ren.ResetCameraClippingRange()
        self.rw.Render()
        w2i = vtk.vtkWindowToImageFilter()
        w2i.SetInput(self.rw)
        w2i.ReadFrontBufferOff()
        w2i.Update()
        im = ns.vtk_to_numpy(w2i.GetOutput().GetPointData().GetScalars()).reshape(h * ss, w * ss, -1)
        rgb = np.ascontiguousarray(lk.finish(im[::-1, :, :3], 1.0, "filmic", ss))
        qimg = QtGui.QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0],
                            QtGui.QImage.Format_RGB888).copy()
        self.canvas.set_frame(qimg)
        if draft:
            self._set_lod(False)
        if refine and level < 2 and not self._interacting:
            self._next_level = level + 1
            self._refine_timer.start(150 if level == 0 else 500)

    # ------------------------------------------------------------ data

    def set_dataset(self, multires):
        if self.ren is not None:
            for a in list(self._plane_actors.values()) + ([self._box_actor] if self._box_actor else []):
                self.ren.RemoveActor(a)
            if self.volume is not None:
                self.ren.RemoveVolume(self.volume)
        self.volume = self.mapper = None
        self._plane_actors, self._box_actor = {}, None
        self._result = self._data = None
        self._position = None
        self._needs_reset = True
        if not VTK_AVAILABLE:
            return
        self.status_label.setText("")
        if multires is not None and self.isVisible() and self._ensure_canvas():
            QtCore.QTimer.singleShot(0, self._maybe_autoload)
        self._render()


    def set_position(self, z, y, x):
        self._position = (int(z), int(y), int(x))
        self._update_overlays()

    def _maybe_autoload(self):
        if (self.auto_load and self.isVisible() and self.loader.multires is not None
                and self.loader.result is None and not self.loader.busy
                and not getattr(self.loader, "user_cancelled", False)):
            self._request_load()

    def _request_load(self, progressive=True):
        if self.loader.multires is None or not self._ensure_canvas():
            return
        self.loader.request(self.quality_combo.currentData(), progressive=progressive)

    def _on_volume_ready(self, res):
        if self.isVisible() and self._ensure_canvas():
            self._show_result(res)

    def _spacing(self, res):
        mi = self.loader.multires
        vs = mi.voxel_size() if mi is not None else (1.0, 1.0, 1.0)
        m = min(vs)
        az, ay, ax = (min(100.0, max(0.01, v / m)) for v in vs)
        wz, wy, wx = res.world_scale
        return (wx * ax, wy * ay, wz * az), (az, ay, ax)

    def _show_result(self, res):
        self._result = res
        if self.smooth_check.isChecked() and res.labels is None:
            self._start_smoothing(res)
        self._upload(res, res.data)

    def _upload(self, res, data):
        self._data = data
        spacing, _ = self._spacing(res)
        args = self._style_args()
        look, light = vr.resolve_style(args)
        if self.mapper is None:
            self.mapper = vr.volume_mapper(vtk, args, light)
            self.volume = vtk.vtkVolume()
            self.volume.SetMapper(self.mapper)
            self.ren.AddVolume(self.volume)
        labels = res.labels is not None
        self._set_label_ui(labels)
        if labels:
            self._layers = vr.label_layers(res.labels, res.code_values)
            # Unsmoothed shading field now (instant), smoothed one from a worker.
            vr.set_label_inputs(vtk, ns, self.mapper, data, self._layers, spacing, sigma=0)
            self._field_key = None
            self._populate_classes(res)
            self._refresh_field()
        else:
            self.mapper.SetInputData(vr.image_data(vtk, ns, data, spacing))
        self._apply_style()
        self._build_box()
        self._update_overlays()
        if self._needs_reset:
            self.reset_camera()
            self._needs_reset = False
        self.status_label.setText(res.describe() + (" · smoothed" if data is not res.data else ""))
        self._render()

    # ------------------------------------------------------------ smoothing

    def _on_smooth_toggled(self, on):
        res = self._result
        if res is None:
            return
        if res.labels is not None:
            self._refresh_field()
            return
        if on:
            self._start_smoothing(res)
        else:
            self._upload(res, res.data)

    def _start_smoothing(self, res):
        self.status_label.setText(res.describe() + " · smoothing…")

        def work():
            try:
                from scipy import ndimage as ndi
                sm = ndi.gaussian_filter(res.data.astype(np.float32), 0.8)
                self._smoothed.emit(res, np.clip(sm + 0.5, 0, 255).astype(np.uint8))
            except Exception as e:
                _log.warn(f"[VTK 3D] smoothing failed: {e}")

        threading.Thread(target=work, daemon=True).start()

    def _on_smoothed(self, res, data):
        if res is self._result and self.smooth_check.isChecked():
            self._upload(res, data)

    # ------------------------------------------------------------ style

    def _style_args(self):
        curve = None if self.curve_combo.currentIndex() == 0 else self.curve_combo.currentText()
        light = lk.get_lighting(self.light_combo.currentText())
        scatter = light["scatter"] * self.scatter_slider.value() / 100.0
        return argparse.Namespace(
            look=self.look_combo.currentText().strip() or "bone", lighting=self.light_combo.currentText(),
            curve=curve, opacity=self.opacity_slider.value() / 100.0, mask_level=20, interior=1.0,
            specular=None, scatter=scatter, gi_reach=0.4, anisotropy=0.0, exposure=None,
            tone_mapping="filmic", supersample=2, background=self.bg_combo.currentText(),
            sample_distance=self.step_slider.value() / 100.0)

    def _window(self):
        lo, hi = self.win_lo.value(), self.win_hi.value()
        if lo >= hi:
            if self.sender() is self.win_lo:
                hi = min(255, lo + 1)
                self.win_hi.blockSignals(True); self.win_hi.setValue(hi); self.win_hi.blockSignals(False)
            else:
                lo = max(0, hi - 1)
                self.win_lo.blockSignals(True); self.win_lo.setValue(lo); self.win_lo.blockSignals(False)
        return float(lo), float(hi)

    def _apply_style(self, *_):
        if self.ren is None:
            return
        try:
            args = self._style_args()
            look, light = vr.resolve_style(args)
        except ValueError as e:
            self.status_label.setText(f"⚠ {e}")
            return
        vr.add_lights(vtk, self.ren, light)
        vr.apply_background(self.ren, look, args.background)
        if self.volume is not None:
            if self._result is not None and self._result.labels is not None and self._layers is not None:
                self.volume.SetProperty(vr.label_volume_property(vtk, self._layers, light, look["material"]))
            else:
                self.volume.SetProperty(vr.volume_property(vtk, look, light, args, self._window()))
        self._render(0)

    def _on_preset(self, idx):
        name = self.preset_combo.itemData(idx)
        if not name:
            return
        c = rcfg.effective({"preset": name})
        self.look_combo.setCurrentText(c["look"] if isinstance(c["look"], str) else "bone")
        self.light_combo.setCurrentText(c["lighting"] if isinstance(c["lighting"], str) else "studio")
        self.curve_combo.setCurrentText(c["curve"] or "(from look)")
        self.opacity_slider.blockSignals(True)
        self.opacity_slider.setValue(int(round(c["opacity"] * 100)))
        self.opacity_slider.blockSignals(False)
        if c.get("scatter") is not None:
            base = lk.get_lighting(c["lighting"])["scatter"] or 1.0
            self.scatter_slider.blockSignals(True)
            self.scatter_slider.setValue(int(round(100 * c["scatter"] / base)) if base else 0)
            self.scatter_slider.blockSignals(False)
        self.smooth_check.setChecked(bool(c.get("smooth")))
        # Preset window percentiles -> 0..255 window on the loaded volume.
        if self._result is not None:
            flat = self._result.data.reshape(-1)
            sample = flat[:: max(1, flat.size // 1_000_000)]
            lo, hi = (int(v) for v in np.percentile(sample, [c["lo_pct"], c["hi_pct"]]))
            for s, v in ((self.win_lo, lo), (self.win_hi, max(hi, lo + 1))):
                s.blockSignals(True); s.setValue(v); s.blockSignals(False)
        self.preset_combo.setCurrentIndex(0)
        self._apply_style()

    # ------------------------------------------------------------ geometry

    def _to_scene(self, z, y, x):
        """Level-0 voxel (z, y, x) -> VTK scene (x, y, z) of the uploaded texture."""
        res = self._result
        spacing, _ = self._spacing(res)
        wz, wy, wx = res.world_scale
        oz, oy, ox = res.world_offset
        tz = (z - oz - (wz - 1) / 2) / wz
        ty = (y - oy - (wy - 1) / 2) / wy
        tx = (x - ox - (wx - 1) / 2) / wx
        return (tx * spacing[0], ty * spacing[1], tz * spacing[2])

    def _line_actor(self, pts, rgb, width=1.5, opacity=0.9):
        poly = vtk.vtkPolyData()
        p = vtk.vtkPoints()
        for q in pts:
            p.InsertNextPoint(*q)
        cells = vtk.vtkCellArray()
        n = len(pts)
        for i in range(0, n, 2):
            cells.InsertNextCell(2)
            cells.InsertCellPoint(i)
            cells.InsertCellPoint(i + 1)
        poly.SetPoints(p)
        poly.SetLines(cells)
        m = vtk.vtkPolyDataMapper()
        m.SetInputData(poly)
        a = vtk.vtkActor()
        a.SetMapper(m)
        pr = a.GetProperty()
        pr.SetColor(*rgb); pr.SetLineWidth(width); pr.SetOpacity(opacity); pr.LightingOff()
        return a

    def _build_box(self):
        mi = self.loader.multires
        if self.ren is None or mi is None or self._result is None:
            return
        if self._box_actor is not None:
            self.ren.RemoveActor(self._box_actor)
        Z, Y, X = mi.get_level_shape(0)
        c = [self._to_scene(z, y, x) for z in (-0.5, Z - 0.5) for y in (-0.5, Y - 0.5) for x in (-0.5, X - 0.5)]
        edges = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
        self._box_actor = self._line_actor([c[i] for e in edges for i in e], (0.6, 0.6, 0.6), 1.0, 0.6)
        self.ren.AddActor(self._box_actor)

    def _update_overlays(self, *_):
        mi = self.loader.multires
        if self.ren is None or mi is None or self._result is None:
            return
        if self._box_actor is not None:
            self._box_actor.SetVisibility(self.box_check.isChecked())
        for a in self._plane_actors.values():
            self.ren.RemoveActor(a)
        self._plane_actors = {}
        if self._position is not None and self.planes_check.isChecked():
            Z, Y, X = mi.get_level_shape(0)
            z, y, x = self._position
            z0, y0, x0, z1, y1, x1 = -0.5, -0.5, -0.5, Z - 0.5, Y - 0.5, X - 0.5
            rects = {'XY': [(z, y0, x0), (z, y0, x1), (z, y1, x1), (z, y1, x0)],
                     'XZ': [(z0, y, x0), (z0, y, x1), (z1, y, x1), (z1, y, x0)],
                     'YZ': [(z0, y0, x), (z0, y1, x), (z1, y1, x), (z1, y0, x)]}
            for name, pts in rects.items():
                seg = []
                for i in range(4):
                    seg += [self._to_scene(*pts[i]), self._to_scene(*pts[(i + 1) % 4])]
                a = self._line_actor(seg, PLANE_COLORS[name])
                self._plane_actors[name] = a
                self.ren.AddActor(a)
        self._render()

    # ------------------------------------------------------------ camera

    def _center_radius(self):
        mi = self.loader.multires
        Z, Y, X = mi.get_level_shape(0)
        p0 = np.array(self._to_scene(-0.5, -0.5, -0.5))
        p1 = np.array(self._to_scene(Z - 0.5, Y - 0.5, X - 0.5))
        return (p0 + p1) / 2, 0.5 * np.linalg.norm(p1 - p0)

    def reset_camera(self, az=-60.0, el=25.0):
        if self.ren is None or self._result is None:
            return
        c, r = self._center_radius()
        dist = r / math.tan(math.radians(15)) * 0.85
        a, e = math.radians(az), math.radians(el)
        cam = self.ren.GetActiveCamera()
        cam.SetFocalPoint(*c)
        cam.SetPosition(*(c + dist * np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), -math.sin(e)])))
        cam.SetViewUp(0, 0, -1)              # z runs down, like the 2D panes
        cam.SetViewAngle(30)
        self.ren.ResetCameraClippingRange()
        self._render()

    def camera_state(self):
        """(azimuth, elevation, zoom) of the current view, as zview-render uses them."""
        cam = self.ren.GetActiveCamera()
        c, r = self._center_radius()
        d = np.array(cam.GetPosition()) - np.array(cam.GetFocalPoint())
        dist = float(np.linalg.norm(d)) or 1.0
        el = math.degrees(math.asin(max(-1.0, min(1.0, -d[2] / dist))))
        az = math.degrees(math.atan2(d[1], d[0])) % 360
        zoom = (r / math.tan(math.radians(15)) * 0.85) / dist
        return round(az, 2), round(el, 2), round(zoom, 3)

    def _start_interaction(self, *_):
        self._refine_timer.stop()
        self._interacting = True

    def _end_interaction(self, *_):
        self._interacting = False
        self._next_level = 1
        self._refine_timer.start(150)

    def _on_drag(self, dx, dy, buttons, mods):
        if self.ren is None or self.volume is None:
            return
        cam = self.ren.GetActiveCamera()
        h = max(1, self.canvas.height())
        pan = bool(buttons & QtCore.Qt.MiddleButton) or bool(mods & QtCore.Qt.ShiftModifier)
        roll = bool(buttons & QtCore.Qt.RightButton) or bool(mods & QtCore.Qt.ControlModifier)
        if pan:
            fp, pos = np.array(cam.GetFocalPoint()), np.array(cam.GetPosition())
            dist = np.linalg.norm(pos - fp)
            up = np.array(cam.GetViewUp())
            right = np.cross(fp - pos, up)
            right /= np.linalg.norm(right) or 1.0
            k = 2 * dist * math.tan(math.radians(cam.GetViewAngle() / 2)) / h
            shift = (-dx * right + dy * up) * k
            cam.SetFocalPoint(*(fp + shift))
            cam.SetPosition(*(pos + shift))
        elif roll:
            cam.Roll(dx * 0.4)
        else:
            cam.Azimuth(-dx * 0.4)
            cam.Elevation(dy * 0.4)
            cam.OrthogonalizeViewUp()
        self._render(0, refine=False)

    def _on_wheel(self, steps):
        if self.ren is None or self.volume is None:
            return
        self.ren.GetActiveCamera().Dolly(1.12 ** steps)
        self._render(0)

    # ------------------------------------------------------------ labels

    def _set_label_ui(self, labels):
        self.classes_panel.setVisible(labels)
        for w in (self.win_lo, self.win_hi, self.opacity_slider, self.curve_combo):
            w.setEnabled(not labels)
        self.smooth_check.setText("Smooth surfaces" if labels else "Smooth")
        if labels and not self.smooth_check.isChecked():
            self.smooth_check.blockSignals(True)
            self.smooth_check.setChecked(True)
            self.smooth_check.blockSignals(False)

    def _populate_classes(self, res):
        t = self.classes_table
        t.blockSignals(True)
        t.setRowCount(0)
        present = {v for _c, v, *_ in self._layers}
        rows = [c for v, c in sorted(res.labels.classes.items()) if v in present]
        for c in rows:
            r = t.rowCount()
            t.insertRow(r)
            it = QtWidgets.QTableWidgetItem(f"{c.value} · {c.name}")
            it.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            it.setCheckState(QtCore.Qt.Checked if c.visible else QtCore.Qt.Unchecked)
            it.setData(QtCore.Qt.UserRole, c.value)
            it.setToolTip(c.name)
            t.setItem(r, 0, it)
            pct = QtWidgets.QTableWidgetItem(f"{c.fraction * 100:.2f}" if c.fraction < 0.1 else f"{c.fraction * 100:.0f}")
            pct.setFlags(QtCore.Qt.ItemIsEnabled)
            t.setItem(r, 1, pct)
            btn = QtWidgets.QPushButton()
            btn.setFixedSize(34, 18)
            self._paint_swatch(btn, c.color)
            btn.clicked.connect(lambda _=False, v=c.value, b=btn: self._pick_color(v, b))
            t.setCellWidget(r, 2, btn)
            sp = QtWidgets.QSpinBox()
            sp.setRange(0, 100)
            sp.setSuffix(" %")
            sp.setValue(int(round(c.opacity * 100)))
            sp.valueChanged.connect(lambda val, v=c.value: self._set_class(v, opacity=val / 100.0))
            t.setCellWidget(r, 3, sp)
        t.blockSignals(False)
        self.classes_title.setText(f"Classes ({len(rows)}) — {res.labels.source}")

    @staticmethod
    def _paint_swatch(btn, rgb):
        r, g, b = (int(round(255 * x)) for x in rgb)
        btn.setStyleSheet(f"QPushButton {{ background: rgb({r},{g},{b}); border: 1px solid #777; }}")

    def _pick_color(self, value, btn):
        c = self._result.labels.classes[value]
        q = QtWidgets.QColorDialog.getColor(QtGui.QColor.fromRgbF(*c.color), self, f"Colour of {c.name}")
        if q.isValid():
            self._paint_swatch(btn, (q.redF(), q.greenF(), q.blueF()))
            self._set_class(value, color=(q.redF(), q.greenF(), q.blueF()))

    def _on_class_item_changed(self, item):
        if item.column() != 0:
            return
        v = item.data(QtCore.Qt.UserRole)
        self._set_class(v, visible=item.checkState() == QtCore.Qt.Checked)

    def _set_class(self, value, **kw):
        res = self._result
        if res is None or res.labels is None:
            return
        c = res.labels.classes[value]
        for k, val in kw.items():
            setattr(c, k, val)
        if kw.get("visible") and c.opacity <= 0:
            c.opacity = 1.0
            self._populate_classes(res)
        self._layers = vr.label_layers(res.labels, res.code_values)
        self._apply_style()          # colours / opacities: instant
        self._refresh_field()        # visibility / glass<->opaque: new shading field

    def _set_all_visible(self, on):
        res = self._result
        if res is None or res.labels is None:
            return
        for c in res.labels.classes.values():
            if c.role != "background":
                c.visible = on
                if on and c.opacity <= 0:
                    c.opacity = 1.0
        self._populate_classes(res)
        self._layers = vr.label_layers(res.labels, res.code_values)
        self._apply_style()
        self._refresh_field()

    def _reset_classes(self):
        from zview.core.labels import build_table  # noqa: PLC0415
        res = self._result
        if res is None or res.labels is None:
            return
        t = res.labels
        fresh = build_table(t.classes.keys(), {v: c.fraction for v, c in t.classes.items()},
                            {v: c.name for v, c in t.classes.items()}, source=t.source)
        t.classes = fresh.classes
        self._populate_classes(res)
        self._layers = vr.label_layers(t, res.code_values)
        self._apply_style()
        self._refresh_field()

    def _layers_key(self):
        return (tuple((code, kind) for code, _v, _rgb, _op, kind in self._layers or ()),
                self.smooth_check.isChecked())

    def _refresh_field(self):
        res = self._result
        if res is None or res.labels is None or self._layers is None:
            return
        if not vr.label_masks_supported(vtk):
            return            # older VTK: codes are rendered directly (no shading field)
        key = self._layers_key()
        if key == self._field_key:
            return
        layers, data = list(self._layers), self._data
        sigma = 1.0 if self.smooth_check.isChecked() else 0.0

        def work():
            try:
                field = vr.label_shading_field(data, layers, sigma)
                self._field_ready.emit(res, key, field)
            except Exception as e:
                _log.warn(f"[VTK 3D] label shading failed: {e}")

        threading.Thread(target=work, daemon=True).start()

    def _on_field_ready(self, res, key, field):
        if res is not self._result or key != self._layers_key() or self.mapper is None:
            return
        spacing, _ = self._spacing(res)
        self.mapper.SetInputData(vr.image_data(vtk, ns, field, spacing))
        self._field_key = key
        self._render(0)

    # ------------------------------------------------------------ export

    def view_state(self):
        """Look / window / camera as zview-render config values."""
        if self._result is None:
            return {}
        args = self._style_args()
        if self._result.labels is not None:
            az, el, zoom = self.camera_state()
            state = {"look": args.look, "lighting": args.lighting, "background": args.background,
                     "preprocess": "labels", "labels": self._result.labels.to_config(),
                     "label_smooth": 1.0 if self.smooth_check.isChecked() else 0.0,
                     "azimuth": az, "elev": el, "elevations": [el], "zoom": float(zoom)}
            light = lk.get_lighting(args.lighting)
            if abs(args.scatter - light["scatter"]) > 1e-6:
                state["scatter"] = round(args.scatter, 3)
            return state
        lo, hi = self._window()
        vlo, vhi = self._result.value_range
        raw = [vlo + lo / 255.0 * (vhi - vlo), vlo + hi / 255.0 * (vhi - vlo)]
        az, el, zoom = self.camera_state()
        state = {"look": args.look, "lighting": args.lighting, "opacity": args.opacity,
                 "background": args.background, "preprocess": "none",
                 "window_values": [round(v, 6) for v in raw], "smooth": 0.8 if self.smooth_check.isChecked() else 0.0,
                 "azimuth": az, "elev": el, "elevations": [el], "zoom": float(zoom)}
        if args.curve:
            state["curve"] = args.curve
        light = lk.get_lighting(args.lighting)
        if abs(args.scatter - light["scatter"]) > 1e-6:
            state["scatter"] = round(args.scatter, 3)
        return state

    def _emit_job(self):
        if self._result is None:
            QtWidgets.QMessageBox.information(self, "Render job", "Load the volume first.")
            return
        self.renderJobRequested.emit(self.view_state())

    def save_still(self):
        if self._result is None or self.ren is None:
            return
        size, ok = QtWidgets.QInputDialog.getItem(self, "Save still", "Image size:", _STILL_SIZES, 0, True)
        if not ok:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save still", "zview_still.png",
                                                        "PNG (*.png);;JPEG (*.jpg)")
        if not path:
            return
        try:
            W, H = (int(v) for v in size.lower().split("x"))
        except ValueError:
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            args = self._style_args()
            spacing, _ = self._spacing(self._result)
            labels = self._layers if self._result.labels is not None else None
            args.label_smooth = 1.0 if self.smooth_check.isChecked() else 0.0
            sc = vr.Scene(self._data, args, W, H, spacing=spacing, window=self._window(), labels=labels)
            cam, src = sc.cam, self.ren.GetActiveCamera()
            cam.SetPosition(*src.GetPosition()); cam.SetFocalPoint(*src.GetFocalPoint())
            cam.SetViewUp(*src.GetViewUp()); cam.SetViewAngle(src.GetViewAngle())
            # Keep the pane's framing at the new aspect ratio.
            sc.ren.ResetCameraClippingRange()
            for a in [self._box_actor] + list(self._plane_actors.values()):
                if a is not None and a.GetVisibility():
                    sc.ren.AddActor(a)
            im = np.ascontiguousarray(sc.grab())
            sc.close()
            qimg = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_RGB888)
            qimg.save(path)
            self.status_label.setText(f"saved {path}")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Save still", str(e))
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self._render()

    # ------------------------------------------------------------ loader UI

    def _on_load_started(self, desc):
        if VTK_AVAILABLE:
            self.status_label.setText(desc + " …")
            self.progress.setRange(0, 0)

    def _on_progress(self, done, total):
        if VTK_AVAILABLE:
            self.progress.setRange(0, max(1, total))
            self.progress.setValue(done)

    def _on_busy(self, busy):
        if VTK_AVAILABLE:
            self.progress.setVisible(busy)
            self.cancel_btn.setVisible(busy)
            self.render_btn.setText("Loading…" if busy else ("Reload" if self.loader.result else "Load"))

    def closeEvent(self, event):
        self._refine_timer.stop()
        if self.rw is not None:
            try:
                self.rw.Finalize()
            except Exception:
                pass
        super().closeEvent(event)
