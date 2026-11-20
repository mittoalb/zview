"""File › Render 3D… — configure and launch an offline render from the viewer.

The dialog writes a render config (zview.render.config) next to the chosen
output folder and runs `python -m zview.render run CONFIG ...` as a
background process, streaming its log. The renderer needs VTK, SciPy and
scikit-image; if the viewer's own Python lacks them, the dialog uses another
interpreter (detected automatically, or chosen by hand) and points it at this
copy of zview via PYTHONPATH.
"""

import glob
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from PyQt5 import QtCore, QtGui, QtWidgets

from zview.render import config as rc
from zview.render import looks as lk

PKG_PARENT = str(Path(__file__).resolve().parent.parent)
# find_spec doesn't import anything, so probing many environments is fast.
_PROBE = ("import importlib.util as u, importlib.metadata as m, sys\n"
          "mods = ['vtk', 'scipy', 'skimage', 'imageio', 'imageio_ffmpeg', 'PIL', 'numpy', 'zarr']\n"
          "missing = [x for x in mods if u.find_spec(x) is None]\n"
          "sys.exit(1) if missing else print(m.version('vtk'))")

SIZES = ["960x540", "1280x720", "1920x1080", "2560x1440", "3840x2160", "2400x2400", "1600x1200"]
MODIFIERS = [("—", None), ("Draft (fast)", "draft"), ("Presentation 4K", "presentation-4k"),
             ("Publication stills", "publication")]
VISPY_TO_LOOK = {"grays": "grays", "viridis": "viridis", "hot": "ember", "fire": "ember", "ice": "ice",
                 "cubehelix": "bone", "coolwarm": "coolwarm", "reds": "tissue", "orange": "copper"}


def probe_python(exe, timeout=15):
    """VTK version string if `exe` can run the renderer, else None."""
    try:
        r = subprocess.run([exe, "-c", _PROBE], capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def candidate_pythons(saved=None):
    seen, out = set(), []

    def add(p):
        if p and os.path.isfile(p) and os.access(p, os.X_OK) and os.path.realpath(p) not in seen:
            seen.add(os.path.realpath(p))
            out.append(p)

    add(saved)
    add(sys.executable)
    roots = [os.environ.get("CONDA_PREFIX", ""), os.environ.get("CONDA_EXE", "")]
    homes = [Path.home() / d for d in ("miniconda3", "anaconda3", "miniforge3", "mambaforge", "conda")]
    envs = []
    for h in homes:
        envs += sorted(glob.glob(str(h / "envs" / "*" / "bin" / "python")))
    for r in roots:
        if r:
            base = Path(r).parent.parent if r.endswith("conda") else Path(r)
            envs += sorted(glob.glob(str(base / "envs" / "*" / "bin" / "python")))
    # Likely render envs first.
    envs.sort(key=lambda p: (0 if any(k in p.lower() for k in ("render", "vtk", "zview")) else 1, p))
    for p in envs:
        add(p)
    return out


class _Detector(QtCore.QThread):
    found = QtCore.pyqtSignal(str, str)   # exe, vtk version ("" = none found)

    def __init__(self, saved, parent=None):
        super().__init__(parent)
        self.saved = saved

    def run(self):
        for exe in candidate_pythons(self.saved):
            ver = probe_python(exe)
            if ver:
                self.found.emit(exe, ver)
                return
        self.found.emit("", "")


class RenderDialog(QtWidgets.QDialog):
    def __init__(self, viewer, parent=None):
        super().__init__(parent or viewer)
        self.viewer = viewer
        self.settings = viewer.settings
        self.proc = None
        self.setWindowTitle("Render 3D")
        self.resize(980, 860)
        self._build()
        mi = getattr(viewer, "multires_image", None)
        if mi is not None and getattr(mi, "labels", None) is not None:
            i = self.preset_combo.findData("labels")
            if i >= 0:
                self.preset_combo.setCurrentIndex(i)
        self._load_preset_values()
        self._init_python()

    # ------------------------------------------------------------ UI

    def _build(self):
        v = QtWidgets.QVBoxLayout(self)
        src = self.viewer.dataset_path or ""
        top = QtWidgets.QFormLayout()
        h5 = getattr(self.viewer, "hdf5_dataset", None)
        self.src_label = QtWidgets.QLabel((src + (f" : {h5}" if h5 else "")) or "(no dataset open)")
        self.src_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        top.addRow("Source:", self.src_label)
        out_row = QtWidgets.QHBoxLayout()
        default_out = self.settings.value("render/last_out", "")
        if src:
            default_out = str(Path(src).parent / (Path(src).stem + "_render"))
        self.out_edit = QtWidgets.QLineEdit(default_out)
        b = QtWidgets.QPushButton("Browse…")
        b.clicked.connect(self._browse_out)
        out_row.addWidget(self.out_edit, 1)
        out_row.addWidget(b)
        top.addRow("Output folder:", out_row)
        v.addLayout(top)

        cols = QtWidgets.QHBoxLayout()
        left = QtWidgets.QVBoxLayout()
        right = QtWidgets.QVBoxLayout()
        cols.addLayout(left, 1)
        cols.addLayout(right, 1)
        v.addLayout(cols)

        # --- preset + outputs
        g = QtWidgets.QGroupBox("Preset")
        f = QtWidgets.QFormLayout(g)
        self.preset_combo = QtWidgets.QComboBox()
        for name, desc in rc.available_presets().items():
            if name in ("draft", "presentation-4k", "publication"):
                continue
            self.preset_combo.addItem(f"{name} — {desc}", name)
        self.preset_combo.currentIndexChanged.connect(self._load_preset_values)
        f.addRow("Base:", self.preset_combo)
        self.mod_combo = QtWidgets.QComboBox()
        for label, key in MODIFIERS:
            self.mod_combo.addItem(label, key)
        self.mod_combo.currentIndexChanged.connect(self._load_preset_values)
        f.addRow("Quality:", self.mod_combo)
        outs = QtWidgets.QHBoxLayout()
        self.out_checks = {}
        for name in ("stills", "turntable", "tour", "surface"):
            c = QtWidgets.QCheckBox(name)
            self.out_checks[name] = c
            outs.addWidget(c)
        f.addRow("Outputs:", outs)
        left.addWidget(g)

        # --- look
        g = QtWidgets.QGroupBox("Look && light")
        f = QtWidgets.QFormLayout(g)
        self.look_combo = QtWidgets.QComboBox()
        self.look_combo.setEditable(True)        # any matplotlib colormap name works too
        self.look_combo.addItems(sorted(lk.LOOKS))
        f.addRow("Look:", self.look_combo)
        self.light_combo = QtWidgets.QComboBox()
        self.light_combo.addItems(list(lk.LIGHTING))
        f.addRow("Lighting:", self.light_combo)
        self.curve_combo = QtWidgets.QComboBox()
        self.curve_combo.addItems(["(from look)", "solid", "soft", "layered", "glass", "mip"])
        f.addRow("Opacity curve:", self.curve_combo)
        self.bg_combo = QtWidgets.QComboBox()
        self.bg_combo.addItems(["look", "black", "grey", "white"])
        f.addRow("Background:", self.bg_combo)
        row = QtWidgets.QHBoxLayout()
        self.filmic_check = QtWidgets.QCheckBox("Filmic tone mapping")
        row.addWidget(self.filmic_check)
        row.addWidget(QtWidgets.QLabel("Exposure"))
        self.exposure_spin = QtWidgets.QDoubleSpinBox()
        self.exposure_spin.setRange(0.0, 5.0)
        self.exposure_spin.setSingleStep(0.05)
        self.exposure_spin.setSpecialValueText("auto")
        row.addWidget(self.exposure_spin)
        row.addWidget(QtWidgets.QLabel("Supersample"))
        self.ss_spin = QtWidgets.QSpinBox()
        self.ss_spin.setRange(1, 4)
        row.addWidget(self.ss_spin)
        f.addRow(row)
        left.addWidget(g)

        # --- data
        g = QtWidgets.QGroupBox("Data")
        f = QtWidgets.QFormLayout(g)
        self.pre_combo = QtWidgets.QComboBox()
        self.pre_combo.addItem("none — window intensities (any data)", "none")
        self.pre_combo.addItem("tomo — sample in a container (medium removal, wall fit)", "tomo")
        self.pre_combo.addItem("labels — segmentation / class IDs", "labels")
        f.addRow("Preprocess:", self.pre_combo)
        row = QtWidgets.QHBoxLayout()
        self.lo_spin, self.hi_spin = QtWidgets.QDoubleSpinBox(), QtWidgets.QDoubleSpinBox()
        for sp in (self.lo_spin, self.hi_spin):
            sp.setRange(0, 100); sp.setDecimals(2); sp.setSuffix(" %")
            row.addWidget(sp)
        f.addRow("Window percentiles:", row)
        row = QtWidgets.QHBoxLayout()
        self.smooth_spin = QtWidgets.QDoubleSpinBox()
        self.smooth_spin.setRange(0, 5); self.smooth_spin.setSingleStep(0.1)
        row.addWidget(self.smooth_spin)
        self.largest_check = QtWidgets.QCheckBox("Keep largest object only")
        row.addWidget(self.largest_check)
        f.addRow("Smoothing (voxels):", row)
        row = QtWidgets.QHBoxLayout()
        self.z0_spin, self.z1_spin = QtWidgets.QSpinBox(), QtWidgets.QSpinBox()
        nz = 1 << 30
        if self.viewer.multires_image is not None:
            nz = self.viewer.multires_image.get_level_shape(0)[0]
        for sp in (self.z0_spin, self.z1_spin):
            sp.setRange(0, nz)
            row.addWidget(sp)
        self.z1_spin.setSpecialValueText("all")
        self.z1_spin.setValue(0)
        zb = QtWidgets.QPushButton("Clear")
        zb.clicked.connect(lambda: (self.z0_spin.setValue(0), self.z1_spin.setValue(0)))
        row.addWidget(zb)
        f.addRow("Z range (full-res):", row)
        self.maxaxis_spin = QtWidgets.QSpinBox()
        self.maxaxis_spin.setRange(64, 4096); self.maxaxis_spin.setSingleStep(128)
        self.maxaxis_spin.setToolTip("Largest side of the volume read for rendering (pyramid level + stride)")
        f.addRow("Render volume size:", self.maxaxis_spin)
        left.addWidget(g)
        left.addStretch(1)

        # --- output
        g = QtWidgets.QGroupBox("Output")
        f = QtWidgets.QFormLayout(g)
        self.size_combo = QtWidgets.QComboBox(); self.size_combo.setEditable(True); self.size_combo.addItems(SIZES)
        f.addRow("Size:", self.size_combo)
        row = QtWidgets.QHBoxLayout()
        self.frames_spin = QtWidgets.QSpinBox(); self.frames_spin.setRange(0, 10000)
        self.elev_spin = QtWidgets.QDoubleSpinBox(); self.elev_spin.setRange(-89, 89)
        row.addWidget(QtWidgets.QLabel("Turntable frames")); row.addWidget(self.frames_spin)
        row.addWidget(QtWidgets.QLabel("Elevation")); row.addWidget(self.elev_spin)
        f.addRow(row)
        self.title_edit = QtWidgets.QLineEdit()
        self.subtitle_edit = QtWidgets.QLineEdit()
        self.credit_edit = QtWidgets.QLineEdit(self.settings.value("render/credit", ""))
        f.addRow("Title:", self.title_edit)
        f.addRow("Subtitle:", self.subtitle_edit)
        f.addRow("Credit:", self.credit_edit)
        right.addWidget(g)

        # --- tour regions
        g = QtWidgets.QGroupBox("Tour regions (close-ups)")
        gl = QtWidgets.QVBoxLayout(g)
        self.regions = QtWidgets.QTableWidget(0, 4)
        self.regions.setHorizontalHeaderLabels(["Name", "z", "y", "x"])
        self.regions.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.regions.setMinimumHeight(110)
        gl.addWidget(self.regions)
        row = QtWidgets.QHBoxLayout()
        add = QtWidgets.QPushButton("Add crosshair position")
        add.setToolTip("Adds the viewer's current (z, y, x) as a close-up region")
        add.clicked.connect(self._add_region)
        rem = QtWidgets.QPushButton("Remove")
        rem.clicked.connect(lambda: self.regions.removeRow(self.regions.currentRow()))
        row.addWidget(add); row.addWidget(rem); row.addStretch(1)
        gl.addLayout(row)
        right.addWidget(g)

        g = QtWidgets.QGroupBox("Advanced overrides (JSON, any config key)")
        gl = QtWidgets.QVBoxLayout(g)
        self.adv_edit = QtWidgets.QPlainTextEdit()
        self.adv_edit.setPlaceholderText('{"cylinder": [256, 256, 220], "tour": {"speed": 0.5}, "scatter": 1.4}')
        self.adv_edit.setMaximumHeight(70)
        gl.addWidget(self.adv_edit)
        right.addWidget(g)

        g = QtWidgets.QGroupBox("Renderer Python")
        gl = QtWidgets.QHBoxLayout(g)
        self.py_edit = QtWidgets.QLineEdit()
        self.py_status = QtWidgets.QLabel("")
        bb = QtWidgets.QPushButton("Browse…")
        bb.clicked.connect(self._browse_python)
        db = QtWidgets.QPushButton("Detect")
        db.clicked.connect(lambda: self._detect(force=True))
        gl.addWidget(self.py_edit, 1); gl.addWidget(bb); gl.addWidget(db)
        right.addWidget(g)
        right.addWidget(self.py_status)
        right.addStretch(1)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        self.log.setStyleSheet("font-family: monospace; font-size: 11px;")
        v.addWidget(self.log, 1)

        btns = QtWidgets.QHBoxLayout()
        self.save_btn = QtWidgets.QPushButton("Save config")
        self.save_btn.setToolTip("Write render.json to the output folder (to run elsewhere with zview-render)")
        self.save_btn.clicked.connect(self._save_only)
        self.copy_btn = QtWidgets.QPushButton("Copy command")
        self.copy_btn.clicked.connect(self._copy_command)
        self.script_btn = QtWidgets.QPushButton("Save script…")
        self.script_btn.setToolTip("Write a standalone Python script that runs this render in the background "
                                   "(e.g. on a GPU node, at a finer volume size, with nohup)")
        self.script_btn.clicked.connect(self._save_script)
        self.preset_btn = QtWidgets.QPushButton("Save as preset…")
        self.preset_btn.clicked.connect(self._save_preset)
        self.open_btn = QtWidgets.QPushButton("Open output folder")
        self.open_btn.clicked.connect(self._open_out)
        self.run_btn = QtWidgets.QPushButton("Render")
        self.run_btn.setDefault(True)
        self.run_btn.setStyleSheet("QPushButton { background:#2a82da; color:white; font-weight:bold; }")
        self.run_btn.clicked.connect(self._toggle_run)
        for w in (self.save_btn, self.copy_btn, self.script_btn, self.preset_btn, self.open_btn):
            btns.addWidget(w)
        btns.addStretch(1)
        btns.addWidget(self.run_btn)
        v.addLayout(btns)

        # Seed the look from the viewer's 3D pane (VTK look or VisPy colormap).
        try:
            v3 = self.viewer.view_3d
            if hasattr(v3, "look_combo"):
                self._initial_look = v3.look_combo.currentText()
            elif v3.cmap_combo.currentText() in VISPY_TO_LOOK:
                self._initial_look = VISPY_TO_LOOK[v3.cmap_combo.currentText()]
        except Exception:
            pass

    # ------------------------------------------------------------ values

    def _presets(self):
        names = [self.preset_combo.currentData()]
        if self.mod_combo.currentData():
            names.append(self.mod_combo.currentData())
        return names

    def _load_preset_values(self, *_):
        c = rc.effective({"preset": self._presets()})
        self._preset_eff = c
        for name, chk in self.out_checks.items():
            chk.setChecked(name in (c.get("outputs") or []))
        look = c["look"] if isinstance(c["look"], str) else "bone"
        look = getattr(self, "_initial_look", None) or look
        self._initial_look = None
        self.look_combo.setCurrentText(look)
        self.light_combo.setCurrentText(c["lighting"])
        self.curve_combo.setCurrentText(c["curve"] or "(from look)")
        self.bg_combo.setCurrentText(c["background"])
        self.filmic_check.setChecked(c["tone_mapping"] == "filmic")
        self.exposure_spin.setValue(c["exposure"] or 0.0)
        self.ss_spin.setValue(int(c["supersample"]))
        self.pre_combo.setCurrentIndex(max(0, self.pre_combo.findData(c["preprocess"])))
        self.lo_spin.setValue(float(c["lo_pct"])); self.hi_spin.setValue(float(c["hi_pct"]))
        self.smooth_spin.setValue(float(c.get("smooth") or 0))
        self.largest_check.setChecked(bool(c["keep_largest"]))
        self.maxaxis_spin.setValue(int(c.get("max_axis") or 768))
        self.size_combo.setCurrentText(c["size"])
        self.frames_spin.setValue(int(c["frames"]))
        self.elev_spin.setValue(float(c["elev"]))

    def _widget_values(self):
        vals = {
            "outputs": [n for n, c in self.out_checks.items() if c.isChecked()],
            "look": self.look_combo.currentText().strip(),
            "lighting": self.light_combo.currentText(),
            "curve": None if self.curve_combo.currentIndex() == 0 else self.curve_combo.currentText(),
            "background": self.bg_combo.currentText(),
            "tone_mapping": "filmic" if self.filmic_check.isChecked() else "none",
            "exposure": self.exposure_spin.value() or None,
            "supersample": self.ss_spin.value(),
            "preprocess": self.pre_combo.currentData(),
            "lo_pct": self.lo_spin.value(), "hi_pct": self.hi_spin.value(),
            "smooth": self.smooth_spin.value(),
            "keep_largest": self.largest_check.isChecked(),
            "max_axis": self.maxaxis_spin.value(),
            "size": self.size_combo.currentText().strip(),
            "frames": self.frames_spin.value(),
            "elev": self.elev_spin.value(),
            "title": self.title_edit.text(), "subtitle": self.subtitle_edit.text(),
            "credit": self.credit_edit.text(),
        }
        z0, z1 = self.z0_spin.value(), self.z1_spin.value()
        vals["z_range"] = [z0, z1] if z1 > z0 else None
        regs = []
        for r in range(self.regions.rowCount()):
            try:
                regs.append({"name": (self.regions.item(r, 0).text() or f"Region {r + 1}"),
                             "z": int(self.regions.item(r, 1).text()), "y": int(self.regions.item(r, 2).text()),
                             "x": int(self.regions.item(r, 3).text())})
            except (AttributeError, ValueError):
                continue
        vals["regions"] = regs
        return vals

    def _config(self):
        """Config dict: preset names + only the values that differ from them."""
        eff = self._preset_eff
        cfg = {"preset": self._presets() if len(self._presets()) > 1 else self._presets()[0],
               "source": self.viewer.dataset_path}
        if getattr(self.viewer, "hdf5_dataset", None):
            cfg["hdf5_dataset"] = self.viewer.hdf5_dataset
        for k, val in self._widget_values().items():
            if k in ("title", "subtitle", "credit", "regions", "z_range") or eff.get(k) != val:
                cfg[k] = val
        text = self.adv_edit.toPlainText().strip()
        if text:
            extra = json.loads(text)
            if not isinstance(extra, dict):
                raise ValueError("advanced overrides must be a JSON object")
            cfg.update(extra)
        return cfg

    def _write_config(self):
        if not self.viewer.dataset_path:
            raise ValueError("open a dataset first")
        out = Path(self.out_edit.text().strip()).expanduser()
        if not str(out):
            raise ValueError("choose an output folder")
        out.mkdir(parents=True, exist_ok=True)
        path = out / "render.json"
        cfg = self._config()
        if path.exists():
            # Keep the cached volume when the data side hasn't changed.
            old = json.loads(path.read_text())
            same = all(old.get(k) == cfg.get(k) for k in ("source", "hdf5_dataset", "max_axis", "preprocess")) and \
                old.get("source") == cfg["source"]
            for k in ("volume", "factor", "voxel_size_um", "pixel_size_um", "hdf5_dataset"):
                if same and old.get(k) is not None:
                    cfg[k] = old[k]
        path.write_text(json.dumps(cfg, indent=2))
        self.settings.setValue("render/last_out", str(out))
        self.settings.setValue("render/credit", self.credit_edit.text())
        return path

    def _command(self, cfg_path):
        what = [n for n, c in self.out_checks.items() if c.isChecked()] or ["stills"]
        return [self.py_edit.text().strip() or sys.executable, "-u", "-m", "zview.render", "run",
                str(cfg_path), *what]

    # ------------------------------------------------------------ view state

    _WIDGET_KEYS = ("look", "lighting", "curve", "background")

    def apply_view_state(self, state):
        """Take look/lighting/window/camera from the interactive 3D pane."""
        st = dict(state)
        if st.get("look"):
            self.look_combo.setCurrentText(str(st.pop("look")))
        if st.get("lighting"):
            self.light_combo.setCurrentText(str(st.pop("lighting")))
        if "curve" in st:
            self.curve_combo.setCurrentText(st.pop("curve") or "(from look)")
        if st.get("background"):
            self.bg_combo.setCurrentText(st.pop("background"))
        if "smooth" in st:
            self.smooth_spin.setValue(float(st.pop("smooth")))
        if st.get("preprocess"):
            self.pre_combo.setCurrentIndex(max(0, self.pre_combo.findData(st.pop("preprocess"))))
        if "elev" in st:
            self.elev_spin.setValue(float(st["elev"]))
        try:
            adv = json.loads(self.adv_edit.toPlainText() or "{}")
        except json.JSONDecodeError:
            adv = {}
        adv.update(st)             # window_values, azimuth, elevations, zoom, opacity, scatter
        self.adv_edit.setPlainText(json.dumps(adv, indent=1))
        self._append("took look, window and camera from the 3D view "
                     "(window / camera are in Advanced overrides)\n")

    # ------------------------------------------------------------ actions

    def _add_region(self):
        z, y, x = self.viewer.position
        r = self.regions.rowCount()
        self.regions.insertRow(r)
        for col, val in enumerate((f"Region {r + 1}", z, y, x)):
            self.regions.setItem(r, col, QtWidgets.QTableWidgetItem(str(val)))
        self.out_checks["tour"].setChecked(True)

    def _browse_out(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Output folder", self.out_edit.text())
        if d:
            self.out_edit.setText(d)

    def _browse_python(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Python with VTK", str(Path.home()))
        if f:
            self.py_edit.setText(f)
            self._check_python(f)

    def _save_only(self):
        try:
            p = self._write_config()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Render 3D", str(e))
            return
        self._append(f"wrote {p}\nrun it with:  {self._shell_cmd(p)}\n")

    def _shell_cmd(self, p):
        cmd = self._command(p)
        return f"PYTHONPATH={shlex.quote(PKG_PARENT)} " + " ".join(shlex.quote(c) for c in cmd)

    def _copy_command(self):
        try:
            p = self._write_config()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Render 3D", str(e))
            return
        QtWidgets.QApplication.clipboard().setText(self._shell_cmd(p))
        self._append(f"copied to clipboard:\n{self._shell_cmd(p)}\n")

    def _save_preset(self):
        name, ok = QtWidgets.QInputDialog.getText(self, "Save preset", "Preset name:")
        if not ok or not name.strip():
            return
        try:
            vals = self._config()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Render 3D", str(e))
            return
        for k in ("source", "title", "subtitle", "regions", "z_range"):
            vals.pop(k, None)
        base = vals.pop("preset")
        merged = rc.effective({"preset": base})
        merged.update(vals)
        keep = {k: merged[k] for k in rc.DEFAULTS if k not in
                ("source", "volume", "hdf5_dataset", "factor", "pixel_size_um", "voxel_size_um",
                 "regions", "z_range", "cylinder", "title", "subtitle", "preset")}
        keep["_description"] = f"saved from the viewer (based on {base})"
        f = rc.save_user_preset(name.strip(), keep)
        self._append(f"saved preset {name!r} -> {f}\n")
        cur = self.preset_combo.currentData()
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        for n, desc in rc.available_presets().items():
            if n not in ("draft", "presentation-4k", "publication"):
                self.preset_combo.addItem(f"{n} — {desc}", n)
        self.preset_combo.setCurrentIndex(max(0, self.preset_combo.findData(cur)))
        self.preset_combo.blockSignals(False)

    def _save_script(self):
        try:
            cfg = self._config()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Save script", str(e))
            return
        out = Path(self.out_edit.text().strip()).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save render script", str(out / "render_job.py"),
                                                        "Python (*.py)")
        if not path:
            return
        cfg["max_axis"] = self.maxaxis_spin.value()
        what = [n for n, c in self.out_checks.items() if c.isChecked()] or ["stills"]
        Path(path).write_text(render_script(cfg, str(out), what, self.py_edit.text().strip()))
        os.chmod(path, 0o755)
        self._append(f"wrote {path}\nrun it in the background, e.g.:\n"
                     f"  nohup {self.py_edit.text().strip() or 'python'} {path} --max-axis 1536 "
                     f"> {Path(path).with_suffix('.log')} 2>&1 &\n")

    def _open_out(self):
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(self.out_edit.text().strip()))

    def _toggle_run(self):
        if self.proc is not None:
            self.proc.kill()
            self._append("\n[stopped]\n")
            return
        det = getattr(self, "_detector", None)
        if det is not None and det.isRunning():
            self._append("waiting for the renderer Python to be detected…\n")
            det.found.connect(lambda *_: QtCore.QTimer.singleShot(0, self._toggle_run))
            return
        if not self._python_ok:
            QtWidgets.QMessageBox.warning(self, "Render 3D", self.py_status.text() or
                                          "Choose a Python with VTK (Renderer Python).")
            return
        try:
            cfg_path = self._write_config()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Render 3D", str(e))
            return
        cmd = self._command(cfg_path)
        self.proc = QtCore.QProcess(self)
        env = QtCore.QProcessEnvironment.systemEnvironment()
        pp = env.value("PYTHONPATH", "")
        env.insert("PYTHONPATH", PKG_PARENT + (os.pathsep + pp if pp else ""))
        env.insert("PYTHONUNBUFFERED", "1")
        self.proc.setProcessEnvironment(env)
        self.proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.proc.setWorkingDirectory(str(cfg_path.parent))
        self.proc.readyReadStandardOutput.connect(self._read_output)
        self.proc.finished.connect(self._finished)
        self._append(f"$ {self._shell_cmd(cfg_path)}\n")
        self.proc.start(cmd[0], cmd[1:])
        self.run_btn.setText("Stop")
        self.run_btn.setStyleSheet("QPushButton { background:#b33; color:white; font-weight:bold; }")

    def _read_output(self):
        data = bytes(self.proc.readAllStandardOutput()).decode(errors="replace")
        self._append(data)

    def _finished(self, code, _status):
        self._append(f"\n[finished, exit code {code}]\n")
        self.proc = None
        self.run_btn.setText("Render")
        self.run_btn.setStyleSheet("QPushButton { background:#2a82da; color:white; font-weight:bold; }")
        if code == 0:
            self._open_out()

    def _append(self, text):
        self.log.moveCursor(QtGui.QTextCursor.End)
        self.log.insertPlainText(text)
        self.log.moveCursor(QtGui.QTextCursor.End)

    # ------------------------------------------------------------ python

    _python_ok = False

    def _init_python(self):
        saved = self.settings.value("render/python", "")
        self.py_edit.setText(saved)
        self._detect(force=False)

    def _detect(self, force=False):
        self.py_status.setText("Looking for a Python with VTK, SciPy, scikit-image, imageio-ffmpeg…")
        self._detector = _Detector(None if force else self.py_edit.text().strip() or None, self)
        self._detector.found.connect(self._on_detected)
        self._detector.start()

    def _on_detected(self, exe, ver):
        self._python_ok = bool(exe)
        if exe:
            self.py_edit.setText(exe)
            self.settings.setValue("render/python", exe)
            self.py_status.setText(f"✓ VTK {ver} — {exe}")
            self.py_status.setStyleSheet("color:#8c8;")
        else:
            self.py_status.setText("✗ No Python with VTK found. Install the render extras "
                                   "(pip install 'zview[render]') or choose an interpreter.")
            self.py_status.setStyleSheet("color:#e88;")

    def _check_python(self, exe):
        ver = probe_python(exe)
        self._on_detected(exe if ver else "", ver or "")

    def closeEvent(self, event):
        if self.proc is not None:
            r = QtWidgets.QMessageBox.question(self, "Render 3D", "A render is running. Stop it?")
            if r != QtWidgets.QMessageBox.Yes:
                event.ignore()
                return
            self.proc.kill()
        super().closeEvent(event)


SCRIPT_TEMPLATE = '''#!/usr/bin/env python3
"""zview render job — generated by the zview viewer on {date}.

Renders {what} of
    {source}
into
    {out}

Run it on a machine that can read the data and has a GPU with an X display
(a VNC/desktop session, or VirtualGL); VTK renders off-screen, nothing
appears on screen. For long jobs:

    nohup {python} {script} > {log} 2>&1 &

Options (all optional):
    --max-axis N   longest side of the volume read for rendering
                   (job default {max_axis}; 1536-2048 gives more detail on big
                   GPUs: N**3 bytes of GPU memory, ~6*N**3 bytes of RAM)
    --what ...     outputs: stills turntable tour surface
    --out DIR      output folder
    --set K=V      override any setting, e.g. --set size=3840x2160 --set frames=720
    --dry-run      print the settings and exit

Needs the zview render extras (VTK, SciPy, scikit-image, imageio-ffmpeg).
"""

import argparse
import json
import sys
from pathlib import Path

ZVIEW = {zview!r}        # zview package location (used if zview is not installed)
if ZVIEW not in sys.path:
    sys.path.insert(0, ZVIEW)

CONFIG = json.loads(r"""
{config}
""")

OUT = {out!r}
WHAT = {what_list!r}


def main():
    ap = argparse.ArgumentParser(description="zview render job")
    ap.add_argument("--max-axis", type=int, default=CONFIG.get("max_axis"))
    ap.add_argument("--what", nargs="*", default=WHAT)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--set", dest="sets", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    from zview.render import config as rc
    from zview.render.cli import run

    out = Path(a.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = dict(CONFIG, max_axis=a.max_axis)
    cfg_path = out / f"render_job_{{a.max_axis}}.json"
    if cfg_path.exists():      # keep the cached volume from an earlier run
        old = json.loads(cfg_path.read_text())
        same = all(old.get(k) == cfg.get(k) for k in ("source", "hdf5_dataset"))
        for k in ("volume", "factor", "voxel_size_um", "pixel_size_um"):
            if same and old.get(k) is not None:
                cfg[k] = old[k]
    cfg_path.write_text(json.dumps(cfg, indent=2))
    overrides = rc.parse_set(a.sets)
    problems = rc.validate(cfg, overrides)
    for msg in problems:
        print(f"CONFIG WARNING: {{msg}}", flush=True)
    if a.dry_run:
        print(json.dumps(rc.effective(cfg, overrides), indent=2, default=str))
        print("config OK" if not problems else f"{{len(problems)}} problem(s) found")
        return
    print(f"config: {{cfg_path}}", flush=True)
    run(cfg_path, a.what, overrides, out)


if __name__ == "__main__":
    main()
'''


def render_script(cfg, out, what, python=""):
    """Source of a standalone background render script for `cfg`."""
    import datetime
    script = str(Path(out) / "render_job.py")
    return SCRIPT_TEMPLATE.format(
        date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), what=", ".join(what),
        source=cfg.get("source"), out=out, python=python or "python", script=script,
        log=str(Path(out) / "render_job.log"), max_axis=cfg.get("max_axis"), zview=PKG_PARENT,
        config=json.dumps(cfg, indent=4), what_list=list(what))
