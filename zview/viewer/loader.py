"""Background, budget-aware 3D volume loading shared by both 3D engines.

`VolumeLoader` picks *what* to show in 3D: it sizes a texel budget from the
machine (GPU 3D-texture limit, free VRAM, free RAM — zview.core.sysinfo),
takes the finest pyramid level that fits (plus a stride when even the
coarsest level is too big) and reads it chunk-by-chunk on a background
thread, with a quick preview first when that is much cheaper to read. Label
maps are read with label-aware pooling and keep their class IDs. The VTK and
VisPy panes (and both the quad cell and the 3D page) share one loader, so
the volume is read once.
"""

import dataclasses
import os
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from PyQt5 import QtCore

from zview.core import log as _log
from zview.core import sysinfo
from zview.core.pyramid import VolumeReadCancelled, io_workers


@dataclass
class VolumeResult:
    data: np.ndarray          # uint8 (z, y, x), C-contiguous
    level: int
    stride: tuple             # (z, y, x) extra subsampling on top of the level
    level_shape: tuple
    world_scale: tuple        # level-0 voxels per texel, (z, y, x)
    world_offset: tuple       # (z, y, x) in level-0 voxels
    value_range: tuple        # raw (lo, hi) mapped to 0..255
    dtype: str
    seconds: float
    final: bool
    budget: object
    chosen: bool = False          # level picked by the user (not automatic)
    labels: object = None         # zview.core.labels.LabelTable for label maps
    code_values: object = None    # data code -> label value (None: codes are the values)

    def describe(self):
        z, y, x = self.data.shape
        st = "" if self.stride == (1, 1, 1) else f" · stride {'×'.join(map(str, self.stride))}"
        mb = self.data.nbytes / 2**20
        tag = "" if self.final else " · preview"
        if self.labels is not None:
            tag = f" · {len(self.labels.classes)} classes" + (" (pooled)" if self.stride != (1, 1, 1) else "") + tag
        lv = f"L{self.level} (chosen)" if self.chosen else f"L{self.level}"
        return f"{lv} · {z}×{y}×{x}{st} · {mb:.0f} MB · {self.seconds:.1f}s{tag}"


def _fmt_bytes(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _to_uint8(vol, cancel=None):
    """Robust-range (0.05-99.95 percentile) conversion to uint8, slab-wise."""
    flat = vol.reshape(-1)
    step = max(1, flat.size // 2_000_000)
    sample = flat[::step]
    if np.issubdtype(sample.dtype, np.floating):
        sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return np.zeros(vol.shape, np.uint8), (0.0, 1.0)
    lo, hi = (float(v) for v in np.percentile(sample, [0.05, 99.95]))
    if not hi > lo:
        lo, hi = float(sample.min()), float(sample.max())
        if not hi > lo:
            hi = lo + 1.0
    if vol.dtype == np.uint8 and lo <= 0 and hi >= 255:
        return np.ascontiguousarray(vol), (0.0, 255.0)

    out = np.empty(vol.shape, np.uint8)
    scale = 255.0 / (hi - lo)
    slab = max(1, (32 * 2**20) // max(1, vol.shape[1] * vol.shape[2] * 4))
    for z0 in range(0, vol.shape[0], slab):
        if cancel is not None and cancel.is_set():
            raise VolumeReadCancelled()
        s = vol[z0:z0 + slab].astype(np.float32)
        s -= lo
        s *= scale
        np.clip(s, 0, 255, out=s)
        if np.issubdtype(vol.dtype, np.floating):
            np.nan_to_num(s, copy=False, nan=0.0)
        out[z0:z0 + slab] = s
    return out, (lo, hi)


class VolumeLoader(QtCore.QObject):
    """Background, budget-aware volume reader shared by all 3D panes."""

    loadStarted = QtCore.pyqtSignal(str)          # human description
    progress = QtCore.pyqtSignal(int, int)        # done, total (blocks)
    volumeReady = QtCore.pyqtSignal(object)       # VolumeResult
    loadFailed = QtCore.pyqtSignal(str)
    loadCancelled = QtCore.pyqtSignal()
    busyChanged = QtCore.pyqtSignal(bool)
    capsChanged = QtCore.pyqtSignal()
    # Worker -> GUI thread: (generation, kind, payload). Every public signal
    # is re-emitted from `_dispatch` only if the generation is still current,
    # so a result posted just before a cancel / dataset switch is dropped.
    _event = QtCore.pyqtSignal(int, str, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.multires = None
        self.gl_caps = None
        self.result = None
        self.quality = "auto"
        self._thread = None
        self._cancel = threading.Event()
        self._gen = 0
        self._busy = False
        self.user_cancelled = False   # suppresses auto-reload until next request
        self.level_override = None    # pyramid level chosen by the user (None = automatic)
        self._event.connect(self._dispatch)
        # Separate from the 2D pool so a big 3D read never starves slicing.
        self._executor = ThreadPoolExecutor(
            max_workers=max(2, io_workers() // 2), thread_name_prefix="zview-3d")
        # Connected first, so `result` is current before any view's handler runs.
        self.volumeReady.connect(self._on_ready_store)

    # ------------------------------------------------------------ state

    @property
    def busy(self):
        return self._busy

    def _set_busy(self, busy):
        if busy != self._busy:
            self._busy = busy
            self.busyChanged.emit(busy)

    def set_dataset(self, multires):
        self._abort()
        self.multires = multires
        self.result = None
        self.user_cancelled = False
        self.level_override = None

    def set_gl_caps(self, caps):
        if caps is not None and self.gl_caps is None:
            self.gl_caps = caps
            _log.info(f"[3D] GPU: {caps.summary()}")
            self.capsChanged.emit()

    def budget(self, quality=None):
        itemsize = 2
        if self.multires is not None:
            itemsize = np.dtype(self.multires.levels[0].dtype).itemsize
        return sysinfo.volume_budget(self.gl_caps, itemsize, quality or self.quality)

    def level_cost(self, level):
        """What rendering pyramid `level` whole would take, and whether it fits.

        Returns dict(fits, reason, gpu_bytes, ram_bytes, read_bytes, budget):
        checked against the machine limit (sysinfo "machine" budget: GPU
        texture size, half the free VRAM, a quarter of free RAM).
        """
        shape = self.multires.levels[level].shape
        itemsize = np.dtype(self.multires.levels[0].dtype).itemsize
        vox = int(np.prod(shape, dtype=np.int64))
        b = self.budget("machine")
        reason = ""
        if max(shape) > b.max_axis:
            reason = f"longest side {max(shape)} > {b.max_axis} (GPU texture limit)"
        elif vox > b.max_voxels:
            reason = f"{vox / 1e9:.2f} G voxels > {b.max_voxels / 1e9:.2f} G ({'; '.join(b.reasons[1:]) or 'RAM / VRAM'})"
        return dict(fits=not reason, reason=reason, gpu_bytes=vox, ram_bytes=vox * (itemsize + 1),
                    read_bytes=self.multires.estimate_read_bytes(level, (1, 1, 1)), budget=b)

    def _cap_read(self, plan, budget):
        """Virtual (strided) pyramids only: coarsen the automatic level until it
        reads at most ZVIEW_3D_MAX_READ_GB (default 2) from the file.

        A stored pyramid level reads about its own size, but a strided level
        of a contiguous file reads whole rows (file size / (fz * fy)), so the
        level that fits the GPU can mean minutes of I/O on network storage.
        A click in the Pyramid panel still renders any level that fits.
        """
        mi = self.multires
        if not getattr(mi, "virtual_levels", False):
            return plan, budget
        try:
            cap = float(os.environ.get("ZVIEW_3D_MAX_READ_GB", "2")) * 2**30
        except ValueError:
            cap = 2 * 2**30
        if cap <= 0:
            return plan, budget
        first = mi.estimate_read_bytes(plan.level, plan.stride)
        L, stride = plan.level, plan.stride
        while L < len(mi.levels) - 1 and mi.estimate_read_bytes(L, stride) > cap:
            L, stride = L + 1, (1, 1, 1)
        if L == plan.level:
            return plan, budget
        shape = tuple(mi.levels[L].shape)
        why = (f"read cap {cap / 2**30:g} GB (L{plan.level} would read {_fmt_bytes(first)}; "
               f"ZVIEW_3D_MAX_READ_GB, or click a level in Pyramid)")
        _log.info(f"[3D] {why} -> L{L}")
        return (sysinfo.VolumePlan(L, (1, 1, 1), shape, shape),
                dataclasses.replace(budget, reasons=list(budget.reasons) + [why]))

    def plan(self, quality=None):
        if self.multires is None:
            return None
        shapes = [lvl.shape for lvl in self.multires.levels]
        return sysinfo.plan_volume(shapes, self.budget(quality))

    # ------------------------------------------------------------ control

    def _abort(self):
        """Stop any load in flight without telling the UI it was cancelled."""
        self._gen += 1
        self._cancel.set()
        self._set_busy(False)

    def cancel(self):
        """User-initiated cancel: stop, report it, and don't auto-restart."""
        was_busy = self._busy
        self._abort()
        if was_busy:
            self.user_cancelled = True
            self.loadCancelled.emit()

    @QtCore.pyqtSlot(int, str, object)
    def _dispatch(self, gen, kind, payload):
        if gen != self._gen:
            return
        if kind == "started":
            self.loadStarted.emit(payload)
        elif kind == "progress":
            self.progress.emit(*payload)
        elif kind == "ready":
            self.volumeReady.emit(payload)
        elif kind == "failed":
            self.loadFailed.emit(payload)
        elif kind == "done":
            self._set_busy(False)

    def request(self, quality=None, progressive=True, level=None, auto=False):
        """(Re)load; a previous load in flight is cancelled.

        level: render this pyramid level whole (kept for later reloads until
        auto=True, e.g. a quality change). Raises ValueError if it does not
        fit the machine (see level_cost).
        """
        if self.multires is None:
            return
        if level is not None:
            cost = self.level_cost(level)
            if not cost["fits"]:
                raise ValueError(f"level {level} is too large for this machine: {cost['reason']}")
            self.level_override = int(level)
        elif auto:
            self.level_override = None
        if quality:
            self.quality = quality
        self._abort()
        self.user_cancelled = False
        self._cancel = threading.Event()
        gen = self._gen

        if self.level_override is not None:
            L = self.level_override
            shape = tuple(self.multires.levels[L].shape)
            final_budget = self.budget("machine")
            final = sysinfo.VolumePlan(L, (1, 1, 1), shape, shape)
        else:
            final_budget = self.budget()
            final = sysinfo.plan_volume([l.shape for l in self.multires.levels], final_budget)
            final, final_budget = self._cap_read(final, final_budget)
        stages = [(final, final_budget, True)]
        if progressive:
            prev_budget = self.budget("preview")
            prev = sysinfo.plan_volume([l.shape for l in self.multires.levels], prev_budget)
            # A preview only helps if it is much cheaper to *read* than the
            # final volume. Without a deep pyramid both decode the same chunks
            # (strides smaller than the chunk size skip nothing), so skip it.
            cheap = (self.multires.estimate_read_bytes(prev.level, prev.stride)
                     < 0.25 * self.multires.estimate_read_bytes(final.level, final.stride))
            if (final.voxels > 8 * prev.voxels and cheap
                    and (self.result is None or not self.result.final)):
                stages.insert(0, (prev, prev_budget, False))

        self._set_busy(True)
        self._thread = threading.Thread(
            target=self._run, args=(gen, self.multires, stages, self._cancel),
            name="zview-3d-loader", daemon=True)
        self._thread.start()

    def _run(self, gen, multires, stages, cancel):
        try:
            for plan, budget, final in stages:
                if gen != self._gen:
                    return
                nbytes = multires.estimate_read_bytes(plan.level, plan.stride)
                desc = (f"{'Preview' if not final else 'Loading'} L{plan.level} → "
                        f"{'×'.join(map(str, plan.out_shape))} · reading {_fmt_bytes(nbytes)}")
                self._event.emit(gen, "started", desc)
                t0 = time.perf_counter()
                def prog(done, total, last=[0.0]):  # noqa: B006 - per-stage state
                    now = time.perf_counter()
                    if now - last[0] > 0.1 or done == total:
                        last[0] = now
                        if gen == self._gen:
                            self._event.emit(gen, "progress", (done, total))

                table = multires.labels
                code_values = None
                if table is not None:
                    # Label map: pooled (rare classes win), IDs kept as-is.
                    from zview.core.labels import render_codes, update_fractions  # noqa: PLC0415
                    if plan.stride != (1, 1, 1):
                        raw = multires.read_volume_pooled(plan.level, plan.stride, table.priority_lut(),
                                                          progress=prog, cancel=cancel, executor=self._executor)
                    else:
                        raw = multires.read_volume(plan.level, plan.stride, progress=prog,
                                                   cancel=cancel, executor=self._executor)
                    data, code_values = render_codes(raw, table)
                    del raw
                    update_fractions(table, data, code_values)
                    vrange = (0.0, 255.0)
                else:
                    raw = multires.read_volume(plan.level, plan.stride, progress=prog,
                                               cancel=cancel, executor=self._executor)
                    data, vrange = _to_uint8(raw, cancel)
                    del raw
                ds = multires.level_downsample(plan.level)
                res = VolumeResult(
                    data=data, level=plan.level, stride=plan.stride,
                    level_shape=plan.level_shape,
                    world_scale=tuple(d * s for d, s in zip(ds, plan.stride)),
                    world_offset=multires.get_world_offset(plan.level),
                    value_range=vrange, dtype=str(multires.levels[0].dtype),
                    seconds=time.perf_counter() - t0, final=final, budget=budget,
                    labels=table, code_values=code_values,
                    chosen=final and self.level_override is not None,
                )
                if gen != self._gen:
                    return
                _log.info(f"[3D] {res.describe()}")
                self._event.emit(gen, "ready", res)
        except VolumeReadCancelled:
            return  # cancel() already updated the UI
        except MemoryError:
            self._event.emit(gen, "failed", "Out of memory — choose a lower quality preset")
        except Exception as e:
            traceback.print_exc()
            self._event.emit(gen, "failed", str(e))
        finally:
            self._event.emit(gen, "done", None)

    def _on_ready_store(self, res):
        self.result = res


