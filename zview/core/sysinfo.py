"""Machine capability probing and resource budgets.

Everything here answers one question: *how much can this machine handle?*
The answers drive three knobs so the viewer adapts to wherever it runs:

* the chunk-cache size (fraction of available RAM),
* the I/O thread-pool width (CPU count),
* the 3D volume budget (GPU 3D-texture limit, free VRAM, free RAM).

All probes are best-effort and never raise; unknown values come back as None
and the budget logic falls back to conservative defaults.
"""

import ctypes
import math
import os
import sys
from dataclasses import dataclass, field


# ---------------------------------------------------------------- GL compat

def apply_gl_compat():
    """Make PyOpenGL use the same libOpenGL/libGLX as Qt in conda envs.

    VisPy uploads 3D textures through PyOpenGL. PyOpenGL looks up the
    *unversioned* `libOpenGL.so`, which conda envs don't ship (they ship only
    `libOpenGL.so.0`), so it falls through to the system copy in /usr/lib64.
    Qt meanwhile uses the env's libglvnd. Two GL dispatch tables means
    PyOpenGL sees no current context, `glTexImage3D` fails with
    GL_INVALID_OPERATION, and every volume renders black.

    Binding PyOpenGL to the versioned sonames reuses whatever copy the dynamic
    linker already resolved for Qt. This is a no-op on non-GLX platforms and
    when the libraries can't be found. Call it before VisPy draws anything.
    """
    if not sys.platform.startswith("linux"):
        return
    if os.environ.get("ZVIEW_NO_GL_COMPAT", "").strip() in ("1", "true", "True"):
        return
    try:
        import OpenGL.platform as _p  # noqa: PLC0415
    except Exception:
        return
    plat = getattr(_p, "PLATFORM", None)
    if type(plat).__name__ != "GLXPlatform":
        return
    # Only override properties PyOpenGL hasn't resolved yet (they're cached
    # in the instance dict on first access).
    for attr, soname in (("GL", "libOpenGL.so.0"), ("GLX", "libGLX.so.0")):
        if attr in vars(plat):
            continue
        try:
            setattr(plat, attr, ctypes.CDLL(soname, mode=ctypes.RTLD_GLOBAL))
        except OSError:
            pass


# ---------------------------------------------------------------- CPU / RAM

def cpu_count():
    """Usable CPU count (respects cgroup/affinity masks on Linux)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def system_memory():
    """Return (total_bytes, available_bytes); either may be None."""
    try:
        import psutil  # noqa: PLC0415
        vm = psutil.virtual_memory()
        return int(vm.total), int(vm.available)
    except Exception:
        pass

    # Linux: /proc/meminfo has MemAvailable (accounts for reclaimable cache).
    try:
        info = {}
        with open("/proc/meminfo") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                info[key] = int(rest.split()[0]) * 1024
        total = info.get("MemTotal")
        avail = info.get("MemAvailable", info.get("MemFree"))
        if total:
            return total, avail
    except Exception:
        pass

    # Generic POSIX.
    try:
        page = os.sysconf("SC_PAGE_SIZE")
        total = page * os.sysconf("SC_PHYS_PAGES")
        try:
            avail = page * os.sysconf("SC_AVPHYS_PAGES")
        except (ValueError, OSError):
            avail = total // 2
        return total, avail
    except (ValueError, OSError, AttributeError):
        pass

    # Windows.
    try:
        class _MEMSTAT(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        st = _MEMSTAT()
        st.dwLength = ctypes.sizeof(_MEMSTAT)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return int(st.ullTotalPhys), int(st.ullAvailPhys)
    except Exception:
        pass

    return None, None


def _env_int(name):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return int(float(raw))
    except ValueError:
        return None


def default_io_workers():
    """Chunk-fetch concurrency: one thread per CPU, clamped to [4, 32].

    I/O threads mostly wait on disk/network (decompression releases the GIL),
    so more threads than cores is fine on network filesystems; 32 is where
    returns flatten out on typical NFS/Lustre mounts.
    """
    env = _env_int("ZVIEW_IO_THREADS")
    if env:
        return max(1, env)
    return int(min(32, max(4, cpu_count())))


def default_cache_mb():
    """Chunk-cache size: 15% of available RAM, clamped to [256 MB, 4 GB]."""
    env = _env_int("ZVIEW_CACHE_MB")
    if env:
        return max(16, env)
    _, avail = system_memory()
    if not avail:
        return 500
    mb = int(avail * 0.15 / 2**20)
    return int(min(4096, max(256, mb)))


# ---------------------------------------------------------------- GPU

@dataclass
class GLCaps:
    """OpenGL capabilities of the context the 3D view renders into."""
    vendor: str = ""
    renderer: str = ""
    version: str = ""
    max_3d_texture: int | None = None
    vram_total_mb: int | None = None
    vram_free_mb: int | None = None

    @property
    def software(self):
        r = self.renderer.lower()
        return any(k in r for k in ("llvmpipe", "softpipe", "swrast", "software"))

    def summary(self):
        parts = [self.renderer or "unknown GPU"]
        if self.vram_total_mb:
            free = f"{self.vram_free_mb / 1024:.1f}/" if self.vram_free_mb else ""
            parts.append(f"{free}{self.vram_total_mb / 1024:.1f} GB VRAM")
        if self.max_3d_texture:
            parts.append(f"max 3D tex {self.max_3d_texture}")
        return " · ".join(parts)


_GL_MAX_3D_TEXTURE_SIZE = 0x8073
_GL_GPU_MEM_TOTAL_NVX = 0x9048       # NVX_gpu_memory_info, KB
_GL_GPU_MEM_AVAIL_NVX = 0x9049
_GL_TEXTURE_FREE_MEMORY_ATI = 0x87FC  # ATI_meminfo, 4 ints KB


def query_gl_caps():
    """Read caps from the *current* GL context (call from a draw callback).

    Uses VisPy's own GL bindings, which always talk to the context VisPy
    made current. Returns a GLCaps; fields that can't be read stay None.
    """
    caps = GLCaps()
    try:
        from vispy.gloo import gl  # noqa: PLC0415
    except Exception:
        return caps

    def _get(enum):
        try:
            val = gl.glGetParameter(enum)
        except Exception:
            val = None
        # Querying an unsupported enum sets GL_INVALID_ENUM; drain it so the
        # next VisPy draw doesn't trip over our error.
        try:
            for _ in range(8):
                if gl.glGetError() == 0:
                    break
        except Exception:
            pass
        return val

    def _as_int(val):
        try:
            if hasattr(val, "__len__"):
                val = val[0] if len(val) else None
            return int(val) if val is not None else None
        except (TypeError, ValueError):
            return None

    caps.vendor = str(_get(gl.GL_VENDOR) or "")
    caps.renderer = str(_get(gl.GL_RENDERER) or "")
    caps.version = str(_get(gl.GL_VERSION) or "")
    caps.max_3d_texture = _as_int(_get(_GL_MAX_3D_TEXTURE_SIZE)) or None

    vendor = caps.vendor.lower()
    if "nvidia" in vendor:
        total = _as_int(_get(_GL_GPU_MEM_TOTAL_NVX))
        free = _as_int(_get(_GL_GPU_MEM_AVAIL_NVX))
        caps.vram_total_mb = total // 1024 if total else None
        caps.vram_free_mb = free // 1024 if free else None
    elif "ati" in vendor or "amd" in vendor:
        free = _as_int(_get(_GL_TEXTURE_FREE_MEMORY_ATI))
        caps.vram_free_mb = free // 1024 if free else None
    return caps


# ---------------------------------------------------------------- 3D budget

# Per-axis texel caps for the quality presets. "auto" derives its cap from the
# hardware (see `volume_budget`).
QUALITY_PRESETS = {
    "auto": None,
    "preview": 128,
    "low": 256,
    "medium": 512,
    "high": 1024,
    "max": 2048,
}


@dataclass
class VolumeBudget:
    max_axis: int            # largest allowed texture dimension
    max_voxels: int          # largest allowed texel count
    reasons: list = field(default_factory=list)

    def summary(self):
        side = round(self.max_voxels ** (1 / 3))
        return f"≤{self.max_axis}/axis, ≤{self.max_voxels / 1e6:.0f} M voxels (~{side}³)"


def volume_budget(caps=None, itemsize=2, quality="auto"):
    """Size limit for the 3D texture on this machine.

    The volume is held once in its native dtype while loading, then once as
    uint8 for upload, so CPU cost is (itemsize + 1) bytes/voxel; on the GPU it
    is a single-channel uint8 texture (1 byte/voxel).

    The result is the tightest of:
      * the quality preset's per-axis cap (or, for "auto", a tier picked from
        VRAM size, dropped for software renderers),
      * GL_MAX_3D_TEXTURE_SIZE,
      * 25% of available RAM,
      * 50% of free VRAM (when the driver reports it).

    ZVIEW_3D_MAX_AXIS / ZVIEW_3D_MAX_VOXELS override the preset and VRAM tiers.
    """
    caps = caps or GLCaps()
    reasons = []

    axis = QUALITY_PRESETS.get(quality)
    if quality == "machine":
        # The hardware limit, not a quality tier: largest 3D texture the GPU
        # takes (unknown -> 4096), then the RAM / VRAM limits below.
        axis = caps.max_3d_texture or 4096
        reasons.append(f"machine limit (3D texture ≤ {axis}{'' if caps.max_3d_texture else ', assumed'})")
    elif axis is None:  # auto
        vram = caps.vram_total_mb
        if caps.software:
            axis, why = 256, "software renderer"
        elif vram is None:
            axis, why = 512, "VRAM unknown"
        elif vram >= 8 * 1024:
            axis, why = 1024, f"{vram / 1024:.0f} GB VRAM"
        elif vram >= 4 * 1024:
            axis, why = 768, f"{vram / 1024:.0f} GB VRAM"
        elif vram >= 2 * 1024:
            axis, why = 512, f"{vram / 1024:.0f} GB VRAM"
        else:
            axis, why = 384, f"{vram} MB VRAM"
        reasons.append(f"auto tier {axis} ({why})")
    else:
        reasons.append(f"preset {quality} ({axis})")

    env_axis = _env_int("ZVIEW_3D_MAX_AXIS")
    if env_axis:
        axis = env_axis
        reasons.append(f"ZVIEW_3D_MAX_AXIS={env_axis}")

    if caps.max_3d_texture and caps.max_3d_texture < axis and quality != "machine":
        axis = caps.max_3d_texture
        reasons.append(f"GL max 3D texture {caps.max_3d_texture}")

    voxels = axis ** 3
    env_vox = _env_int("ZVIEW_3D_MAX_VOXELS")
    if env_vox:
        voxels = env_vox
        reasons.append(f"ZVIEW_3D_MAX_VOXELS={env_vox}")

    _, avail = system_memory()
    if avail:
        ram_vox = int(avail * 0.25 / (itemsize + 1))
        if ram_vox < voxels:
            voxels = ram_vox
            reasons.append(f"RAM {avail / 2**30:.1f} GB free")

    if caps.vram_free_mb:
        gpu_vox = int(caps.vram_free_mb * 2**20 * 0.5)
        if gpu_vox < voxels:
            voxels = gpu_vox
            reasons.append(f"VRAM {caps.vram_free_mb} MB free")

    return VolumeBudget(max_axis=max(16, int(axis)), max_voxels=max(16**3, int(voxels)),
                        reasons=reasons)


@dataclass
class VolumePlan:
    level: int
    stride: tuple            # per-axis (z, y, x) extra subsampling on top of the level
    out_shape: tuple         # shape of the texture that will be produced
    level_shape: tuple

    @property
    def voxels(self):
        return math.prod(self.out_shape)


def plan_volume(level_shapes, budget):
    """Pick the finest pyramid level (plus stride, if needed) within `budget`.

    `level_shapes` lists (z, y, x) per level, finest first. If even the
    coarsest level is too big (single-level or shallow pyramids), it is read
    with a per-axis stride; the stride grows on the axis with the largest
    output extent until the texel budget is met, which keeps voxels as close
    to cubic as possible.
    """
    def fits(shape):
        return max(shape) <= budget.max_axis and math.prod(shape) <= budget.max_voxels

    for lvl, shape in enumerate(level_shapes):
        if fits(shape):
            return VolumePlan(lvl, (1, 1, 1), tuple(shape), tuple(shape))

    lvl = len(level_shapes) - 1
    shape = tuple(level_shapes[lvl])
    stride = [max(1, math.ceil(d / budget.max_axis)) for d in shape]

    def out(st):
        return tuple(math.ceil(d / s) for d, s in zip(shape, st))

    while math.prod(out(stride)) > budget.max_voxels:
        o = out(stride)
        ax = max(range(3), key=lambda i: o[i])
        stride[ax] += 1
    return VolumePlan(lvl, tuple(stride), out(stride), shape)


def describe_machine(caps=None):
    """One-line hardware summary for the UI."""
    total, avail = system_memory()
    parts = [f"{cpu_count()} CPUs"]
    if total:
        parts.append(f"RAM {avail / 2**30:.0f}/{total / 2**30:.0f} GB free")
    if caps is not None and caps.renderer:
        parts.append(caps.summary())
    return " · ".join(parts)
