"""Label maps (segmentations, annotations): detection, class tables, pooling.

A label volume stores *class IDs*, not intensities: there is no "brighter",
interpolating between IDs invents classes that don't exist, and picking one
voxel out of N when downsampling drops small structures. This module gives
the rest of zview what it needs to treat such data correctly:

* `detect_labels(multires)` — a `LabelTable` (or None) from OME-NGFF
  ``image-label`` metadata, a ``lookup``/``labels`` name table in the
  attributes, or — for unannotated data — an integer volume with only a few
  distinct values.
* `LabelTable` — per-class name, colour, opacity and visibility. Defaults come
  from the class names (vessel → red, nucleus → blue, void → hidden…) and from
  how much of the volume a class fills (bulk classes become faint glass,
  small ones are opaque).
* `pool_labels` — block downsampling where, inside each cell, the
  highest-priority class present wins. Priority is rarity, with background
  last, so thin vessels and small cells survive at coarse scales instead of
  vanishing (stride subsampling) or being outvoted (majority).
"""

import colorsys
import re
from dataclasses import dataclass, field

import numpy as np

# Words in class names -> (rgb, role). Roles: "background" (hidden),
# "bulk" (faint glass when large), None (opaque structure).
_KEYWORDS = [
    (r"\b(void|background|outside|empty|unlabel+ed|ignore|air)\b", (0.0, 0.0, 0.0), "background"),
    (r"lumen", (0.90, 0.12, 0.10), None),
    (r"vessel|capillar|arter|vein|blood", (0.65, 0.12, 0.12), None),
    (r"nucle", (0.35, 0.40, 0.95), None),
    (r"neuron|soma", (1.00, 0.80, 0.25), None),
    (r"astro", (0.30, 0.85, 0.45), None),
    (r"oligo", (0.98, 0.52, 0.15), None),
    (r"microglia", (0.85, 0.35, 0.85), None),
    (r"opc|ng2", (0.30, 0.80, 0.90), None),
    (r"ependym", (0.70, 0.65, 0.30), None),
    (r"white matter|myelin", (0.88, 0.88, 0.92), "bulk"),
    (r"gr[ae]y matter|neuropil|cortex", (0.85, 0.62, 0.62), "bulk"),
    (r"mitochond", (0.95, 0.30, 0.30), None),
    (r"membrane|wall|border|limitans", (0.75, 0.55, 0.35), None),
    (r"axon|dendrit|fiber|fibre", (0.55, 0.75, 0.95), None),
    (r"bone|mineral", (0.95, 0.93, 0.85), "bulk"),
    (r"pore|void space|crack", (0.20, 0.50, 0.95), None),
    (r"tissue|matrix|body|organ", (0.85, 0.70, 0.62), "bulk"),
]


def _distinct_color(i):
    """Well-separated colours for unnamed classes (golden-ratio hue walk)."""
    h = (0.11 + i * 0.618033988749895) % 1.0
    s = 0.65 if i % 2 else 0.85
    v = 0.95 if i % 3 else 0.80
    return colorsys.hsv_to_rgb(h, s, v)


@dataclass
class LabelClass:
    value: int
    name: str
    color: tuple
    opacity: float = 1.0
    visible: bool = True
    fraction: float = 0.0          # share of the (sampled) volume
    role: str | None = None        # background | bulk | None


@dataclass
class LabelTable:
    classes: dict = field(default_factory=dict)    # value -> LabelClass
    source: str = ""                               # where the names came from

    # ---------------------------------------------------------- queries

    def __contains__(self, v):
        return int(v) in self.classes

    def name(self, v):
        c = self.classes.get(int(v))
        return c.name if c else f"label {int(v)}"

    @property
    def values(self):
        return sorted(self.classes)

    @property
    def background(self):
        return {v for v, c in self.classes.items() if c.role == "background"}

    def visible_classes(self):
        return [c for v, c in sorted(self.classes.items()) if c.visible and c.opacity > 0]

    # ---------------------------------------------------------- lookups

    def lut_rgba(self, size=256):
        """(size, 4) uint8 colour table for 2D display (index = label value)."""
        lut = np.zeros((size, 4), np.uint8)
        lut[:, 3] = 255
        for v, c in self.classes.items():
            if 0 <= v < size:
                rgb = (0.05, 0.05, 0.05) if c.role == "background" else c.color
                lut[v, :3] = [int(round(255 * x)) for x in rgb]
        return lut

    def priority_lut(self, size=None):
        """Rank per label value for pooling: rare classes high, background 0."""
        size = size or (max(self.classes) + 1 if self.classes else 1)
        rank = np.zeros(size, np.int32)
        order = sorted(self.classes.values(),
                       key=lambda c: (c.role != "background", -c.fraction))   # background first, then common
        for i, c in enumerate(order):
            if c.value < size:
                rank[c.value] = i + 1
        return rank

    # ---------------------------------------------------------- (de)serialise

    def to_config(self):
        out = {}
        for v, c in sorted(self.classes.items()):
            d = {"name": c.name, "color": [round(x, 4) for x in c.color],
                 "opacity": round(c.opacity, 4), "visible": c.visible}
            if c.role:
                d["role"] = c.role
            out[str(v)] = d
        return out

    def update_from_config(self, cfg):
        for k, d in (cfg or {}).items():
            v = int(k)
            c = self.classes.get(v) or LabelClass(v, d.get("name", f"label {v}"), _distinct_color(v))
            c.name = d.get("name", c.name)
            if "color" in d:
                c.color = tuple(float(x) for x in d["color"][:3])
            if "opacity" in d:
                c.opacity = float(d["opacity"])
            if "visible" in d:
                c.visible = bool(d["visible"])
            if "role" in d:
                c.role = d["role"]
            elif c.role is None and re.search(_KEYWORDS[0][0], c.name.lower()):
                c.role = "background"
            self.classes[v] = c
        return self

    @classmethod
    def from_config(cls, cfg):
        return cls(source="config").update_from_config(cfg)


# ------------------------------------------------------------------ building

def build_table(values, fractions=None, names=None, colors=None, source=""):
    """Class table with sensible defaults for `values`."""
    fractions = fractions or {}
    names = names or {}
    colors = colors or {}
    table = LabelTable(source=source)
    fg_total = sum(f for v, f in fractions.items()
                   if not re.search(_KEYWORDS[0][0], str(names.get(v, "")).lower())) or 1.0
    for i, v in enumerate(sorted(int(x) for x in values)):
        name = str(names.get(v, f"label {v}"))
        low = name.lower()
        color, role = None, None
        for pat, rgb, r in _KEYWORDS:
            if re.search(pat, low):
                color, role = rgb, r
                break
        if v in colors:
            color = colors[v]
        if color is None:
            color = _distinct_color(i)
        frac = float(fractions.get(v, 0.0))
        c = LabelClass(v, name, tuple(color), 1.0, True, frac, role)
        if role == "background":
            c.visible, c.opacity = False, 0.0
        elif role == "bulk" or frac / fg_total > 0.15:
            c.role = "bulk"
            c.opacity = 0.06        # faint glass: context without hiding the small classes
        table.classes[v] = c
    # Unnamed data: the most common value at the volume border-ish is usually
    # background; if nothing was named background and value 0 exists, hide it.
    if not table.background and 0 in table.classes and not names:
        c0 = table.classes[0]
        c0.role, c0.visible, c0.opacity = "background", False, 0.0
    return table


def _names_from_attrs(attrs):
    """{value: name} and {value: rgb} from common metadata layouts."""
    names, colors = {}, {}
    il = attrs.get("image-label") if isinstance(attrs, dict) else None
    if isinstance(il, dict):
        for c in il.get("colors", []) or []:
            try:
                colors[int(c["label-value"])] = tuple(x / 255 for x in c["rgba"][:3])
            except (KeyError, TypeError, ValueError):
                pass
        for p in il.get("properties", []) or []:
            try:
                v = int(p["label-value"])
                names[v] = p.get("name") or p.get("class") or p.get("label") or f"label {v}"
            except (KeyError, TypeError, ValueError):
                pass
        if not names and colors:
            names = {v: f"label {v}" for v in colors}
    for key in ("lookup", "labels", "label_names", "class_names", "classes"):
        d = attrs.get(key) if isinstance(attrs, dict) else None
        if isinstance(d, dict):
            for k, v in d.items():
                try:
                    names[int(k)] = str(v if not isinstance(v, dict) else v.get("name", k))
                except (TypeError, ValueError):
                    pass
        elif isinstance(d, list) and d and all(isinstance(x, str) for x in d):
            names.update({i: s for i, s in enumerate(d)})
    return names, colors


def _flatness(arr):
    """Share of neighbouring voxel pairs (along the last two axes) that are equal.

    Label maps are piecewise constant (~0.95+); images, even low-dynamic-range
    8-bit ones, have noise and gradients that break this.
    """
    a = np.asarray(arr)
    if a.ndim == 1:
        return 1.0
    eq, n = 0, 0
    for ax in (-1, -2):
        if a.shape[ax] > 1:
            d = np.diff(a, axis=ax)
            eq += int((d == 0).sum())
            n += d.size
    return eq / n if n else 1.0


def detect_labels(multires, max_unique=32, sample_voxels=4_000_000, min_flatness=0.85, force=False):
    """LabelTable if `multires` looks like a label map, else None.

    Metadata names (OME-NGFF image-label, a lookup table) are decisive.
    Without them, an integer volume needs few distinct values *and* to be
    piecewise constant (`min_flatness`). force=True skips the heuristics
    (View › Treat as label map).
    """
    lvl0 = multires.levels[0]
    if not np.issubdtype(np.dtype(lvl0.dtype), np.integer):
        return None
    attrs = {}
    try:
        attrs = dict(getattr(multires.zarr_group, "attrs", {}) or {})
        if isinstance(attrs.get("ome"), dict):
            attrs = {**attrs, **attrs["ome"]}
    except Exception:
        pass
    names, colors = _names_from_attrs(attrs)

    # Sample a few chunk-sized blocks of the coarsest level (cheap even on
    # TB-scale data, where a strided pass would still decode every chunk).
    blocks = _sample_blocks(multires.levels[-1], sample_voxels, flat=False)
    if not blocks:
        return None
    smp = np.concatenate([b.ravel() for b in blocks])
    vals, counts = np.unique(smp, return_counts=True)
    if not names and not force:
        if len(vals) > max_unique:
            return None
        flat = np.mean([_flatness(b) for b in blocks])
        if flat < min_flatness:
            return None
    total = counts.sum()
    fractions = {int(v): c / total for v, c in zip(vals, counts)}
    present = set(fractions) | set(names)
    return build_table(present, fractions, names, colors,
                       source="metadata" if names else f"{len(vals)} distinct values")


def _sample_blocks(arr, max_voxels=4_000_000, n=9, flat=True):
    """Up to `n` blocks spread over a (z, y, x) array, ~max_voxels in total.

    Returns one concatenated 1-D array (flat=True) or the list of blocks.
    """
    shape = arr.shape
    side = max(8, int((max_voxels / n) ** (1 / 3)))
    chunks = arr.chunks or shape
    side = min(side, *[max(8, c) for c in chunks]) if arr.chunks else side
    pts = [0.5] if n == 1 else [0.2, 0.5, 0.8]
    out = []
    for fz in pts:
        for fy in pts:
            for fx in (0.5,) if n <= 9 else pts:
                lo = [max(0, min(int(f * d) - side // 2, d - side)) for f, d in zip((fz, fy, fx), shape)]
                try:
                    out.append(np.asarray(arr[lo[0]:lo[0] + side, lo[1]:lo[1] + side, lo[2]:lo[2] + side]))
                except Exception:
                    return None
    if not flat:
        return out
    return np.concatenate([b.ravel() for b in out]) if out else None


def update_fractions(table, codes, code_values=None):
    """Refresh class fractions (and bulk/glass defaults) from a loaded volume."""
    counts = np.bincount(codes.ravel(), minlength=256)
    total = counts.sum() or 1
    for code, cnt in enumerate(counts):
        if cnt == 0:
            continue
        v = code if code_values is None else (code_values[code] if code < len(code_values) else None)
        if v is not None and v in table.classes:
            table.classes[v].fraction = cnt / total
    return table


# ------------------------------------------------------------------ pooling

def pool_labels(block, factors, rank_lut):
    """Downsample a label block by integer `factors`, rare classes winning.

    Each output voxel takes the class with the highest rank present in its
    cell (rank_lut[value]; background ranks lowest). Edges are padded with
    rank 0. Returns the pooled label values (same dtype as `block`).
    """
    fz, fy, fx = (int(f) for f in factors)
    if (fz, fy, fx) == (1, 1, 1):
        return block
    nz, ny, nx = block.shape
    oz, oy, ox = -(-nz // fz), -(-ny // fy), -(-nx // fx)
    vals = block.astype(np.int64, copy=False)
    ranks = rank_lut[np.clip(vals, 0, len(rank_lut) - 1)] if len(rank_lut) else np.zeros_like(vals)
    ranks = np.where(vals < len(rank_lut), ranks, 0)
    pad = ((0, oz * fz - nz), (0, oy * fy - ny), (0, ox * fx - nx))
    if any(p[1] for p in pad):
        ranks = np.pad(ranks, pad, constant_values=-1)
    best = ranks.reshape(oz, fz, oy, fy, ox, fx).max(axis=(1, 3, 5))
    inv = np.zeros(int(rank_lut.max()) + 2, block.dtype)
    nz_rank = rank_lut > 0
    inv[rank_lut[nz_rank]] = np.nonzero(nz_rank)[0].astype(block.dtype)
    # rank 0 cells (unknown/background values): fall back to the first voxel
    out = inv[np.clip(best, 0, len(inv) - 1)]
    zero = best <= 0
    if zero.any():
        out[zero] = block[::fz, ::fy, ::fx][:oz, :oy, :ox][zero]
    return out


def render_codes(data, table):
    """Compact uint8 codes for rendering a label volume.

    VTK's label-map rendering indexes per-class transfer functions by code,
    and needs them compact. Background classes all map to 0 (never drawn);
    the other values present map to 1..N in value order. Values missing from
    the table are added as unnamed classes. With more than 255 classes
    (instance segmentations) codes fall back to value % 255 + 1.

    Returns (codes uint8, code_values) where code_values[code] is the label
    value drawn for that code (code 0: a background value or None).
    """
    flat = data.ravel()
    if np.issubdtype(data.dtype, np.integer) and data.dtype.itemsize <= 2 and flat.min() >= 0:
        counts = np.bincount(flat, minlength=1)
        present = np.nonzero(counts)[0]
    else:
        present = np.unique(flat)
    present = [int(v) for v in present]
    for i, v in enumerate(present):
        if v not in table.classes:
            table.classes[v] = LabelClass(v, f"label {v}", _distinct_color(len(table.classes) + i))
    bg = [v for v in present if table.classes[v].role == "background"]
    fg = [v for v in present if v not in bg]
    if len(fg) > 255:
        codes = (data % 255).astype(np.uint8) + 1
        for v in bg:
            codes[data == v] = 0
        return codes, None
    lut = np.zeros(max(present) + 1 if present else 1, np.uint8)
    for i, v in enumerate(fg):
        lut[v] = i + 1
    codes = lut[data]
    return codes, [bg[0] if bg else None] + fg

