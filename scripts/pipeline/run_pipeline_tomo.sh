#!/usr/bin/env bash
# Launcher for a host where rec.h5 sits on a LOCAL disk: reads it in place
# (no copy), converts to sharded Zarr v3, then renders.
#
# Put your paths in config.local.sh next to this script (git-ignored; see
# config.example.sh), or pass them as env vars:
#
#   SRC_H5=/path/to/rec.h5 OUT_BASE=/path/to/output \
#   nohup bash run_pipeline_tomo.sh > /tmp/pipeline.log 2>&1 & disown
#
# Follow with:
#
#   tail -f /tmp/pipeline.log

set -euo pipefail

SCRIPT_DIR="${SCRIPT_DIR:-$(dirname "$(readlink -f "$0")")}"
# Site-specific settings (paths, env) — not tracked in git
[[ -f "$SCRIPT_DIR/config.local.sh" ]] && source "$SCRIPT_DIR/config.local.sh"

# ---------- CONFIG --------------------------------------------------------
# Source HDF5 — read in place (no copy)
SRC_H5="${SRC_H5:?set SRC_H5=/path/to/rec.h5 (env var or config.local.sh)}"

# Output folder (needs TB-scale free space): rec.zarr, renders/, logs/ go inside
OUT_BASE="${OUT_BASE:?set OUT_BASE=/path/to/output (env var or config.local.sh)}"
OUT_ZARR="$OUT_BASE/rec.zarr"
RENDER_DIR="$OUT_BASE/renders"
LOG_DIR="$OUT_BASE/logs"

# Python from the conda env (used directly — no `conda activate` needed)
CONDA_ENV="${CONDA_ENV:-zarr3_render}"
ENV_PY="${ENV_PY:-$HOME/miniconda3/envs/${CONDA_ENV}/bin/python}"

# zview package (scripts/pipeline/ -> repo root); importable without installing it.
ZVIEW_ROOT="${ZVIEW_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
export PYTHONPATH="$ZVIEW_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Conversion knobs
OUT_DTYPE="${OUT_DTYPE:-uint16}"          # uint8 for smaller/faster renders
CHUNK="${CHUNK:-128 128 128}"
SHARD="${SHARD:-512 512 512}"
COMPRESSION="${COMPRESSION:-blosclz}"
PIXEL_SIZE="${PIXEL_SIZE:-1.0}"           # micrometers; set the real voxel size
SLAB="${SLAB:-1024}"              # slices per slab; RAM ~ (2.4*SLAB+200) x Y x X x 2 bytes. 1024 @ 12288^2 = ~800 GB (see README)
MAX_LEVELS="${MAX_LEVELS:-6}"

# Rendering knobs
TURNTABLE_FRAMES="${TURNTABLE_FRAMES:-180}"
SLICE_FRAMES="${SLICE_FRAMES:-240}"
COLORMAP="${COLORMAP:-gray}"
# Renderer uses the finest pyramid level <= BUDGET_GB; keep it under one GPU's VRAM
BUDGET_GB="${BUDGET_GB:-32}"
# ---------------------------------------------------------------------------

# Python block-buffers stdout when piped to tee; force line-by-line progress in the logs
export PYTHONUNBUFFERED=1

say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

mkdir -p "$LOG_DIR" "$RENDER_DIR" "$(dirname "$OUT_ZARR")"

# Sanity checks up front — fail fast instead of midway
if [[ ! -e "$SRC_H5" ]]; then
    say "ERROR: source does not exist: $SRC_H5"
    exit 1
fi
if [[ ! -x "$ENV_PY" ]]; then
    say "ERROR: python not found at $ENV_PY — override with ENV_PY=... or create the env first."
    exit 1
fi
for f in "$ZVIEW_ROOT/zview/tools/convert.py" "$ZVIEW_ROOT/zview/render/cpu.py"; do
    if [[ ! -f "$f" ]]; then
        say "ERROR: missing module: $f (set ZVIEW_ROOT to the zview repo path)"
        exit 1
    fi
done

say "=== zarr-auto-render (read in place) ==="
say "source      : $SRC_H5"
say "zarr out    : $OUT_ZARR"
say "renders out : $RENDER_DIR"
say "logs        : $LOG_DIR"
say "python      : $ENV_PY"
"$ENV_PY" -c "import zarr, h5py; print('zarr', zarr.__version__, ' h5py', h5py.__version__)"

# ---------- Wait briefly in case source is still being written -----------
say "checking that $SRC_H5 is stable (no active writers)..."
prev_size=-1
stable_checks=0
while :; do
    writers=$( { timeout 60 lsof "$SRC_H5" 2>/dev/null || true; } | awk 'NR>1 && $4 ~ /w/' | wc -l)
    size=$(stat -c '%s' "$SRC_H5")
    hsize=$(numfmt --to=iec "$size")
    if (( writers == 0 )) && (( size == prev_size )); then
        stable_checks=$((stable_checks + 1))
        say "no writers, size stable at $hsize ($stable_checks/3)"
        if (( stable_checks >= 3 )); then
            say "source stable. proceeding."
            break
        fi
    else
        stable_checks=0
        say "waiting: writers=$writers size=$hsize"
    fi
    prev_size=$size
    sleep 20
done

# ---------- Probe the HDF5 ------------------------------------------------
say "probing $SRC_H5"
"$ENV_PY" - <<PY
import h5py, numpy as np
with h5py.File("$SRC_H5", "r") as f:
    def visit(name, obj):
        if isinstance(obj, h5py.Dataset) and obj.ndim == 3:
            nbytes = int(np.prod(obj.shape)) * obj.dtype.itemsize
            print(f"  /{name}  shape={obj.shape}  dtype={obj.dtype}  size={nbytes/1024**3:.2f} GB")
    f.visititems(visit)
PY

# ---------- Convert HDF5 -> sharded Zarr v3 -------------------------------
if [[ -e "$OUT_ZARR" ]]; then
    say "WARN: $OUT_ZARR exists; passing --overwrite"
    OVERWRITE="--overwrite"
else
    OVERWRITE=""
fi

say "launching conversion -> $OUT_ZARR"
# shellcheck disable=SC2086
"$ENV_PY" -m zview.tools.convert "$SRC_H5" "$OUT_ZARR" \
    --dtype "$OUT_DTYPE" \
    --chunk-size $CHUNK \
    --shard-size $SHARD \
    --compression "$COMPRESSION" \
    --pixel-size "$PIXEL_SIZE" \
    --slab "$SLAB" \
    --max-levels "$MAX_LEVELS" \
    $OVERWRITE \
    2>&1 | tee "$LOG_DIR/convert.log"

say "conversion complete"

# Pyramid summary + file count
say "pyramid on disk:"
find "$OUT_ZARR" -maxdepth 1 -mindepth 1 -type d | sort | while read -r d; do
    n=$(find "$d" -type f | wc -l)
    s=$(du -sh "$d" 2>/dev/null | awk '{print $1}')
    echo "  $(basename "$d"):  files=$n  size=$s"
done
total_files=$(find "$OUT_ZARR" -type f | wc -l)
say "total files in $OUT_ZARR: $total_files"

# ---------- Render --------------------------------------------------------
say "launching renders -> $RENDER_DIR"
export QT_QPA_PLATFORM=offscreen
"$ENV_PY" -m zview.render cpu "$OUT_ZARR" \
    --budget-gb "$BUDGET_GB" \
    --cmaps "$COLORMAP" \
    --frames "$TURNTABLE_FRAMES" \
    --slice-sweep "$SLICE_FRAMES" \
    --out "$RENDER_DIR" \
    2>&1 | tee "$LOG_DIR/render.log"

say "renders complete -> $RENDER_DIR"
say "pipeline finished"
