#!/usr/bin/env bash
# One-shot pipeline:
#   wait for rec.h5 copy to finish -> convert HDF5 -> multiscale sharded Zarr v3
#   -> render 3D (ortho + canonical views + turntable MP4 + slice-sweep MP4)
#
# Everything runs headless with its own timestamped logs. Designed to be launched
# with nohup so you can disconnect:
#
#   SRC_H5=/path/to/rec.h5 OUT_BASE=/path/to/output \
#   nohup bash run_pipeline.sh > pipeline.log 2>&1 &
#
# Settings can also live in config.local.sh next to this script (git-ignored).
#
# Watch progress:
#   tail -f pipeline.log
#   tail -f "$OUT_BASE/logs/convert.log"
#   tail -f "$OUT_BASE/logs/render.log"

set -euo pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
# Site-specific settings (paths, env) — not tracked in git
[[ -f "$SCRIPT_DIR/config.local.sh" ]] && source "$SCRIPT_DIR/config.local.sh"

# ---------- CONFIG --------------------------------------------------------
SRC_H5="${SRC_H5:?set SRC_H5=/path/to/rec.h5 (env var or config.local.sh)}"
OUT_BASE="${OUT_BASE:?set OUT_BASE=/path/to/output (env var or config.local.sh)}"
OUT_ZARR="$OUT_BASE/rec.zarr"
RENDER_DIR="$OUT_BASE/renders"
LOG_DIR="$OUT_BASE/logs"

CONDA_ENV="${CONDA_ENV:-zarr3_render}"
# Use the env's python directly (no `conda activate` needed)
ENV_PY="${ENV_PY:-$HOME/miniconda3/envs/${CONDA_ENV}/bin/python}"

# zview package (scripts/pipeline/ -> repo root); importable without installing it.
ZVIEW_ROOT="${ZVIEW_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
export PYTHONPATH="$ZVIEW_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Conversion defaults (override by editing these)
OUT_DTYPE="uint16"                      # uint8 for smaller/faster renders
CHUNK="128 128 128"                     # inner chunk (OME-Zarr v3)
SHARD="512 512 512"                     # shard size (one shard = many chunks)
COMPRESSION="blosclz"
PIXEL_SIZE=1.0                          # micrometers; set your real voxel size
SLAB=512                                # Z-slab height per read (multiple of shard Z)
MAX_LEVELS=6

# Rendering defaults (override by editing)
TURNTABLE_FRAMES=180
SLICE_FRAMES=240
RENDERING="attenuated_mip"              # mip | attenuated_mip | iso | translucent | additive | average | minip
COLORMAP="gray_r"
BUDGET_GB=8                             # memory budget (GB) to pick mip level for render.
                                        # Keep <= GPU VRAM.
# ---------------------------------------------------------------------------

mkdir -p "$LOG_DIR" "$RENDER_DIR"

export PYTHONUNBUFFERED=1

say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ---------- 1. Wait for copy to finish ------------------------------------
say "waiting for $SRC_H5 copy to finish"
if [[ ! -e "$SRC_H5" ]]; then
    say "ERROR: $SRC_H5 does not exist yet. Start the copy and rerun."
    exit 1
fi

prev_size=-1
stable_checks=0
while :; do
    # Is anyone writing to it?
    writers=$( { timeout 60 lsof "$SRC_H5" 2>/dev/null || true; } | awk 'NR>1 && $4 ~ /w/' | wc -l)
    size=$(stat -c '%s' "$SRC_H5")
    hsize=$(numfmt --to=iec "$size")
    if (( writers == 0 )) && (( size == prev_size )); then
        stable_checks=$((stable_checks + 1))
        say "no writers, size stable at $hsize ($stable_checks/3)"
        if (( stable_checks >= 3 )); then
            say "copy finished, final size $hsize"
            break
        fi
    else
        stable_checks=0
        say "still copying: writers=$writers size=$hsize"
    fi
    prev_size=$size
    sleep 60
done

# ---------- 2. Env check (we use ENV_PY directly, not `conda activate`) ---
say "using python: $ENV_PY"
"$ENV_PY" -c "import zarr, h5py, numpy; print('zarr', zarr.__version__, '  h5py', h5py.__version__, '  numpy', numpy.__version__)"

# ---------- 3. Probe the HDF5 ---------------------------------------------
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

# ---------- 4. Convert HDF5 -> Zarr v3 ------------------------------------
if [[ -e "$OUT_ZARR" ]]; then
    say "WARN: $OUT_ZARR exists; passing --overwrite"
    OVERWRITE="--overwrite"
else
    OVERWRITE=""
fi

say "launching conversion -> $OUT_ZARR"
# shellcheck disable=SC2086
"$ENV_PY" -m zview.convert "$SRC_H5" "$OUT_ZARR" \
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

# Quick pyramid summary and file count
say "pyramid on disk:"
find "$OUT_ZARR" -maxdepth 1 -mindepth 1 -type d | sort | while read -r d; do
    n=$(find "$d" -type f | wc -l)
    s=$(du -sh "$d" 2>/dev/null | awk '{print $1}')
    echo "  $(basename "$d"):  files=$n  size=$s"
done
total_files=$(find "$OUT_ZARR" -type f | wc -l)
say "total files in $OUT_ZARR: $total_files"

# ---------- 5. Render -----------------------------------------------------
say "launching renders -> $RENDER_DIR"
export QT_QPA_PLATFORM=offscreen
"$ENV_PY" -m zview.render napari "$OUT_ZARR" \
    --batch \
    --budget-gb "$BUDGET_GB" \
    --rendering "$RENDERING" \
    --colormap "$COLORMAP" \
    --turntable "$TURNTABLE_FRAMES" \
    --slice-sweep "$SLICE_FRAMES" \
    --out "$RENDER_DIR" \
    2>&1 | tee "$LOG_DIR/render.log"

say "renders complete -> $RENDER_DIR"
say "pipeline finished"
