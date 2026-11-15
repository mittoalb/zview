#!/usr/bin/env bash
# High-performance file / dataset copy via GridFTP.
#
# Two modes are supported, picked by the shape of the arguments:
#
#   1) Direct gsiftp:// URLs  -> uses `globus-url-copy` with parallel streams
#      ./gridftp_copy.sh gsiftp://src/host/path/file gsiftp://dst/host/path/
#
#   2) Globus endpoint UUIDs  -> uses the modern `globus transfer` CLI
#      ./gridftp_copy.sh SRC_UUID:/path/to/file DST_UUID:/path/to/
#
# Notes:
# - For (1), you need a proxy certificate: `grid-proxy-init` first.
# - For (2), you need to be logged in: `globus login` first, and both endpoints
#   must be activated on https://app.globus.org
# - Local-to-local or regular ssh copies should use `rsync`/`scp` instead;
#   this script is for high-throughput network transfers.
#
# Env knobs (override on the command line):
#   PARALLEL=16        # parallel data streams per transfer (globus-url-copy)
#   CONCURRENCY=4      # concurrent files when src is a directory
#   TCP_BS=16M         # TCP buffer size (per-stream)
#   BLOCK_SIZE=16M     # application block size
#   LABEL="..."        # label for the Globus transfer (mode 2)
#   SYNC_LEVEL=checksum  # checksum | mtime | size | exists  (mode 2)

set -euo pipefail

PARALLEL="${PARALLEL:-16}"
CONCURRENCY="${CONCURRENCY:-4}"
TCP_BS="${TCP_BS:-16M}"
BLOCK_SIZE="${BLOCK_SIZE:-16M}"
LABEL="${LABEL:-zarr-auto-render $(date +%Y%m%d_%H%M%S)}"
SYNC_LEVEL="${SYNC_LEVEL:-checksum}"

if [[ $# -ne 2 ]]; then
    cat <<EOF
Usage: $0 <source> <destination>

Modes:
  gsiftp:// URLs          -> globus-url-copy with -p \$PARALLEL -cc \$CONCURRENCY
  UUID:/path endpoint IDs -> globus transfer --sync-level=\$SYNC_LEVEL

Examples:
  # mode 1: direct gsiftp URLs
  $0 gsiftp://src.example.org:2811//exchange/rec.h5 \\
     gsiftp://host.example.org:2811//path/to/rec.h5

  # mode 1: directory (trailing slash)
  $0 gsiftp://src.example.org:2811//exchange/dataset/ \\
     gsiftp://host.example.org:2811//path/to/dataset/

  # mode 2: Globus endpoint UUIDs (preferred)
  $0 SRC_ENDPOINT_UUID:/share/godata/file1.txt \\
     DST_ENDPOINT_UUID:/~/file1.txt

  # tuning via env vars:
  PARALLEL=32 TCP_BS=32M $0 ...
EOF
    exit 1
fi

SRC="$1"
DST="$2"

# Mode detection: UUIDs look like 8-4-4-4-12 hex
if [[ "$SRC" =~ ^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}: ]]; then
    MODE="globus"
elif [[ "$SRC" == gsiftp://* ]] || [[ "$SRC" == ftp://* ]] || [[ "$SRC" == sshftp://* ]]; then
    MODE="guc"
else
    echo "ERROR: cannot tell what kind of source this is: $SRC" >&2
    echo "Expected either a UUID:/path or a gsiftp:// URL." >&2
    exit 2
fi

case "$MODE" in
    guc)
        if ! command -v globus-url-copy >/dev/null 2>&1; then
            echo "ERROR: globus-url-copy not found in PATH." >&2
            exit 3
        fi
        # Confirm we have a proxy certificate (common gotcha)
        if command -v grid-proxy-info >/dev/null 2>&1; then
            if ! grid-proxy-info -exists 2>/dev/null; then
                echo "WARN: no active grid proxy. Run `grid-proxy-init` first if transfer fails."
            fi
        fi

        # Decide between single-file and recursive directory mode
        if [[ "$SRC" == */ ]] || [[ "$DST" == */ ]]; then
            RECURSIVE="-r"
        else
            RECURSIVE=""
        fi

        echo "[gridftp] mode: globus-url-copy"
        echo "[gridftp] src: $SRC"
        echo "[gridftp] dst: $DST"
        echo "[gridftp] options: -p $PARALLEL -cc $CONCURRENCY -tcp-bs $TCP_BS -bs $BLOCK_SIZE $RECURSIVE"
        echo ""

        # -fast: use mode E (extended block mode, fewer round trips)
        # -vb:   per-transfer bytes/sec
        # -sync: skip if dest already has it (idempotent reruns)
        exec globus-url-copy \
            -p "$PARALLEL" \
            -cc "$CONCURRENCY" \
            -tcp-bs "$TCP_BS" \
            -bs "$BLOCK_SIZE" \
            -fast \
            -vb \
            -sync \
            $RECURSIVE \
            "$SRC" "$DST"
        ;;

    globus)
        if ! command -v globus >/dev/null 2>&1; then
            echo "ERROR: globus CLI not found in PATH." >&2
            exit 3
        fi
        # Confirm login
        if ! globus whoami >/dev/null 2>&1; then
            echo "ERROR: not logged in. Run `globus login` first." >&2
            exit 4
        fi

        # Decide between file and recursive
        RECURSIVE=""
        if [[ "$SRC" == */ ]] || [[ "$DST" == */ ]]; then
            RECURSIVE="--recursive"
        fi

        echo "[gridftp] mode: globus transfer"
        echo "[gridftp] src: $SRC"
        echo "[gridftp] dst: $DST"
        echo "[gridftp] label: $LABEL"
        echo "[gridftp] sync-level: $SYNC_LEVEL"
        echo "[gridftp] recursive: ${RECURSIVE:-no}"
        echo ""

        # Returns a task id you can watch with:  globus task wait <TASK>
        globus transfer \
            --label "$LABEL" \
            --sync-level "$SYNC_LEVEL" \
            --notify off \
            $RECURSIVE \
            "$SRC" "$DST" \
            --format json \
            | tee /tmp/last_globus_submit.json

        TASK=$(python3 -c "import json,sys; print(json.load(open('/tmp/last_globus_submit.json'))['task_id'])" 2>/dev/null || true)
        if [[ -n "$TASK" ]]; then
            echo ""
            echo "Task submitted: $TASK"
            echo "Watch:    globus task show $TASK"
            echo "Wait:     globus task wait $TASK"
            echo "Cancel:   globus task cancel $TASK"
        fi
        ;;
esac
