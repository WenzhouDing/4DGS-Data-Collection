#!/usr/bin/env bash
# organize_episodes.sh — KITTI-style per-episode organization
# ============================================================
# Runs sync_pipeline.py + run_calibration.py end-to-end and arranges
# output/ as a set of self-contained episode_NNNN folders, each with its
# own calibration/ folder. Episodes within the same --groups arg share
# calibration: the first index in each group is the calibration source,
# and its calibration/ folder is copied into every other episode in the
# group (intentional duplication so each episode is shippable on its own).
#
# Usage example:
#   ./organize_episodes.sh --base . --cams 12 --board 9x12 \
#       --square-size 0.03 --ref-cam 3 --groups "1" "2,3,4"
#
# Meaning of --groups "1" "2,3,4":
#   * Group A = [episode 1] — calibration captured from episode 1 itself
#   * Group B = [episode 2, 3, 4] — episodes 3 and 4 reuse the calibration
#     captured during episode 2 (which is itself a dedicated calib clip).
#
# Migrates any pre-existing output/session_NN/ folders to output/episode_NNNN/.

set -euo pipefail

# ─── Defaults ────────────────────────────────────────────────────
BASE="."
CAMS=""
BOARD=""
SQUARE=""
REF_CAM=1
MAX_FRAMES=0
GROUP_LIST=()
SKIP_SYNC=0
SKIP_CALIB=0

# ─── Arg parsing ─────────────────────────────────────────────────
print_help() {
    cat <<EOF
Usage: $0 [options]

Required:
  --cams N             Number of cameras
  --board COLSxROWS    Checkerboard size (e.g. 9x12)
  --square-size METRES Square side in metres (e.g. 0.03)
  --groups G1 G2 ...   One arg per group; comma-separated episode indices,
                       first index is the calibration source.

Optional:
  --base DIR           Project root (default: .)
  --ref-cam N          Reference camera (default: 1)
  --max-frames N       run_calibration --max-frames (default: 0 = no cap)
  --skip-sync          Don't re-run sync_pipeline (assume episodes exist)
  --skip-calib         Don't run calibration; only copy existing calib folders

Example:
  $0 --base . --cams 12 --board 9x12 --square-size 0.03 \\
      --ref-cam 3 --groups "1" "2,3,4"
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base)         BASE="$2"; shift 2 ;;
        --cams)         CAMS="$2"; shift 2 ;;
        --board)        BOARD="$2"; shift 2 ;;
        --square-size)  SQUARE="$2"; shift 2 ;;
        --ref-cam)      REF_CAM="$2"; shift 2 ;;
        --max-frames)   MAX_FRAMES="$2"; shift 2 ;;
        --skip-sync)    SKIP_SYNC=1; shift ;;
        --skip-calib)   SKIP_CALIB=1; shift ;;
        --groups)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                GROUP_LIST+=("$1"); shift
            done
            ;;
        -h|--help) print_help; exit 0 ;;
        *) echo "ERROR: unknown arg: $1" >&2; print_help >&2; exit 1 ;;
    esac
done

# ─── Validation ──────────────────────────────────────────────────
errs=0
[[ -z "$CAMS"    ]] && { echo "ERROR: --cams required"        >&2; errs=1; }
[[ -z "$BOARD"   ]] && { echo "ERROR: --board required"       >&2; errs=1; }
[[ -z "$SQUARE"  ]] && { echo "ERROR: --square-size required" >&2; errs=1; }
[[ ${#GROUP_LIST[@]} -eq 0 ]] && { echo "ERROR: at least one --groups arg required" >&2; errs=1; }
[[ $errs -ne 0 ]] && { print_help >&2; exit 1; }

OUTPUT_DIR="$BASE/output"

# ─── Helpers ─────────────────────────────────────────────────────
episode_name() {
    # Format an integer episode index as episode_NNNN (4-digit zero-padded).
    printf "episode_%04d" "$((10#$1))"
}

# ─── Step 1: Sync ────────────────────────────────────────────────
if [[ $SKIP_SYNC -eq 0 ]]; then
    echo "════════════════════════════════════════════════════════════"
    echo "STEP 1: Sync (sync_pipeline.py)"
    echo "════════════════════════════════════════════════════════════"
    uv run python sync_pipeline.py \
        --base "$BASE" --cams "$CAMS" --ref-cam "$REF_CAM"
    echo
else
    echo "[skip-sync] Assuming episodes already exist in $OUTPUT_DIR"
fi

# ─── Step 2: Migrate any session_NN -> episode_NNNN ──────────────
echo "════════════════════════════════════════════════════════════"
echo "STEP 2: Migrate any session_NN/ -> episode_NNNN/ (if needed)"
echo "════════════════════════════════════════════════════════════"
migrated=0
for d in "$OUTPUT_DIR"/session_*; do
    [[ -d "$d" ]] || continue
    old=$(basename "$d")
    num=${old#session_}
    new=$(episode_name "$num")
    newpath="$OUTPUT_DIR/$new"
    if [[ -e "$newpath" ]]; then
        echo "  $old -> $new : target exists, skipping"
        continue
    fi
    mv "$d" "$newpath"
    # Rename embedded files that include the old session name.
    for f in "$newpath"/${old}_*; do
        [[ -e "$f" ]] || continue
        mv "$f" "${f//${old}/${new}}"
    done
    if [[ -f "$newpath/metadata/${old}_metadata.json" ]]; then
        mv "$newpath/metadata/${old}_metadata.json" "$newpath/metadata/${new}_metadata.json"
    fi
    echo "  $old -> $new"
    migrated=$((migrated + 1))
done
[[ $migrated -eq 0 ]] && echo "  (nothing to migrate)"
echo

# Verify each episode referenced by the groups exists.
echo "════════════════════════════════════════════════════════════"
echo "STEP 3: Verify all referenced episodes exist"
echo "════════════════════════════════════════════════════════════"
all_indices=()
for group in "${GROUP_LIST[@]}"; do
    IFS=',' read -ra parts <<< "$group"
    for p in "${parts[@]}"; do
        all_indices+=("$p")
    done
done
missing=0
for idx in "${all_indices[@]}"; do
    ep=$(episode_name "$idx")
    if [[ ! -d "$OUTPUT_DIR/$ep" ]]; then
        echo "  ERROR: $OUTPUT_DIR/$ep not found"
        missing=$((missing + 1))
    else
        echo "  ✓ $ep"
    fi
done
if [[ $missing -gt 0 ]]; then
    echo "Missing $missing episode folder(s). Did sync_pipeline create them?" >&2
    exit 1
fi
echo

# ─── Step 4: Calibrate the source episode of each group ──────────
echo "════════════════════════════════════════════════════════════"
echo "STEP 4: Calibrate each group's source episode"
echo "════════════════════════════════════════════════════════════"
for group in "${GROUP_LIST[@]}"; do
    IFS=',' read -ra parts <<< "$group"
    src_idx="${parts[0]}"
    src_ep=$(episode_name "$src_idx")
    src_calib="$OUTPUT_DIR/$src_ep/calibration"

    echo "── Group [${parts[*]}] — calibration source: $src_ep ──"

    if [[ -f "$src_calib/calibration_all_cameras.json" ]]; then
        echo "  $src_ep: calibration already present ($src_calib)"
    elif [[ $SKIP_CALIB -eq 1 ]]; then
        echo "  [skip-calib] expected $src_calib/calibration_all_cameras.json but not found" >&2
        exit 1
    else
        echo "  Running calibration on $src_ep..."
        OPENCV_OPENCL_DEVICE=disabled uv run python run_calibration.py \
            --base "$BASE" --cams "$CAMS" --episode "$src_ep" \
            --ref-cam "$REF_CAM" --board "$BOARD" --square-size "$SQUARE" \
            --max-frames "$MAX_FRAMES"
    fi

    # ─── Step 5: Copy calibration to other episodes in the group ──
    for idx in "${parts[@]:1}"; do
        ep=$(episode_name "$idx")
        target="$OUTPUT_DIR/$ep/calibration"
        if [[ -d "$target" ]]; then
            echo "  $ep/calibration: already present, replacing"
            rm -rf "$target"
        fi
        cp -R "$src_calib" "$target"
        # Mark provenance so the destination knows the calib didn't come
        # from its own checkerboard footage.
        cat > "$target/calibration_source.json" <<JSON
{
  "source_episode": "$src_ep",
  "copied_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "note": "Calibration captured during ${src_ep}; reused here because this episode shares the same physical rig configuration."
}
JSON
        echo "  $ep: copied calibration from $src_ep"
    done
    echo
done

# ─── Step 6: Generate 3D camera-pose HTML in each episode ───────
echo "════════════════════════════════════════════════════════════"
echo "STEP 6: Render 3D camera-pose HTML for each episode"
echo "════════════════════════════════════════════════════════════"
OPENCV_OPENCL_DEVICE=disabled uv run python viz_calibration.py --base "$BASE"
echo

# ─── Step 7: Final summary ───────────────────────────────────────
echo "════════════════════════════════════════════════════════════"
echo "DONE — episode layout:"
echo "════════════════════════════════════════════════════════════"
for d in "$OUTPUT_DIR"/episode_*; do
    [[ -d "$d" ]] || continue
    ep=$(basename "$d")
    has_synced=$([[ -d "$d/synced_raw" ]] && echo "synced_raw/✓" || echo "synced_raw/✗")
    has_calib=$([[ -f "$d/calibration/calibration_all_cameras.json" ]] && echo "calibration/✓" || echo "calibration/✗")
    src=""
    [[ -f "$d/calibration/calibration_source.json" ]] && \
        src=" (calib from $(uv run python -c "import json; print(json.load(open('$d/calibration/calibration_source.json'))['source_episode'])" 2>/dev/null))"
    echo "  $ep — $has_synced  $has_calib$src"
done
