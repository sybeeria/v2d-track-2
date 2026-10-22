#!/usr/bin/env bash
# Convert one released Track 2 tier and retarget every episode to G1.
#
# Usage: scripts/retarget_tier.sh <tier-root> <work-dir> [jobs]
#    or: scripts/retarget_tier.sh --sequences <work-dir>/sequences <work-dir> [jobs]
#   tier-root  e.g. data/challenge/track_2/tier_1_multiview_caption
#   work-dir   receives sequences/, motion/ and logs/
# Env: PYTHON (absolute path, default <project>/.venv/bin/python), RG_ROOT (robotic_grounding checkout).
set -euo pipefail

TIER_ROOT=
if [ "${1:-}" = --sequences ]; then
    SEQUENCES=$(cd "${2:?sequences dir}" && pwd)
    WORK=$(mkdir -p "${3:?work dir}" && cd "$3" && pwd)
    JOBS=${4:-3}
else
    TIER_ROOT=$(cd "${1:?tier root}" && pwd)
    WORK=$(mkdir -p "${2:?work dir}" && cd "$2" && pwd)
    SEQUENCES=$WORK/sequences
    JOBS=${3:-4}
fi
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || { echo "jobs must be a positive integer" >&2; exit 1; }
PROJECT=$(cd "$(dirname "$0")/.." && pwd)
PYTHON=${PYTHON:-$PROJECT/.venv/bin/python}  # absolute: retargeting runs from RG_ROOT
RG_ROOT=${RG_ROOT:-$PROJECT/external/video_to_data/robotic_grounding}
bash "$PROJECT/scripts/apply_retarget_patch.sh" "$RG_ROOT"

mkdir -p "$WORK/logs"
if [ -n "${TIER_ROOT:-}" ]; then
    "$PYTHON" "$PROJECT/scripts/lerobot_to_soma.py" --tier-root "$TIER_ROOT" \
        --out "$SEQUENCES" --motion-root "$WORK/motion"
fi

retarget_one() {
    local seq_dir=$1 name
    seq_dir=$(cd "$seq_dir" && pwd -P)
    name=$(basename "$seq_dir")
    if (cd "$RG_ROOT" && "$PYTHON" scripts/retarget/soma_to_g1.py "$seq_dir" --save --fps 30 \
            --motion-root "$WORK/motion") > "$WORK/logs/$name.log" 2>&1; then
        echo "ok   $name"
    else
        echo "FAIL $name (see logs/$name.log)"
        return 1
    fi
}
export -f retarget_one
export PYTHON RG_ROOT WORK

find -L "$SEQUENCES" -mindepth 1 -maxdepth 1 -type d | sort |
    xargs -P "$JOBS" -I{} bash -c 'retarget_one "$@"' _ {} | tee "$WORK/logs/summary.txt"

expected=0
written=0
while IFS= read -r seq_dir; do
    seq_dir=$(cd "$seq_dir" && pwd -P)
    expected=$((expected + 1))
    name=$(basename "$seq_dir")
    if [ -s "$WORK/motion/whole_body/soma/sequence_id=$name/robot_name=g1/data.parquet" ]; then
        written=$((written + 1))
    fi
done < <(find -L "$SEQUENCES" -mindepth 1 -maxdepth 1 -type d)
echo "retargeted $written / $expected episodes into $WORK/motion"
[ "$written" -eq "$expected" ]
