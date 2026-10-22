#!/usr/bin/env bash
# Evaluate one tier sequentially, pack its CSV, and validate it without contacting Kaggle.
# Usage: scripts/eval_and_pack.sh tier1 runs/tier1 data/results/tier1 COMMIT_URL [work-dir]
# Tier 2/3 use the no-clean builder; supply their exact frozen work-dir explicitly.
# Env: CHECKPOINT_MAP (frozen per-episode map), POLICY_STEPS, PYTHON (FlashCHORD), PACK_PYTHON (kit dependencies), SOURCE_ROOT.
set -euo pipefail
PROJECT=$(cd "$(dirname "$0")/.." && pwd)
cd "$PROJECT"
tier=${1:?tier1, tier2 or tier3}
runs=${2:?training output root}
out=${3:?results output root}
commit=${4:?GitHub commit URL}
if [ "$tier" != tier1 ] && [ "$#" -lt 5 ]; then
    echo "Tier 2/3 require the exact frozen work-dir as argument 5" >&2
    exit 2
fi
work=${5:-$PROJECT/data/work/$tier}
case "$tier" in
    tier1) released=tier_1_multiview_caption ;;
    tier2) released=tier_2_synthetic_noise ;;
    tier3) released=tier_2_synthetic_noise ;;  # no Tier 3 release; same episodes and frame grid
    *) echo "unknown tier: $tier" >&2; exit 2 ;;
esac
KIT=$PROJECT/external/kit/v2d_submission_kit
FLASH=$PROJECT/external/video_to_data/robotic_grounding/flash_chord
PYTHON=${PYTHON:-$PROJECT/.venv-flash/bin/python}
PACK_PYTHON=${PACK_PYTHON:-$PROJECT/.venv/bin/python}
dataset=$PROJECT/data/challenge
tier_root=$dataset/track_2/$released
mkdir -p "$out"
out=$(cd "$out" && pwd)
default_steps=349999104
if [ "$tier" != tier1 ]; then default_steps=49999872; fi
checkpoint_args=(--runs-root "$runs")
if [ -n "${CHECKPOINT_MAP:-}" ]; then checkpoint_args=(--checkpoint-map "$CHECKPOINT_MAP"); fi
"$PACK_PYTHON" scripts/write_checkpoints.py --tier-root "$tier_root" "${checkpoint_args[@]}" \
    --steps "${POLICY_STEPS:-$default_steps}" -o "$out/checkpoints.json"
frame_args=()
if [ "$tier" != tier1 ]; then
    "$PACK_PYTHON" scripts/frame_references.py --work-dir "$work" --tier-root "$tier_root" \
        -o "$out/frame_references"
    frame_args=(--frame-references "$out/frame_references")
fi
source_args=()
if [ -n "${SOURCE_ROOT:-}" ]; then
    source_args=(--source-root "$SOURCE_ROOT")
fi
"$PYTHON" "$KIT/examples/run_policy_evaluation.py" --checkpoints "$out/checkpoints.json" \
    --results "$out/evaluation" --runtime-root "$FLASH" --world-count 1 \
    ${source_args[@]+"${source_args[@]}"}
if [ "$tier" = tier1 ]; then
    "$PACK_PYTHON" "$KIT/tools/pack_submission.py" --track "track_2_$tier" \
        --results "$out/evaluation" --dataset-root "$dataset" --code-commit "$commit" \
        -o "$out/track2_$tier.csv"
else
    "$PACK_PYTHON" scripts/pack_no_clean.py --track "track_2_$tier" \
        --results "$out/evaluation" --code-commit "$commit" \
        "${frame_args[@]}" -o "$out/track2_$tier.csv"
fi
# The uploader's --dry-run still authenticates and calls Kaggle. Call only its offline validator.
"$PACK_PYTHON" scripts/validate_offline.py "$KIT" "$out/track2_$tier.csv" "track_2_$tier"
