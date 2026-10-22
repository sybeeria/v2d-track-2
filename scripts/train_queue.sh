#!/usr/bin/env bash
# Train one FlashSAC policy per G1 reference, sequentially on one GPU, resuming interrupted runs.
#
# Usage: scripts/train_queue.sh <gpu-index> <out-root> <data.parquet>...
#   Spread a tier over 4 GPUs (run each line in its own tmux window):
#     P=($(find data/work/tier1/motion -path '*robot_name=g1/data.parquet' | sort))
#     scripts/train_queue.sh 0 runs/tier1 "${P[@]:0:8}"
#     scripts/train_queue.sh 1 runs/tier1 "${P[@]:8:8}"   # ...and so on
#   Timing run (stop after 10M steps; read throughput from train.log):
#     EXTRA_ARGS="training.total_environment_steps=10000384" scripts/train_queue.sh 0 runs/timing "${P[0]}"
# Env: EXTRA_ARGS (space-separated overrides), PYTHON, FLASH_ROOT.
set -euo pipefail

GPU=${1:?gpu index}
OUT_ROOT=$(mkdir -p "${2:?out root}" && cd "$2" && pwd)
shift 2
[ "$#" -gt 0 ] || { echo "no reference parquets supplied" >&2; exit 2; }
PROJECT=$(cd "$(dirname "$0")/.." && pwd)
FLASH=${FLASH_ROOT:-$PROJECT/external/video_to_data/robotic_grounding/flash_chord}
PYTHON=${PYTHON:-$PROJECT/.venv-flash/bin/python}
FINAL_STEPS=349999104
failures=0
extra=()
if [ -n "${EXTRA_ARGS:-}" ]; then
    read -r -a extra <<< "$EXTRA_ARGS"
fi

for parquet in "$@"; do
    parquet=$(cd "$(dirname "$parquet")" && pwd)/$(basename "$parquet")
    name=$(echo "$parquet" | sed -E 's#.*/sequence_id=([^/]+)/.*#\1#')
    [ -f "$parquet" ] && [ "$name" != "$parquet" ] || { echo "invalid reference: $parquet" >&2; exit 2; }
    out=$OUT_ROOT/$name
    mkdir -p "$out"
    if [ -z "${EXTRA_ARGS:-}" ] && [ -f "$out/policy_${FINAL_STEPS}.safetensors" ]; then
        echo "done  $name"; continue
    fi
    # A released empty stage means no support; the runtime rejects an enabled empty stage.
    support=$("$PYTHON" - "$parquet" <<'PY'
import sys
from flash_chord.assets.registry import support_usda_for_parquet
from pxr import Usd
path = support_usda_for_parquet(sys.argv[1], robot_name="g1")
if path is None:
    raise FileNotFoundError(f"no support stage for {sys.argv[1]}")
stage = Usd.Stage.Open(str(path))
if stage is None:
    raise ValueError(f"cannot open support stage: {path}")
print(str(any(p.GetTypeName() in ("Cylinder", "Cube") for p in stage.Traverse())).lower())
PY
    ) || { echo "FAIL  $name (support stage)"; failures=$((failures + 1)); continue; }
    resume=()
    latest=$(ls "$out" | grep -E '^state_[0-9]+\.safetensors$' | sort -t_ -k2 -n | tail -1 || true)
    if [ -n "$latest" ]; then
        resume=(training.resume=true "training.checkpoint_path='$out/$latest'")
        echo "resume $name from $latest"
    else
        echo "start $name"
    fi
    start=$(date +%s)
    (cd "$FLASH" && CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" scripts/train_flash_sac.py \
        experiment=g1_recon_body_flash_sac "task.parquet='$parquet'" "scene.support=$support" reset.seed=42 \
        logging.mode=disabled "output_dir='$out'" "hydra.run.dir='$out/hydra'" \
        ${resume[@]+"${resume[@]}"} ${extra[@]+"${extra[@]}"}) >> "$out/train.log" 2>&1 \
        || { echo "FAIL  $name (see $out/train.log)"; failures=$((failures + 1)); continue; }
    echo "ok    $name in $(( ($(date +%s) - start) / 60 )) min"
done
[ "$failures" -eq 0 ]
