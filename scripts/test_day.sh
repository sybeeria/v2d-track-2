#!/usr/bin/env bash
# Round-2 test day (data/brainstorm2/PLAN.md): one pilot_queue job per arm/input on the
# non-challenge screen, spread over GPUs longest-first. Each job gets its own out-root, so
# no two processes share an arm's recipe.json or summary.
#   scripts/test_day.sh GPUS OUT_ROOT            # run; per-job logs in OUT_ROOT/<job>.log
#   scripts/test_day.sh GPUS OUT_ROOT --dry-run  # print every command; create nothing
# JOBS_FILE=FILE replaces the built-in job list with FILE's lines (same format; blank and # lines skipped).
# MANIFEST=FILE replaces the screen manifest (e.g. the superset written by finger_targets.py screen).
set -euo pipefail
PROJECT=$(cd "$(dirname "$0")/.." && pwd)
gpus=${1:?gpu count} out=${2:?out root} dry=${3:-}
shards=${SHARD_COUNT:-1} shard=${SHARD_INDEX:-0}
[[ "$gpus" =~ ^[1-9][0-9]*$ && "$shards" =~ ^[1-9][0-9]*$ && "$shard" =~ ^[0-9]+$ ]] &&
    [ "$shard" -lt "$shards" ] || { echo "invalid GPU count or shard" >&2; exit 2; }
MANIFEST=${MANIFEST:-$PROJECT/data/screen/tier2/manifest.json}
SNACK=2026-03-06_10-24-18_snack_box_pick_and_place_01
LATE="40001536 49999872"
FULL="250009600 290011136 310011904 330012672 349999104"
# job | arms file | arms (comma-separated) | input | sequence (all = whole manifest) | steps
JOBS=(
    "F1-control|pilot_arms_350m.txt|F1-control|clean|$SNACK|$FULL"
    "F4-orientation|pilot_arms_350m.txt|F4-orientation|clean|$SNACK|$FULL"
    "E1-clean|pilot_arms.txt|E1-control|clean|all|$LATE"
    "E4-clean|pilot_arms.txt|E4-orientation|clean|all|$LATE"
    "E6-clean|pilot_arms.txt|E6-rebalance-tail|clean|all|$LATE"
    "X5-finger-clean|pilot_arms.txt|X5-finger|clean|all|$LATE"
    "E1-raw_upstream|pilot_arms.txt|E1-control|raw_upstream|all|$LATE"
    "E1-raw_repaired|pilot_arms.txt|E1-control|raw_repaired|all|$LATE"
    "E1-denoised_upstream|pilot_arms.txt|E1-control|denoised_upstream|all|$LATE"
    "E1-denoised_repaired|pilot_arms.txt|E1-control|denoised_repaired|all|$LATE"
    "T2C-contact-raw_repaired|pilot_arms.txt|T2C-contact|raw_repaired|all|$LATE"
    "T2C-firststarts-raw_repaired|pilot_arms.txt|T2C-firststarts|raw_repaired|all|$LATE"
    "throughput|throughput_arms.txt|X4-control,X4-w4096-u8,X4-u2|clean|$SNACK|10002432"
)
if [ -n "${JOBS_FILE:-}" ]; then
    JOBS=()
    while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in ''|'#'*) ;; *) JOBS+=("$line") ;; esac
    done < "$JOBS_FILE"
fi

run_job() {  # gpu job-spec [--dry-run]
    local name arms_file arms input sequence steps
    IFS='|' read -r name arms_file arms input sequence steps <<< "$2"
    local command=("$PROJECT/scripts/pilot_queue.sh" --arms "$PROJECT/scripts/$arms_file"
                   --manifest "$MANIFEST" --inputs "$input" --out-root "$out/$name" --gpu "$1")
    for arm in ${arms//,/ }; do command+=(--arm "$arm"); done
    [ "$sequence" = all ] || command+=(--sequences "$sequence")
    # shellcheck disable=SC2206
    command+=(--steps $steps ${3:+"$3"})
    "${command[@]}"
}

# Identical frozen job lists, SHARD_COUNT=N and distinct SHARD_INDEX values partition boxes.
selected=()
names="|"
for i in "${!JOBS[@]}"; do
    name=${JOBS[$i]%%|*}
    [[ "$name" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ && "$names" != *"|$name|"* ]] || {
        echo "invalid or duplicate job name: $name" >&2; exit 2;
    }
    names+="$name|"
    if [ $((i % shards)) -eq "$shard" ]; then selected+=("${JOBS[$i]}"); fi
done
JOBS=("${selected[@]}")
if [ "$dry" = --dry-run ]; then
    for job in "${JOBS[@]}"; do run_job 0 "$job" --dry-run; done
    exit 0
fi
mkdir -p "$out"
# Kernel-owned lock survives exec and is released even on a crash; old claim directories are irrelevant.
if [ "${TEST_DAY_LOCKED:-}" != "$out" ]; then
    exec "${PACK_PYTHON:-$PROJECT/.venv/bin/python}" - "$0" "$out" "$@" <<'PYLOCK'
import fcntl, os, sys
lock = open(os.path.join(sys.argv[2], '.runner.lock'), 'a')
try:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    sys.exit('another test_day runner owns this out-root')
os.set_inheritable(lock.fileno(), True)
os.environ['TEST_DAY_LOCKED'] = sys.argv[2]
os.execvp('bash', ['bash', sys.argv[1], *sys.argv[3:]])
PYLOCK
fi
worker() {
    local i name failed=0
    for ((i = $1; i < ${#JOBS[@]}; i += gpus)); do
        name=${JOBS[$i]%%|*}
        echo "GPU $1: $name"
        run_job "$1" "${JOBS[$i]}" >> "$out/$name.log" 2>&1 || {
            echo "GPU $1: $name FAILED (see $out/$name.log)"; failed=1;
        }
    done
    return "$failed"
}
pids=()
for ((gpu = 0; gpu < gpus; gpu++)); do worker "$gpu" & pids+=("$!"); done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
first=1
for job in "${JOBS[@]}"; do
    table="$out/${job%%|*}/summary.tsv"
    [ -f "$table" ] || continue
    if [ "$first" ]; then cat "$table"; first=; else tail -n +2 "$table"; fi
done > "$out/summary.tsv"
cat "$out/summary.tsv"
exit "$failed"
