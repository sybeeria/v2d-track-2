#!/usr/bin/env python3
"""Stage the I1 screen (T2-STRATEGY/REPORT.md) as a launch.sh run directory: OUT/i1screen/{jobs.txt,post.sh}.

Box: bootstrap + a frozen external Tier 2 screen, the 12 training jobs, then post.sh runs exactly k=3
evaluate-only repeats of every final checkpoint (prepare_repeats.py). Decide on the Mac with T2-STRATEGY/i1_decide.py.
usage: .venv/bin/python -B data/brainstorm7/stage_i1.py OUT_DIR
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAME, K, FINAL = 'i1screen', 3, 49999872

out = Path(sys.argv[1]) / NAME
out.mkdir(parents=True)
lines = [l for l in (HERE / 'T2-STRATEGY/i1_jobs.txt').read_text().splitlines() if l and not l.startswith('#')]
candidates = []
for line in lines:
    job, arms_file, arm, label, sequence, steps = line.split('|')
    if steps != str(FINAL) or label != 'raw_repaired':
        raise SystemExit(f'unexpected I1 job: {line}')
    candidates.append({'name': job, 'source': f'{job}/{arm}', 'arm': arm, 'arms_file': arms_file, 'input': label,
                       'sequence': sequence, 'steps': [FINAL], 'final_step': FINAL, 'group': 'i1'})
(out / 'jobs.txt').write_text('\n'.join(lines) + '\n')
(out / 'env.sh').write_text('export BUILD_SCREEN=0\n')
(out / 'pre.sh').write_text("""set -euo pipefail
archive=runs/i1screen/frozen-screen
if [ -d "$archive" ]; then
    rm -rf data/screen/tier2
    mkdir -p data/screen
    cp -a "$archive" data/screen/tier2
else
    .venv/bin/python -B scripts/make_tier2_screen.py --jobs 6
    mkdir -p runs/i1screen
    rm -rf runs/i1screen/.staging
    cp -a data/screen/tier2 runs/i1screen/.staging
    mv runs/i1screen/.staging "$archive"
fi
""")
plan = json.dumps({'repeats': K, 'candidates': candidates}, indent=1)
run = f'runs/{NAME}'
(out / 'post.sh').write_text(f"""set -euo pipefail
cd /workspace/v2d-track-2
cat > {run}/repeat_plan.json <<'PLAN'
{plan}
PLAN
.venv/bin/python -B scripts/prepare_repeats.py --plan {run}/repeat_plan.json --source-root {run} --out-root {run}/repeats
PILOT_EVALUATE_ONLY=1 JOBS_FILE={run}/repeats/jobs-i1.txt scripts/test_day.sh "$(nvidia-smi -L | wc -l)" {run}/repeats/i1
""")
print(out, len(candidates), 'jobs,', len(candidates) * K, 'repeat evaluations')
