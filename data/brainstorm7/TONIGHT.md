# Tonight — exact commands (commit a443a86+, branch prod-prep; all times UTC)

Labels: [M] measured, [I] inferred, [G] guess. Repo root is the working directory. Never `--explain`/`--curl` with
vastai. T0 = top-up + `vastai tfa login` (user). Order and gates are SCHEDULE.md's; this card is only the commands.

```bash
P=data/brainstorm6/PROD-READY
# Offer searches (gpu_name tokens measured in offers_2026-10-22T2300.txt). Pick the cheapest; check gpu_name in the row.
search() { vastai search offers "verified=true rentable=true rented=false direct_port_count>=1 $1" --storage "$2" --type on-demand --raw -o dph_total |
  python3 -I -c "import json,sys;[print(o['id'],o['gpu_name'],o['num_gpus'],o['cpu_cores_effective'],round(o['cpu_ram']/1024),o['disk_space'],round(o['dph_total'],3)) for o in json.load(sys.stdin)[:8]]"; }
TRAIN2='num_gpus=2 gpu_name=RTX_PRO_5000 cpu_cores_effective>=16 cpu_ram>=64 disk_space>=200 dph_total<=1.85'
TRAIN1='num_gpus=1 gpu_name=RTX_PRO_5000 cpu_cores_effective>=8 cpu_ram>=32 disk_space>=200 dph_total<=0.92'
RECON2='num_gpus=2 gpu_name=RTX_6000Ada cpu_cores_effective>=24 cpu_ram>=96 disk_space>=300 dph_total<=1.40'
RECON1='num_gpus=1 gpu_name=RTX_6000Ada cpu_cores_effective>=16 cpu_ram>=64 disk_space>=300 dph_total<=0.79'
# Create: prints only the new instance id.
rent() { vastai create instance "$1" --image "$2" --disk "$3" --ssh --direct --raw | python3 -I -c "import json,sys;print(json.load(sys.stdin)['new_contract'])"; }
CUDA=nvidia/cuda:13.0.2-devel-ubuntu24.04; TORCH=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
# One paid run = watchdog armed first, then launch (cap; credit floor default $4). If launch fails (container not up
# yet), rerun the SAME command: the original watchdog keeps enforcing the cap and is reused.
# bash scripts/vast/start_run.sh RUN_DIR INSTANCE_ID CAP
# Wallet rule (B8 C4): before each tranche, credit must cover the caps of every box about to start plus the unspent
# caps of boxes already running, plus $5. Otherwise top up or launch fewer hosts; the $4 floor is only a last resort.
```

## T0 — canary (2 GPUs) and reconstruction

```bash
search "$TRAIN2" 200; ID_CANARY=$(rent <offer> $CUDA 200)
bash scripts/vast/start_run.sh $P/runs/canary $ID_CANARY 16       # ~7 h x $1.8 incl. CANARY.md step 3 replacement
# Then CANARY.md steps 1-4 (first pull timing, deadman "armed", preemption + restore, failed-backup injection).

search "$RECON2" 300; search "$RECON1" 300   # 2 GPUs if listed, else 1
ID_RECON=$(rent <offer> $TORCH 300)
df -h ~ | tail -1                                                   # need >=150 GiB free before the ~121 GiB pull [M 191 GiB now]
RUN="$PWD/data/brainstorm6/T3-BATCH/prepared"
caffeinate -dimsu .venv/bin/python -B scripts/tier3_native/morning.py "$RUN" "$ID_RECON" \
  --cap 18 --credit-floor 4 --hourly-rate-ceiling 1.40 --gpu-memory-mib 48000 --confirm-paid-reconstruction   # .79 for 1 GPU
```

## ~T0+1:30 — after the canary's first complete pull: Tier 2 (3 hosts), ep18 pilot, I1 screen

`--frozen-from` now verifies every archived hash on the Mac; "has no complete manifest" means wait for the next pull.

```bash
for k in 0 1 2; do .venv/bin/python -B $P/make_run.py --tier 2 --name t2h$k \
  --jobs $P/generated/tier2/split3/jobs-host$k-of3.txt --frozen-from $P/runs/canary/pulled/canary/frozen-tier2; done
search "$TRAIN1" 200      # rent 5 with: rent <offer> $CUDA 200  (115 GB must stay free after setup)
for k in 0 1 2; do bash scripts/vast/start_run.sh $P/runs/t2h$k <ID_T2H$k> 9; done   # 10 x 50M + 30 evals, ~6.2 h base / 8.9 h stress (B8 BOX)

.venv/bin/python -B $P/verify_payload.py --pilot $P/runs/ep18-pilot-d1
bash scripts/vast/start_run.sh $P/runs/ep18-pilot-d1 <ID_PILOT> 5     # cap ~ rate x 4.5 h + $1

bash scripts/vast/start_run.sh data/brainstorm7/runs/i1screen <ID_I1> 11  # 12 x 50M + 36 repeat evals [I 7.6 h base / 10.8 h stress]
```

## ~T0+4 — ep18 pilot decision; ~T0+3.5-7 — reconstruction pulled

```bash
cat $P/runs/ep18-pilot-d1/pulled/ep18-pilot-d1/decision.json
```

## ~T0+6 — sentinel gate, then Tier 1 bulk (7 hosts)

```bash
if .venv/bin/python -B scripts/sentinel_gate.py --qualification $P/runs/canary/pulled/canary/qualification; then
  for k in 0 1 2 3 4 5 6; do .venv/bin/python -B $P/make_run.py --tier 1 --name t1h$k \
    --jobs $P/generated/tier1/split7/remaining-host$k-of7.txt --frozen-from $P/runs/canary/pulled/canary/frozen-inputs; done
  for k in 0 1 2 3 4 5; do bash scripts/vast/start_run.sh $P/runs/t1h$k <ID_T1H$k> 14; done   # 4 x 350M, ~12 h
  bash scripts/vast/start_run.sh $P/runs/t1h6 <ID_T1H6> 11                                 # 3 x 350M
else
  echo "HOLD: no Tier 1 bulk staging or launch" >&2
fi
```

## I1 decision (frozen rule, on the Mac)

```bash
R=data/brainstorm7/runs/i1screen/pulled/i1screen
.venv/bin/python -I -B data/brainstorm7/T2-STRATEGY/i1_decide.py --root "$R" \
  --pair T2C-E4-seed44=T2C-contact-seed44 --pair T2C-E4-seed45=T2C-contact-seed45 \
  --out data/brainstorm7/T2-STRATEGY/i1_decision.json
```
GO only if the joint six-motion/seed decision prints GO (+2/6 MP, T2-STRATEGY §I1); then a T2C-E4 Tier 2 cohort, merged with
`merge --cohort <T2C-E4> --rescue <T2C-contact> ...`. NO-GO: $0 more.

## ~T0+8-12 — Mac bridge (29 new episodes, ~4.5 h)

```bash
RUN="$PWD/data/brainstorm6/T3-BATCH/prepared"
.venv/bin/python -B scripts/tier3_bridge_batch.py \
  --plan "$RUN/bridge_plan.json" --native-root "$RUN/pulled/$(basename "$RUN")/episodes" \
  --existing-ep18 data/tier3_native/ep18 --tier1-root data/challenge/track_2/tier_1_multiview_caption \
  --protocol data/brainstorm5/T3-RECON/PROTOCOL.md --rest-evidence-dir data/brainstorm6/T3-EVIDENCE \
  --rgb-dir data/brainstorm6/T3-BATCH/rgb --out data/brainstorm6/T3-REST/bridge_oct25 \
  --released-scene-assets-engineering-only
```

## ~T0+12-16 — Tier 3 coverage (needs the asset decision "use" and the pilot verdict)

```bash
# Set T3_ASSET_DECISION=use only after the user's recorded asset decision. Any failing step stops the whole block;
# `jobs` also refuses anything but all 30 released episodes and re-checks the pilot verdict's binding.
( set -euo pipefail
  [ "${T3_ASSET_DECISION:-}" = use ] || { echo "HOLD: user's asset decision required" >&2; exit 1; }
  COV=data/brainstorm6/T3-REST/bridge_oct25/coverage_manifest.json
  PILOT=$P/runs/ep18-pilot-d1/pulled/ep18-pilot-d1/decision.json
  .venv/bin/python -B - "$PILOT" <<'PILOT_GATE'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1]);d=json.loads(p.read_text());q=p.parent/'qualification/qualification.json'
assert d['decision'] in ('retain 50M','selective rescue','diagnose'), 'unrecognized pilot verdict'
assert q.is_file() and d['qualification_sha256']==hashlib.sha256(q.read_bytes()).hexdigest(), 'pilot qualification absent or changed'
PILOT_GATE
  .venv/bin/python -B scripts/tier3_coverage.py jobs --coverage $COV --pilot-decision $PILOT > $P/generated/tier3/coverage-jobs.txt
  .venv/bin/python -B scripts/split_jobs.py $P/generated/tier3/coverage-jobs.txt 5 $P/generated/tier3/split5
  for k in 0 1 2 3 4; do .venv/bin/python -B $P/make_run.py --tier 3 --coverage $COV --pilot-decision $PILOT --name t3h$k \
    --jobs $P/generated/tier3/split5/coverage-jobs-host$k-of5.txt; .venv/bin/python -B $P/verify_payload.py --pilot $P/runs/t3h$k; done
  # 6 x 50M + 18 evals, ~5 h; the host carrying a selective-rescue 350M ep18 job gets cap 10.
  for k in 0 1 2 3 4; do if grep -q '|T3P-contact350|' $P/runs/t3h$k/jobs.txt 2>/dev/null; then cap=10; else cap=7; fi
    bash scripts/vast/start_run.sh $P/runs/t3h$k <ID_T3H$k> $cap; done
)
```

Do not rent the coverage hosts until all five staged payloads verify. The pilot verdict is bound to its retained
qualification: `retain 50M` and D2's `diagnose` use 50M coverage; `selective rescue` uses the registered 350M recipe
for episode 18 only. That host needs an allowance for the longer job; check its cap before launch, without changing
the recipe or choosing by another score.

A bridge HOLD can still produce complete D2 coverage. A missing/FAIL non-18 episode blocks coverage spend: repair
the failed native/bridge stage and rerun the **full same bridge command**, which reuses completed stages when all
hashes match. Do not use `--episodes` with the canonical output: it republishes only that subset. Changed inputs
require a fresh retained root. Never turn a structurally invalid FAIL into HOLD just to reach 30.

## Before the first Tier 2/3 export (Oct 25 evening) — RELEASE.md plus two REVIEW additions

- On the export host, restore both canary archives and run `generate_jobs.py --production-only` before `merge` (L7).
- RELEASE.md's `t1a/t1b/t2a/t2b` are old examples. Tonight's cohorts (B8 RUNBOOK F8):

```bash
cohorts=(--cohort runs/canary/qualification)
for k in 0 1 2 3 4 5 6; do cohorts+=(--cohort "runs/t1h$k/qualification"); done
.venv/bin/python -B scripts/production_repeats.py merge "${cohorts[@]}" \
  --tier-root data/challenge/track_2/tier_1_multiview_caption --out runs/release-tier1
cohorts=()
for k in 0 1 2; do cohorts+=(--cohort "runs/t2h$k/qualification"); done
.venv/bin/python -B scripts/production_repeats.py merge "${cohorts[@]}" \
  --tier-root data/challenge/track_2/tier_2_synthetic_noise --out runs/release-tier2
```
- Run `data/brainstorm7/REVIEW/frame_preflight.py` on the export host before `export`; for Tier 3 pass the work
  directory whose rest-repaired `poses.npy` matches each trained reference (I2). `EXPORT_STARTED` forbids a retry.
- `code_commit_url` must name a pushed commit: needs the user's GitHub repo + push approval.

Tier 3 merge takes the five `runs/t3hK/qualification` cohorts after their trees have been restored at their original
paths. The merge writes `runs/release-tier3/manifest.json` containing all 30 frozen inputs; use that manifest for
the frame preflight. The exact per-episode converter `poses.npy` files (ep18 from candidate_v1) still need to be
assembled/transferred into the `--work` directory, as required by I2; training asset payloads alone do not provide it.

## Money [I] (operating ceilings; per-host caps above; the credit floor is the only cross-host stop)

| Phase | Est. | Cumulative cap |
|---|---|---|
| Canary $13, recon $6, Tier 2 $15, pilot $4, I1 $7 | $45 | $48 |
| + Tier 1 bulk 7 hosts | +$70 | $120 |
| + Tier 3 coverage 5 hosts | +$22 | $140 |
| + improvement rounds, reseeds, export hosts | | $180 |
