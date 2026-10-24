# First real box — fixed custody and restart gates

[I] Run only after the parent authorizes real rental/canary spend and resolves the provider login. These commands were NOT run here. `P=data/brainstorm6/PROD-READY`; commands start at the repo root. All destructive actions below address only the explicitly selected canary. Keep the Mac awake/networked. The watchdog cap/floor ALWAYS override backup retention; a failed final backup at either hard limit records `BACKUP_FAILED` and still requests destruction. A stopped instance is not evidence of stopped storage billing.

[I] After each authorized instance creation and before launch, require a running container with an SSH port. This bounded read-only readiness command was not run against Vast tonight; set `ID_CANARY` (or the replacement ID) from the create result:

```bash
.venv/bin/python - "$ID_CANARY" <<'PYWAIT'
import json,subprocess,sys,time
end=time.monotonic()+600
while time.monotonic()<end:
 r=subprocess.run(['vastai','show','instance',sys.argv[1],'--raw'],check=True,stdout=subprocess.PIPE,text=True,timeout=min(20,end-time.monotonic()))
 o=json.loads(r.stdout);state=o.get('actual_status')
 if state=='running' and o.get('public_ipaddr') and o.get('ports',{}).get('22/tcp'):break
 if state in {'exited','unknown','offline'}:raise SystemExit(f'NO-GO: instance {state}')
 time.sleep(min(10,max(0,end-time.monotonic())))
else:raise SystemExit('NO-GO: container not ready within 10 min')
print('container running; SSH endpoint allocated')
PYWAIT
```

1. [I] Start the prepared sentinel run and watchdog as in MORNING.md. Read `host` into controller variables; never print a provider response that might contain credentials:

```bash
P=data/brainstorm6/PROD-READY
D="$PWD/$P/runs/canary"; name=$(basename "$D")
read -r ID HOST PORT < "$D/host"
R=/workspace/v2d-track-2
S=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -i "$HOME/.ssh/id_ed25519" -p "$PORT")
stop_controller() {
  .venv/bin/python - "$1" <<'PYCTRL'
import os,signal,sys,time
from pathlib import Path
group=int(Path(sys.argv[1]).read_text())
assert group>1
try:
 os.killpg(group,signal.SIGTERM);time.sleep(.25);os.killpg(group,signal.SIGKILL)
except ProcessLookupError:pass
PYCTRL
}
start_controller() {
  .venv/bin/python - "$D" "$1" <<'PYCTRL'
import subprocess,sys
from pathlib import Path
d=Path(sys.argv[1]);record=d/sys.argv[2]
with (d/'controller.log').open('a') as log:
 p=subprocess.Popen(['caffeinate','-i','bash','scripts/vast/watchdog.sh',str(d),'12','1'],stdout=log,stderr=log,start_new_session=True)
record.write_text(str(p.pid)+'\n')
PYCTRL
}
ssh "${S[@]}" root@"$HOST" "df -h '$R'; nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free --format=csv"
```

[M/I] Require >=46000 MiB free/GPU and driver >=580 (bootstrap's existing admission), >=115 GB disk free after bootstrap for each assigned 15-long/15-short payload plus margin, exact source/recipe hashes, actual training progress. [G] Escalate setup after 60 min rather than extrapolating a warm install. Price/speed qualification starts after >=20% transitions: record cumulative steps/time for BOTH concurrent trainers, include failures and idle GPU time. A printed command or bootstrap exit is not training evidence.

2. [I] Time the first full, uncached periodic pull (bytes and wall time; target <600 s, otherwise replan transfer cadence/cost), while the sentinel is still incomplete; select an actor by maximum step (never by score), inventory and rehash every retained policy/state. Preserve the resulting JSON and pick a policy only from those verified bytes:

```bash
.venv/bin/python - "$D" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]);tree=root/'pulled'/root.name
items={str(p.relative_to(tree)):hashlib.sha256(p.read_bytes()).hexdigest() for p in tree.rglob('*.safetensors')}
assert any('/policy_' in p for p in items) and any('/state_' in p for p in items), 'NO-GO: no weights/state pulled'
(root/'custody.json').write_text(json.dumps(items,indent=2)+'\n')
print('verified local checkpoint files:',len(items))
PY
# A fresh directory on the GPU host is the independent restore/load target.
ssh "${S[@]}" root@"$HOST" 'test ! -e /workspace/canary-restore && mkdir /workspace/canary-restore'
rsync -az -e "ssh ${S[*]}" "$D/pulled/$name/" root@"$HOST":/workspace/canary-restore/
rsync -az -e "ssh ${S[*]}" "$D/custody.json" root@"$HOST":/workspace/
ssh "${S[@]}" root@"$HOST" "cd '$R' && .venv-flash/bin/python -" <<'PY'
import hashlib,json,subprocess
from pathlib import Path
root=Path('/workspace/canary-restore');expected=json.loads(Path('/workspace/custody.json').read_text())
for rel,h in expected.items():assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==h, rel
policies=[root/p for p in expected if Path(p).name.startswith('policy_')]
checkpoint=max(policies,key=lambda p:int(p.stem.split('_')[1]))
out=Path('/workspace/canary-reload-evaluation');assert not out.exists();out.mkdir()
kit=Path('/workspace/v2d-track-2/external/kit/v2d_submission_kit')
flash=Path('/workspace/v2d-track-2/external/video_to_data/robotic_grounding/flash_chord')
subprocess.run([str(Path.cwd()/'.venv-flash/bin/python'),str(kit/'flash_chord/scripts/evaluate_policy.py'),
 f'evaluation.checkpoint={checkpoint}','evaluation.world_count=1','evaluation.reset_mode=explicit',
 f'evaluation.metrics_output={out}/metrics.json',f'evaluation.object_trajectories_output={out}/trajectory.parquet',
 f'hydra.run.dir={out}/hydra','logging.mode=disabled'],cwd=flash,check=True)
m=json.loads((out/'metrics.json').read_text())
assert m['checkpoint_sha256']==hashlib.sha256(checkpoint.read_bytes()).hexdigest()
assert m['world_count']==1 and m['reset_mode']=='explicit' and m['non_finite_worlds']==[]
assert m['environment_steps']==int(checkpoint.stem.split('_')[1])
assert (out/'trajectory.parquet').stat().st_size>0
recipe=json.loads((checkpoint.parents[3]/'recipe.json').read_text())
key=f'{checkpoint.parents[2].name}/{checkpoint.parent.name}'
reference=recipe['references'][key]['training']['path']
subprocess.run([str(Path('/workspace/v2d-track-2/.venv/bin/python')),str(Path('/workspace/v2d-track-2/scripts/score_rollout.py')),
 '--rollout',str(out/'trajectory.parquet'),'--reference',reference,'--output',str(out/'scores.json')],check=True)
print('REHASH + REAL POLICY RELOAD + FULL FRAME SCORE PASS',checkpoint)
PY
```

[I] Copy reload output/logs back with `rsync -az -e "ssh ${S[*]}" root@"$HOST":/workspace/canary-reload-evaluation/ "$D/reload-evidence/"`. The real scorer above validates the exact full frame grid against the frozen policy input (T1 only for this sentinel). NO-GO if one missing/nonfinite frame, mismatched digest, deserialization failure, different evaluated physics, or no actual full rollout. Never use this diagnostic rollout as a submission candidate or selection replicate.

3. [I] Inject actual preemption only after step 2 passes and while a final policy is still missing. Stop the controller watchdog to prevent its old instance binding from racing launch; stop the remote process group, then destroy the original canary (the parent must already authorize this irreversible canary action). Record the PID when starting the watchdog in MORNING.md; do not kill an unrelated watchdog.

```bash
stop_controller "$D/controller-watchdog.pid"
ssh "${S[@]}" root@"$HOST" "read -r p < /workspace/$name.pid; case \$p in ''|*[!0-9]*) exit 1;; esac; test \$p -gt 1 && kill -TERM -- -\$p"
# Final state was already pulled and independently loaded in step 2.
# Only a successful destroy may publish the stop receipt that lets a different instance take over this run (B8 C7).
vastai destroy instance "$ID" -y && echo "$(date -u +%FT%TZ) manual canary preemption: destroyed $ID" > "$D/STOPPED"
cp "$D/STOPPED" "$D/canary-evidence-STOPPED-$ID"
# Parent chooses a fresh compatible offer using MORNING.md, creates it, sets NEW_ID.
bash scripts/vast/start_run.sh "$D" "$NEW_ID" 16
```

[I] Require `resume ... state_<latest pulled step>.safetensors` in the job log, continued progress to the SAME total, matching original recipe/input hashes, no replacement training seed, intact earlier actor digests, and complete final policy/evaluations. Local namespace `/workspace/v2d-track-2` remains constant; if source paths differ, use the evaluator's documented `evaluation.source_root`, never rewrite checkpoint bytes. Clear an old `STOPPED` marker only by archiving it as owned canary evidence before rearming. The launch script itself does not reconcile old instance bindings/markers; stop its old controller first.

4. [I] Inject a failed backup while the replacement finishes, below both hard financial limits. Replace only the controller's rsync boundary temporarily; no evaluator/trainer change. Run `stop_controller "$D/controller-watchdog.pid"` first, then:

```bash
mkdir -p "$D/fail-bin"
command -v rsync > "$D/real-rsync"
cat > "$D/fail-bin/rsync" <<'SH'
#!/bin/sh
case "$*" in *:/workspace/v2d-track-2/runs/*) exit 23;; esac
exec "$CANARY_REAL_RSYNC" "$@"
SH
chmod +x "$D/fail-bin/rsync"
CANARY_REAL_RSYNC=$(cat "$D/real-rsync") PATH="$D/fail-bin:$PATH" \
  start_controller failure-watchdog.pid
```

[I] Once remote `box done` is present and watchdog logs `backup failed (finished)`, REQUIRE no newly created `STOPPED`, no destroy request in watchdog.log, and the same live provider instance. The failure watchdog must stay alive and retry, not silently exit. Check `kill -0` on its recorded PID. If cap/floor has already tripped this test is INVALID; the hard rule is expected to destroy, so do not call that a durability defect. Run `stop_controller "$D/failure-watchdog.pid"`, remove only `D/fail-bin` and `D/real-rsync`, run `start_controller controller-watchdog.pid` to rearm the real watchdog, and require a successful final pull and destroy request. Independently confirm provider removal/billing; `STOPPED` proves only a successful request. Preserve all logs including the injected failure.

[G] Canary GO only when ALL of backup, byte-identical fresh restore, real policy reload, actual resume, failure-blocked soft destroy, and independent final provider termination pass. Otherwise no second training host. The Mac fixture proves transport/coordinator behavior, not this real gate. Cap $12 for the canary/replacement combined is an operating envelope, not a promise that the original box alone enforces a cross-instance sum; parent tracks both instances in its spend ledger.
