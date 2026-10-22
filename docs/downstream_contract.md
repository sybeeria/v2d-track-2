# Track 2 downstream readiness — 2026-10-21

**CPU end-to-end is blocked by a reproducible native graph-replay crash.** No trained policy,
evaluation trajectory, or submission CSV was produced. The conditional 30-episode CPU packing
run was not started. The contract and handoff scripts are checked offline; a real GPU
train → official evaluator → Tier 1 pack remains the outstanding runtime gate before a full queue.

All paths below are relative to `/Users/siberia/.t3/projects/v2d-track-2`.
`FC` means `external/video_to_data/robotic_grounding/flash_chord`;
`KIT` means `external/kit/v2d_submission_kit`. Upstream HEAD is
`7eb9721088615fe9201634a6be5cf48b049d1090`. Shared `.venv` and `data/work/` were not modified.
No commits, pushes, uploads, submissions, Kaggle commands/API calls, or Tier 2/3 packing occurred.

## Contract (verified from the checked-out code)

- **Evaluation takes the actor policy**, `runs/<tier>/<sequence_id>/policy_349999104.safetensors`
  for the default G1 recipe. Short runs use `policy_<actual environment steps>.safetensors`.
  `FC/src/flash_chord/training/flash_sac/checkpoint.py:123` constructs `policy_<step>` and
  `state_<step>` separately; `:149` rejects the wrong `checkpoint_kind`; `:162` loads only
  `policy`. `FC/src/flash_chord/training/flash_sac/evaluation.py:128` calls that actor loader.
  Resume training with `training.resume=true training.checkpoint_path=/absolute/state_<step>.safetensors`.
  Replay is not saved by default: restoring compact state is not a bit-for-bit simulation/replay resume.
  See checkpoint.py:172 and `configs/training/reference_flash_sac.yaml:87`.
- **Checkpoint map:** a JSON object with decimal episode ID strings as keys and path strings
  as values, e.g. `{"0": "/workspace/v2d-track-2/runs/tier1/<sequence>/policy_349999104.safetensors"}`.
  There is no enclosing `episodes` object or nested checkpoint object. Values are resolved from
  the runner's working directory, not the map file's directory; our writer emits absolute paths.
  See `KIT/examples/run_policy_evaluation.py:53-65`.
- **Runner:** `examples/run_policy_evaluation.py:69-84` invokes the kit evaluator once per ID,
  requests one world, explicit reset, disabled logging, and writes `episode_XXXXXX.parquet`
  plus `episode_XXXXXX.json`. It reconstructs the experiment from checkpoint metadata.
  The underlying evaluator YAML defaults to 4096 worlds; use the batch runner's default **1**,
  matching `KIT/config/tracks.json`'s Track 2 `worlds_per_episode`.
- **Trajectory:** one file per scored episode, all scored world/frame/object tuples present
  exactly once. The native schema is `flash_chord_object_trajectories_v1`; it contains
  `environment_id`, `reference_frame`, `object_body_id`, achieved and reference positions and
  xyzw quaternions, and `environment_valid`. Position units are metres. See
  `FC/src/flash_chord/evaluation/rollout.py:59-105`, `:130`, and
  `KIT/v2dlb/submission_parquet.py:119`. A single G1 object's name may be `object`
  (`KIT/tools/pack_submission.py:357`). Sequence-ID reference paths do not encode episode IDs;
  the output filename `episode_XXXXXX.parquet` supplies that identity as a fallback
  (`KIT/v2dlb/submission_parquet.py:92`).
- **Metrics JSON is optional.** It is useful for Tier 1 cross-checking and supplies a checkpoint
  SHA fallback; it does not replace trajectory metadata for evaluation-code provenance.
  The outcome JSON has top-level `checkpoint_sha256` (`harness.py:92`, `:619`);
  the script/code digests are in parquet schema metadata. Native exports carry
  `flash_chord.checkpoint_sha256`, `flash_chord.eval_script_sha256`, and
  `flash_chord.eval_code_sha256`. `harness.py:138` hashes the checkpoint and script;
  `:127` hashes the sorted names, NUL separators, and bytes of four evaluation modules:
  `__init__.py`, `harness.py`, `metrics.py`, `rollout.py`. This digest is not a hash of the
  entire simulator/runtime. `rollout.py:75-78` adds the `flash_chord.` prefix.
- **Packing provenance:** `KIT/tools/pack_submission.py:289-308` reads parquet hashes, with
  JSON fallback only for checkpoint SHA; `:411` optionally hashes `--eval-script` if needed.
  `:419-444` requires one uniform script/code digest across episodes and allows distinct
  policy hashes. `:486` rejects any blank provenance cells. The packer adds its own SHA,
  uploader SHA, and GitHub `/commit/<sha>` URL (`:462-482`). Missing checkpoints are not
  repaired by inventing hashes. Keep evaluator outputs in their own directory because
  `load_results()` recursively reads every parquet and JSON (`:52`).
- **Runtime selection:** all four files shipped in `KIT/flash_chord` match upstream byte for
  byte (measured by `audit_contract.py`). The kit contains the evaluator script and three
  config files, not an installable complete FlashCHORD runtime. Install pinned upstream
  `.[train,dev]`, and run the kit entry point. No Isaac Lab import is needed by this version;
  that wording in the example runner's docstring is stale. Native export is implemented in
  `FC/src/flash_chord/evaluation/rollout.py:130`, called from `harness.py:498`.
  Script SHA: `160d24f5fbeb70ccd569ca4192056f1892b2a9f2999066db9d8279461648fdf1`.
  Evaluation-code SHA: `b1b3ebe3ae4629d1687a83f420149e3f6fdfc154d70acad55c12fc8fa69811a1`.

## Paths, frames, and visibility

Training saves the resolved experiment as `resolved_config_json`, including absolute
`task.parquet` and the resolved reference path (`FC/scripts/train_flash_sac.py:76-85`,
`training/checkpoint_metadata.py:19`, `:97`, `:203`). This is established from source;
there is no newly trained checkpoint header to inspect in this run. The G1/SONIC default
asset directories are resolved by the installed package; explicitly supplied absolute
asset overrides would also remain absolute in saved config.

`--source-root` has deliberately narrow behavior (`harness.py:283-290`):

- `/workspace/X/data.parquet` becomes `<source-root>/X/data.parquet`.
- Relative `X/data.parquet` becomes `<source-root>/X/data.parquet`.
- `/Users/.../data.parquet` and absolute paths outside `/workspace` are unchanged.

For a checkpoint trained at `/workspace/v2d-track-2`, moving the project to this Mac would
require source-root `/Users/siberia/.t3/projects`, not the project directory itself.
`--runtime-root` only selects the child process's working directory
(`KIT/examples/run_policy_evaluation.py:50`, `:84`); it neither installs a package nor rewrites
checkpoint paths or parquet cells.

All 60 local reference parquets contain absolute `object_mesh_paths` / `object_urdf_paths`
under this Mac's `data/work/tier{1,2}/motion/.../object/`. The loader remaps missing paths
only when they contain `/assets/` or `/data/object_assets/`; these local paths contain
neither (`FC/src/flash_chord/assets/registry.py:43-66`). Copying them to Linux without
regenerating or deliberately relocating their asset closure fails. Bootstrap now regenerates
references when their stored assets are missing, even if the parquet count is already 30.
Full exact paths are recorded in `contract/source-audit.json`.

The G1 recipe keeps source frame numbering: `source_frame_playback=true`, speed 1, control
FPS 50 (`configs/experiment/g1_recon_body_flash_sac.yaml:17`, `configs/sim/recon_body.yaml:2`).
`data/motion_v1.py:272-303` bypasses resampling, preserving frame count despite the released
source FPS. Evaluation advances through the entire reference and records
`reference_frame = step + start_frame` (`evaluation/rollout.py:189`, `:346`); it disables early
tracking-failure resets and assistance (`evaluation/harness.py:216-249`).

The packer's clean reference reader drops invisible frames while retaining their original
indices (`KIT/v2dlb/trajectory.py:92-97`). Extra unscored trajectory rows are ignored after
selecting the required grid (`tools/pack_submission.py:311-352`). Our converter fills invisible
frames by interpolation/nearest endpoint for simulation, without dropping rows
(`scripts/lerobot_to_soma.py:108-130`). Each tier has 22,990 input frames, including one
filled/invisible frame, and 22,989 scored frames. Filled poses are simulated/exported but
not scored. Evaluator JSON metrics can therefore differ from packer metrics; the Tier 1
cross-check warns rather than failing (`pack_submission.py:590-617`). No cross-check was disabled.

The converter keeps the released world transform unchanged in `poses.npy`: it only converts
wxyz quaternions to matrices and interpolates invisible samples. Our new writer reverses
that representation to xyzw, with no axis flip, centering, or use of clean answers. The
retargeter's separate world/base transform is accounted for by the packer's rigid conversion
(`pack_submission.py:82-129`). Its maximum tolerances are **1 mm and 0.001 radians**.
For edited Tier 2/3 inputs, pass frame references derived from the exact edited `poses.npy`
that generated the retargeted references; using the released unedited input would erase that
contract and usually fail the rigid check.

## CPU experiment and exact blockers

Environment: CPython 3.11.1, M1/16 GB, JAX 0.10.2 CPU, Newton 1.2.0, MuJoCo and mujoco-warp
3.8.1, Warp 1.14.0, onnxruntime 1.28.0 CPU, and every other direct FlashCHORD pin unchanged.
`contract/cpu-requirements.txt`, `cpu-install.log`, and `cpu-freeze.txt` preserve installation
inputs and resolution. pandas 3.0.6 was added for kit tooling. `.venv` was not used for installs.

The one-episode command selects released Tier 1 episode 0, uses one world, a 16-transition
budget, replay capacity 64/minimum 8/batch 8/n-step 1, small networks, float32, and saves every
8 transitions. Logging is disabled. The normal 350M-step recipe and 10M replay were never
allocated on this Mac.

1. **Setup failure, resolved privately:** upstream FlashCHORD assets were still LFS pointers.
   Newton loaded the first G1 STL as an empty mesh and raised
   ```text
   ValueError: cannot call `vectorize` on size 0 inputs unless `otypes` is set
   ```
   See `contract/cpu-train-lfs-blocked.log` and Newton's `utils/mesh.py:721`.
   A private copy under `data/smoke/flash_chord` has identical source plus 65 materialized
   G1/SONIC LFS files, each size/SHA-verified against its pinned pointer (122,557,955 bytes).
   The shared upstream checkout was not materialized/edited. `prepare_cpu_runtime.py` reruns this.
2. **Real recipe failure, fixed in the queue:** episode 0's released support stage is an
   empty Xform. `FC/src/flash_chord/scene/support.py:94` raises
   `ValueError: support surface contains no Cylinder or Cube collision shapes: ...`.
   There are **16 empty and 14 populated stages in each tier** (measured, full stage traversal).
   For an empty stage `scene.support=false` is the correct config; populated stages retain
   support. Missing/unreadable stages still fail. See `contract/cpu-train-empty-support.log`.
3. **Unresolved CPU blocker:** with actual assets, CPU device, empty-stage support disabled,
   and `action.enable_cuda_graph=false`, the simulator builds and JAX compiles, but the first
   collection segfaults (exit **139**). Both attempts print
   `APIC: Error - memset dst pointer not in any registered region` during capture. The
   faulthandler retry pinpoints:

   ```text
   Fatal Python error: Segmentation fault
   warp/_src/context.py:10463 in capture_launch
   flash_chord/envs/rl.py:484 in step
   flash_chord/training/environment.py:143 in step
   flash_chord/training/flash_sac/runner.py:560 in _run_training
   ```

   `FC/scripts/train_flash_sac.py:73` unconditionally calls `env.capture_step()`;
   `envs/rl.py:487-493` unconditionally captures the RL step. There is no Hydra flag to
   disable this environment graph. `action.enable_cuda_graph=false` only affects SONIC's
   ONNX CUDA provider (`runtime/actions/sonic.py:523-537`, `:581`), which correctly selects
   CPUExecutionProvider on CPU. Warp 1.14 does implement CPU graph replay using APIC
   (`warp/_src/context.py:9787-9849`); this is a reproduced CPU replay failure, not evidence
   that Warp capture universally requires CUDA. The evaluator also captures unconditionally
   (`evaluation/harness.py:427`), so the same unsupported path remains a source-inferred
   evaluation risk. No evaluator rollout was run without a checkpoint.

Native crash evidence: `contract/cpu-train.log`, `cpu-train-capture-retry.log`, and
`cpu-crash.ips` (EXC_BAD_ACCESS / SIGSEGV / null address). No monkeypatch, pin change, eager
fallback, simulator substitution, or altered evaluator was used to manufacture a successful run.

**Measured CPU environment steps/s: unavailable.** Both configured attempts completed zero
training collections and produced no checkpoints. Treating setup/JIT time or a crashed
first transition as throughput would measure the wrong work. No GPU throughput, cost, score,
or training-time estimate is inferred from this.

## Frame-reference proof and verification

`verify_frame_references.py` executes the writer as a CLI, then uses the packer's own
`input_frame_reference()` on both output NPZs and that tier's released dataset, selecting
the entire public sample-submission grid. Tier 2 comparisons use only Tier 2 poses and public
row IDs. It never calls `pack_submission.main`, clean-reference scoring, or any metric.
It also loads each robot reference with source-frame playback and calls only the rigid
frame-conversion helper against its same-tier writer output. Per-episode source hashes and
residuals are retained in `contract/frame-verification.json`.

| Measured check | Tier 1 | Tier 2 |
|---|---:|---:|
| Episodes / scored frames checked | 30 / 22,989 | 30 / 22,989 |
| Max NPZ-vs-released position residual, m | 0 | 0 |
| Max NPZ-vs-released angle residual, rad | 7.1101e-16 | 7.0793e-16 |
| Max retarget rigid-conversion position residual, m | 1.6831e-7 | 1.4493e-7 |
| Max retarget rigid-conversion angle residual, rad | 1.6859e-7 | 1.5197e-7 |

Angles compare normalized rotations, so quaternion sign and original float32 quaternion norm
do not introduce false rotation residuals. This is a full-grid representation check, not a
trajectory-quality score or a performance benchmark. The original Tier 2 input at the recorded
hashes is unmodified; later denoised inputs should not be expected to match it.

Baseline and final focused upstream suites: **158 passed, 7 skipped, 1 warning**. Skips require
CUDA; the warning is the existing tensorflow_probability/JAX deprecation. Existing reference
loader checks passed for all 60 references before and after. Three new CLI behavior tests
passed (exact checkpoint mapping + missing episode rejection; modified work poses + invalid
matrix rejection; queue resume selection + failure exit). Queue process tests use a stub at
the external trainer boundary and do not prove training. Real Hydra CLI composition through
the queue was checked with one empty and one populated stage; real kit evaluator composition
also passed. These probes are explicitly config-only. `bash -n` passed all three shell scripts.

## Files and changes

- New `scripts/write_checkpoints.py`: requires every released episode's final policy in
  `runs-root/<sequence_id>/policy_<steps>.safetensors`; default steps 349999104; fails before
  replacing a map if any episode is absent. It maps paths, not checkpoint tensor validity.
- New `scripts/frame_references.py`: reads work poses, validates rigid finite matrices and
  matching frame counts, writes the packer's `[T,1,7]` NPZ contract including filled frames.
- New executable `scripts/eval_and_pack.sh`: writes the map, evaluates sequentially with
  the kit script, packs, then calls only the uploader's offline `validate_submission()`.
  Tier 1 uses released input; Tier 2/3 generate and pass `--frame-references` from their work
  directory. The script supports final Tier 2/3 packaging but was never run for those tiers.
- Changed `scripts/train_queue.sh`: handles empty support stages, aggregates failures into
  a nonzero exit, rejects an empty reference list, safely expands override arrays, puts Hydra
  outputs in each run, and accepts interpreter/runtime overrides for checks. Corrected the
  timing example from 10,000,000 steps (not divisible by 2048) to 10,000,384 steps. Existing
  experiment, final checkpoint name, seed, and resume override names were correct.
- Changed `scripts/gpu_bootstrap.sh`: checks disk, uses a unique cleaned kit archive,
  installs kit requirements even for an existing retarget venv, repairs FlashCHORD installation
  on reruns, explicitly retains pinned upstream as runtime, and regenerates stale absolute-path
  references. No GPU bootstrap was executed on this Mac. Linux install/driver behavior remains
  unverified. Before/after copies and unified diffs are in `contract/`.
- Evidence and rerunnable checks under `data/smoke/`: `verify_all.sh`, `test_handoff.py`,
  `verify_frame_references.py`, `audit_contract.py`, `probe_configs.sh`, CPU preparation/run
  scripts, per-tier NPZs, logs, freeze, crash report, and the isolated runtime asset copy.
  Proposed README addition is `PROPOSED_README.md`; the existing README was not edited.

`submit/v2d_submit.py --dry-run` is **not offline**: `:544-551` builds the API and performs
Kaggle preflight before checking dry-run. The isolated `validate_submission()` at `:69`
checks CSV columns, values, provenance, and exact public row order without invoking Kaggle.
No CSV exists here, so that final validation was not run. `tools/preflight.py --skip-kaggle`
is the kit's explicit offline mode, but cannot certify missing trajectories/checkpoints.

## Reproduce on this Mac

Run from the project root. The first two installation commands are unnecessary if retaining
the completed environment. Do not install into `.venv`.

```bash
uv venv --python 3.11 .venv-flash-cpu
uv pip install --python .venv-flash-cpu/bin/python -r data/smoke/contract/cpu-requirements.txt
.venv-flash-cpu/bin/python data/smoke/prepare_cpu_runtime.py
uv pip install --python .venv-flash-cpu/bin/python --no-deps -e data/smoke/flash_chord
uv pip install --python .venv-flash-cpu/bin/python pandas==3.0.6
bash data/smoke/verify_all.sh
bash data/smoke/probe_configs.sh
# Expected native crash, exit 139; not a successful smoke test:
PYTHONFAULTHANDLER=1 bash data/smoke/run_cpu_smoke.sh scene.support=false
```

Use the artifact writers independently:

```bash
.venv/bin/python scripts/write_checkpoints.py \
  --tier-root data/challenge/track_2/tier_1_multiview_caption \
  --runs-root runs/tier1 -o data/smoke/checkpoints.json
.venv/bin/python scripts/frame_references.py \
  --work-dir data/work/tier2 \
  --tier-root data/challenge/track_2/tier_2_synthetic_noise \
  -o data/smoke/custom-frame-references/tier2
```

The first command intentionally fails today because no trained policies exist. On a GPU box
after successful training of all episodes, the Tier 1 end-to-end command is:

```bash
scripts/eval_and_pack.sh tier1 runs/tier1 data/smoke/tier1 \
  https://github.com/example/example/commit/0000000
```

That placeholder URL is only for an offline smoke artifact; use the actual code commit for
a real final submission. No upload command is part of the script.

## Disk and cleanup

Measured retained allocation: `.venv-flash-cpu` **1,369,736 KiB (1.306 GiB)** and
`data/smoke` about **140 MiB (0.134 GiB)**, roughly **1.44 GiB combined**. The smoke tree
includes the required private runtime/assets (~122.5 MiB), CPU Warp cache (~11 MiB), and
frame references (~3 MiB). These remain reusable by the reproduction commands. Exact final
sizes are in `contract/disk.json`; small report/log changes can move the smoke total slightly.
The shared uv download cache was used, but its net increase cannot be isolated from concurrent
workers. Filesystem free space went from 196 GiB to about 194 GiB; that system-wide delta is
not attributed exclusively to this task. No process launched by this task remains running.
Temporary handoff fixtures and archive downloads are removed; retained before/after copies
are review evidence, not abandoned worktrees or builds.
