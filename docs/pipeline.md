# Pipeline reference

Script by script notes, commands and rules. For an overview, see the [README](../README.md). Paths are relative to the repository root.

## Scripts

```
released LeRobot episode ──lerobot_to_soma.py──> soma_params.npz + poses.npy + mesh + ground plane
                         ──soma_to_g1.py (upstream)──> G1 reference data.parquet
                         ──train_flash_sac.py (upstream)──> one policy per episode
                         ──kit evaluator + packer──> submission CSV
```

| Script | Runs on | Does |
|---|---|---|
| `scripts/lerobot_to_soma.py` | Mac or Linux | Converts a released tier into retargeter input folders. Interpolates invisible object frames (listed in `filled_frames.json`) and stages support surfaces. |
| `scripts/retarget_tier.sh` | Mac or Linux | Converts and retargets a whole tier in parallel; applies the mesh refresh patch first. |
| `scripts/apply_retarget_patch.sh` | Mac or Linux | Run `bash scripts/apply_retarget_patch.sh` to apply `patches/retarget_mesh_refresh.patch` idempotently; also called by retargeting and GPU bootstrap. Each load regenerates the OBJ from the current GLB. |
| `scripts/tier3_initializers.py`, `tier3_export.py`, `tier3_materialize.py`, `tier3_quality.py`, `tier3_prompts.py`, `tier3_native/` | Mac preparation / native CUDA inference | Stage-05 initializer bridge, portable G1 assets, label-free checks, video-only prompts, and resumable native batch; engineering evidence and asset limits in `data/tier3_bridge/REPORT.md`. |
| `scripts/check_references.py` | Mac or Linux | Loads every G1 reference with FlashCHORD's own loader; fails on empty or incomplete builds, load errors, non-finite values or loaded/source frame-count mismatches. |
| `scripts/contacts.py` | Mac or Linux | Rewrites `hand_contact_active` in retargeted Tier 2 references with noise-robust labels (see below). |
| `scripts/eval_contacts.py` | Mac or Linux | Scores contact-label rules on self-corrupted clean sequences. |
| `scripts/noise_stats.py` | Mac or Linux | Per-sequence noise statistics (jitter, carrying gap, foot height); moving-object statistics include rotation about a fixed origin. |
| `scripts/gpu_bootstrap.sh` | Rented GPU box | Checks driver/memory, installs both venvs without Docker, applies `patches/`, downloads data, retargets Tier 1 and 2. |
| `scripts/train_queue.sh` | Rented GPU box | Trains episodes sequentially on one GPU and resumes from the latest `state_*.safetensors`. Disables support for the 16 episodes whose released support stage is empty. |
| `scripts/eval_and_pack.sh` | GPU evaluation / Mac packing | Evaluates exactly `CHECKPOINT_MAP=selected.json` when supplied, and validates offline with `validate_offline.py` (kit validator plus a per-episode checkpoint check; the kit itself stays unmodified so stamped packer/uploader hashes match `KIT_MANIFEST.json`). Tier 2/3 require the exact frozen work-dir as argument 5 and default to step 49999872. Every tier packs with `pack_no_clean.py` (17 significant digits; the kit packer's `%.6f` can flip a boundary MP-SR). The older clean-packer guidance below is superseded for these tiers. Never uploads. |
| `scripts/pack_no_clean.py` | Mac or Linux | Packs all three tiers using the public sample grid and frame-reference NPZs; preserves achieved poses and checkpoint provenance, rejects duplicate/incomplete rollouts, and never calls the kit packer or clean scorer. |
| `scripts/write_checkpoints.py` | Mac or Linux | Builds a uniform-step map or validates `--checkpoint-map FILE` (plain map or tournament `checkpoint_map`); converts `episode_000000` keys to decimal 0–29, requires complete episode coverage and nonempty checkpoint files, and preserves the selected per-episode steps. |
| `scripts/frame_references.py` | Mac or Linux | Writes the packer's `--frame-references` NPZs from `sequences/*/poses.npy`, for denoised or reconstructed inputs. |
| `scripts/make_tier2_screen.py` | Mac or Linux | Builds the Tier 2 recipe screen in `data/screen/tier2/` from the 3 upstream example sequences: a clean control plus raw/denoised × original/repaired contacts, corrupted with an unused seed, with a manifest of hashes. Rerun on the GPU box to rebind asset paths. |
| `scripts/finger_targets.py` | Mac or Linux | Fills the Dex3 finger targets, which `soma_to_g1.py` leaves at zero (open hand), by per-frame IK toward the human thumb/index/middle tips, weighted by each tip's distance to the object. Frames where a hand link would go more than 8 mm deeper into the object (FlashCHORD's own CoACD pieces) or a support than the open hand already does are backed off toward open; frame 0 may not go deeper at all. Body joints, objects and labels are unchanged. `tree` rewrites a work dir into a new one; `screen` adds a `clean_fingers` input to a screen manifest. |
| `scripts/fit_support.py` | Mac or Linux | Writes support disks under the clean object's elevated, untouched rest poses (the upstream example stages don't match our retarget frame). Called by `make_tier2_screen.py`. |
| `scripts/screen_report.py` | Mac | Applies the frozen screen decision rules (`data/brainstorm2/PLAN.md`) to pulled `summary.tsv` files: F4 vs F1 at 350M, 2-of-3 arm rules, seed agreement, the R4 two-seed gate, noisy-input arms read against the input's own score (no credit or blame for gates the input already fails; MPPE > 15 cm counts as a catastrophe), and the re-evaluation check. Missing rows give no verdict. |
| `scripts/score_rollout.py` | Mac or Linux | Scores evaluator parquets with the kit's four metrics against an explicitly supplied clean reference (never the parquet's own `reference_*` columns). MPPE in cm. |
| `scripts/pilot_queue.sh` | Rented GPU box | Trains recipe arms (`scripts/pilot_arms.txt`) × screen inputs (`--manifest`, `--inputs`), evaluates chosen checkpoints with the kit evaluator, scores against the manifest's clean reference (or, for a manifest with `scoring_mode: self` such as the Tier 3 one from `scripts/tier3_manifest.py`, against each input itself), and writes `summary.tsv`. Supports `--dry-run`. A cached evaluation is reused only while the checkpoint, inputs, evaluator script and FlashCHORD source state (including applied patches) are unchanged. `--evaluate-only` or `PILOT_EVALUATE_ONLY=1` forbids implicit training; Tier 3 requires self-scoring, and summary references must match the frozen recipe. |
| `scripts/prepare_repeats.py` | Mac or Linux | `--plan FILE --source-root DIR --out-root DIR` stages separate repeat roots and job lists after checking checkpoint digests. Run the jobs through `test_day.sh` with `PILOT_EVALUATE_ONLY=1`; all rollouts are retained. |
| `scripts/test_day.sh` | Rented GPU box | Runs the round-2 screen (`data/brainstorm2/PLAN.md`): 13 `pilot_queue.sh` jobs from `pilot_arms.txt`, `pilot_arms_350m.txt` (full-length check) and `throughput_arms.txt`, spread over N GPUs, then merges `summary.tsv`. Supports `--dry-run`. `JOBS_FILE=FILE` swaps in another job list (e.g. `data/r4screen/jobs.txt`: the round-4 Tier 1 arms in `pilot_arms_r4.txt`); `MANIFEST=FILE` swaps the screen manifest. A failed sequence no longer cancels the arm's other sequences; the job exits non-zero naming it. Reruns validate artifacts, propagate failed jobs, lock each output root, and partition a frozen job list with `SHARD_COUNT` / `SHARD_INDEX`. |
| `scripts/vast/` | Mac controller | Launch restores retained run artifacts before starting the box. Watchdog retains every actor and the newest two learner states, checks billing/progress, stops writers and requires a successful final backup before a soft destroy; the budget cap and credit floor destroy even without one (`BACKUP_FAILED`). Failed stop/backup/destroy remains visible; `STOPPED` records a successful destroy request. Each cycle arms `deadman.sh` on the box, which stops the instance itself 30 min past the cap or 90 min after the box script ends, using the per-instance key. Real GPU recovery and billing termination still need the canary. |
| `patches/flash_chord_object_orientation.patch` | Applied by bootstrap | Opt-in object-orientation reward `exp(-(θ/σ)²)`: `objective.object_orientation_weight=1.0` enables it in all curriculum stages. Weight 0 (default) is bitwise identical to stock. |

## Local setup (Mac)

```bash
uv venv --python 3.11 .venv
# Same package list as the "Retargeting venv" step in scripts/gpu_bootstrap.sh, then:
scripts/retarget_tier.sh data/challenge/track_2/tier_1_multiview_caption data/work/tier1 3
```

`external/` (upstream clone, kit) and most of `data/` are not tracked; `scripts/gpu_bootstrap.sh` recreates the upstream code and challenge data. Reports and plans under `data/` that this README names but the repo lacks are local records. A fresh clone therefore can't run the whole test suite: many tests need the retargeted data, the upstream checkout and the trained policies, none of which are included.

## Tier 2: contact labels

`soma_to_g1.py` marks a hand in contact when a fingertip is within 2 cm of the object mesh, and FlashCHORD's force-closure reward only fires on those frames. Tier 2 moves hands and objects apart: compared with the three clean upstream examples, the median fingertip-to-object gap while the object moves is 5-15 cm in the worst episodes instead of 0.3-1 cm, so most grasps lose their labels. `scripts/contacts.py` relabels with hysteresis, hole filling and a per-episode threshold raised to that episode's carrying gap plus 4 cm (skipped when a foot, not a hand, moves the object).

Measured with `eval_contacts.py` on the three upstream examples, corrupted at five noise levels x five seeds, against clean 2 cm labels (per-hand F1):

| Rule | jitter | mild | medium | severe | extreme | mean |
|---|---|---|---|---|---|---|
| upstream 2 cm | 0.990 | 0.932 | 0.851 | 0.689 | 0.488 | 0.790 |
| `contacts.py` | 0.891 | 0.889 | 0.888 | 0.874 | 0.819 | 0.872 |

```bash
scripts/retarget_tier.sh data/challenge/track_2/tier_2_synthetic_noise data/work/tier2 3
.venv/bin/python scripts/contacts.py data/work/tier2
cd scripts && ../.venv/bin/python eval_contacts.py ../external/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/human_motion_data/whole_body/soma/sequence_id=*
```

## Evaluation and packing

```bash
scripts/eval_and_pack.sh tier1 runs/tier1 data/results/tier1 "$COMMIT_URL"
scripts/eval_and_pack.sh tier2 runs/tier2 data/results/tier2 "$COMMIT_URL" data/work/tier2_denoised
```

The packer accepts a reference only if it matches the released input within 1 mm / 1e-3 rad after one rigid transform. A Tier 2 or Tier 3 reference that differs from the released input therefore has to be packed with `--frame-references` built from the same `poses.npy` it was retargeted from. `eval_and_pack.sh` does this for Tier 2 and Tier 3. For unmodified inputs, `frame_references.py` reproduces the released poses exactly on all 22,989 scored frames per tier (0 m, below 1e-15 rad).

Checkpoints store absolute reference paths. Retarget on the GPU host; `gpu_bootstrap.sh` regenerates references whose asset paths are stale. FlashCHORD cannot train on a Mac CPU: Warp's graph replay segfaults on the first transition. [downstream_contract.md](downstream_contract.md) has the evidence and the full train/evaluate/pack contract.

## Tier 3: monocular reconstruction

Tier 3 reconstructs each episode from the exo video with upstream's SAM2 → CARI4D (MHR) pipeline. Those modules are Docker-only, so they need a GPU VM, not a container rental (Lambda A6000, $1.09/hr on 2026-10-21).
- `scripts/tier3_reconstruct.sh` runs the pipeline, one episode at a time.
- `scripts/tier3_export.py` converts CARI4D's output into SOMA inputs.
- `scripts/tier3_align.py` aligns the result to the released world using only Tier 1 frame 0, as the FAQ allows. It then writes the usual sequence folders; retarget them with `retarget_tier.sh --sequences`.

Status:
- The alignment math is tested on CPU: `.venv/bin/python -m unittest discover -s tests -p 'test_tier3_align.py'`.
- All 30 episodes were reconstructed on rented GPUs. None passed the quality gates (see [Results](../README.md#results)).
- Access to the gated `facebook/sam-3d-body-dinov3` and `nvidia/cari4d_commercial` weights is confirmed (2026-10-21; the CARI4D checkpoint's SHA-256 matches the pin). Whether Tier 3 may use the released object meshes and scene files was never resolved with the organizers.

[tier3.md](tier3.md) has the commands, pinned weights, costs and kill criteria.

## Rules this repo follows

- Tier 2 and Tier 3 inputs never use the clean Tier 1 trajectories, except the first Tier 1 frame for Tier 3 alignment, as the FAQ allows.
- Tier 2/3 methods are validated on self-corrupted non-challenge data, not on the clean reference.
- Tier 2/3 are packed only for final submissions, because the packer prints clean-reference scores.
