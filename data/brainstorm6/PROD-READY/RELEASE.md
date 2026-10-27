# Repeat qualification and the single fresh full-tier export

[M/I] The supplied patch adds `scripts/production_repeats.py` on top of existing `prepare_repeats.py`, `test_day.sh` and evaluate-only. Production requires the frozen k=3 (`--k 3`); `make_run.py` refuses other values before staging. Diagnostic screens can use other k through `prepare_repeats.py`. For Tier 1, “k-mean” is explicitly mean **AUC**, tie -> final; no arbitrary sum of metrics with different units. All cohort trajectories remain on disk. The first trial can favor 330M while the mean selects final; the CLI test exercises this and a separate cohort where 330M wins.

## On each production GPU box

[I] `make_run.py` already stages these exact operations as its post hook, before `box done`; do NOT run them a second time. Example for a manual fresh declared run:

```bash
# RUN is the existing production run name, TIER is 1 or 2; use its exact frozen input.
RUN=t1a
MANIFEST="$PWD/data/brainstorm6/PROD-READY/generated/tier1/manifest.json"
.venv/bin/python -B scripts/production_repeats.py prepare \
  --jobs "/workspace/$RUN.jobs" --manifest "$MANIFEST" --source "runs/$RUN" \
  --tier-root data/challenge/track_2/tier_1_multiview_caption --tier 1 --k 3 \
  --out "runs/$RUN/qualification"
.venv/bin/python -B scripts/production_repeats.py run --gpus 2 --out "runs/$RUN/qualification"
.venv/bin/python -B scripts/production_repeats.py qualify --out "runs/$RUN/qualification"
```

[M/I] Tier 1 stage requests exactly 330012672 and 349999104, k times each, separate roots. Every requested policy and its initial checkpoint receipt must exist before staging. A repeat run is evaluate-only; it cannot train. In Tier 1, a missing/invalid member, changed policy/input/runtime, duplicate/additional trajectory or non-finite artifact blocks qualification. In Tiers 2 and 3, a member whose evaluation ran but failed without sealing is recorded as a catastrophe and qualification completes (BOX-8); a member that never completed still blocks, as do the other defects. `RUN_STARTED` prevents rerunning a failed or completed cohort; diagnosis requires a new declared experiment/root, with the old attempts retained, never repeated until a favorable score.

[I] Tier 2 uses the same commands with `--tier 2`, its raw_repaired manifest and tier_2_synthetic_noise metadata; only policy_49999872 is eligible. Mean position <=.10 m, mean orientation <=.5 rad, self-MPPE <=15 cm must pass in EVERY member. Tier 3 production is also final-only 50M/own-input; the ep18 pilot uses `--tier 3 --ep18`, one budget job at a time in separate roots. It uses the stricter self-MPPE <=10 cm veto. The existing T3-POLICY protocol still owns the 858-frame/drop/carry checks on EVERY member; this generic cohort tool does not replace those checks or grant physical/RGB/input/asset approval. No score-based choice between the 50M and 350M pilot; use the registered 50M-first protocol.

## Assemble 30 selected policies, then export once

[I] Watchdog pulls retain run trees, qualification cohorts, policies and frozen input closures before destruction. On a fresh compatible evaluation host, restore the complete retained trees under the SAME `/workspace/v2d-track-2/runs/<name>` paths and restore the canary's frozen tier input using `freeze_inputs.py restore`. Do not regenerate a changed reference under a trained policy. Use the existing launch restore path per run (`launch.sh` restores `RUN_DIR/pulled/<name>/` before starting writers), or explicitly transfer already-finished trees without starting trainers. Verify all archive hashes after transfer. First real cross-host transfer and actual GPU reload are CANARY.md gates, not Mac performance evidence.

[M/I] Tier 1/2 export requires the work tree restored by `freeze_inputs.py restore --archive runs/<name>/frozen-inputs`, including `data/work/tier{1,2}/sequences/<sequence>/poses.npy`. Export checks its bytes against every retained cohort's archive before `EXPORT_STARTED`. The Mac's regenerated poses fail all 30 hashes in each tier despite float differences at roundoff scale; use the restored frozen tree. Tier 3 requires the sealed tree from `assemble_tier3_work.py`; export runs its existing verifier.

[I] A complete `launch.sh` upload writes `/workspace/v2d-track-2/LAUNCH_COMMIT`. The export URL must name that full 40-hex commit and the record must say `changed_files=0`. This is the export host's fresh evaluation/packing code; training launches may have different commits. Hosts without this record need a complete clean upload. Partial native uploads invalidate it. Publication of the named commit remains an external prerequisite.

```bash
# On the final evaluation GPU host, Tier 1 includes the 3 sentinels plus disjoint 14+13 cohorts.
.venv/bin/python -B scripts/production_repeats.py merge \
  --cohort runs/canary/qualification --cohort runs/t1a/qualification --cohort runs/t1b/qualification \
  --tier-root data/challenge/track_2/tier_1_multiview_caption --out runs/release-tier1
.venv/bin/python -B scripts/production_repeats.py export --out runs/release-tier1 \
  --work data/work/tier1 --commit "$PINNED_PUBLIC_COMMIT_URL"
# Tier 2: same merge/export, with its two cohorts, Tier 2 metadata, work and a fresh release root.
.venv/bin/python -B scripts/production_repeats.py merge \
  --cohort runs/t2a/qualification --cohort runs/t2b/qualification \
  --tier-root data/challenge/track_2/tier_2_synthetic_noise --out runs/release-tier2
.venv/bin/python -B scripts/production_repeats.py export --out runs/release-tier2 \
  --work data/work/tier2 --commit "$PINNED_PUBLIC_COMMIT_URL"
```

[M/I] `merge` requires every metadata episode exactly once, identical k/tier/input-manifest/runtime identities and intact selected hashes; duplicate/missing coverage fails before creating a release. `export` checks frozen work and the clean launch commit before claiming an attempt, then freezes one new output directory and starts exactly one fresh per-policy evaluation through the real kit coordinator in `eval_and_pack.sh`. It uses `pack_no_clean.py` for every tier, then `validate_offline.py`. The output score cannot select another rollout or trigger a rerun. `EXPORT_STARTED` prevents a second invocation, including after a failed attempt; failed outputs remain. No cohort member is itself selected as the submission rollout.

[I] A technical export failure requires diagnosis and an explicitly recorded new experiment, never a hidden retry. The final map/hash and full attempt ledger travel with the CSV. Tier 2/3 must never run the kit packer or clean scorer. The test places a Python/Arrow access fence around the full harder-tier export coordinator. Only fixtures are packed here; real all-30 achieved exports and organizer acceptance remain unproved. `pack_no_clean.py` stamps its own builder hash; whether that provenance is accepted remains an organizer ruling, not something offline validation settles.
