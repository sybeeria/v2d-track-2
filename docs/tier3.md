# Tier 3 preparation — 2026-10-21

[M] Updated 2026-10-22: episode 18 stages 01–05 now bridge **on the M1 CPU** through SOMA, guarded frame-0 alignment and G1 retargeting. A portable 858-frame reference passes the real FlashCHORD loader and train-queue configuration path. [M] Label-free checks flag substantial acceleration and ground-penetration issues. [I] This establishes an engineering training input, not a policy-quality or challenge-eligibility result. Released mesh/plane/support use remains unresolved. See `data/tier3_bridge/REPORT.md` for current commands, evidence, 30 video-only prompt pairs, native dry-run qualification and open gates. The older preparation notes below retain their original source-review scope.

## Rules and unresolved asset permission

The [official FAQ, page 3](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/assets/v2d_challenge_2026_faqs.pdf), checked 2026-10-21, says exactly:

> Yes! Using the first frame of Tier 1 data to compute alignment is fine, provided the Tier 1 caption isn’t used in Tier 3 otherwise.

The FAQ requires learned-policy object trajectories for all 30 episodes and separate tier submissions. Whole-body control is required; pretrained controllers, pretrained models, and external datasets are allowed. Later clean Tier 1 trajectory frames cannot be used to recover, tune, or replace Tier 3 trajectories. This task additionally excludes Tier 2 trajectories and submission packing.

**Not found in the published FAQ/site/dataset card:** an explicit Track 2 Tier 3 permission or prohibition for released object meshes, ground-plane sidecars, or support USDA files. It does not expressly require reconstructing the mesh for Track 2 Tier 3. Its explicit own-mesh requirement is for **Track 3**, and its prohibition on Track 2 assets is for **Track 1**; neither establishes a Track 2 Tier 3 ruling. Therefore do not claim these assets are approved. Parent should resolve that narrow rules question with the organizers before using clean scene assets. No contact was sent. The scripts require explicit mesh/plane/support inputs and never fetch clean assets automatically.

The [challenge site](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/) identifies CARI4D as the Tier 3 off-the-shelf method and scores every tier against the clean object trajectory. Freeze: **2026-11-04 17:00 EST (22:00 UTC)**. The [dataset card](https://huggingface.co/datasets/nvidia/video_to_data_challenge/blob/main/README.md) describes the tiers but adds no Tier 3 asset ruling. Source captures and hashes are in `data/tier3/evidence/sources.json`; the FAQ PDF is retained there.

## Pipeline traced at upstream 7eb9721088615fe9201634a6be5cf48b049d1090

Paths below are relative to `external/video_to_data/`. This is the vendored **MHR commercial CARI4D implementation**, not the separate public NVlabs/CARI4D SMPL-H/custom-video recipe, whose model preparation and scale procedure differ.

`robotic_grounding/run_pipeline_local.sh:1` is a dataset-loader/retargeting orchestrator for ARCTIC/TACO/HOT3D/etc., **not** the monocular reconstruction entry point. The applicable reconstruction entry is `reconstruction/modules/v2d_cari4d/docker/run_inference.py:14`, which invokes `lib/run_inference.py:90`. Host Python launches Docker; heavy models stay inside containers. The generic multi-view pipeline (`reconstruction/modules/v2d_pipelines/run_mv_hoi_reconstruction.py`) requires calibrated multi-camera inputs and must not be substituted for Tier 3.

| Step | Exact upstream code / outputs under `<output>/<sequence>/` |
|---|---|
| SAM2 masks | `modules/v2d_sam2/docker/run_video_to_masks.py`; video + video-only prompts, human ID 0/object ID 1 → `0/000000.png`, `1/000000.png`, etc. |
| Pack masks | `v2d_cari4d/lib/pack_masks.py`; strict count/dimension/stem check → H5 `<sequence>/<frame>-k0.person_mask.png` and `.obj_rend_mask.png` datasets. |
| 1. MoGe 2 | `lib/run_inference.py:129`, `prep/mhr_wild_depth.py` → `depth/moge2/raw.npy`, `intrinsics.pkl`, `depth_report.json`. Batch 8. |
| 2. Export RGB/masks/mesh | `lib/run_inference.py:136`, `prep/prepare_mhr_wild_export.py:79` → `export/<sequence>/wild_export.json`, image/mask H5s, `edex`, `object_mesh/output_aligned.glb`. Object geometry is centered/reoriented using oriented bounds; source-to-aligned matrix retained in JSON. Scale is preserved. |
| 3. Human | `lib/run_inference.py:139`, `prep/run_sam3d_mhr_export.py` → `mhr_init/<sequence>.pkl` and `.sam3d.h5`. SAM2 bbox+mask prompts, chunk 16, refit batch 512, `--no-align-to-gt-depth`. |
| 4. Depth scale | `lib/run_inference.py:143`, `prep/align_depth_to_mhr_wild.py:47` → `depth/moge2/aligned.h5`. Robust Huber log-ratio alignment of monocular depth to decoded initialized MHR human depth. |
| 5. Object | `lib/run_inference.py:146`, `prep/run_foundationpose_mhr_export.py` → `foundationpose/<sequence>.pkl`. First usable registration then tracking; 5 iterations, up to 5 attempts; explicit `--no-first-usable-frame-gt-rotation-oracle`. |
| 6. CoCoNet | `lib/run_inference.py:149`, `tools/run_mhr_wild_inference.py:523` → `inference/coconet.pth`, `materialized_inputs.h5`. Stride 96; offline checkpoint supervision contract; `gt={}` and `ground_truth_used=False`. |
| 7. Contact refinement | `lib/run_inference.py:152`, `learning/training/mhr_opt_refineout.py:1256` → `inference/refined.pth`, `refinement_checkpoint.pth`. Full clip, 300 steps, temporal weight 100, human pose prior 200; fixed initial <0.05 m hand-surface contact gate. Object rotation optimization defaults off. |
| 8. Render | `lib/run_inference.py:156` → final side-by-side and three-stage MP4s; `pipeline_report.json` records elapsed seconds and reused/new stages. `.stages/` markers permit validated reuse. |

All `prep/`, `tools/`, `lib_mhr/` and `learning/` paths in the table are under `reconstruction/modules/v2d_cari4d/lib/cari4d/`. Full exact subprocess argument lists are in the pinned `lib/run_inference.py:129`; our runner delegates to that code rather than maintaining another implementation of those eight stages.

### Images and weights

Build commands, from a host with the wrappers installed:

```bash
python -m v2d.sam2.docker.build
python -m v2d.cari4d.docker.build
python -m v2d.sam2.docker.run_download_weights --output_dir data/tier3/weights/sam2
python -m v2d.cari4d.docker.run_download_weights --output_dir data/tier3/weights/cari4d
```

Images are locally built `v2d_sam2:latest` and `v2d_cari4d:latest`, both based on `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`. No published ready-made image digest was established. CARI4D includes SAM-3D-Body and the SOMA exporter; a third Body image is unnecessary. Dockerfiles compile CUDA architectures 8.0/8.6/8.9/9.0, install PyTorch3D v0.7.9, Kaolin 0.18.0, py-soma-x 0.2.1, and constrain transformers to 5.3.0. Some build dependencies remain upstream-floating despite the source pin: save image IDs and treat a successful GPU build as an outstanding gate.

Exact weight identities (`v2d_cari4d/lib/download_weights.py:19`):

| Artifact | Source / revision |
|---|---|
| CARI4D | `nvidia/cari4d_commercial` @ `1f7287ac6fd5f72c30ce2222fb345a3e7d779fc9`, `2026-08-25-09-35-57/step200000.pth`, `resolved_config.yaml`, `manifest.json` |
| Checkpoint SHA256 | `78ff5cb874dd012a272382e3f2d8bc11226d5b7d0ecc739a60fbb4a97a5a5ba3` |
| SAM-3D-Body | `facebook/sam-3d-body-dinov3` @ `11aaa346c7204874a1cbafe3d39a979080b2c55a`; `model.ckpt`, `assets/mhr_model.pt`; manually gated approval required, currently pending per task brief |
| MoGe 2 | `Ruicheng/moge-2-vitl-normal` @ `b135031bae30b5ac2ae141a0e68717795ce38340`, `model.pt` |
| DINOv3 source | `facebookresearch/dinov3` @ `6876159a11b4df116f30f667f8c9888617df0751` |
| DINOv2 source | `facebookresearch/dinov2` @ `7764ea0f912e53c92e82eb78a2a1631e92725fc8`; pretrained `vitb14`/`vits14` downloaded and hash-checked |
| FoundationPose | NVlabs PyTorch scorer `2024-01-11-20-02-45`, refiner `2023-10-28-18-33-37`; Google Drive folder IDs `12Te_3TELLes5cim1d7F7EBTwUSe7iRBj` and `1BEQLZH69UO5EOfah-K9bfI3JyP9Hf7wC` (`v2d_foundation_pose/lib/download_weights.py:36`) |
| SAM2 | `facebook/sam2.1-hiera-large`, `sam2.1_hiera_large.pt`; upstream downloader follows main |
| SOMA export assets | `nvidia/SOMA-X` @ `466879a83d57eabf3d875ded2d869f2075f90348`, matching `robotic_grounding/scripts/setup_soma_assets.py:57`; legacy embedded rig required |

HF token must read both CARI4D and gated Body artifacts. No W&B credentials are needed. The runner checks access using HEAD metadata without printing the token. FoundationPose requires working Google Drive downloads too; HF preflight alone does not establish that.

**GPU memory/runtime:** the pinned CARI4D README and runner give batch sizes and instrument elapsed time but state no end-to-end VRAM minimum or minutes/sequence. Do not quote the separate public repository's SAM3 chunking/24 GB statement as a requirement for this MHR pipeline. Our 40+ GB and 150 GiB free-disk preflight thresholds are conservative **planning choices**, not measured minimums. Full-clip refinement and 1536×1152 masks/depth may still exceed resources.

**Inferred disk bound:** at the released 1536×1152 resolution, all 22,990 frames occupy 75.77 GiB for uint16 raw depth alone (`prep/mhr_wild_depth.py:72`). The two uncompressed full-resolution bool mask datasets in `tools/run_mhr_wild_inference.py:483` add another 75.77 GiB before crop caches, aligned depth, bundles, export RGB, images, and weights. Thus 150 GiB free is an initial-pilot gate, not permission to retain every intermediate for all 30. The 512 GiB Lambda root disk needs a measured retention plan or additional storage. Retain compact sequence outputs, provenance, and visual evidence; only remove owned regenerable intermediates after validation. The runner never deletes them automatically.

### Intrinsics, world frame, metric scale

Released `meta/info.json` explicitly has `camera_calibration: null` and states calibration was not in the source export. No calibration is needed as an input to this CARI4D wrapper: MoGe 2 estimates intrinsics; export creates one camera with identity extrinsics (`prepare_mhr_wild_export.py:191`). Therefore the raw reconstruction world is **OpenCV camera 0**, not the released challenge world.

MHR decoding yields metric geometry (`lib_mhr/mhr_layer.py:420`); the human prior establishes an estimated metric scale. Raw MoGe depth is subsequently scaled to that estimated human. CoCoNet centers XYZ on MHR joint 1 and normalizes initialized neutral height to 2 m, then returns translation updates to metric units. The supplied object's physical scale remains an independent required input: CARI4D does not automatically calibrate a generated mesh. These learned priors do not eliminate monocular uncertainty or guarantee correct scale.

If meshes must be reconstructed, upstream documents an **optional separate** SAM-3D-Objects path (`v2d_cari4d/README.md:20`): build `v2d_sam3d`, approve `facebook/sam-3d-objects`, extract a well-visible RGB frame and matching object mask, and run `python -m v2d.sam3d.docker.run_image_to_mesh` with `--image_path`, `--mask_path`, `--mesh_path`, `--transform_path`, `--intrinsics_path`, `--weights_dir`. Its GLB and estimated layout JSON are separate; CARI4D ignores that JSON. Generated geometry needs physical autoscaling before use. A separate calibrated stereo scan route is also documented, but the challenge monocular video is not such a scan. The current runner deliberately requires an already metric mesh rather than claiming that optional path is complete.

### Reconstruction → retargeter

`scripts/tier3_export.py` runs inside the CARI4D image. It validates full contiguous frame IDs, explicit no-GT metadata, and the object mesh frame. It uses the **median of reconstructed shape/scale only** to supply the one identity `read_soma` expects, decodes through the pinned MHR head, checks native/CV vertices against the CARI4D decoder, then calls upstream `v2d.sam3d_body.lib.export_soma.export_soma` (PoseInversion, defaults body/full/lie = 2/1/3). It injects the pinned SOMA asset directory into that factory; following current HF main would select an incompatible slim rig. This bridge is source-reviewed and compile-checked, **not GPU-executed**. Per-sequence identity stabilization changes the decoded reconstruction slightly and is recorded in `tier3_export.json`.

Raw output is `data/tier3/raw/<sequence>/`: `soma_params.npz` (native SOMA Y-up), `poses.npy` (OpenCV camera frame), reconstructed local GLB, and `filled_frames.json`. CARI4D estimates all frames, so the bridge records no explicit interpolation fills. It does not claim every occluded frame is accurate.

`scripts/tier3_align.py` uses only reference frame 0 to estimate **one SE(3)** from 16 body joints. It uses its Sim(3) fit only as a scale diagnostic. This is deliberate: upstream requires metric reconstruction, and changing root translations alone cannot uniformly scale a fixed SOMA body. This exporter does **not** repair global scale errors; a Sim(3)-correcting body-identity/refit path would be additional work if the pilot proves necessary.

Object mesh origin/orientation also differs because CARI4D recenters it. After the global world transform, a single constant **object-local coordinate change** is calibrated from frame-0 object pose and applied to every object pose; its inverse goes into the GLB. This is a mesh gauge conversion, not another time-dependent world fit. It preserves the same posed geometry and expresses poses in the released object's frame. Object symmetry and erroneous frame-0 poses remain real ambiguity/accuracy risks. World alignment cannot repair reconstruction drift.

The alignment uses `lerobot_to_soma.write_sequence` and `object_poses_matrix`. It writes exactly the existing consumer schema: `[T,77,3]` local rotvec poses, transl, 77 names, MHR identity, `[1,45]` identity, `[1,68]` scales, meters, keep_root=False, bone flexibles; object `[T,4,4]`, GLB, ground plane, fill list, plus staged support USDA. Reconstructed identity/local articulation are retained; clean identity is used only to decode the alignment frame. Plane must already be in released OpenCV world; USDA must already be in G1 retarget world. Neither is automatically loaded from Tier 1 or transformed twice.

`load_tier1_frame0` calls `ParquetFile.iter_batches(batch_size=1, columns=FRAME0_COLUMNS)` and consumes exactly its first batch, with an explicit frame_index=0 check. It never calls `load_episode` or returns a parquet iterator. Parquet may internally decompress a whole column chunk; application code sees exactly one row. A test rejects any request for a second batch/full-table read. All 30 actual episodes passed this loader; only frame-zero caption columns were exposed.

## Commands

CPU checks (no pytest installation required for our tests):

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_tier3_align.py' -v
bash -n scripts/tier3_reconstruct.sh
.venv/bin/python -m py_compile scripts/tier3_*.py
.venv/bin/python scripts/tier3_inspect.py                 # sizes only
# Already done; do not broaden beyond the two inspected episodes:
.venv/bin/python scripts/tier3_inspect.py --download --episodes 0 1
```

GPU VM, after copying this checkout and installing `huggingface_hub` into an isolated Python environment:

```bash
export TIER3_PYTHON=/absolute/path/to/gpu-host-venv/bin/python
# Supply HF_TOKEN privately in the VM environment; do not paste it into logs.
bash scripts/tier3_reconstruct.sh preflight
bash scripts/tier3_reconstruct.sh setup

# Prompt only from the video. If using the annotation server, invoke explicitly:
"$TIER3_PYTHON" -m v2d.sam2.docker.run_annotate \
  --video_path /absolute/video/sequence.0.color.mp4 \
  --prompts_path /absolute/video/sequence_prompts.json --port 8080
# Stop annotation before starting reconstruction. IDs: human=0, object=1.
bash scripts/tier3_reconstruct.sh run \
  /absolute/video/sequence.0.color.mp4 /absolute/video/sequence_prompts.json \
  /absolute/allowed_or_reconstructed_metric_mesh.glb 778

.venv/bin/python scripts/tier3_align.py \
  --source data/tier3/raw/sequence \
  --tier1-root data/challenge/track_2/tier_1_multiview_caption --episode 0 \
  --out data/tier3/sequences --motion-root data/tier3/motion \
  --ground-plane /absolute/permitted_released_world_plane.json \
  --support-surface /absolute/permitted_g1_world_support.usda

# The existing sequence mode is available; use one heavy process at a time:
bash scripts/retarget_tier.sh --sequences data/tier3/sequences data/tier3 1
```

Use the actual sequence name and episode frame count, not the literal placeholder `sequence`/778. Copy or hardlink inspected MP4s into the `<sequence>.0.color.mp4` naming convention under `data/tier3/`; avoid reencoding or truncating. The inspection directory has a cumulative two-episode download cap. The run wrapper processes one episode at a time, keeps upstream defaults, records logs/image IDs, and refuses to overwrite a raw export. After an interrupted export, inspect the partial folder before removing only regenerable files; upstream reconstruction stage markers can resume validated stage outputs.

## Measured checks and limits

Baseline: project `.venv` has no pytest. Its attempt failed with `No module named pytest`. Existing system `pytest` ran the upstream host-safe CARI4D contract suite: **12 passed in 0.43s**, and the final unchanged-source rerun reported **12 passed in 0.09s**. This is source-contract evidence only. Full upstream GPU/Docker tests were unavailable.

Our CPU suite: **4 tests passed**. It exercises actual output folders and representative saved human frames through the real retargeter `SOMA.load_motion`, verifies the entire object pose array and local mesh after a known rigid world change plus independent mesh-frame change, rejects degenerate/nonfinite fits, and tests the frame-zero reader. The numerical fixture is the non-challenge upstream `sequence_id=2026-02-20_17-37-45_blue_trash_can_drag_007`, all 742 object frames. No later challenge Tier 1 frame or Tier 2 trajectory enters the tests.

`data/tier3/alignment_test_results.json` contains rerunnable results:

| Synthetic check | Measured result |
|---|---|
| Known random Sim(3), scale 1.37, no noise | Max translation error 1.33e-15 m; full pose matrices asserted within 1e-12. This tests the math helper, **not scaled SOMA folder export**. |
| 1 cm independent Gaussian noise in source frame-zero body points, seed 20261021 | Fitted scale 1.38585, rotation error 1.80687°, whole object-path translation RMSE 0.02141 m, maximum 0.03655 m. No later frames were used for fitting. |
| Rigid folder roundtrip | Entire object-path max translation error 2.98e-7 m; decoded representative joint max coordinate error 4.92e-7 m. Float32 model/export roundoff limits these checks. |
| Frame-zero orientation bias: 0.5° / 1° / 3° / 5° | Whole-path translation RMSE 0.00474 / 0.00949 / 0.02846 / 0.04742 m about the frame-zero body centroid. |

Lever-arm error is `2*d*sin(theta/2)` for perpendicular distance d from the rotation axis: a 1° error gives 1.745 cm at 1 m, 5.236 cm at 3 m, 8.727 cm at 5 m. This is geometric sensitivity, **not a measured Tier 3 reconstruction accuracy**. Translation bias adds a constant offset; scale bias grows with distance too. These tests establish coordinate handling under a frozen synthetic perturbation, not monocular estimation quality or simulator score.

Shell syntax and Python compilation passed. On the M1, GPU preflight intentionally exits with `ERROR: requires x86_64 Linux GPU VM`. No Docker or CUDA execution was attempted. GPU bridge correctness beyond its source/runtime guards remains unverified.

### Video evidence

Measured HF dataset revision: `5f68335f3acc802033d1e80728c1633197521de8`. The inventory includes Track 1 plus both Track 2 tiers; **each Track 2 tier has 30 exo MP4s totaling 686,124,469 bytes**. All 30 corresponding Tier 1/Tier 2 LFS SHA256 hashes and sizes match. No Tier 2 trajectories were read.

Only episodes 0/1 were downloaded into `data/tier3/videos/` (45,869,034 bytes). `ffprobe -count_frames -show_frames` decoded their timelines:

| Episode | Size | Resolution / rate | Decoded frames = parquet rows | First/last PTS |
|---|---|---|---|---|
| 0: stool | 24,069,401 B | 1536×1152, 30/1 | 778 | 0 / 25.900000 s |
| 1: vacuum | 21,799,633 B | 1536×1152, 30/1 | 747 | 0 / 24.866667 s |

PTS increments are 0.033333–0.033334 s. Max video/parquet timestamp difference is 9.6912e-7 s, explained by serialized float precision and decimal PTS rounding; frame indices are contiguous. Only `frame_index` and `timestamp` columns were read for this comparison. Downloaded file SHA256s were verified against HF LFS. Raw probe summaries are retained in `video_inspection.json`, full remote sizes/hashes in `video_inventory.json`. Metadata-only totals for all 30 episodes: 22,990 frames, 440–1,003 frames each, 766.33 s of video at 30 fps.

## Provider and cost, sources checked 2026-10-21

**Recommend 1× Lambda RTX A6000 48 GB, $1.09/hour** before tax, 14 vCPU / 100 GiB RAM / 512 GiB SSD. It is the cheapest fixed-price compatible configuration established in this check, not a proof of the cheapest rental across the entire market or available stock. The [official price table](https://lambda.ai/instances) lists that 1-GPU configuration; [Lambda VM/image documentation](https://docs.lambda.ai/public-cloud/on-demand/) establishes a real Linux VM, Docker, NVIDIA container toolkit, and driver availability. Select Lambda Stack; run preflight before builds. The cheaper $0.69/h Quadro RTX 6000 is Turing and falls outside the pinned native extension targets; A10 24 GB is $1.29/h and provides less memory; A100 PCIe 40 GB is $1.99/h. No account/capacity check was made.

**Correction to the task premise:** [Vast full VMs](https://docs.vast.ai/guides/instances/virtual-machines) support nested Docker; its Ubuntu 22.04 VM template includes CUDA and Docker and filters `vms_enabled=true`. Ordinary container offers are a different product. A read-only [official search API](https://docs.vast.ai/api-reference/search/search-offers) query for verified, rentable, 1-GPU, 40+ GB, Ampere/Hopper VM-capable machines with 200+ GB disk returned `offers=[]`. Exact request/response are retained in `data/tier3/evidence/vast_query.json` and `vast_offers.json`. This filtered public response is not a general unavailability claim. The [official pricing page](https://vast.ai/pricing) provides market pricing but no inspectable VM-specific numerical quote in this capture; therefore no cheaper Vast VM price is verified. Do not substitute an advertised container price for a compatible VM quote. A cheaper qualifying VM can replace Lambda after the same preflight and a full pilot.

**Guess, pending pilot:** allow 0.5–2 GPU-hours per episode for masks + CARI4D + SOMA export, i.e. 15–60 GPU-hours for 30; add 2–6 billed hours for image builds/download/setup. At $1.09/h this is **$18.53–$71.94**, or about **$23–$90 with 25% contingency**. An input mesh generation/scale workflow, manual prompting, storage expansion, and RL/evaluation are excluded. These are budget scenarios, not upstream timings or measured throughput. At 4 h/episode the same formula yields 122–126 billed hours, $133–$137 before contingency. No finite upper runtime bound is established.

After one complete uncached pilot, record each stage's elapsed seconds, peak GPU memory, decoded/output frame counts, and per-episode disk growth. Exclude weight downloads/builds from inference timings but include their paid time in cost. Reject reused-stage timings as throughput evidence. Inspect overlay videos and load the generated folders; a `pipeline_report` PASS alone does not establish accurate reconstruction. Scale the remaining estimate by the 22,990-frame workload and include prompt repairs, failed attempts, and the longest episode. GPU availability, full-clip optimization memory, mask occlusion, mesh scale, and compile compatibility are the main uncertainties.

## Critical path and kill criteria

1. **Now–Oct 23:** obtain actual gated Body approval; resolve Track 2 Tier 3 scene-asset permission. If meshes cannot be reused, establish a legal metric mesh route and any separate SAM-3D-Objects approval. This is upstream input preparation, not a small flag change.
2. **Oct 23–25:** run preflight/build/download; reconstruct one complete episode with unchanged defaults; execute the MHR-to-SOMA bridge; inspect human/object overlays and metric scale; align using only frame 0; retarget; train and evaluate one policy end to end. Preserve failure evidence. Confirm local mesh basis and retarget/reference world inversion with the other worker's frame-reference writer.
3. **Oct 25–28:** process all 30 in a sequential queue, prepare masks/scene assets, fix input-only failures, and start episode policies as sequences pass validation. GPU-hour numbers must be replaced by pilot measurements. Training and reconstruction should use separate resources/queues when both are needed, rather than competing on one GPU.
4. **By roughly Oct 29:** all gates above should be resolved; all 30 reconstructed/aligned sequences and G1 references should be ready or on a measured queue that finishes immediately. At least one learned G1 policy must have completed a valid simulator evaluation with correct world-frame object outputs. All 30 per-episode policies need a credible completion schedule for Nov 1–2, leaving Nov 3–4 for evaluation/retries and submission checks.

**Kill/stop-expanding criteria:** no gated access or allowed metric-mesh path by Oct 25; no complete reconstruction→SOMA→G1→learned-policy rollout by Oct 27–29; unreconciled object frame/scale; remaining measured GPU work cannot finish before the evaluation buffer; or any episode lacks a valid learned-policy output. Do not replace missing episodes with clean trajectories, simulated references, default poses, or policy-free rollouts. Missing one episode is not an all-30 valid Tier 3 result.

**Judgment:** upstream defaults are a plausible minimum-effort starting point if allowed meshes and the full GPU bridge work quickly. Their existence is not evidence that they can complete this challenge's 30 varied objects, occlusions, and whole-body tasks. Access and asset legality can currently kill this path before compute cost does. A fixed-scale rigid export also needs a credible metric reconstruction; the tested Sim(3) helper alone does not provide uniform body-scale correction.

## Files and proposed README text

Created only `scripts/tier3_align.py`, `scripts/tier3_export.py`, `scripts/tier3_inspect.py`, `scripts/tier3_reconstruct.sh`, this report, and `tests/test_tier3_align.py`. Data/evidence are under `data/tier3/`. No existing README or script was edited by this task.

Proposed README addition (not applied because existing-file edits are forbidden):

```markdown
## Tier 3: monocular reconstruction

`scripts/tier3_reconstruct.sh` prepares and runs the pinned SAM2 → MHR CARI4D
pipeline sequentially on an x86_64 Docker/NVIDIA GPU VM. It requires gated
SAM-3D-Body access, video-only masks/prompts, and an allowed metric object mesh.
`scripts/tier3_export.py` bridges its refined MHR bundle to SOMA inputs.
`scripts/tier3_align.py` reads only Tier 1 frame zero for rigid world alignment
and object-local frame calibration, then writes the existing retargeter format.
Use `retarget_tier.sh --sequences` on those outputs; preserve one learned
FlashCHORD policy and valid evaluated object trajectory per episode.

CPU verification: `.venv/bin/python -m unittest discover -s tests -p 'test_tier3_align.py' -v`.
The GPU bridge is not yet runtime-qualified. Released scene-asset permission
needs confirmation; the scripts do not infer permission from release availability.
See `docs/tier3.md` for pins, commands, tests, costs, and kill criteria.
```

Disk: data files total 46,244,206 bytes; with the new source/report/test files, approximately **44.2 MiB** was added. Downloaded videos account for 45,869,034 bytes. Test temporary sequence folders and this task's generated bytecode were removed. No build directories, Docker images, installed environments, or background services were created.


## Stage-05 engineering bridge — 2026-10-22

[M] `scripts/tier3_initializers.py` accepts trusted upstream MHR and FoundationPose pickles plus `wild_export.json`; it rejects incomplete frame timelines, GT-capable object metadata and human GT/depth alignment. It projects only small, positive-determinant rotation drift (orthogonality residual at most 1e-4) to SO(3), recording the exact change. `--released-mesh-engineering-only SOURCE.glb --mesh-out NEW.glb` reconstructs the export-aligned mesh using the retained source-to-aligned matrix. This switch does not establish permission to use released meshes.

[M] `tier3_export.py --device cpu` runs the upstream MHR decoder and SOMA PoseInversion in bounded batches. It uses the installed full-LOD rig because the pinned local bundle lacks `MHR/mhr_model_lod6.pt`; iterations remain 2 body / 1 full / 0 finger / 3 Lie. It avoids the upstream export wrapper's unconditional CUDA synchronization. `--checkpoint`, `--mhr-model` and `--python-deps` allow explicit local assets/dependencies without image-model imports. Per-sequence median inferred identity/scale is retained; no reference identity is substituted.

[M] Alignment uses the existing guarded loader only. `--released-scene-assets-engineering-only` records the unresolved released fitted plane/support provenance in `tier3_alignment.json`. Ep18's support USDA is empty, so the unchanged training queue resolves it and sets `scene.support=false`; the fitted plane still levels the world to robot z=0. The retargeter applies its existing capped constant frame-0 object lift. [I] No scene-independent ground/support reconstruction is implemented by this unit.

[M] `tier3_materialize.py --motion-root WORK/motion --out WORK/assets` copies the generated asset closure and rewrites only mesh/URDF locators to `/assets/...`. FlashCHORD re-roots these against the parquet's own `assets` ancestor after transfer. Copy the whole closure, rather than only `data.parquet`. A relocation test verifies real mesh/URDF/support resolution while preserving motion arrays.

[M] `tier3_quality.py` reads only the generated motion and permitted frame 0. It checks video/SOMA/object/G1 coverage, unsmoothed finite-difference continuity, complete-mesh ground distances, human fingertip and G1 palm distances on retargeter contact labels, support-stage resolution, and projected RGB overlays at first/middle/last frames. [I] Geometrically generated contact labels and calibrated frame-zero agreement are not independent accuracy labels.

[M] `tier3_prompts.py` opens only MP4s and a video-derived annotation JSON. The generated 30 prompt pairs use human ID 0 and object ID 1. Boxes or points are mutually exclusive under the upstream SAM2 schema. Ep28 uses foreground broom points and a background chair point. [I] SAM2 propagation and mask quality for these prompts need a later authorized CUDA run; they are not masks.

[M] Native scripts live in `scripts/tier3_native/{setup,env,run,batch}.sh`, with the retained SAM2 patch. They take project/work/weights paths explicitly, accept `--dry-run`, and run only stages 01–05 plus initializer validation. Batch TSV columns are sequence, frame count, video, prompt JSON and metric mesh. The runner names input-video symlinks as `<sequence>.0.color.mp4` and validates frame count. Stage records retain failed logs, resume owned partial outputs, skip unchanged successes, and reject changed completed inputs/outputs. [I] Setup/build changes were syntax/dry-run checked on Mac only; a fresh-box replay is outstanding.

Mac engineering replay into a new destination (explicit released-asset acknowledgement):

```bash
bash data/tier3_bridge/reproduce.sh --released-assets-engineering-only data/tier3_bridge_replay
```

[M] Unit-1 full-suite baseline: 31 tests. Final suite: 43 tests; exact retained output is `data/tier3_bridge/evidence/final_patch_full_tests.log`. [M] No GPU host, upload, rental, clean trajectory scoring, packer or submission was used in that unit. Its report retains the future native-box commands; the 50M training-command correction and queue gate are documented below.

## Ep18 label-free repair and pilot gate — unit 2

[M] See `data/tier3_bridge/REPORT_unit2.md`. `tier3_repair.py` imports the existing denoiser's fixed 15-frame cubic/continuous-quaternion kernels, smooths object and SOMA local motion, then applies the minimum full-mesh plane-normal lift equally to human and object before G1 IK. The fixed policy is `data/tier3_bridge/repair_policy.json`. The complete portable repaired closure lives at `data/tier3_bridge/repaired/assets/`. Replay in a new directory with `data/tier3_bridge/reproduce_unit2.sh --released-scene-assets-engineering-only NEW`.

[M] Both replay scripts take mesh/plane/support assets from `tier_2_synthetic_noise`; the ep18 asset-only byte comparison is retained in `evidence/unit2_scene_asset_identity.json`. The only Tier 1 reference access in the replay is the guarded frame-zero loader. [I] Released asset eligibility remains unresolved.

[M] `tier3_contact_baseline.py` uses the same gap function as `tier3_quality.py` on the three external screen `clean` references. These are external data, not challenge Tier 1/2 trajectories. [I] This is retargeter context, not a contact-accuracy threshold.

[M] `tier3_manifest.py` regenerates absolute paths, hashes and self-scoring records for `tier3_raw` and `tier3_repaired` on each host. The current protected `pilot_queue.py` rejects non-`clean` self-scoring; the manifest is **not queue-compatible yet**. `--check-queue` fails closed; the saved `unit2_run/pre.sh` exports `MANIFEST` only after that check succeeds. No ready-to-train claim is made. Use the frozen `T2C-contact` arm with its 50M curriculum when the queue gains explicit self-scoring support; the old total-steps-only command in the unit-1 report has been removed.

[M] `.venv-flash-cpu/bin/python -I scripts/tier3_verify_unit2.py` checks both actual closures with the source-frame loader, evaluator parquet writer and official metric adapter, and retains the exact dry-run rejection. [I] Its identity/offset achieved poses are synthetic recorder controls, not simulator or policy evidence.
