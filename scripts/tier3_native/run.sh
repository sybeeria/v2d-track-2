#!/usr/bin/env bash
# Native stages 01--05 only. All paths are explicit; no host provisioning.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
dry=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --work-root) NATIVE_ROOT=$2; shift 2;;
    --project) NATIVE_PROJECT=$2; shift 2;;
    --weights) NATIVE_WEIGHTS=$2; shift 2;;
    --video) video=$2; shift 2;;
    --prompts) prompts=$2; shift 2;;
    --mesh) mesh=$2; shift 2;;
    --sequence) sequence=$2; shift 2;;
    --frames) frames=$2; shift 2;;
    --released-mesh-engineering-only) released=true; shift;;
    --dry-run) dry=true; shift;;
    *) echo "Unknown argument: $1" >&2; exit 2;;
  esac
done
: "${video:?--video required}" "${prompts:?--prompts required}" "${mesh:?--mesh required}" "${sequence:?--sequence required}" "${frames:?--frames required}"
[[ $sequence =~ ^[A-Za-z0-9_-]+$ && $frames =~ ^[1-9][0-9]*$ ]] || exit 2
source "$HERE/env.sh"
# Resolve before changing directory. Dry runs need no files or CUDA environment.
NATIVE_ROOT=$(cd "$(dirname "$NATIVE_ROOT")" && pwd)/$(basename "$NATIVE_ROOT")
root=$NATIVE_ROOT/$sequence
C=$NATIVE_MODULES/v2d_cari4d/lib/cari4d
W=$NATIVE_WEIGHTS
O=$root/output/$sequence
E=$O/export/$sequence
entry=$HERE/python_entry.py
step() {
  local name=$1; shift
  local -a cmd=("$NATIVE_PYTHON" -I "$HERE/stage.py" --root "$root" --name "$name" --input "$HERE/run.sh" --input "$HERE/env.sh" --input "$HERE/stage.py")
  while [[ $1 != -- ]]; do cmd+=("$1" "$2"); shift 2; done
  shift
  cmd+=(-- "$@")
  if $dry; then printf '%q ' "${cmd[@]}"; printf '\n'; else "${cmd[@]}"; fi
}
if ! $dry; then
  [[ $video = /* && $prompts = /* && $mesh = /* && $NATIVE_PROJECT = /* && $W = /* ]] || { echo 'Use absolute paths' >&2; exit 2; }
  "$NATIVE_PYTHON" -I -c 'import torch; assert torch.cuda.is_available(), "Native inference requires CUDA"'
  mkdir -p "$root"
  printf '%s\n' "mesh=$mesh released_mesh_engineering_only=${released:-false} eligibility_unresolved=${released:-false}" > "$root/asset_provenance.txt"
fi
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACEHUB_API_TOKEN
step video --input "$video" --input "$HERE/prepare_video.py" --output "$root/input" -- "$NATIVE_PYTHON" -I "$HERE/prepare_video.py" "$video" "$root/input/$sequence.0.color.mp4" "$frames"
video=$root/input/$sequence.0.color.mp4
step sam2 --input "$W/sam2/sam2.1_hiera_large.pt" --input "$NATIVE_MODULES/v2d_sam2/lib/sam2_utils.py" --input "$video" --input "$prompts" --output "$root/masks" -- "$NATIVE_PYTHON" -I -m v2d.sam2.lib.video_to_masks --video_path "$video" --prompts_path "$prompts" --masks_dir "$root/masks" --weights_dir "$W/sam2"
step pack_masks --input "$video" --input "$root/masks" --output "$root/masks.h5" -- "$NATIVE_PYTHON" -I -m v2d.cari4d.lib.pack_masks --video_path "$video" --human_masks_path "$root/masks/0" --object_masks_path "$root/masks/1" --output_path "$root/masks.h5"
step 01_depth --input "$W/hf_home/hub/models--Ruicheng--moge-2-vitl-normal" --input "$video" --input "$C/prep/mhr_wild_depth.py" --output "$O/depth/moge2" -- "$NATIVE_PYTHON" -I "$entry" "$C/prep/mhr_wild_depth.py" --video "$video" --output-depth "$O/depth/moge2/raw.npy" --output-intrinsics "$O/depth/moge2/intrinsics.pkl" --output-report "$O/depth/moge2/depth_report.json" --device cuda --batch-size 8 --expected-frames "$frames" --local-files-only
step 02_export --input "$video" --input "$root/masks.h5" --input "$mesh" --input "$O/depth/moge2/intrinsics.pkl" --output "$O/export" -- "$NATIVE_PYTHON" -I "$entry" "$C/prep/prepare_mhr_wild_export.py" --video "$video" --mask-h5 "$root/masks.h5" --object-mesh "$mesh" --intrinsics-file "$O/depth/moge2/intrinsics.pkl" --output-root "$O/export"
step 03_human --input "$W/sam3d_body/checkpoints/sam-3d-body-dinov3/model.ckpt" --input "$W/sam3d_body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt" --input "$E" --output "$O/mhr_init" -- "$NATIVE_PYTHON" -I "$entry" "$C/prep/run_sam3d_mhr_export.py" "$E" --out-file "$O/mhr_init/$sequence.pkl" --camera-id 0 --depth-root "$E" --sam3d-ckpt "$W/sam3d_body/checkpoints/sam-3d-body-dinov3/model.ckpt" --mhr-path "$W/sam3d_body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt" --chunk-size 16 --sam3d-cache "$O/mhr_init/$sequence.sam3d.h5" --alignment-workers 1 --refit-batch-size 512 --human-prompt-mode sam2_bbox_mask --no-align-to-gt-depth
step 04_align --input "$E" --input "$O/depth/moge2/raw.npy" --input "$O/mhr_init/$sequence.pkl" --output "$O/aligned_depth" -- "$NATIVE_PYTHON" -I "$entry" "$C/prep/align_depth_to_mhr_wild.py" "$E" --raw-depth "$O/depth/moge2/raw.npy" --depth-report "$O/depth/moge2/depth_report.json" --mhr-init "$O/mhr_init/$sequence.pkl" --output "$O/aligned_depth/aligned.h5" --render-batch-size 4 --encoding-workers 16 --write-batch-size 64
step 05_object --input "$W/foundationpose/nvlabs_pytorch" --input "$E" --input "$O/aligned_depth/aligned.h5" --output "$O/foundationpose" -- "$NATIVE_PYTHON" -I "$entry" "$C/prep/run_foundationpose_mhr_export.py" "$E" --depth-root "$O/aligned_depth/aligned.h5" --out-file "$O/foundationpose/$sequence.pkl" --camera-id 0 --iteration 5 --max-attempts 5 --register-first-then-track --no-first-usable-frame-gt-rotation-oracle
step validate --input "$O/mhr_init/$sequence.pkl" --input "$O/foundationpose/$sequence.pkl" --input "$E/wild_export.json" --output "$root/initializers.pth" -- "$NATIVE_PYTHON" -I "$NATIVE_PROJECT/scripts/tier3_initializers.py" --human "$O/mhr_init/$sequence.pkl" --object "$O/foundationpose/$sequence.pkl" --wild-export "$E/wild_export.json" --expected-frames "$frames" --out "$root/initializers.pth"
