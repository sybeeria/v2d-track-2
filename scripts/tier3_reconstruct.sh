#!/usr/bin/env bash
# GPU VM only. No accounts, rentals, training, evaluations, or submission packing.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
UPSTREAM="$ROOT/external/video_to_data"
WORK="$ROOT/data/tier3"
PYTHON=${TIER3_PYTHON:-python3}
PIN=7eb9721088615fe9201634a6be5cf48b049d1090

fail() { echo "ERROR: $*" >&2; exit 1; }
usage() {
    cat <<'EOF'
Usage:
  scripts/tier3_reconstruct.sh preflight
  scripts/tier3_reconstruct.sh setup
  scripts/tier3_reconstruct.sh run VIDEO PROMPTS_JSON METRIC_MESH EXPECTED_FRAMES

Run on an x86_64 Linux GPU VM with Docker, NVIDIA runtime, Python >=3.10,
and huggingface_hub installed in TIER3_PYTHON. Set HF_TOKEN privately.
setup builds v2d_sam2:latest and v2d_cari4d:latest sequentially, then fetches weights.
run expects a <sequence>.0.color.mp4, video-only SAM2 prompts (human ID 0,
object ID 1), and a metric mesh. It creates data/tier3/raw/<sequence>/.
Do not use released clean assets unless their Tier 3 use is independently allowed.
No clean trajectory is read here. Run tier3_align.py separately for alignment.
EOF
}

preflight() {
    [[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || fail "requires x86_64 Linux GPU VM"
    for exe in nvidia-smi docker nvidia-container-cli ffprobe "$PYTHON"; do
        command -v "$exe" >/dev/null || fail "missing $exe"
    done
    [[ $(git -C "$UPSTREAM" rev-parse HEAD) == "$PIN" ]] || fail "upstream must be $PIN"
    nvidia-smi
    nvidia-smi --query-gpu=driver_version --format=csv,noheader | "$PYTHON" -c 'import sys; assert all(int(v.split(".")[0]) >= 550 for v in sys.stdin if v.strip()), "use driver 550+ for the CUDA 12.4 images"'
    nvidia-container-cli --version
    docker info >/dev/null || fail "Docker daemon not accessible; a container-only rental is insufficient"
    docker info --format '{{json .Runtimes}}' | "$PYTHON" -c 'import json,sys; assert "nvidia" in json.load(sys.stdin), "configure NVIDIA Docker runtime"'
    nvidia-smi --query-gpu=memory.total,compute_cap --format=csv,noheader,nounits | "$PYTHON" -c '
import sys
rows=[r.strip().split(",") for r in sys.stdin if r.strip()]
assert len(rows)==1, "use one visible GPU per sequential worker"
memory,cap=map(float,rows[0])
assert memory >= 40000, "40+ GB is a conservative pilot requirement, not an upstream measured minimum"
assert 8 <= cap < 10, "pinned extensions target Ampere/Hopper, not Turing/Blackwell"
'
    mkdir -p "$WORK"
    df -h "$WORK"
    docker_root=$(docker info --format '{{.DockerRootDir}}')
    "$PYTHON" - "$WORK" "$docker_root" <<'PY'
import shutil,sys
for path in sys.argv[1:]:
    free=shutil.disk_usage(path).free
    print(f"{path}: {free/1024**3:.1f} GiB free")
    assert free >= 150*1024**3, "reserve 150 GiB for initial images/weights/pilot; remeasure before all 30"
PY
    "$PYTHON" - <<'PY'
import os,sys
from huggingface_hub import get_hf_file_metadata,hf_hub_url
assert sys.version_info >= (3,10)
assert os.environ.get("HF_TOKEN"), "set HF_TOKEN privately; login without gated approval is insufficient"
checks=[("facebook/sam-3d-body-dinov3","11aaa346c7204874a1cbafe3d39a979080b2c55a","model.ckpt"),
        ("facebook/sam-3d-body-dinov3","11aaa346c7204874a1cbafe3d39a979080b2c55a","assets/mhr_model.pt"),
        ("nvidia/cari4d_commercial","1f7287ac6fd5f72c30ce2222fb345a3e7d779fc9","2026-08-25-09-35-57/step200000.pth"),
        ("Ruicheng/moge-2-vitl-normal","b135031bae30b5ac2ae141a0e68717795ce38340","model.pt"),
        ("facebook/sam2.1-hiera-large","main","sam2.1_hiera_large.pt"),
        ("nvidia/SOMA-X","466879a83d57eabf3d875ded2d869f2075f90348","SOMA_neutral.npz")]
for repo,revision,filename in checks:
    try:
        meta=get_hf_file_metadata(hf_hub_url(repo,filename,revision=revision), token=os.environ["HF_TOKEN"])
    except Exception as exc:
        raise SystemExit(f"HF access failed for {repo}/{filename}: {type(exc).__name__}; check approval and read scope") from None
    print(f"HF access OK: {repo}/{filename}, bytes={meta.size}")
PY
}

case ${1:---help} in
    --help|-h) usage; exit 0 ;;
    preflight) preflight; exit 0 ;;
    setup|run) ACTION=$1; shift ;;
    *) usage; exit 2 ;;
esac
preflight
if [[ $ACTION == setup ]]; then
    "$PYTHON" -m pip install -e "$UPSTREAM/reconstruction/modules/v2d_common" \
        -e "$UPSTREAM/reconstruction/modules/v2d_docker" \
        -e "$UPSTREAM/reconstruction/modules/v2d_sam2/docker" \
        -e "$UPSTREAM/reconstruction/modules/v2d_cari4d/docker"
    "$PYTHON" -m v2d.sam2.docker.build
    "$PYTHON" -m v2d.cari4d.docker.build
    docker run --rm --runtime=nvidia --gpus all v2d_cari4d:latest \
        python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())'
    "$PYTHON" -m v2d.sam2.docker.run_download_weights --output_dir "$WORK/weights/sam2"
    "$PYTHON" -m v2d.cari4d.docker.run_download_weights --output_dir "$WORK/weights/cari4d"
    "$PYTHON" - "$WORK/weights/soma" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download("nvidia/SOMA-X",revision="466879a83d57eabf3d875ded2d869f2075f90348",local_dir=sys.argv[1],
                  allow_patterns=["SOMA_neutral.npz","correctives_model.pt","MHR/mhr_model_lod1.pt","MHR/base_body_lod1.obj","MHR/SOMA_wrap_lod1.obj"])
PY
    docker image inspect v2d_sam2:latest v2d_cari4d:latest --format '{{.RepoTags}} {{.Id}}' > "$WORK/docker_images.txt"
    exit 0
fi

[[ $# == 4 ]] || { usage; exit 2; }
VIDEO=$1 PROMPTS=$2 MESH=$3 FRAMES=$4
[[ $VIDEO == *.0.color.mp4 ]] || fail "video must end in .0.color.mp4"
[[ $FRAMES =~ ^[1-9][0-9]*$ ]] || fail "EXPECTED_FRAMES must be positive"
for input in "$VIDEO" "$PROMPTS" "$MESH"; do [[ -f $input ]] || fail "missing $input"; done
SEQUENCE=$(basename "$VIDEO" .0.color.mp4)
mkdir -p "$WORK/masks/$SEQUENCE" "$WORK/reconstruction" "$WORK/raw" "$WORK/logs"
[[ ! -e $WORK/raw/$SEQUENCE ]] || fail "raw output exists; inspect it before a new export"
"$PYTHON" - "$PROMPTS" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))["prompts"]
assert {v["object_id"] for v in p} == {0,1}, "human=0 and object=1 required"
PY
"$PYTHON" -m v2d.sam2.docker.run_video_to_masks --video_path "$VIDEO" --prompts_path "$PROMPTS" \
    --masks_dir "$WORK/masks/$SEQUENCE" --weights_dir "$WORK/weights/sam2" 2>&1 | tee "$WORK/logs/${SEQUENCE}_sam2.log"
"$PYTHON" -m v2d.cari4d.docker.run_pack_masks --video_path "$VIDEO" \
    --human_masks_path "$WORK/masks/$SEQUENCE/0" --object_masks_path "$WORK/masks/$SEQUENCE/1" \
    --output_path "$WORK/masks/${SEQUENCE}_masks_k0.h5"
"$PYTHON" -m v2d.cari4d.docker.run_inference --video_path "$VIDEO" \
    --mask_h5_path "$WORK/masks/${SEQUENCE}_masks_k0.h5" --object_mesh_path "$MESH" \
    --weights_path "$WORK/weights/cari4d" --output_dir "$WORK/reconstruction" \
    --expected_frames "$FRAMES" --skip_weight_download 2>&1 | tee "$WORK/logs/${SEQUENCE}_cari4d.log"
docker run --rm --runtime=nvidia --gpus all --user "$(id -u):$(id -g)" \
    -e HF_TOKEN -e HF_HOME=/tier3/weights/cari4d/hf_home -e TORCH_HOME=/tier3/weights/cari4d/sam3d_body/torch_home \
    -v "$ROOT/scripts:/tier3_scripts:ro" -v "$WORK:/tier3" v2d_cari4d:latest \
    python /tier3_scripts/tier3_export.py \
    --bundle "/tier3/reconstruction/$SEQUENCE/inference/refined.pth" \
    --mesh "/tier3/reconstruction/$SEQUENCE/export/$SEQUENCE/object_mesh/output_aligned.glb" \
    --weights /tier3/weights/cari4d/sam3d_body --soma-assets /tier3/weights/soma \
    --out "/tier3/raw/$SEQUENCE" 2>&1 | tee "$WORK/logs/${SEQUENCE}_export.log"
