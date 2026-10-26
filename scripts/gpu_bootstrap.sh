#!/usr/bin/env bash
# One-shot setup on a rented Linux GPU box (RunPod/Vast container, no Docker needed).
#
# Copy this project to the box first (it is not on GitHub yet), e.g.:
#   rsync -az --exclude .venv --exclude external --exclude data ./ root@HOST:/workspace/v2d-track-2/
# Then on the box:
#   bash /workspace/v2d-track-2/scripts/gpu_bootstrap.sh
#
# Result: external/ checkouts with LFS assets, .venv (retargeting) and
# .venv-flash (FlashCHORD training), Track 2 Tier 1/2 data, and G1 references in
# data/work/tier{1,2}/motion. Safe to re-run; finished steps are skipped.
set -euo pipefail

PROJECT=$(cd "$(dirname "$0")/.." && pwd)
UPSTREAM_COMMIT=7eb9721088615fe9201634a6be5cf48b049d1090
MIN_DRIVER=580
MIN_FREE_MIB=46000
cd "$PROJECT"
df -h "$PROJECT"

echo "== GPU check"
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free --format=csv
driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
[ "${driver%%.*}" -ge "$MIN_DRIVER" ] || { echo "driver $driver < $MIN_DRIVER: pick another host"; exit 1; }
[ "$free_mib" -ge "$MIN_FREE_MIB" ] || { echo "only $free_mib MiB free (< $MIN_FREE_MIB): pick another host"; exit 1; }

echo "== System packages"
# Tools plus the GL/X11 client libraries FlashCHORD's own Dockerfile installs.
if ! command -v git-lfs >/dev/null || ! command -v unzip >/dev/null || ! ldconfig -p | grep -q libEGL.so.1; then
    apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
        git git-lfs rsync curl unzip ca-certificates \
        libglvnd0 libgl1 libglx0 libegl1 libgles2 libopengl0 libglu1-mesa libosmesa6 libglfw3 \
        libx11-6 libxext6 libxrender1 libxi6 libxrandr2 libxcursor1 libxinerama1 libxfixes3 \
        libxxf86vm1 libxkbcommon0 libxcb1 fontconfig libfreetype6 >/dev/null
fi
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }

echo "== Upstream repo @ $UPSTREAM_COMMIT"
if [ ! -d external/video_to_data/.git ]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone -q --filter=blob:none https://github.com/nvidia-isaac/video_to_data.git external/video_to_data
fi
git -C external/video_to_data checkout -q "$UPSTREAM_COMMIT"
git -C external/video_to_data lfs install --local >/dev/null
RG_ASSETS=robotic_grounding/source/robotic_grounding/robotic_grounding/assets
git -C external/video_to_data lfs pull --include="robotic_grounding/flash_chord/**,$RG_ASSETS/meshes/**,$RG_ASSETS/urdfs/**,$RG_ASSETS/human_motion_data/**"
# Opt-in object-orientation reward (weight 0 by default, so the stock recipe is unchanged).
orientation_patch=$PROJECT/patches/flash_chord_object_orientation.patch
if ! git -C external/video_to_data apply --reverse --check "$orientation_patch" 2>/dev/null; then
    git -C external/video_to_data apply "$orientation_patch"
fi

bash scripts/apply_retarget_patch.sh

echo "== Submission kit"
if [ ! -d external/kit/v2d_submission_kit ]; then
    kit_zip=$(mktemp)
    trap 'rm -f "$kit_zip"' EXIT
    curl -sfL https://nvidia-isaac.github.io/video_to_data/v2d_challenge/assets/v2d_submission_kit.zip -o "$kit_zip"
    mkdir -p external/kit && unzip -q -o "$kit_zip" -d external/kit
    rm -f "$kit_zip"
    trap - EXIT
fi
echo "== Retargeting venv (.venv)"
if [ ! -x .venv/bin/python ]; then
    uv venv -q --python 3.11 .venv
fi
# An interpreter can survive an interrupted install; dependency installation must be retried.
# pinocchio's native closure is pinned as in robotic_grounding/workflow/Dockerfile.
uv pip install -q --python .venv/bin/python torch scipy trimesh pyarrow pandas "viser==1.0.15" \
    "py-soma-x==0.2.1" mujoco "judo-rai>=0.0.5" "pin==3.7.0" "pin-pink==4.2.0" "coal==3.0.1" \
    "eigenpy==3.10.3" "cmeel==0.57.3" "cmeel-boost==1.87.0.1" "cmeel-urdfdom==4.0.1" \
    "cmeel-console-bridge==1.0.2.3" "cmeel-tinyxml2==10.0.0" "cmeel-tinyxml==2.6.2.3" \
    "cmeel-assimp==5.4.3.1" "cmeel-octomap==1.10.0" "cmeel-qhull==8.0.2.1" "cmeel-zlib==1.3.1" \
    usd-core deprecation osqp "qpsolvers[clarabel,daqp,proxqp,scs,osqp]==4.9.0" loop-rate-limiters \
    tqdm psutil toml setuptools huggingface_hub
uv pip install -q --python .venv/bin/python --no-build-isolation \
    -e external/video_to_data/robotic_grounding/source/robotic_grounding
# Also install into an existing retarget-only venv.
uv pip install -q --python .venv/bin/python -r external/kit/v2d_submission_kit/requirements.txt
(cd external/video_to_data/robotic_grounding && "$PROJECT/.venv/bin/python" scripts/setup_soma_assets.py)

echo "== FlashCHORD venv (.venv-flash)"
if [ ! -x .venv-flash/bin/python ]; then
    uv venv -q --python 3.11 .venv-flash
fi
# 7eb9721 includes the kit's trajectory exporter; the kit is not a complete runtime package.
(cd external/video_to_data/robotic_grounding/flash_chord && \
    uv pip install -q --python "$PROJECT/.venv-flash/bin/python" -e ".[train,dev]")

if [ "${BOOTSTRAP_CHALLENGE:-1}" = 1 ]; then
echo "== Challenge data (Track 2 Tier 1/2, no videos)"
.venv/bin/python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("nvidia/video_to_data_challenge", repo_type="dataset", local_dir="data/challenge",
                  allow_patterns=["track_2/README.md", "track_2/tier_1_multiview_caption/*",
                                  "track_2/tier_2_synthetic_noise/*"],
                  ignore_patterns=["*/videos/*"])
EOF
fi

echo "== Retarget Tier 1 and Tier 2 to G1"
cpus=$(nproc)
# Rented containers often report every host core but enforce a smaller cgroup quota.
if read -r quota period < /sys/fs/cgroup/cpu.max 2>/dev/null && [ "$quota" != max ]; then
    cpus=$(( quota / period ))
fi
jobs=$(( cpus / 2 )); [ "$jobs" -ge 1 ] || jobs=1
# Every retarget process runs on GPU 0 and peaks near 9 GiB: 16 at once ran a 48 GB card out of memory (Oct 26 canary).
gpu_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 2>/dev/null | head -1)
[[ "$gpu_free" =~ ^[0-9]+$ ]] || gpu_free=20480
gpu_jobs=$(( gpu_free / 10240 )); [ "$gpu_jobs" -ge 1 ] || gpu_jobs=1
[ "$jobs" -le "$gpu_jobs" ] || jobs=$gpu_jobs
echo "retarget jobs: $jobs (cpus $cpus, GPU 0 free ${gpu_free} MiB)"
# BOOTSTRAP_TIERS=tier1 skips Tier 2, e.g. on a short-lived timing box.
for tier in ${BOOTSTRAP_TIERS:-tier1 tier2}; do
    case $tier in tier1) src=tier_1_multiview_caption ;; tier2) src=tier_2_synthetic_noise ;; esac
    dst=$tier
    count=$(find "data/work/$dst/motion/whole_body/soma" -path '*robot_name=g1/data.parquet' 2>/dev/null | wc -l || true)
    # Mac-generated references contain absolute object paths; regenerate after moving hosts.
    if [ "$count" -ne 30 ] || ! .venv/bin/python - "data/work/$dst" <<'PYREF'
import sys
from pathlib import Path
import pyarrow.parquet as pq
for path in Path(sys.argv[1]).glob("motion/whole_body/soma/sequence_id=*/robot_name=g1/data.parquet"):
    row = pq.ParquetFile(path).read(columns=["object_mesh_paths", "object_urdf_paths"]).to_pylist()[0]
    for field in ("object_mesh_paths", "object_urdf_paths"):
        for asset in row[field]:
            if asset and not Path(asset).is_file():
                print(f"stale reference asset: {asset}", file=sys.stderr)
                sys.exit(1)
PYREF
    then
        scripts/retarget_tier.sh "data/challenge/track_2/$src" "data/work/$dst" "$jobs"
    fi
done
if [ "${BOOTSTRAP_CHALLENGE:-1}" = 1 ] && [ -d data/work/tier2/sequences ]; then
    echo "== Repair Tier 2 contact labels (idempotent)"
    .venv/bin/python scripts/contacts.py data/work/tier2
fi

echo "== Done. Next: scripts/train_queue.sh (see its header)."
