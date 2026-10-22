#!/usr/bin/env bash
# Native probe recipe: pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel; root Linux.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
step=all; dry=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --work-root) NATIVE_ROOT=$2; shift 2;;
    --project) NATIVE_PROJECT=$2; shift 2;;
    --weights) NATIVE_WEIGHTS=$2; shift 2;;
    --dry-run) dry=true; shift;;
    all|apt|python|extensions|packages|tensorrt|weights) step=$1; shift;;
    *) echo "Unknown argument: $1" >&2; exit 2;;
  esac
done
source "$HERE/env.sh"
export DEBIAN_FRONTEND=noninteractive SAM2_BUILD_ALLOW_ERRORS=0 FORCE_CUDA=1
export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1
export PIP_CONSTRAINT=$NATIVE_MODULES/v2d_cari4d/lib/pip-constraints.txt
run() { printf '%q ' "$@"; printf '\n'; if ! $dry; then "$@"; fi; }
if ! $dry; then
  [[ $(uname -s) = Linux ]] || { echo 'Setup requires Linux/CUDA; use --dry-run on Mac' >&2; exit 2; }
  df -h "$(dirname "$NATIVE_ROOT")"
  mkdir -p "$NATIVE_ROOT/third_party" "$NATIVE_ROOT/setup" "$NATIVE_WEIGHTS"
fi
phase() {
  current_phase=$1
  [[ $step = "$1" || $step = all ]] || return 1
  if ! $dry && [[ -f $NATIVE_ROOT/setup/$1.done ]]; then
    cmp -s "$0" "$NATIVE_ROOT/setup/$1.done" || { echo 'Setup recipe changed; use a fresh build root' >&2; exit 1; }
    echo "SKIP setup $1"; return 1
  fi
}
finish_phase() { if ! $dry; then cp "$0" "$NATIVE_ROOT/setup/$current_phase.done"; fi; }
archive() {
  local repo=$1 ref=$2 dest=$3 tmp
  if $dry; then
    printf 'ARCHIVE %q %q %q\n' "$repo" "$ref" "$dest"; return
  fi
  if [[ -f $dest/.native_archive_pin ]]; then
    [[ $(cat "$dest/.native_archive_pin") = "$repo $ref" ]]; return
  fi
  [[ ! -e $dest ]] || { echo "Refusing unowned partial source: $dest" >&2; exit 1; }
  tmp=$(mktemp -d "$NATIVE_ROOT/third_party/download.XXXXXX")
  curl -fsSL --retry 2 --connect-timeout 20 "https://codeload.github.com/$repo/tar.gz/$ref" -o "$tmp/source.tar.gz"
  mkdir "$tmp/source"
  tar -xzf "$tmp/source.tar.gz" --strip-components=1 -C "$tmp/source"
  printf '%s %s\n' "$repo" "$ref" > "$tmp/source/.native_archive_pin"
  mv "$tmp/source" "$dest"
  rm "$tmp/source.tar.gz"; rmdir "$tmp"
}
if phase apt; then
  run df -h "$NATIVE_ROOT"
  run apt-get update --fix-missing
  run apt-get install -y python3-dev wget curl zip unzip ffmpeg g++ gcc build-essential cmake gfortran pkg-config yasm libgl1 libglib2.0-0 libgtk2.0-dev libjpeg8-dev libtiff5-dev libavcodec-dev libavformat-dev libswscale-dev libxine2-dev libv4l-dev libtbb-dev libatlas-base-dev libfaac-dev libmp3lame-dev libtheora-dev libvorbis-dev libxvidcore-dev libopencore-amrnb-dev libopencore-amrwb-dev x264 v4l-utils libprotobuf-dev protobuf-compiler libgoogle-glog-dev libgflags-dev libgphoto2-dev libhdf5-dev libflann-dev libboost-all-dev proj-data libproj-dev libyaml-cpp-dev libzmq3-dev freeglut3-dev libxrender1 libxext6 libsm6 libx11-6 libegl1-mesa libegl1-mesa-dev libgles2-mesa libgbm1 libosmesa6-dev libspatialindex-dev libeigen3-dev time
  finish_phase
fi
if phase python; then
  run "$NATIVE_PYTHON" -I -m pip install numpy==1.26.3 scipy joblib scikit-learn ruamel.yaml trimesh pyyaml opencv-python imageio imageio-ffmpeg 'imageio[pyav]' open3d transformations warp-lang einops kornia pyrender av h5py hdf5plugin videoio==0.3.0 scikit-image meshcat webdataset omegaconf pypng roma seaborn opencv-contrib-python openpyxl wandb matplotlib tqdm pillow accelerate timm yacs smplx tensorboardx pycocotools rtree xatlas pytorch-lightning hydra-core hydra-submitit-launcher hydra-colorlog pyrootutils networkx==3.2.1 rich dill pandas jsonlines loguru optree fvcore cython py-soma-x==0.2.1 fast_simplification gdown setuptools wheel ninja flask decord pytransform3d imgaug==0.4.0 transformers==5.3.0 pytest==9.1.1
  run "$NATIVE_PYTHON" -I -m pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu124.html
  finish_phase
fi
if phase extensions; then
  archive pybind/pybind11 v2.10.0 "${NATIVE_ROOT}/third_party/pybind11"
  run cmake -S "${NATIVE_ROOT}/third_party/pybind11" -B "${NATIVE_ROOT}/third_party/pybind11/build" -DCMAKE_BUILD_TYPE=Release -DPYBIND11_INSTALL=ON -DPYBIND11_TEST=OFF
  run cmake --build "${NATIVE_ROOT}/third_party/pybind11/build" -j2
  run cmake --install "${NATIVE_ROOT}/third_party/pybind11/build"
  archive facebookresearch/pytorch3d v0.7.9 "${NATIVE_ROOT}/third_party/pytorch3d"
  run "$NATIVE_PYTHON" -I -m pip install "${NATIVE_ROOT}/third_party/pytorch3d" --no-build-isolation
  archive NVlabs/nvdiffrast 253ac4fcea7de5f396371124af597e6cc957bfae "${NATIVE_ROOT}/third_party/nvdiffrast"
  run "$NATIVE_PYTHON" -I -m pip install "${NATIVE_ROOT}/third_party/nvdiffrast" --no-build-isolation
  run cmake -S "${NATIVE_MODULES}/v2d_foundation_pose/lib/FoundationPose/mycpp" -B "${NATIVE_MODULES}/v2d_foundation_pose/lib/FoundationPose/mycpp/build" -DPYTHON_EXECUTABLE="$(command -v "$NATIVE_PYTHON")"
  run cmake --build "${NATIVE_MODULES}/v2d_foundation_pose/lib/FoundationPose/mycpp/build" -j2
  run "$NATIVE_PYTHON" -I -m pip install -e "${NATIVE_MODULES}/v2d_foundation_pose/lib/FoundationPose/bundlesdf/mycuda" --no-build-isolation
  finish_phase
fi
if phase packages; then
  archive EasternJournalist/utils3d 3fab839f0be9931dac7c8488eb0e1600c236e183 "${NATIVE_ROOT}/third_party/utils3d"
  run "$NATIVE_PYTHON" -I -m pip install --no-deps "${NATIVE_ROOT}/third_party/utils3d"
  archive microsoft/MoGe 925b8ed835a7a9cdb7578ba15c658a0afc969030 "${NATIVE_ROOT}/third_party/MoGe"
  run "$NATIVE_PYTHON" -I -m pip install "${NATIVE_ROOT}/third_party/MoGe" --no-deps
  archive EasternJournalist/pipeline 866f059d2a05cde05e4a52211ec5051fd5f276d6 "${NATIVE_ROOT}/third_party/pipeline"
  run "$NATIVE_PYTHON" -I -m pip install moderngl gradio transformers==5.3.0
  run "$NATIVE_PYTHON" -I -m pip install "${NATIVE_ROOT}/third_party/pipeline" --no-deps
  archive facebookresearch/sam2 2b90b9f5ceec907a1c18123530e92e794ad901a4 "${NATIVE_ROOT}/third_party/sam2"
  run "$NATIVE_PYTHON" -I -m pip install "${NATIVE_ROOT}/third_party/sam2" --no-build-isolation
  run "$NATIVE_PYTHON" -I -m pip install -e "${NATIVE_MODULES}/v2d_sam2/lib" --no-deps
  run "$NATIVE_PYTHON" -I -m pip install --no-deps -e "${NATIVE_MODULES}/v2d_common[io]" -e "${NATIVE_MODULES}/v2d_mv[math,vis]" -e "${NATIVE_MODULES}/v2d_mesh/lib" -e "${NATIVE_MODULES}/v2d_foundation_pose/lib" -e "${NATIVE_MODULES}/v2d_moge/lib" -e "${NATIVE_MODULES}/v2d_sam3d_body/lib" -e "${NATIVE_MODULES}/v2d_cari4d/lib"
  run "$NATIVE_PYTHON" -I "$HERE/apply_native_patches.py"
  finish_phase
fi

if phase tensorrt; then
  run apt-get update
  run apt-get install -y libnvinfer-bin=10.7.0.23-1+cuda12.6 libnvinfer10=10.7.0.23-1+cuda12.6 libnvinfer-lean10=10.7.0.23-1+cuda12.6 libnvinfer-plugin10=10.7.0.23-1+cuda12.6 libnvinfer-vc-plugin10=10.7.0.23-1+cuda12.6 libnvinfer-dispatch10=10.7.0.23-1+cuda12.6 libnvonnxparsers10=10.7.0.23-1+cuda12.6
  run "$NATIVE_PYTHON" -I -m pip install tensorrt-cu12==10.7.0.post1
  finish_phase
fi

if phase weights; then
  run "$NATIVE_PYTHON" -I "$HERE/download_public.py"
  finish_phase
fi
printf 'NATIVE_DONE %s %s\n' "$step" "$(date -u +%FT%TZ)"
