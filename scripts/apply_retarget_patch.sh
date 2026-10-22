#!/usr/bin/env bash
# Keep the retargeter's OBJ synchronized with the supplied GLB; safe to rerun.
set -euo pipefail
PROJECT=$(cd "$(dirname "$0")/.." && pwd)
rg=${1:-$PROJECT/external/video_to_data/robotic_grounding}
upstream=$(cd "$rg/.." && pwd)
patch_file=$PROJECT/patches/retarget_mesh_refresh.patch
if ! git -C "$upstream" apply --reverse --check "$patch_file" 2>/dev/null; then
    git -C "$upstream" apply --check "$patch_file"
    git -C "$upstream" apply "$patch_file"
fi
