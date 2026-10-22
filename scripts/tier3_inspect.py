"""Inventory exo videos and inspect at most two without reading reference trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, hf_hub_download

REPO = "nvidia/video_to_data_challenge"
TIER = "track_2/tier_2_synthetic_noise"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/tier3"))
    parser.add_argument("--episodes", type=int, nargs="*", default=[0, 1])
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if len(set(args.episodes)) > 2:
        parser.error("inspection is capped at two episodes")
    api = HfApi()
    repo = api.dataset_info(REPO, files_metadata=True)
    videos = [f for f in repo.siblings if "/observation.images.exo_camera/" in f.rfilename]
    inventory = [{"path": f.rfilename, "bytes": f.size, "sha256": f.lfs.sha256 if f.lfs else None} for f in videos]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "video_inventory.json").write_text(json.dumps({"revision": repo.sha, "files": inventory}, indent=2) + "\n")
    print(json.dumps({"revision": repo.sha, "files": len(videos), "bytes_per_tier": {
        tier: sum(f.size for f in videos if f.rfilename.startswith(f"track_2/{tier}/"))
        for tier in ("tier_1_multiview_caption", "tier_2_synthetic_noise")}}, indent=2))
    if not args.download:
        return
    selected = {f"episode_{i:06d}.mp4" for i in args.episodes}
    existing = {p.name for p in (args.out / "videos").rglob("*.mp4")}
    if len(existing | selected) > 2:
        parser.error("this inspection directory already contains other episodes; cumulative cap is two")
    results = []
    for index in dict.fromkeys(args.episodes):
        filename = f"{TIER}/videos/chunk-000/observation.images.exo_camera/episode_{index:06d}.mp4"
        entry = next(f for f in videos if f.rfilename == filename)
        if shutil.disk_usage(args.out).free < entry.size + 1024**3:
            raise OSError("need video size plus 1 GiB free")
        video = Path(hf_hub_download(REPO, filename, repo_type="dataset", revision=repo.sha,
                                   local_dir=args.out / "videos"))
        probe = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_streams",
            "-show_frames", "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,nb_read_frames,duration,start_time:frame=best_effort_timestamp_time",
            "-of", "json", str(video)], text=True))
        stream = probe["streams"][0]
        pts = np.array([float(f["best_effort_timestamp_time"]) for f in probe["frames"]])
        if len(pts) != int(stream['nb_frames']) or not np.isfinite(pts).all() or np.any(np.diff(pts) <= 0):
            raise ValueError("video frame count or presentation timestamps are invalid")
        digest = hashlib.sha256(video.read_bytes()).hexdigest()
        if entry.lfs and digest != entry.lfs.sha256:
            raise ValueError("download hash mismatch")
        results.append({"episode": index, "path": str(video),
                        "bytes": video.stat().st_size, "sha256": digest, "stream": stream,
                        "video_frames": len(pts), "pts_first": float(pts[0]), "pts_last": float(pts[-1]),
                        "pts_delta_min_s": float(np.diff(pts).min()), "pts_delta_max_s": float(np.diff(pts).max())})
    (args.out / "video_inspection.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
