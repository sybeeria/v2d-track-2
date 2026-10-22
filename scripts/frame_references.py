"""Export converter-frame poses.npy files as packer input references.

lerobot_to_soma.py keeps released world positions and rotations unchanged, converting
wxyz quaternions to matrices and filling invisible frames. This reverses the matrix
representation, including any subsequent edits to poses.npy. It does not undo the
retargeter's separate robot-base transform; the packer estimates that transform.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--tier-root", type=Path, required=True)
    parser.add_argument("-o", "--out", type=Path, required=True)
    args = parser.parse_args()
    info = json.loads((args.tier_root / "meta/info.json").read_text())
    metadata = [json.loads(line) for line in
                (args.tier_root / "meta/episodes_metadata.jsonl").read_text().splitlines()]
    if sorted(row["episode_index"] for row in metadata) != list(range(info["total_episodes"])):
        parser.error("episode metadata must contain each released episode exactly once")
    args.out.mkdir(parents=True, exist_ok=True)
    for row in sorted(metadata, key=lambda row: row["episode_index"]):
        episode = row["episode_index"]
        source = args.work_dir / "sequences" / row["sequence_id"] / "poses.npy"
        poses = np.load(source, allow_pickle=False)
        chunk = episode // info["chunks_size"]
        table = pq.ParquetFile(args.tier_root / info["data_path"].format(
            episode_chunk=chunk, episode_index=episode)).read(columns=["frame_index"])
        frames = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)
        if not np.array_equal(frames, np.arange(len(frames))):
            parser.error(f"episode {episode}: converter requires contiguous frame indices")
        if poses.shape != (len(frames), 4, 4) or not np.isfinite(poses).all():
            parser.error(f"{source}: expected finite [{len(frames)},4,4] poses")
        rotation = poses[:, :3, :3]
        if not (np.allclose(poses[:, 3], [0, 0, 0, 1], rtol=0, atol=1e-6)
                and np.allclose(rotation.transpose(0, 2, 1) @ rotation, np.eye(3), rtol=0, atol=1e-5)
                and np.allclose(np.linalg.det(rotation), 1, rtol=0, atol=1e-5)):
            parser.error(f"{source}: poses are not rigid transforms")
        xyzw = np.concatenate((poses[:, :3, 3], Rotation.from_matrix(rotation).as_quat()), axis=1)
        path = args.out / f"episode_{episode:06d}.npz"
        np.savez(path, episode_index=np.asarray(episode),
                 coordinate_frame=np.asarray("released_track2_world"),
                 frame_index=frames, pose_xyzw=xyzw[:, None, :])
        print(f"episode {episode:02d}: {len(frames)} frames -> {path}")


if __name__ == "__main__":
    main()
