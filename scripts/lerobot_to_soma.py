"""Convert released Track 2 LeRobot episodes into soma_to_g1.py sequence folders.

For each episode this writes ``<out>/<sequence_id>/`` containing:

- ``soma_params.npz``: SOMA-X motion in the layout ``read_soma.SOMA.load_motion`` reads.
- ``poses.npy``: ``[T, 4, 4]`` ``world_T_object`` in the source OpenCV world frame.
- ``reconstructed_mesh/output_aligned.glb``: the released object mesh.
- ``ground_plane.json``: the released ground-plane sidecar.
- ``filled_frames.json``: frames whose invisible object pose was interpolated.

With ``--motion-root`` it also stages the released G1 support surface where the
FlashCHORD loader expects it.

Usage:
    python scripts/lerobot_to_soma.py --tier-root data/challenge/track_2/tier_1_multiview_caption \
        --out data/sequences/tier1 --motion-root data/motion/tier1 [--episodes 0 3 7]
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation, Slerp

NUM_JOINTS = 77


@dataclass(frozen=True)
class Episode:
    index: int
    sequence_id: str
    object_name: str
    joint_names: list[str]
    poses_rotvec: np.ndarray  # [T, 77, 3]
    transl: np.ndarray  # [T, 3] metres
    identity_coeffs: np.ndarray  # [45]
    scale_params: np.ndarray  # [68]
    bone_length_flexibles: np.ndarray  # [T, 6]
    object_pose_wxyz: np.ndarray  # [T, 7] x y z qw qx qy qz
    object_visible: np.ndarray  # [T] bool
    mesh_path: Path
    ground_plane_path: Path
    support_surface_path: Path


def _column(table, name: str) -> np.ndarray:
    return np.asarray(table.column(name).to_pylist(), dtype=np.float32)


def load_episode(tier_root: Path, index: int) -> Episode:
    info = json.loads((tier_root / "meta" / "info.json").read_text())
    metadata = {
        row["episode_index"]: row
        for row in map(json.loads, (tier_root / "meta" / "episodes_metadata.jsonl").read_text().splitlines())
    }[index]

    pose_names = info["features"]["observation.human.pose"]["names"]
    joint_names = [name.rsplit(".", 1)[0] for name in pose_names[::3]]
    expected = [f"{joint}.{axis}" for joint in joint_names for axis in ("rx", "ry", "rz")]
    if pose_names != expected or len(joint_names) != NUM_JOINTS:
        raise ValueError(f"unexpected pose column layout in {tier_root}/meta/info.json")

    table = pq.read_table(tier_root / info["data_path"].format(episode_chunk=0, episode_index=index))
    frame_index = np.asarray(table.column("frame_index").to_pylist())
    if not np.array_equal(frame_index, np.arange(len(frame_index))):
        raise ValueError(f"episode {index}: frame_index is not a contiguous 0..T-1 range")

    identity = _column(table, "observation.human.identity_coeffs")
    scale = _column(table, "observation.human.scale_params")
    for name, values in (("identity_coeffs", identity), ("scale_params", scale)):
        if not np.all(values == values[0]):
            raise ValueError(f"episode {index}: {name} varies over time; read_soma uses frame 0 only")

    return Episode(
        index=index,
        sequence_id=metadata["sequence_id"],
        object_name=metadata["object"],
        joint_names=joint_names,
        poses_rotvec=_column(table, "observation.human.pose").reshape(-1, NUM_JOINTS, 3),
        transl=_column(table, "observation.human.translation"),
        identity_coeffs=identity[0],
        scale_params=scale[0],
        bone_length_flexibles=_column(table, "observation.human.bone_length_flexibles"),
        object_pose_wxyz=_column(table, "observation.object.pose"),
        object_visible=np.asarray(table.column("observation.object.visible").to_pylist(), dtype=bool),
        mesh_path=tier_root / metadata["mesh"],
        ground_plane_path=tier_root / metadata["ground_plane"],
        support_surface_path=tier_root / metadata["support_surface"],
    )


def object_poses_matrix(pose_wxyz: np.ndarray, visible: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Return ``[T, 4, 4]`` poses, interpolating invisible frames from visible neighbours.

    The simulator needs a pose on every frame; the released data zero-fills invisible ones.
    """
    if not visible.any():
        raise ValueError("object is never visible")
    frames = np.arange(len(pose_wxyz))
    known = frames[visible]
    rotations = Rotation.from_quat(pose_wxyz[visible, 3:7], scalar_first=True)
    # Hold the nearest visible pose before the first and after the last visible frame.
    query = np.clip(frames, known[0], known[-1])
    if len(known) == 1:
        rot_all = Rotation.from_quat(np.repeat(rotations.as_quat()[:1], len(frames), axis=0))
    else:
        rot_all = Slerp(known, rotations)(query)
    transl_all = np.stack([np.interp(query, known, pose_wxyz[visible, axis]) for axis in range(3)], axis=-1)

    matrices = np.tile(np.eye(4), (len(frames), 1, 1))
    matrices[:, :3, :3] = rot_all.as_matrix()
    matrices[:, :3, 3] = transl_all
    return matrices, [int(f) for f in frames[~visible]]


def write_sequence(episode: Episode, out_root: Path, motion_root: Path | None) -> Path:
    folder = out_root / episode.sequence_id
    (folder / "reconstructed_mesh").mkdir(parents=True, exist_ok=True)

    # No joint_orient: read_soma then uses the rig T-pose, the same reference the SOMA exporter saves.
    np.savez(
        folder / "soma_params.npz",
        poses=episode.poses_rotvec,
        transl=episode.transl,
        joint_names=np.asarray(episode.joint_names),
        identity_model_type=np.asarray("mhr"),
        identity_coeffs=episode.identity_coeffs[None],
        scale_params=episode.scale_params[None],
        unit=np.asarray("meters"),
        keep_root=np.asarray(False),
        bone_length_flexibles=episode.bone_length_flexibles,
    )

    poses, filled = object_poses_matrix(episode.object_pose_wxyz, episode.object_visible)
    np.save(folder / "poses.npy", poses)
    (folder / "filled_frames.json").write_text(json.dumps({"object": episode.object_name, "frames": filled}))

    shutil.copyfile(episode.mesh_path, folder / "reconstructed_mesh" / "output_aligned.glb")
    shutil.copyfile(episode.ground_plane_path, folder / "ground_plane.json")

    if motion_root is not None:
        stage_dir = motion_root / "whole_body" / "reconstructed_stage"
        stage_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(episode.support_surface_path, stage_dir / f"{episode.sequence_id}_support.usda")
    return folder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tier-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--motion-root", type=Path)
    parser.add_argument("--episodes", type=int, nargs="*", help="Default: every episode in the tier.")
    args = parser.parse_args()

    total = json.loads((args.tier_root / "meta" / "info.json").read_text())["total_episodes"]
    for index in args.episodes if args.episodes else range(total):
        episode = load_episode(args.tier_root, index)
        folder = write_sequence(episode, args.out, args.motion_root)
        filled = int((~episode.object_visible).sum())
        print(f"episode {index:02d} -> {folder} ({len(episode.transl)} frames, {filled} filled)")


if __name__ == "__main__":
    main()
