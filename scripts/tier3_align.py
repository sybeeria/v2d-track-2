"""Align metric monocular sequence folders using only Tier 1 frame zero.

Human SOMA parameters use the native Y-up frame; object poses use OpenCV.
The exporter uses SE(3): upstream CARI4D requires a metric mesh and metric MHR.
Sim(3) estimation is exposed for scale diagnostics and synthetic verification.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import trimesh
from scipy.spatial.transform import Rotation

from lerobot_to_soma import Episode, object_poses_matrix, write_sequence

CV_TO_SOMA = np.diag([1.0, -1.0, -1.0])
ALIGN_JOINTS = ("Hips", "Spine1", "Chest", "Head", "LeftArm", "RightArm",
                "LeftForeArm", "RightForeArm", "LeftHand", "RightHand",
                "LeftLeg", "RightLeg", "LeftShin", "RightShin", "LeftFoot", "RightFoot")
FRAME0_COLUMNS = ("frame_index", "observation.human.pose", "observation.human.translation",
                  "observation.human.identity_coeffs", "observation.human.scale_params",
                  "observation.human.bone_length_flexibles", "observation.object.pose",
                  "observation.object.visible")


@dataclass(frozen=True)
class Similarity:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    def points(self, points: np.ndarray) -> np.ndarray:
        return self.scale * np.asarray(points) @ self.rotation.T + self.translation

    def poses(self, poses: np.ndarray) -> np.ndarray:
        result = np.asarray(poses, dtype=np.float64).copy()
        result[:, :3, :3] = self.rotation @ result[:, :3, :3]
        result[:, :3, 3] = self.points(result[:, :3, 3])
        return result


def estimate_alignment(source: np.ndarray, target: np.ndarray, *, similarity: bool = False) -> Similarity:
    """Least-squares proper rotation, optional positive scale, and translation."""
    source, target = np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or len(source) < 3:
        raise ValueError("need matching [N,3] frame-zero points, N >= 3")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("non-finite alignment points")
    x, y = source - source.mean(0), target - target.mean(0)
    if np.linalg.matrix_rank(x, tol=1e-8) < 2 or np.linalg.matrix_rank(y, tol=1e-8) < 2:
        raise ValueError("frame-zero alignment is collinear or collapsed")
    u, values, vt = np.linalg.svd(x.T @ y)
    signs = np.ones(3)
    signs[-1] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ np.diag(signs) @ u.T
    scale = float((values * signs).sum() / np.square(x).sum()) if similarity else 1.0
    if scale <= 0:
        raise ValueError("alignment scale must be positive")
    return Similarity(scale, rotation, target.mean(0) - scale * rotation @ source.mean(0))


def load_tier1_frame0(tier_root: Path, index: int) -> Episode:
    """Expose one projected row; never materialize a reference trajectory."""
    info = json.loads((tier_root / "meta/info.json").read_text())
    metadata = next(r for r in map(json.loads, (tier_root / "meta/episodes_metadata.jsonl").read_text().splitlines())
                    if r["episode_index"] == index)
    pose_names = info["features"]["observation.human.pose"]["names"]
    names = [n.rsplit(".", 1)[0] for n in pose_names[::3]]
    if pose_names != [f"{n}.{axis}" for n in names for axis in ("rx", "ry", "rz")] or len(names) != 77:
        raise ValueError("unexpected released joint layout")
    path = tier_root / info["data_path"].format(episode_chunk=index // info["chunks_size"], episode_index=index)
    # Parquet decompresses column chunks internally; only the first one-row batch
    # is yielded to application code, and no iterator is returned to the caller.
    with pq.ParquetFile(path) as parquet:
        row = next(parquet.iter_batches(batch_size=1, columns=list(FRAME0_COLUMNS), use_threads=False)).to_pylist()[0]
    if row["frame_index"] != 0 or not row["observation.object.visible"]:
        raise ValueError("alignment requires a visible object at frame zero")
    array = lambda key: np.asarray(row[f"observation.human.{key}"], dtype=np.float64)
    return Episode(index, metadata["sequence_id"], metadata["object"], names,
                   array("pose").reshape(1, 77, 3), array("translation")[None],
                   array("identity_coeffs"), array("scale_params"), array("bone_length_flexibles")[None],
                   np.asarray(row["observation.object.pose"], dtype=np.float64)[None], np.ones(1, dtype=bool),
                   Path(), Path(), Path())


def load_soma(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    frames = len(data["poses"])
    data.setdefault("bone_length_flexibles", np.zeros((frames, 6)))
    if data["poses"].shape != (frames, 77, 3) or data["transl"].shape != (frames, 3) or frames == 0:
        raise ValueError("invalid SOMA motion dimensions")
    if data["bone_length_flexibles"].shape != (frames, 6) or data["joint_names"].shape != (77,):
        raise ValueError("invalid bone-flexible or joint-name dimensions")
    if str(data["unit"]) != "meters" or str(data["identity_model_type"]) != "mhr" or bool(data["keep_root"]):
        raise ValueError("expected meter MHR SOMA parameters with keep_root=False")
    if bool(data.get("absolute_pose", False)):
        raise ValueError("absolute SOMA poses are unsupported")
    for key, width in (("identity_coeffs", 45), ("scale_params", 68)):
        if data[key].ndim != 2 or data[key].shape[1] != width or data[key].shape[0] not in (1, frames):
            raise ValueError(f"invalid {key} dimensions")
        if not np.allclose(data[key], data[key][:1], rtol=0, atol=1e-6):
            raise ValueError(f"{key} varies; retargeter uses a single identity")
    for key in ("poses", "transl", "identity_coeffs", "scale_params", "bone_length_flexibles"):
        if not np.isfinite(data[key]).all():
            raise ValueError(f"non-finite {key}")
    return data


def soma_frame0_joints(data: dict[str, np.ndarray], soma, temp: Path) -> np.ndarray:
    """Decode exactly one body frame, independent of trajectory length."""
    np.savez(temp, **{key: value[:1] if key in ("poses", "transl", "bone_length_flexibles", "identity_coeffs", "scale_params") else value
                     for key, value in data.items()})
    return soma.load_motion(temp)["joints"][0] @ CV_TO_SOMA.T


def align_sequence(source: Path, reference: Episode, out: Path, motion_root: Path,
                   ground_plane: Path, support_surface: Path, soma, *, scene_assets_mode='provided') -> Path:
    """Write a metric sequence with a frame-zero-calibrated object local frame."""
    data = load_soma(source / "soma_params.npz")
    names = [str(n) for n in data["joint_names"]]
    if names != reference.joint_names:
        raise ValueError("reconstructed and reference SOMA joints differ")
    if (out / reference.sequence_id).exists():
        raise FileExistsError(out / reference.sequence_id)
    poses = np.load(source / "poses.npy", allow_pickle=False)
    if poses.shape != (len(data["poses"]), 4, 4) or not np.isfinite(poses).all():
        raise ValueError("object/human frame counts differ or object poses are invalid")
    if not np.allclose(poses[:, 3], [0, 0, 0, 1]) or not np.allclose(
            poses[:, :3, :3].transpose(0, 2, 1) @ poses[:, :3, :3], np.eye(3), atol=1e-5):
        raise ValueError("object poses must be rigid homogeneous transforms")
    if not np.allclose(np.linalg.det(poses[:, :3, :3]), 1, atol=1e-5):
        raise ValueError("object rotations contain a reflection")
    ids = [names.index(n) for n in ALIGN_JOINTS]
    with tempfile.TemporaryDirectory(prefix="tier3-frame0-", dir=out.parent) as temp_dir:
        temp = Path(temp_dir)
        ref_data = dict(poses=reference.poses_rotvec, transl=reference.transl, joint_names=np.array(names),
                        identity_model_type=np.array("mhr"), identity_coeffs=reference.identity_coeffs[None],
                        scale_params=reference.scale_params[None], bone_length_flexibles=reference.bone_length_flexibles,
                        unit=np.array("meters"), keep_root=np.array(False))
        x = soma_frame0_joints(data, soma, temp / "source.npz")[ids]
        y = soma_frame0_joints(ref_data, soma, temp / "target.npz")[ids]
        transform = estimate_alignment(x, y)
        scale_diagnostic = estimate_alignment(x, y, similarity=True)
        human_rotation = CV_TO_SOMA @ transform.rotation @ CV_TO_SOMA
        human_translation = CV_TO_SOMA @ transform.translation
        body_poses = data["poses"].copy()
        body_poses[:, 0] = Rotation.from_matrix(human_rotation @ Rotation.from_rotvec(body_poses[:, 0]).as_matrix()).as_rotvec()
        body_translation = data["transl"] @ human_rotation.T + human_translation
        aligned = transform.poses(poses)
        reference_object, _ = object_poses_matrix(reference.object_pose_wxyz, reference.object_visible)
        # CARI4D recenters and reorients its mesh. A world transform alone cannot
        # align the local mesh basis/origin used by the score's object poses.
        local = np.linalg.inv(aligned[0]) @ reference_object[0]
        aligned = aligned @ local
        mesh = trimesh.load(source / "reconstructed_mesh/output_aligned.glb", force="scene", process=False)
        mesh.apply_transform(np.linalg.inv(local))
        mesh_path = temp / "mesh.glb"
        mesh.export(mesh_path)
        filled = json.loads((source / "filled_frames.json").read_text())["frames"]
        if any(not isinstance(i, int) or i < 0 or i >= len(poses) for i in filled):
            raise ValueError("invalid reconstructed filled-frame indices")
        object_wxyz = np.c_[aligned[:, :3, 3], Rotation.from_matrix(aligned[:, :3, :3]).as_quat(scalar_first=True)]
        episode = Episode(reference.index, reference.sequence_id, reference.object_name, names, body_poses,
                          body_translation, data["identity_coeffs"][0], data["scale_params"][0],
                          data["bone_length_flexibles"], object_wxyz, np.ones(len(poses), bool),
                          mesh_path, ground_plane, support_surface)
        folder = write_sequence(episode, out, motion_root)
        (folder / "filled_frames.json").write_text(json.dumps({"object": reference.object_name, "frames": filled}) + "\n")
        report = {"method": "SE(3), metric input mesh and human; no identity rescaling",
                  "reference_episode": reference.index, "reference_frames_used": [0], "alignment_joints": list(ALIGN_JOINTS),
                  "rotation": transform.rotation.tolist(), "translation": transform.translation.tolist(), "scale": 1.0,
                  "diagnostic_sim3_scale": scale_diagnostic.scale,
                  "frame0_joint_rmse_m": float(np.sqrt(np.square(transform.points(x)-y).sum(1).mean())),
                  "object_local_basis": local.tolist(), "source": str(source.resolve()),
                  "scene_assets_mode": scene_assets_mode,
                  "ground_plane": str(ground_plane.resolve()), "support_surface": str(support_surface.resolve())}
        (folder / "tier3_alignment.json").write_text(json.dumps(report, indent=2) + "\n")
    return folder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--tier1-root", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--motion-root", type=Path, required=True)
    parser.add_argument("--ground-plane", type=Path, required=True, help="Released-world plane; provenance must be permitted independently of captions")
    parser.add_argument("--support-surface", type=Path, required=True, help="G1 retarget-world USDA; do not apply the camera/world transform")
    parser.add_argument("--released-scene-assets-engineering-only", action="store_true",
                        help="Flag use of released plane/support; Tier 3 eligibility unresolved")
    args = parser.parse_args()
    from robotic_grounding.retarget.read_soma import SOMA
    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(align_sequence(args.source, load_tier1_frame0(args.tier1_root, args.episode), args.out,
                         args.motion_root, args.ground_plane, args.support_surface, SOMA(),
                         scene_assets_mode="released-engineering-only; ruling pending" if args.released_scene_assets_engineering_only else "provided"))


if __name__ == "__main__":
    main()
