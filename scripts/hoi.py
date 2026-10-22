"""Shared helpers for soma_to_g1 sequence folders (soma_params.npz + poses.npy + mesh + ground plane)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from robotic_grounding.retarget.params import SOMA_JOINTS_ORDER
from robotic_grounding.retarget.read_soma import SOMA
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

CV_TO_SOMA = np.diag([1.0, -1.0, -1.0])  # source OpenCV world -> SOMA world, as in soma_to_g1.py
SIDES = ("Left", "Right")
FINGERTIPS = {side: [i for i, n in enumerate(SOMA_JOINTS_ORDER) if n.startswith(f"{side}Hand") and n.endswith("End")]
              for side in SIDES}
FEET = [i for i, n in enumerate(SOMA_JOINTS_ORDER) if "Foot" in n or "Toe" in n]
MESH_POINTS = 4000
FPS = 30.0
MOVING_SPEED = 0.2  # m/s


@dataclass(frozen=True)
class Sequence:
    folder: Path
    object_poses_cv: np.ndarray  # [T, 4, 4] world_T_object, OpenCV world
    joints: np.ndarray  # [T, J, 3] SOMA joint positions, SOMA world
    mesh_points: np.ndarray  # [N, 3] object-frame surface samples

    @classmethod
    def load(cls, folder: Path, soma: SOMA) -> Sequence:
        mesh = trimesh.load(folder / "reconstructed_mesh" / "output_aligned.glb", force="mesh")
        points = (trimesh.sample.sample_surface(mesh, MESH_POINTS, seed=0)[0] if len(mesh.vertices) > MESH_POINTS
                  else mesh.vertices)
        return cls(
            folder=folder,
            object_poses_cv=np.load(folder / "poses.npy"),
            joints=soma.load_motion(folder / "soma_params.npz")["joints"],
            mesh_points=np.asarray(points),
        )

    def object_moving(self, speed: float = MOVING_SPEED) -> np.ndarray:
        """[T] frames where some point of the smoothed object surface moves faster than `speed` m/s.

        Surface points, not the origin, so that turning or tipping an object in place counts as moving.
        """
        quats = Rotation.from_matrix(self.object_poses_cv[:, :3, :3]).as_quat()
        quats[np.cumsum(np.r_[False, np.sum(quats[1:] * quats[:-1], axis=1) < 0]) % 2 == 1] *= -1
        quats = savgol_filter(quats, 15, 3, axis=0)
        rotation = Rotation.from_quat(quats / np.linalg.norm(quats, axis=1, keepdims=True)).as_matrix()
        translation = savgol_filter(self.object_poses_cv[:, :3, 3], 15, 3, axis=0)
        points = self.mesh_points[:: max(1, len(self.mesh_points) // 64)]
        world = np.einsum("tij,nj->tni", rotation, points) + translation[:, None]
        return np.linalg.norm(np.gradient(world, axis=0), axis=-1).max(axis=1) * FPS > speed

    def _gaps(self, joint_ids: list[int]) -> np.ndarray:
        """[T] smallest distance from the given joints to the object surface, metres."""
        rotation = CV_TO_SOMA @ self.object_poses_cv[:, :3, :3]
        translation = self.object_poses_cv[:, :3, 3] @ CV_TO_SOMA.T
        local = np.einsum("tkj,tji->tki", self.joints[:, joint_ids] - translation[:, None], rotation)
        distances = cKDTree(self.mesh_points).query(local.reshape(-1, 3))[0]
        return distances.reshape(len(translation), -1).min(axis=1)

    def hand_gaps(self) -> np.ndarray:
        """[2, T] smallest fingertip-to-object-surface distance per hand (Left, Right), metres."""
        return np.stack([self._gaps(FINGERTIPS[side]) for side in SIDES])

    def foot_gaps(self) -> np.ndarray:
        """[T] smallest foot/toe-joint-to-object-surface distance, metres."""
        return self._gaps(FEET)
