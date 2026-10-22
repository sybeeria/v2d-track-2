"""Per-sequence noise statistics for soma_to_g1 sequence folders (soma_params.npz + poses.npy + mesh).

High-frequency residual = signal minus a Savitzky-Golay fit (15 frames, cubic): real motion
leaves little residual at that scale, white jitter leaves almost all of it. The carrying gap is the
smallest fingertip-to-object-mesh distance on frames where the object moves faster than 0.2 m/s.

Usage: python scripts/noise_stats.py <label>=<glob of sequence folders> [...]
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
from hoi import CV_TO_SOMA, FEET, FPS, MOVING_SPEED, Sequence
from robotic_grounding.retarget.read_soma import SOMA
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

WINDOW, ORDER = 15, 3
HANDS_AWAY = 0.25  # m: no fingertip this close to the object, so it should be resting


def rotation_residual_deg(quats_xyzw: np.ndarray) -> np.ndarray:
    """Angle between each rotation and its Savitzky-Golay smoothed version, for [T, ..., 4] quaternions."""
    q = quats_xyzw.copy()
    for t in range(1, len(q)):  # remove double-cover sign flips before filtering
        flip = np.sum(q[t] * q[t - 1], axis=-1) < 0
        q[t][flip] *= -1
    smooth = savgol_filter(q, WINDOW, ORDER, axis=0)
    smooth /= np.linalg.norm(smooth, axis=-1, keepdims=True)
    dot = np.abs(np.sum(q * smooth, axis=-1)).clip(0, 1)
    return np.degrees(2 * np.arccos(dot))


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def sequence_stats(folder: Path, soma: SOMA) -> dict[str, float]:
    params = np.load(folder / "soma_params.npz")
    seq = Sequence.load(folder, soma)
    poses, joints = seq.object_poses_cv, seq.joints
    obj_t = poses[:, :3, 3]
    obj_q = Rotation.from_matrix(poses[:, :3, :3]).as_quat()
    human_q = Rotation.from_rotvec(params["poses"].reshape(-1, 3)).as_quat().reshape(len(poses), -1, 4)

    speed = np.linalg.norm(np.gradient(savgol_filter(obj_t, WINDOW, ORDER, axis=0), axis=0), axis=-1) * FPS
    all_gaps = seq.hand_gaps().min(axis=0)
    moving = seq.object_moving(speed=MOVING_SPEED)
    gaps = all_gaps[moving] if moving.any() else np.full(1, np.nan)
    hands_away = all_gaps > HANDS_AWAY

    plane = json.loads((folder / "ground_plane.json").read_text())["plane"]
    normal_soma, offset = np.asarray(plane[:3]) @ CV_TO_SOMA, plane[3]
    foot_height = (joints[:, FEET] @ normal_soma + offset).min(axis=1)

    return {
        "frames": len(poses),
        "obj_t_hf_mm": rms(np.linalg.norm(obj_t - savgol_filter(obj_t, WINDOW, ORDER, axis=0), axis=-1)) * 1e3,
        "obj_r_hf_deg": rms(rotation_residual_deg(obj_q)),
        "root_t_hf_mm": rms(np.linalg.norm(params["transl"] - savgol_filter(params["transl"], WINDOW, ORDER, axis=0),
                                           axis=-1)) * 1e3,
        "joint_r_hf_deg": float(np.median([rms(r) for r in rotation_residual_deg(human_q).T])),
        "moving_frac": float(np.mean(moving)),
        "gap_p50_cm": float(np.nanmedian(gaps)) * 100,
        "gap_p90_cm": float(np.nanpercentile(gaps, 90)) * 100,
        "gap_lt2cm_frac": float(np.mean(gaps < 0.02)),
        "away_frac": float(np.mean(hands_away)),
        "away_speed_mm_s": float(np.median(speed[hands_away]) * 1e3) if hands_away.any() else float("nan"),
        "foot_h_p10_cm": float(np.percentile(foot_height, 10) * 100),
        "foot_h_p90_cm": float(np.percentile(foot_height, 90) * 100),
    }


def main() -> None:
    soma = SOMA(identity_model_type="mhr")
    keys = None
    for arg in sys.argv[1:]:
        label, pattern = arg.split("=", 1)
        rows = [sequence_stats(Path(f), soma) for f in sorted(glob.glob(pattern))]
        keys = keys or list(rows[0])
        if label.endswith("+"):
            for f, r in zip(sorted(glob.glob(pattern)), rows, strict=True):
                print(f"  {Path(f).name[:44]:44} " + " ".join(f"{k}={r[k]:.3g}" for k in keys))
        med = {k: float(np.median([r[k] for r in rows])) for k in keys}
        print(f"{label.rstrip('+'):12} n={len(rows):2d} median: " + " ".join(f"{k}={med[k]:.3g}" for k in keys))


if __name__ == "__main__":
    main()
