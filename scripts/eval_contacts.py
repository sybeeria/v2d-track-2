"""Choose contact-label parameters on self-corrupted clean sequences (never on challenge Tier 1 data).

Corruption mimics what scripts/noise_stats.py measured on Tier 2 alone: ~1 mm / ~1 deg object
jitter, a slow bounded random walk on the object pose, and a bounded random walk on the human root.
Clean labels use soma_to_g1's 2 cm rule on the clean sequence. Scores are per-hand F1 averaged over
sequences, seeds and noise levels.

Usage: python scripts/eval_contacts.py <clean sequence folder>...
"""

from __future__ import annotations

import itertools
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from contacts import BASELINE_THRESHOLD, DEFAULT_PARAMS, LabelParams, carrying_gap, robust_labels
from hoi import CV_TO_SOMA, Sequence
from robotic_grounding.retarget.read_soma import SOMA
from scipy.spatial.transform import Rotation

# (object translation walk cm, object rotation walk deg, human root walk cm)
LEVELS = {"jitter": (0.0, 0.0, 0.0), "mild": (3.0, 5.0, 2.0), "medium": (6.0, 10.0, 3.0),
          "severe": (12.0, 15.0, 4.0), "extreme": (20.0, 25.0, 5.0)}
JITTER_M, JITTER_DEG = 0.001, 1.0
SEEDS = range(5)


def bounded_walk(rng: np.random.Generator, frames: int, bound: float) -> np.ndarray:
    """Zero-mean 3D random walk whose largest excursion has norm `bound`."""
    walk = np.cumsum(rng.standard_normal((frames, 3)), axis=0)
    walk -= walk.mean(axis=0)
    return walk * bound / np.linalg.norm(walk, axis=1).max()


def corrupt(seq: Sequence, rng: np.random.Generator, level: tuple[float, float, float]) -> Sequence:
    obj_cm, obj_deg, root_cm = level
    frames = len(seq.object_poses_cv)
    poses = seq.object_poses_cv.copy()
    poses[:, :3, 3] += bounded_walk(rng, frames, obj_cm / 100) + rng.normal(0, JITTER_M, (frames, 3))
    rotvec = bounded_walk(rng, frames, np.radians(obj_deg)) + rng.normal(0, np.radians(JITTER_DEG) / np.sqrt(3),
                                                                          (frames, 3))
    poses[:, :3, :3] = Rotation.from_rotvec(rotvec).as_matrix() @ poses[:, :3, :3]
    root_shift_soma = bounded_walk(rng, frames, root_cm / 100) @ CV_TO_SOMA.T
    return replace(seq, object_poses_cv=poses, joints=seq.joints + root_shift_soma[:, None, :])


def f1(pred: np.ndarray, truth: np.ndarray) -> float:
    tp = np.sum(pred & truth)
    denom = pred.sum() + truth.sum()
    return 1.0 if denom == 0 else 2 * tp / denom


def main() -> None:
    soma = SOMA(identity_model_type="mhr")
    clean = [Sequence.load(Path(f), soma) for f in sys.argv[1:]]
    cases = []  # (level, clean labels, corrupted hand gaps, corrupted object-moving mask, corrupted foot gaps)
    for seq in clean:
        truth = seq.hand_gaps() < BASELINE_THRESHOLD
        for (name, level), seed in itertools.product(LEVELS.items(), SEEDS):
            noisy = corrupt(seq, np.random.default_rng(seed), level)
            cases.append((name, truth, noisy.hand_gaps(), noisy.object_moving(), noisy.foot_gaps()))
    for name in LEVELS:  # compare with the per-episode carrying gaps noise_stats.py reports for Tier 2
        cm = [carrying_gap(g, m) * 100 for n, _, g, m, _ in cases if n == name]
        print(f"carrying gap {name:8} median {np.median(cm):5.1f} cm  max {np.max(cm):5.1f} cm")

    def score(label_fn) -> dict[str, float]:
        per_level = {name: [] for name in LEVELS}
        for name, truth, gaps, moving, feet in cases:
            pred = label_fn(gaps, moving, feet)
            per_level[name] += [f1(pred[h], truth[h]) for h in range(len(truth)) if truth[h].any()]
        out = {name: float(np.mean(v)) for name, v in per_level.items()}
        out["mean"] = float(np.mean(list(out.values())))
        return out

    def show(label: str, s: dict[str, float]) -> None:
        print(f"{label:58} " + " ".join(f"{k}={v:.3f}" for k, v in s.items()))

    show("clean labels from clean data (sanity, must be 1.0)",
         {"mean": float(np.mean([f1(s.hand_gaps()[h] < BASELINE_THRESHOLD, s.hand_gaps()[h] < BASELINE_THRESHOLD)
                                 for s in clean for h in range(2)]))})
    show("baseline: soma_to_g1 2 cm rule on corrupted data", score(lambda g, m, f: g < BASELINE_THRESHOLD))
    for tau in (0.04, 0.06, 0.08, 0.10):
        show(f"plain threshold {tau * 100:.0f} cm", score(lambda g, m, f, tau=tau: g < tau))

    grid = [LabelParams(on, extra, med, fill, 5, adaptive)
            for on, extra, med, fill, adaptive in itertools.product(
                (0.02, 0.03, 0.05, 0.08), (0.02, 0.04), (1, 9), (0, 15), (None, 0.01, 0.02, 0.04))]
    results = sorted(((score(lambda g, m, f, p=p: robust_labels(g, m, p, f)), p) for p in grid),
                     key=lambda r: -r[0]["mean"])
    for s, p in results[:6]:
        show(f"on={p.on} off+={p.off_extra} med={p.median_frames} fill={p.fill_frames} adapt={p.adaptive_margin}", s)
    show("DEFAULT_PARAMS", score(lambda g, m, f: robust_labels(g, m, DEFAULT_PARAMS, f)))


if __name__ == "__main__":
    main()
