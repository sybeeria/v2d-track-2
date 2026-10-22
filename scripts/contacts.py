"""Hand-contact labels that survive noisy hand/object poses, and a CLI that writes them into G1 references.

soma_to_g1.py marks a hand active when a fingertip is within 2 cm of the object mesh. On noisy input
(Tier 2) grasps routinely sit 3-15 cm apart, so most labels vanish and FlashCHORD's force-closure
reward, which is gated by these labels, never fires. These labels use hysteresis on a median-filtered
gap and clean up short gaps and blips. Parameters are chosen by scripts/eval_contacts.py on
self-corrupted non-challenge sequences.

Usage:
    python scripts/contacts.py <work-dir>   # e.g. data/work/tier2; rewrites hand_contact_active in place
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.ndimage import median_filter

BASELINE_THRESHOLD = 0.02  # soma_to_g1.py --contact-threshold default


@dataclass(frozen=True)
class LabelParams:
    on: float  # activate below this gap (m); the floor when adaptive
    off_extra: float  # deactivate above on + off_extra (m)
    median_frames: int  # odd window for the gap median filter
    fill_frames: int  # close inactive holes up to this length
    min_frames: int  # drop active runs shorter than this
    # If set, raise `on` to (median carrying gap of this episode) + margin. The carrying gap is
    # ~1 cm on clean data, so its size measures how far this episode's grasps were pushed apart.
    adaptive_margin: float | None = None


# Best mean F1 in scripts/eval_contacts.py (0.872 vs 0.790 for the 2 cm rule; 0.816 vs 0.488 at the harshest
# level). The adaptive margin matters on Tier 2, whose worst carrying gaps (5-15 cm) exceed anything the
# synthetic corruption produced (max 5 cm), so a fixed threshold tuned there would under-label them.
DEFAULT_PARAMS = LabelParams(on=0.03, off_extra=0.04, median_frames=1, fill_frames=15, min_frames=5,
                             adaptive_margin=0.04)


def carrying_gap(gaps: np.ndarray, moving: np.ndarray) -> float:
    """Median closest-hand gap over frames where the object moves; 0 if it never moves."""
    return float(np.median(gaps.min(axis=0)[moving])) if moving.any() else 0.0


def _runs(mask: np.ndarray, value: bool) -> list[tuple[int, int]]:
    """[start, end) runs where mask == value."""
    padded = np.concatenate([[not value], mask == value, [not value]]).astype(int)
    edges = np.flatnonzero(np.diff(padded))
    return list(zip(edges[::2], edges[1::2], strict=True))


def robust_labels(gaps: np.ndarray, moving: np.ndarray, p: LabelParams,
                  foot_gaps: np.ndarray | None = None) -> np.ndarray:
    """[H, T] bool labels from [H, T] fingertip-to-object gaps and a [T] object-moving mask.

    No adaptation when a foot is closer than any hand to the moving object: it is being kicked or pushed
    by foot, and widening the hand threshold would invent grasps.
    """
    carry = carrying_gap(gaps, moving)
    foot_moved = foot_gaps is not None and carrying_gap(foot_gaps[None], moving) < carry
    adapt = p.adaptive_margin is not None and not foot_moved
    on = max(p.on, carry + p.adaptive_margin) if adapt else p.on
    off = on + p.off_extra
    labels = np.zeros(gaps.shape, dtype=bool)
    for h, gap in enumerate(gaps):
        smooth = median_filter(gap, size=p.median_frames, mode="nearest")
        active = False
        for t, g in enumerate(smooth):
            active = g < on if not active else g <= off
            labels[h, t] = active
        for start, end in _runs(labels[h], False):
            if 0 < start and end < len(gap) and end - start <= p.fill_frames:
                labels[h, start:end] = True
        for start, end in _runs(labels[h], True):
            if end - start < p.min_frames:
                labels[h, start:end] = False
    return labels


def rewrite_reference(parquet: Path, labels: np.ndarray) -> None:
    """Replace hand_contact_active ([1, H, T]) in a motion_v1 G1 reference."""
    table = pq.read_table(parquet)
    old = np.asarray(table.column("hand_contact_active")[0].as_py())
    if old.shape != labels.shape:
        raise ValueError(f"{parquet}: label shape {labels.shape} != reference {old.shape}")
    column = pa.array([labels.astype(np.float32).tolist()], type=table.schema.field("hand_contact_active").type)
    index = table.schema.get_field_index("hand_contact_active")
    pq.write_table(table.set_column(index, table.schema.field(index), column), parquet)


def main() -> None:
    from hoi import Sequence  # heavy: imports SOMA
    from robotic_grounding.retarget.read_soma import SOMA

    work = Path(sys.argv[1])
    soma = SOMA(identity_model_type="mhr")
    for folder in sorted((work / "sequences").iterdir()):
        parquet = work / "motion/whole_body/soma" / f"sequence_id={folder.name}" / "robot_name=g1/data.parquet"
        seq = Sequence.load(folder, soma)
        labels = robust_labels(seq.hand_gaps(), seq.object_moving(), DEFAULT_PARAMS, seq.foot_gaps())
        before = np.asarray(pq.read_table(parquet, columns=["hand_contact_active"]).column(0)[0].as_py())
        rewrite_reference(parquet, labels)
        print(f"{folder.name[:58]:58} L {before[0].mean():.2f}->{labels[0].mean():.2f}  "
              f"R {before[1].mean():.2f}->{labels[1].mean():.2f}")


if __name__ == "__main__":
    main()
