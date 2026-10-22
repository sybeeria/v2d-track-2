"""Load every retargeted G1 reference with FlashCHORD's own loader and print per-episode stats.

Fails if any reference does not load, has non-finite values, or has a frame count that
differs from its source sequence. Needs flash_chord's src on PYTHONPATH.

Usage:
    PYTHONPATH=external/video_to_data/robotic_grounding/flash_chord/src \
        python scripts/check_references.py data/work/tier1 [data/work/tier2 ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from flash_chord.data.motion_v1 import load_motion_v1

# The G1 recipe's playback settings (configs/sim/recon_body.yaml, experiment/g1_recon_body_flash_sac.yaml).
CONTROL_FPS = 50.0


def check_work_dir(work: Path) -> int:
    failures = 0
    parquets = sorted(work.glob("motion/whole_body/soma/sequence_id=*/robot_name=g1/data.parquet"))
    sources = {p.name for p in (work / "sequences").glob("*") if p.is_dir()}
    written = {p.parent.parent.name.removeprefix("sequence_id=") for p in parquets}
    if not sources:
        print(f"  FAIL {work}: no source sequences")
        failures += 1
    for sequence in sorted(sources - written):
        print(f"  FAIL {sequence}: missing G1 reference")
        failures += 1
    print(f"{work}: {len(parquets)} references for {len(sources)} source sequences")
    print(f"  {'sequence':58} {'T':>4} {'ik med':>6} {'ik p95':>6} {'contact L/R':>11} {'obj path m':>10}")
    for parquet in parquets:
        sequence = parquet.parent.parent.name.removeprefix("sequence_id=")
        try:
            reference = load_motion_v1(str(parquet), control_fps=CONTROL_FPS, source_frame_playback=True)
            row = pq.read_table(parquet, columns=["ik_error_per_frame", "hand_contact_active",
                                                  "object_root_position", "robot_joint_positions"])
            ik = np.asarray(row.column("ik_error_per_frame")[0].as_py())
            contact = np.asarray(row.column("hand_contact_active")[0].as_py(), dtype=float)
            obj = np.asarray(row.column("object_root_position")[0].as_py())
            joints = np.asarray(row.column("robot_joint_positions")[0].as_py())
            source_frames = len(np.load(work / "sequences" / sequence / "poses.npy", mmap_mode="r"))
            problems = []
            if reference.num_frames != source_frames:
                problems.append(f"{reference.num_frames} loaded frames vs {source_frames} in source")
            if len(ik) != source_frames:
                problems.append(f"{len(ik)} frames vs {source_frames} in source")
            if not (np.isfinite(obj).all() and np.isfinite(joints).all()):
                problems.append("non-finite values")
        except Exception as error:  # noqa: BLE001 - report every broken reference, then fail
            problems = [f"{type(error).__name__}: {error}"]
        if problems:
            failures += 1
            print(f"  FAIL {sequence}: {'; '.join(problems)}")
            continue
        path_length = float(np.linalg.norm(np.diff(obj, axis=0), axis=-1).sum())
        print(f"  {sequence[:58]:58} {len(ik):4d} {np.median(ik):6.3f} {np.percentile(ik, 95):6.3f} "
              f"{contact[0].mean():5.2f}/{contact[1].mean():4.2f} {path_length:10.2f}")
    return failures


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: check_references.py WORK_DIR [WORK_DIR ...]")
    failures = sum(check_work_dir(Path(arg)) for arg in sys.argv[1:])
    print(f"{failures} failing references")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
