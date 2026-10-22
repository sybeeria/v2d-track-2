"""Paired physical statistics on frozen original Tier 2 masks, plus upstream clean controls."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from analyze_denoise import CLEAN, EVIDENCE, ROOT, load_cached
from denoise import Params, plane, rest_mask, smooth
from hoi import CV_TO_SOMA, FEET, FPS, Sequence
from robotic_grounding.retarget.read_soma import SOMA


def masks(seq: Sequence, gaps: np.ndarray) -> tuple[np.ndarray, ...]:
    speed = np.linalg.norm(np.gradient(smooth(seq.object_poses_cv[:, :3, 3]), axis=0), axis=1) * FPS
    return speed > 0.2, seq.object_moving(), gaps.min(axis=0) > 0.25, rest_mask(seq, Params(), gaps)


def physical_row(seq: Sequence, group: str, frozen: tuple[np.ndarray, ...], gaps: np.ndarray) -> dict:
    moving, surface_moving, away, rest = frozen
    velocity = np.linalg.norm(np.gradient(smooth(seq.object_poses_cv[:, :3, 3]), axis=0), axis=1) * FPS
    n, d = plane(seq.folder)
    foot_height = (seq.joints[:, FEET] @ CV_TO_SOMA @ n + d).min(axis=1)
    return dict(sequence=seq.folder.name, group=group, frames=len(seq.joints),
        moving_frames=int(moving.sum()), away_frames=int(away.sum()), rest_frames=int(rest.sum()),
        surface_moving_frames=int(surface_moving.sum()),
        carrying_gap_cm=float(np.median(gaps.min(axis=0)[moving])*100) if moving.any() else None,
        carrying_surface_gap_cm=float(np.median(gaps.min(axis=0)[surface_moving])*100) if surface_moving.any() else None,
        away_speed_mm_s=float(np.median(velocity[away])*1000) if away.any() else None,
        rest_speed_mm_s=float(np.median(velocity[rest])*1000) if rest.any() else None,
        foot_p10_cm=float(np.percentile(foot_height,10)*100),
        foot_p90_cm=float(np.percentile(foot_height,90)*100))


def main() -> None:
    soma = SOMA(identity_model_type='mhr')
    rows = []
    for folder in sorted((ROOT / 'data/work/tier2/sequences').iterdir()):
        original = load_cached(folder, soma)
        saved = ROOT / 'data/work/tier2_denoised/sequences' / folder.name
        with np.load(folder / 'soma_params.npz') as a, np.load(saved / 'soma_params.npz') as b:
            shift = b['transl'] - a['transl']
        output = replace(original, folder=saved, object_poses_cv=np.load(saved / 'poses.npy'),
                         joints=original.joints + shift[:, None])
        before = original.hand_gaps()
        frozen = masks(original, before)
        for label, seq in [('tier2', original), ('denoised', output)]:
            row = physical_row(seq, label, frozen, before if label == 'tier2' else seq.hand_gaps())
            work = 'tier2' if label == 'tier2' else 'tier2_denoised'
            reference = ROOT / 'data/work' / work / 'motion/whole_body/soma' / f'sequence_id={folder.name}/robot_name=g1/data.parquet'
            ik = np.asarray(pq.ParquetFile(reference).read(columns=['ik_error_per_frame']).column(0)[0].as_py())
            assert ik.shape == (len(seq.joints),) and np.isfinite(ik).all()
            row.update(ik_median_stored=float(np.median(ik)), ik_p95_stored=float(np.percentile(ik,95)),
                       initial_root_shift_mm=float(np.linalg.norm(shift[0])*1000) if label == 'denoised' else 0.0)
            rows.append(row)
    for folder in sorted(CLEAN.glob('sequence_id=*')):
        seq = load_cached(folder, soma)
        gaps = seq.hand_gaps()
        row = physical_row(seq, 'clean', masks(seq, gaps), gaps)
        row['initial_root_shift_mm'] = 0.0
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(EVIDENCE / 'paired_physical_stats.csv', index=False)
    summary = frame.groupby('group').median(numeric_only=True)
    summary.to_csv(EVIDENCE / 'paired_physical_summary.csv')
    print(summary.to_string())
    print('Tier 2 masks frozen from original data. Clean masks use each upstream example. Undefined values remain NaN.')


if __name__ == '__main__':
    main()
