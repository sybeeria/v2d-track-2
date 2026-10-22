"""Compare sampled object/support clearance on frozen original Tier 2 rest frames."""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
import pandas as pd
from analyze_denoise import EVIDENCE, ROOT, load_cached
from denoise import Params, denoise, plane, rest_mask, support_clearance
from robotic_grounding.retarget.read_soma import SOMA
from scipy.spatial.transform import Rotation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compare-frames', action='store_true')
    args = parser.parse_args()
    soma = SOMA(identity_model_type='mhr')
    rows = []
    for folder in sorted((ROOT / 'data/work/tier2/sequences').iterdir()):
        original = load_cached(folder, soma)
        rest = rest_mask(original, Params())
        before, height = support_clearance(original)
        n, d = plane(folder)
        # Freeze the original surface assignment; do not improve a result by switching supports.
        candidate = rest & ((height > 0) | (before + height <= 0.25))
        for mode in ('world', 'local') if args.compare_frames else ('output',):
            if mode == 'output':
                output = np.load(ROOT / 'data/work/tier2_denoised/sequences' / folder.name / 'poses.npy')
            else:
                p = Params(root_strength=0.5, rest_strength=0.5, propagate_rest=True, vertical_only=True, rotation_frame=mode)
                output = denoise(original, p)[0].object_poses_cv
            direction = np.einsum('tij,i->tj', output[:, :3, :3], n)
            ground_after = (original.mesh_points @ direction.T).min(axis=0) + output[:, :3, 3] @ n + d
            after = ground_after - height
            change = Rotation.from_matrix(output[:, :3, :3] @ original.object_poses_cv[:, :3, :3].transpose(0,2,1)).magnitude()
            row = dict(sequence=folder.name, frame=mode, rest_frames=int(rest.sum()), matched_frames=int(candidate.sum()),
                unmatched_elevated_frames=int((rest & ~candidate).sum()),
                rotation_change_max_deg=float(np.degrees(change).max()))
            for label, values in [('before', before), ('after', after)]:
                row[label+'_signed_clearance_cm'] = float(np.median(values[candidate])*100) if candidate.any() else None
                row[label+'_abs_clearance_cm'] = float(np.median(np.abs(values[candidate]))*100) if candidate.any() else None
                row[label+'_mean_abs_clearance_cm'] = float(np.mean(np.abs(values[candidate]))*100) if candidate.any() else None
                row[label+'_abs_clearance_p90_cm'] = float(np.percentile(np.abs(values[candidate]),90)*100) if candidate.any() else None
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(EVIDENCE / ('rotation_frame_support.csv' if args.compare_frames else 'support_comparison.csv'), index=False)
    print(frame.sort_values('rotation_change_max_deg',ascending=False).head(5).to_string(index=False))
    print('Median of per-episode median absolute sampled clearances (cm):')
    print(frame.groupby('frame')[['before_abs_clearance_cm','after_abs_clearance_cm','before_mean_abs_clearance_cm','after_mean_abs_clearance_cm']].median().to_string())
    print('Rest/matched/unmatched-elevated frame totals:',frame[['rest_frames','matched_frames','unmatched_elevated_frames']].sum().to_dict())


if __name__ == '__main__':
    main()
