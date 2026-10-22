"""Regenerate the host-local Tier 3 pilot_queue manifest in its label-free `scoring_mode: self`.

Each input (tier3_raw, tier3_repaired) is scored against its own training reference, never a
released clean trajectory; the manifest has no `clean` input.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'external/video_to_data/robotic_grounding/flash_chord/src'))
from score_rollout import write_json


def file_record(path):
    path = path.resolve(strict=True)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def generate(bridge):
    from flash_chord.assets.registry import support_usda_for_parquet
    from pxr import Usd
    suffix = 'assets/human_motion_data/whole_body/soma'
    raw = {p.parent.parent.name.removeprefix('sequence_id='): p
           for p in (bridge / suffix).glob('sequence_id=*/robot_name=g1/data.parquet')}
    repaired = {p.parent.parent.name.removeprefix('sequence_id='): p
                for p in (bridge / 'repaired' / suffix).glob('sequence_id=*/robot_name=g1/data.parquet')}
    if not raw or raw.keys() != repaired.keys():
        raise ValueError('raw/repaired sequence coverage differs or is empty')
    sequences = []
    for sequence in sorted(raw):
        arms, counts = {}, []
        for label, path in [('tier3_raw', raw[sequence]), ('tier3_repaired', repaired[sequence])]:
            reference = file_record(path)
            rows = pq.ParquetFile(path).read(columns=['coord_frame', 'object_body_position']).to_pylist()
            if len(rows) != 1 or rows[0]['coord_frame'] not in ('world', 'robot_base_z_up'):
                raise ValueError('expected one evaluator-world motion row')
            counts.append(len(rows[0]['object_body_position']))
            support = support_usda_for_parquet(str(path), robot_name='g1')
            if support is None:
                raise FileNotFoundError('support stage did not resolve')
            stage = Usd.Stage.Open(str(support))
            if stage is None:
                raise ValueError('support stage failed to open')
            arms[label] = {'training_reference': reference, 'clean_scoring_reference': reference,
                           'support': {**file_record(support), 'enabled': any(p.GetTypeName() in ('Cube', 'Cylinder') for p in stage.Traverse())}}
        if counts[0] <= 0 or counts[0] != counts[1]:
            raise ValueError('raw and repaired frame grids differ or are empty')
        sequences.append({'sequence': sequence, 'frames': counts[0], 'arms': arms})
    return {'schema_version': 1, 'tier': 3, 'scoring_mode': 'self',
            'paths': 'Absolute paths and hashes regenerated on this host; copy complete bridge/assets and bridge/repaired/assets.',
            'metric_space': 'Each input G1 world frame, source-frame playback, no metric alignment; label-free tracking fidelity only.',
            'sequences': sequences}



def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bridge', type=Path, default=ROOT / 'data/tier3_bridge')
    p.add_argument('--out', type=Path, default=ROOT / 'data/tier3_bridge/manifest.json')
    p.add_argument('--check-queue', action='store_true', help='Fail closed unless the actual queue accepts both self-scoring inputs')
    a = p.parse_args()
    write_json(a.out, generate(a.bridge.resolve()))
    print(a.out.resolve())
    if a.check_queue:
        from pilot_queue import read_inputs
        try:
            jobs = read_inputs(a.out, None, ['tier3_raw', 'tier3_repaired'])
        except (ValueError, KeyError, OSError) as error:
            p.exit(1, f'Queue incompatible with Tier 3 self-scoring: {error}\n')
        if any(job['training'] != job['scoring'] for job in jobs.values()):
            raise ValueError('queue did not preserve Tier 3 self-scoring')


if __name__ == '__main__':
    main()
