#!/usr/bin/env python3
"""Copy and seal exact Tier 3 converter poses before the one-shot export.

Uses only Tier 2 metadata, reconstructed poses and self-scoring training references.
Run `verify` again on the export host after transferring the complete work tree.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from ep18_pilot import LABEL as EP18_LABEL, SEQUENCE as EP18_SEQUENCE
from pack_no_clean import convert_frame


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def release_inputs(manifest, tier_root):
    content = json.loads(manifest.read_text())
    if (content.get('tier'), content.get('scoring_mode')) != (3, 'self'):
        raise ValueError('requires a Tier 3 self-scoring release manifest')
    metadata = [json.loads(line) for line in (tier_root / 'meta/episodes_metadata.jsonl').read_text().splitlines()]
    info = json.loads((tier_root / 'meta/info.json').read_text())
    roster = {r['sequence_id']: r for r in metadata}
    if (len(roster) != len(metadata) or
            sorted(r['episode_index'] for r in metadata) != list(range(info['total_episodes']))):
        raise ValueError('invalid released episode roster')
    entries = content['sequences']
    if len(entries) != len(roster) or {e['sequence'] for e in entries} != set(roster):
        raise ValueError('release must contain every sequence exactly once')
    result = {}
    for entry in entries:
        sequence = entry['sequence']
        row = roster[sequence]
        if not re.fullmatch(r'[A-Za-z0-9_-]+', sequence) or entry['frames'] != row['frames']:
            raise ValueError('release sequence/frame count differs from the roster')
        (label, arm), = entry['arms'].items()
        reference = arm['training_reference']
        if arm['clean_scoring_reference'] != reference or digest(reference['path']) != reference['sha256']:
            raise ValueError('self-scoring training reference changed')
        result[sequence] = {'episode': row['episode_index'], 'frames': row['frames'],
                            'label': label, 'reference': reference}
    return result


def check_frame(poses_path, entry):
    poses = np.load(poses_path, allow_pickle=False)
    if poses.shape != (entry['frames'], 4, 4) or not np.isfinite(poses).all():
        raise ValueError('converter pose frame count/nonfinite mismatch')
    rotation = poses[:, :3, :3]
    if (not np.allclose(poses[:, 3], [0, 0, 0, 1], rtol=0, atol=1e-6) or
            not np.allclose(rotation.transpose(0, 2, 1) @ rotation, np.eye(3), rtol=0, atol=1e-5) or
            not np.allclose(np.linalg.det(rotation), 1, rtol=0, atol=1e-5)):
        raise ValueError('converter poses are not rigid transforms')
    rows = pq.ParquetFile(entry['reference']['path']).read(
        columns=['object_body_position', 'object_body_wxyz']).to_pylist()
    if len(rows) != 1:
        raise ValueError('expected one training-reference row')
    pos = np.asarray(rows[0]['object_body_position'], float)
    quat = np.asarray(rows[0]['object_body_wxyz'], float)
    if pos.shape != (entry['frames'], 1, 3) or quat.shape != (entry['frames'], 1, 4):
        raise ValueError('training-reference frame grid differs')
    evaluator = np.c_[pos[:, 0], quat[:, 0, 1:], quat[:, 0, :1]]
    target = np.c_[poses[:, :3, 3], Rotation.from_matrix(rotation).as_quat()]
    # The packer's actual admission function, without evaluating or scoring anything.
    convert_frame(evaluator, evaluator, target)


def verify(manifest, tier_root, work):
    entries = release_inputs(manifest, tier_root)
    receipt = json.loads((work / 'work_manifest.json').read_text())
    if receipt['manifest_sha256'] != digest(manifest) or set(receipt['sequences']) != set(entries):
        raise ValueError('work tree belongs to a different merged manifest')
    expected = {'work_manifest.json'}
    for sequence, entry in entries.items():
        rel = f'sequences/{sequence}/poses.npy'
        expected.add(rel)
        record = receipt['sequences'][sequence]
        if (record['reference_sha256'] != entry['reference']['sha256'] or
                digest(work / rel) != record['poses_sha256']):
            raise ValueError('sealed converter poses/reference changed: ' + sequence)
        check_frame(work / rel, entry)
    if any(p.is_symlink() for p in work.rglob('*')) or {
            str(p.relative_to(work)) for p in work.rglob('*') if p.is_file()} != expected:
        raise ValueError('work tree inventory changed')
    return len(entries)


def assemble(manifest, tier_root, work, bridge, ep18_work):
    entries = release_inputs(manifest, tier_root)
    covered = json.loads((bridge / 'coverage_manifest.json').read_text())
    if (covered.get('tier'), covered.get('scoring_mode')) != (3, 'self'):
        raise ValueError('requires the bridge coverage manifest')
    rows = covered['sequences']
    by_sequence = {r['sequence']: r for r in rows}
    if len(by_sequence) != len(rows):
        raise ValueError('duplicate bridge sequence')
    records, sources = {}, {}
    for sequence, entry in entries.items():
        if entry['episode'] == 18:
            if (sequence, entry['label']) != (EP18_SEQUENCE, EP18_LABEL):
                raise ValueError('episode 18 must be the registered candidate_v1')
            source_root = ep18_work
        else:
            row = by_sequence[sequence]
            if (row['episode'], row['frames']) != (entry['episode'], entry['frames']) or row['gate']['status'] not in ('PASS', 'HOLD'):
                raise ValueError('bridge coverage identity/status differs')
            if row['arms'][entry['label']]['training_reference']['sha256'] != entry['reference']['sha256']:
                raise ValueError('merged release trained a different bridge reference')
            source_root = bridge / f"ep{entry['episode']:02d}"
        source = source_root / 'sequences' / sequence / 'poses.npy'
        reference = source_root / f'assets/human_motion_data/whole_body/soma/sequence_id={sequence}/robot_name=g1/data.parquet'
        if digest(reference) != entry['reference']['sha256']:
            raise ValueError('source reference differs from the merged release')
        if entry['episode'] != 18:
            stage = {'tier3_repaired': 'repair', 'tier3_rest_repaired': 'video_rest_repair'}[entry['label']]
            completed = json.loads((source_root / f'.{stage}.done.json').read_text())['artifacts']
            suffix = f'/sequences/{sequence}/poses.npy'
            hashes = [sha for path, size, sha in completed if path.endswith(suffix)]
            if hashes != [digest(source)]:
                raise ValueError('converter poses differ from the completed bridge stage')
        check_frame(source, entry)
        sources[sequence] = source
        records[sequence] = {'episode': entry['episode'], 'frames': entry['frames'],
            'source': str(source.resolve()), 'poses_sha256': digest(source),
            'reference_sha256': entry['reference']['sha256']}
    receipt = {'schema_version': 1, 'manifest_sha256': digest(manifest), 'sequences': records}
    if work.exists():
        if json.loads((work / 'work_manifest.json').read_text()) != receipt:
            raise ValueError('existing work tree differs; choose a fresh destination')
        return verify(manifest, tier_root, work)
    work.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.tier3-work-', dir=work.parent) as temporary:
        staged = Path(temporary) / 'work'
        staged.mkdir()
        for sequence, source in sources.items():
            target = staged / 'sequences' / sequence / 'poses.npy'
            target.parent.mkdir(parents=True)
            shutil.copyfile(source, target)
        (staged / 'work_manifest.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
        count = verify(manifest, tier_root, staged)
        staged.rename(work)
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    for action in ('assemble', 'verify'):
        command = sub.add_parser(action)
        for name in ('manifest', 'tier-root', 'work'):
            command.add_argument('--' + name, type=Path, required=True)
        if action == 'assemble':
            command.add_argument('--bridge', type=Path, required=True)
            command.add_argument('--ep18-work', type=Path, required=True)
    args = parser.parse_args()
    try:
        count = assemble(args.manifest, args.tier_root, args.work, args.bridge, args.ep18_work) if args.action == 'assemble' else verify(args.manifest, args.tier_root, args.work)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f'error: {error}\n')
    print(f'PASS: {count} exact converter files sealed; all packer frame checks pass')


if __name__ == '__main__':
    main()
