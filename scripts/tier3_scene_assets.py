"""Explicit ep18 asset-only identity audit; never opens either tier's trajectory."""
import argparse
import hashlib
import json
from pathlib import Path

SEQUENCE = '2026-05-06_10-00-19_blue_trash_can_drag_sideways_08'
ASSETS = (f'ground_plane/{SEQUENCE}.json', f'support_surface/{SEQUENCE}_support.usda',
          'mesh/blue_trash_can/blue_trash_can.glb')


def compare_assets(track_root):
    records = []
    for name in ASSETS:
        paths = [track_root / tier / name for tier in ('tier_1_multiview_caption', 'tier_2_synthetic_noise')]
        values = [p.read_bytes() for p in paths]
        if values[0] != values[1]:
            raise ValueError(f'released scene asset differs: {name}')
        records.append({'asset': name, 'byte_identical': True, 'bytes': len(values[0]),
                        'sha256': hashlib.sha256(values[0]).hexdigest(),
                        'comparison_only_tier1': str(paths[0].resolve()), 'tier3_source': str(paths[1].resolve())})
    return {'evidence': 'measured byte comparison; explicit asset-only audit exception; no trajectory read', 'assets': records}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--track-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    result = compare_assets(a.track_root)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2) + '\n')
    print(a.out)


if __name__ == '__main__':
    main()
