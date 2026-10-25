"""Validate a packed Track 2 CSV offline, without contacting Kaggle.

Checks every kit file against KIT_MANIFEST.json, runs the kit uploader's own validator, then rejects checkpoint
mixing within an episode, which the host scorer rejects but the stock uploader misses (it only checks when an
episode_index column exists), and stamped packer/uploader hashes that don't name the code that wrote the rows.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROW_EPISODE = r'^track_\d+/e(\d{6})/w\d{3}/f\d{6}/o\d+$'
PACK_NO_CLEAN = Path(__file__).resolve().with_name('pack_no_clean.py')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(kit, csv, track):
    manifest = json.loads((kit / 'KIT_MANIFEST.json').read_text())['files']
    for name, expected in manifest.items():
        # Tiers 2/3 never open the kit packer: it reads the clean reference.
        if name == 'tools/pack_submission.py' and track != 'track_2_tier1':
            continue
        path = kit / name
        if not path.is_file() or digest(path) != expected:
            raise SystemExit(f'{path}: differs from KIT_MANIFEST.json')
    sys.path.insert(0, str(kit))
    from submit.v2d_submit import validate_submission
    frame = validate_submission(csv, track, json.loads((kit / 'config/tracks.json').read_text()))
    episodes = frame['row_id'].astype(str).str.extract(ROW_EPISODE, expand=False)
    if episodes.isna().any():
        raise SystemExit(f'{csv}: malformed row_id')
    counts = frame.groupby(episodes)['checkpoint_sha256'].nunique()
    mixed = sorted(counts[counts > 1].index)
    if mixed:
        raise SystemExit(f"{csv}: 'checkpoint_sha256' must be identical within each episode; episode(s) {mixed[:5]}")
    # Every tier uses the same precision-preserving local builder; stamp its actual bytes.
    packer = str(frame['packer_script_sha256'].iloc[0]).strip().lower()
    allowed = {digest(PACK_NO_CLEAN)}
    if track == 'track_2_tier1':
        allowed.add(manifest['tools/pack_submission.py'])
    if packer not in allowed:
        raise SystemExit(f'{csv}: unknown packer_script_sha256 {packer}')
    for column, expected in [('packer_script_sha256', packer), ('uploader_script_sha256', manifest['submit/v2d_submit.py'])]:
        if not frame[column].astype(str).str.strip().str.lower().eq(expected).all():
            raise SystemExit(f'{csv}: {column} is not {expected}, the code that packs {track}')
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kit', type=Path)
    parser.add_argument('csv', type=Path)
    parser.add_argument('track')
    args = parser.parse_args()
    frame = validate(args.kit, args.csv, args.track)
    print(f'offline uploader validation: {len(frame)} rows -> {args.csv}')


if __name__ == '__main__':
    main()
