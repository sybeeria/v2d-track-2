"""Validate a packed Track 2 CSV offline, without contacting Kaggle.

Runs the kit uploader's own validator, then rejects checkpoint mixing within an episode, which the host
scorer rejects but the stock uploader misses (it only checks when an episode_index column exists).
The kit files stay unmodified so the packer/uploader hashes stamped into each row match KIT_MANIFEST.json.
"""

import argparse
import json
import sys
from pathlib import Path

ROW_EPISODE = r'^track_\d+/e(\d{6})/w\d{3}/f\d{6}/o\d+$'


def validate(kit, csv, track):
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
