"""Validate a frozen episode checkpoint map or build a uniform final-policy map."""

import argparse
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier-root", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--runs-root", type=Path)
    source.add_argument("--checkpoint-map", type=Path,
                        help="decimal episode keys or tournament checkpoint_map with episode_XXXXXX keys")
    parser.add_argument("--steps", type=int, default=349999104)
    parser.add_argument("-o", "--out", type=Path, required=True)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    metadata = [json.loads(line) for line in
                (args.tier_root / "meta/episodes_metadata.jsonl").read_text().splitlines()]
    total = json.loads((args.tier_root / "meta/info.json").read_text())["total_episodes"]
    indices = [row["episode_index"] for row in metadata]
    if sorted(indices) != list(range(total)):
        parser.error("episode metadata must contain each released episode exactly once")
    if len({row["sequence_id"] for row in metadata}) != total:
        parser.error("duplicate sequence IDs in episode metadata")
    mapping = {}
    if args.checkpoint_map:
        try:
            supplied = json.loads(args.checkpoint_map.read_text())
        except (OSError, ValueError) as error:
            parser.error(str(error))
        if isinstance(supplied, dict):
            supplied = supplied.get('checkpoint_map', supplied)
        if not isinstance(supplied, dict) or not supplied:
            parser.error('checkpoint map must be a nonempty JSON object')
        for key, value in supplied.items():
            match = re.fullmatch(r'(0|[1-9][0-9]*)|episode_([0-9]{6})', key)
            if not match or not isinstance(value, str):
                parser.error(f'invalid checkpoint map entry: {key!r}')
            episode = str(int(match[1] or match[2]))
            if episode in mapping:
                parser.error(f'duplicate episode in checkpoint map: {episode}')
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = args.checkpoint_map.resolve().parent / path
            mapping[episode] = str(path.resolve())
        if set(mapping) != set(map(str, indices)):
            parser.error(f'checkpoint map must cover exactly episodes 0 through {total-1}')
    else:
        mapping = {str(row['episode_index']): str((args.runs_root / row['sequence_id'] /
                   f'policy_{args.steps}.safetensors').resolve()) for row in metadata}
    missing = [f'episode {episode}: {path}' for episode, path in mapping.items()
               if not Path(path).is_file() or Path(path).stat().st_size == 0]
    if missing:
        parser.error("missing final policies:\n" + "\n".join(missing))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(sorted(mapping.items(), key=lambda item: int(item[0]))), indent=2) + "\n")
    print(f"{len(mapping)} policies -> {args.out}")


if __name__ == "__main__":
    main()
