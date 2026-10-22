#!/usr/bin/env bash
# CPU orchestration uses kit dependencies; training/evaluation use PYTHON (.venv-flash).
set -euo pipefail
PROJECT=$(cd "$(dirname "$0")/.." && pwd)
exec "${PACK_PYTHON:-$PROJECT/.venv/bin/python}" "$PROJECT/scripts/pilot_queue.py" "$@"
