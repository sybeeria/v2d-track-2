set -euo pipefail
archive=runs/i1screen/frozen-screen
if [ -d "$archive" ]; then
    rm -rf data/screen/tier2
    mkdir -p data/screen
    cp -a "$archive" data/screen/tier2
else
    .venv/bin/python -B scripts/make_tier2_screen.py --jobs 6
    mkdir -p runs/i1screen
    rm -rf runs/i1screen/.staging
    cp -a data/screen/tier2 runs/i1screen/.staging
    mv runs/i1screen/.staging "$archive"
fi
