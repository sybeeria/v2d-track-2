#!/usr/bin/env python3
"""Tier 1 sentinel scale gate (PROD-READY/MORNING.md), fixed before any GPU result.

PASS only if: every sentinel has a complete k cohort for both checkpoints; no member of either checkpoint exceeds
15 cm MPPE; and at least 2 of the 3 sentinels have mean SP-SR >= 2/3 on the selected checkpoint.
Reads the canary's qualification directory (selection.json from production_repeats.py qualify). HOLD exits 1.
"""
import argparse
import json
import sys
from pathlib import Path

SENTINELS = 3
MPPE_LIMIT_CM = 15
SP_MEAN = 2 / 3
SP_EPISODES = 2


def gate(qualification):
    plan = json.loads((qualification / 'plan.json').read_text())
    reasons = []
    if not (qualification / 'RUN_COMPLETE').exists() or not (qualification / 'selection.json').exists():
        return {'status': 'HOLD', 'reasons': ['cohort incomplete or qualification missing']}
    rows = json.loads((qualification / 'selection.json').read_text())
    if plan['tier'] != 1 or len(rows) != SENTINELS:
        reasons.append(f"expected {SENTINELS} Tier 1 sentinels, found tier {plan['tier']} with {len(rows)}")
    sp_ok = []
    for row in rows:
        metrics = row['member_metrics']
        for step, members in metrics.items():
            if len(members) != plan['repeats']:
                reasons.append(f"{row['name']} {step}: {len(members)} of {plan['repeats']} members")
            for i, member in enumerate(members, 1):
                if member['MPPE'] > MPPE_LIMIT_CM:
                    reasons.append(f"{row['name']} {step} member {i}: MPPE {member['MPPE']:.2f} cm > {MPPE_LIMIT_CM}")
        selected = metrics[str(row['selected_step'])]
        sp = sum(m['SP-SR'] for m in selected) / len(selected)
        sp_ok.append(sp >= SP_MEAN - 1e-9)
        if not sp_ok[-1]:
            reasons.append(f"{row['name']}: mean SP-SR {sp:.3f} < 2/3 on the selected checkpoint")
    if sum(sp_ok) >= SP_EPISODES:
        reasons = [r for r in reasons if 'SP-SR' not in r]
    return {'status': 'HOLD' if reasons else 'PASS', 'reasons': reasons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qualification', type=Path, required=True)
    args = parser.parse_args()
    result = gate(args.qualification)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result['status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
