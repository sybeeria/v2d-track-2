#!/usr/bin/env python3
"""Apply the frozen screen decision rules (data/brainstorm2/PLAN.md) to pulled pilot_queue summaries.

  python3 -I scripts/screen_report.py data/testday/pulled data/r4screen/pulled data/r4b/pulled data/r5/pulled

Reads every summary.tsv below the given roots. These are external screen sequences scored against their clean
upstream references, never challenge data. A summary.tsv with per-job summaries below it is test_day.sh's merged
copy and is skipped. Rows from a job directory named 'reeval' are kept apart for the
evaluation-repeatability check. A comparison with missing rows says so and gets no verdict.
"""
import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEQUENCES = {'snack': '2026-03-06_10-24-18_snack_box_pick_and_place_01',
             'trash': '2026-02-20_17-37-45_blue_trash_can_drag_007',
             'corn': '2026-03-23_18-10-01_corn_can_right_left_handover_01'}
FINAL_50M, FINAL_350M = '49999872', '349999104'
CATASTROPHE_CM = 15.0

# title, candidate (arm, input), control (arm, input), rule, group (rule 'r4' passes only if every comparison in its group does)
COMPARISONS = [
    ('E4 vs E1, clean (base objective, 50M half)', ('E4-orientation', 'clean'), ('E1-control', 'clean'), 'two_of_three', None),
    ('E6 vs its base E4, clean', ('E6-rebalance-tail', 'clean'), ('E4-orientation', 'clean'), 'two_of_three', None),
    ('X5 vs its base E1, clean', ('X5-finger', 'clean'), ('E1-control', 'clean'), 'two_of_three', None),
    ('finger targets vs E1-clean', ('E1-control', 'clean_fingers'), ('E1-control', 'clean'), 'two_of_three', None),
    ('E1 seed 43 vs seed 42', ('E1-seed43', 'clean'), ('E1-control', 'clean'), 'seed', None),
    ('E4 seed 43 vs seed 42', ('E4-seed43', 'clean'), ('E4-orientation', 'clean'), 'seed', None),
    ('R4 exact starts, seed 42', ('R4-exactstarts', 'clean'), ('E4-orientation', 'clean'), 'r4', 'R4-exactstarts'),
    ('R4 exact starts, seed 43', ('R4-exactstarts-seed43', 'clean'), ('E4-seed43', 'clean'), 'r4', 'R4-exactstarts'),
    ('R4 closure tail, seed 42', ('R4-closure-tail', 'clean'), ('E4-orientation', 'clean'), 'r4', 'R4-closure-tail'),
    ('R4 closure tail, seed 43', ('R4-closure-tail-seed43', 'clean'), ('E4-seed43', 'clean'), 'r4', 'R4-closure-tail'),
    ('finger targets on E4 vs E4-clean', ('E4-orientation', 'clean_fingers'), ('E4-orientation', 'clean'), 'r4', None),
    ('T2C contact vs E1, raw_repaired', ('T2C-contact', 'raw_repaired'), ('E1-control', 'raw_repaired'), 'noisy', None),
    ('T2C first starts vs E1, raw_repaired', ('T2C-firststarts', 'raw_repaired'), ('E1-control', 'raw_repaired'), 'noisy', None),
    ('E4 vs E1, raw_repaired (Tier 2/3 objective guard)', ('E4-orientation', 'raw_repaired'), ('E1-control', 'raw_repaired'), 'noisy_strict', None),
    ('T2C-E4 vs T2C-contact, raw_repaired, seed 42', ('T2C-E4', 'raw_repaired'), ('T2C-contact', 'raw_repaired'), 'noisy', 'T2C-E4'),
    ('T2C-E4 vs T2C-contact, raw_repaired, seed 43', ('T2C-E4-seed43', 'raw_repaired'), ('T2C-contact', 'raw_repaired'), 'noisy', 'T2C-E4'),
]
# 350M runs on the snack box: title, candidate arm, control arm, minimum final-checkpoint AUC gain
# (None: the E4 rule, candidate >= control by successes, AUC, MPPE at both its best and final checkpoints).
FULL_LENGTH = [
    ('F4 (E4 objective) vs F1 (stock)', 'F4-orientation', 'F1-control', None),
    ('F6 (E6 objective) vs F4', 'F6-rebalance-tail', 'F4-orientation', 0.02),
]
NOISY_INPUTS = ('raw_upstream', 'raw_repaired', 'denoised_upstream', 'denoised_repaired')


def load(roots):
    rows, reeval = {}, {}
    for root in roots:
        for path in sorted(Path(root).rglob('summary.tsv')):
            if any(path.parent.glob('*/summary.tsv')):
                continue  # merged copy written by test_day.sh; the per-job tables are read instead
            target = reeval if path.parent.name == 'reeval' else rows
            with path.open(newline='') as f:
                for r in csv.DictReader(f, delimiter='\t'):
                    key = (r['arm'], r['input'], r['sequence'], r['step'])
                    value = {k: float(r[k]) for k in ('AUC', 'SP-SR', 'MP-SR', 'MPPE')}
                    if key in target and target[key] != value:
                        raise SystemExit(f'conflicting rows for {key} in {path}')
                    target[key] = value
    return rows, reeval


def input_ceiling():
    with (ROOT / 'data/screen/tier2/reference_metrics.csv').open(newline='') as f:
        return {(r['arm'], r['sequence']): {k: float(r[k]) for k in ('AUC', 'SP-SR', 'MP-SR', 'MPPE')}
                for r in csv.DictReader(f)}


def fmt(m):
    return f"AUC {m['AUC']:.3f} SP {m['SP-SR']:.0f} MP {m['MP-SR']:.0f} MPPE {m['MPPE']:6.2f}"


def compare(rows, ceiling, candidate, control, rule):
    lines, lost, better, catastrophes = [], 0, 0, 0
    for short, sequence in SEQUENCES.items():
        a = rows.get((*candidate, sequence, FINAL_50M))
        b = rows.get((*control, sequence, FINAL_50M))
        if a is None or b is None:
            missing = [name for name, m in (('candidate', a), ('control', b)) if m is None]
            return [f'  {short}: MISSING {" and ".join(missing)}'], 'incomplete'
        gates = ['SP-SR', 'MP-SR']
        if rule in ('noisy', 'noisy_strict'):  # credit or blame successes only where the noisy input itself passes the gate
            gates = [g for g in gates if ceiling[(candidate[1], sequence)][g] >= 0.5]
        d_auc, d_mppe = a['AUC'] - b['AUC'], a['MPPE'] - b['MPPE']
        lose = any(a[g] < b[g] for g in gates)
        gain = sum(a[g] for g in gates) > sum(b[g] for g in gates)
        if rule == 'seed':
            note = 'agree' if all(a[g] == b[g] for g in ('SP-SR', 'MP-SR')) else 'DISAGREE'
            lost += note == 'DISAGREE'
        elif rule == 'r4':
            improved = d_auc >= 0.02 or d_mppe <= -1.0
            lost += lose
            better += improved and not lose
            note = 'LOST SUCCESS' if lose else ('improved' if improved else 'no gain')
        else:
            is_better = gain or (not lose and sum(a[g] for g in gates) == sum(b[g] for g in gates) and d_auc > 0 and d_mppe < 0)
            if rule == 'noisy_strict':
                is_better = not lose and d_auc > 0 and d_mppe < 0
            new_catastrophe = a['MPPE'] > CATASTROPHE_CM >= b['MPPE']
            lost += lose
            catastrophes += new_catastrophe
            better += is_better and not lose
            note = 'LOST SUCCESS' if lose else ('better' if is_better else 'not better')
            if new_catastrophe:
                note += ', NEW CATASTROPHE'
            if rule in ('noisy', 'noisy_strict'):
                note += f" (gates counted: {','.join(g[:2] for g in gates) or 'none'})"
        lines.append(f'  {short:5s} cand {fmt(a)} | ctrl {fmt(b)} | dAUC {d_auc:+.3f} dMPPE {d_mppe:+6.2f} | {note}')
    if rule == 'seed':
        verdict = 'seeds agree on every success' if not lost else f'seeds disagree on {lost} of 3 sequences'
    elif rule == 'r4':
        verdict = 'PASS' if not lost and better >= 2 else 'fail'
    else:
        verdict = 'PASS' if not lost and not catastrophes and better >= 2 else 'fail'
    return lines, verdict


def best(rows, arm, sequence):
    found = [(step, m) for (a, i, s, step), m in rows.items() if (a, i, s) == (arm, 'clean', sequence)]
    if not found:
        return None
    key = lambda item: (item[1]['SP-SR'] + item[1]['MP-SR'], item[1]['AUC'], -item[1]['MPPE'])
    return max(found, key=key), dict(found).get(FINAL_350M)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('roots', nargs='+', type=Path)
    args = parser.parse_args()
    rows, reeval = load(args.roots)
    ceiling = input_ceiling()

    order = lambda m: (m['SP-SR'] + m['MP-SR'], m['AUC'], -m['MPPE'])
    for title, candidate, control, margin in FULL_LENGTH:
        a, b = best(rows, candidate, SEQUENCES['snack']), best(rows, control, SEQUENCES['snack'])
        if not (a and b and a[1] and b[1]):
            print(f'## 350M snack, {title}: incomplete\n  MISSING {candidate} or {control} rows\n')
            continue
        pairs = (('best', a[0][1], b[0][1]), ('final', a[1], b[1]))
        if margin is None:
            passed = all(order(x) >= order(y) for _, x, y in pairs)
        else:
            x, y = a[1], b[1]
            passed = x['SP-SR'] >= y['SP-SR'] and x['MP-SR'] >= y['MP-SR'] and x['AUC'] - y['AUC'] >= margin
        print(f'## 350M snack, {title}: {"PASS" if passed else "fail"}')
        for label, x, y in pairs:
            print(f"  {label:5s} cand {fmt(x)} | ctrl {fmt(y)} | dAUC {x['AUC'] - y['AUC']:+.3f}")
        print()

    groups = {}
    for title, candidate, control, rule, group in COMPARISONS:
        lines, verdict = compare(rows, ceiling, candidate, control, rule)
        print(f'## {title}: {verdict}')
        print('\n'.join(lines) + '\n')
        if group:
            groups.setdefault(group, []).append(verdict)
    for group, verdicts in groups.items():
        overall = 'PASS' if verdicts and all(v == 'PASS' for v in verdicts) else 'fail or incomplete'
        print(f'## {group}, both seeds: {overall}')

    print('\n## Tier 2 input: E1 on each noisy input vs the input itself (x = gate the input already fails)')
    for name in NOISY_INPUTS:
        cells = []
        for short, sequence in SEQUENCES.items():
            m, ref = rows.get(('E1-control', name, sequence, FINAL_50M)), ceiling[(name, sequence)]
            if m is None:
                cells.append(f'{short} MISSING')
                continue
            slots = ''.join('x' if ref[g] < 0.5 else ('+' if m[g] >= 0.5 else '-') for g in ('SP-SR', 'MP-SR'))
            catastrophe = ' CATASTROPHE' if m['MPPE'] > CATASTROPHE_CM else ''
            cells.append(f"{short} {slots} dAUC {m['AUC'] - ref['AUC']:+.3f} dMPPE {m['MPPE'] - ref['MPPE']:+6.2f}{catastrophe}")
        print(f'  {name:17s} ' + ' | '.join(cells))

    print('\n## Tier 3, label-free: scored against its own input, never a clean trajectory (veto: final MPPE > 10 cm)')
    tier3 = sorted((key, m) for key, m in rows.items() if key[1].startswith('tier3_'))
    if not tier3:
        print('  MISSING tier3 rows')
    for (arm, input_name, sequence, step), m in tier3:
        veto = ' VETO' if step == FINAL_50M and m['MPPE'] > 10 else ''
        print(f'  {arm} {input_name:15s} {step:>9s} {fmt(m)}{veto}')

    print('\n## Evaluation repeatability: re-evaluated checkpoints')
    if not reeval:
        print('  MISSING reeval rows')
    for key, m in sorted(reeval.items()):
        first = rows.get(key)
        same = first is not None and all(abs(first[k] - m[k]) < 1e-9 for k in m)
        print(f"  step {key[3]}: {fmt(m)} | first {fmt(first) if first else 'MISSING'} | {'identical' if same else 'DIFFERENT'}")


if __name__ == '__main__':
    main()
