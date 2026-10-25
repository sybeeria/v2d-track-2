"""Frozen GO/NO-GO for idea I1 (T2C-E4 first, T2C-contact when its veto class is strictly better). External screen only.

For each motion and each --pair PRIMARY=CONTROL (arm directory names), collect every final-checkpoint rollout under the
--root directories (exactly the three fresh repeats under repeats/i1; the training evaluation is excluded). Each arm gets the production veto class
(qualify_veto.patch): catastrophe if any member has own-input position >10 cm or MPPE >15 cm, else orientation if
any member >.5 rad, else pass. Rule R takes PRIMARY unless CONTROL's class is strictly better, exactly as
`merge --cohort PRIMARY --rescue CONTROL` does. Successes are a majority of members;
AUC/MPPE are member means, scored against the clean upstream reference (the screen's external truth).
GO iff MP(R) >= MP(C)+2, SP(R) >= SP(C), no clean catastrophe (>15 cm) under R, mean AUC(R) >= mean AUC(C)-.02.
usage: .venv/bin/python -I -B data/brainstorm7/T2-STRATEGY/i1_decide.py --root DIR... --pair A=B...
"""
import argparse
import hashlib
import re
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def fence(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        path = str(Path(os.fsdecode(args[0])).resolve()).lower()
        if any(x in path for x in ('tier_1', '/tier1/', 'pack_submission', 'data/challenge')):
            raise PermissionError(f'forbidden read: {path}')


sys.addaudithook(fence)
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(ROOT / 'scripts'))
from score_rollout import load_reference, load_rollout, score_rollout

FINAL = 49999872


def members(roots, arm, seq):
    found = set()
    for root in roots:
        for p in (root/'repeats/i1').glob(f'*-r*/{arm}/raw_repaired/evaluation/{seq}/{FINAL}/trajectory.parquet'):
            if p.resolve()!=root.resolve()/p.relative_to(root) or p.stat().st_nlink!=1:
                raise SystemExit('separate regular I1 repeat artifacts required')
            found.add(p.resolve())
    repeats=sorted(p.parents[5].name.rsplit('-r',1)[-1] for p in found)
    if repeats != ['01','02','03']:
        raise SystemExit(f'incomplete or extra fresh cohort for {arm} {seq}: expected r01,r02,r03, found {repeats}')
    identities=set()
    for path in found:
        metadata=pq.ParquetFile(path).schema_arrow.metadata or {}
        values=tuple(metadata.get(('flash_chord.'+key).encode(),b'').decode() for key in
                     ('checkpoint_sha256','eval_script_sha256','eval_code_sha256'))
        if any(not re.fullmatch('[0-9a-fA-F]{64}',v) for v in values):
            raise SystemExit('missing I1 checkpoint/evaluator provenance')
        identities.add(values)
    if len(identities)!=1:
        raise SystemExit('mixed I1 checkpoint/evaluator provenance')
    return sorted(found)


def frozen_reference(archive,record):
    parts=Path(record['path']).parts
    anchor=next((i for i in range(len(parts)-2) if parts[i:i+3]==('data','screen','tier2')),None)
    if anchor is None:
        raise SystemExit('I1 reference is outside the frozen external screen')
    path=archive.joinpath(*parts[anchor+3:])
    if archive.resolve() not in path.resolve().parents or hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:
        raise SystemExit(f'{path}: frozen I1 reference SHA-256 mismatch')
    return path


def trained_on_screen(path, seq, arm):
    recorded = json.loads((path.parents[4]/'recipe.json').read_text())['references'][f'raw_repaired/{seq}']
    expected = {'training': arm['training_reference'], 'scoring': arm['clean_scoring_reference'], 'support': arm['support']}
    if (any(recorded[k]['sha256'] != v['sha256'] for k, v in expected.items())
            or recorded['support']['enabled'] != arm['support']['enabled']):
        raise SystemExit(f'{path}: I1 member was trained on inputs other than the frozen screen')
    return path


def assess(path, own_ref, clean_ref):
    mesh = own_ref.parent.parent / 'object/textured_mesh.obj'
    own = score_rollout(path, own_ref, mesh)['metrics']
    truth, _, _, name = load_reference(own_ref, mesh)
    pred = load_rollout(path, len(truth), name)
    position_cm = float(np.linalg.norm(pred[:, :3] - truth[:, :3], axis=1).mean() * 100)
    q = truth[:, 3:] / np.linalg.norm(truth[:, 3:], axis=1, keepdims=True)
    r = pred[:, 3:] / np.linalg.norm(pred[:, 3:], axis=1, keepdims=True)
    angle = float((2 * np.arccos(np.clip(np.abs((q * r).sum(axis=1)), 0, 1))).mean())
    return {'rollout': str(path), 'own_position_cm': position_cm, 'own_orientation_rad': angle, 'own_mppe_cm': own['MPPE'],
            'catastrophe': position_cm > 10 or own['MPPE'] > 15, 'orientation': angle > .5,
            'clean': score_rollout(path, clean_ref, clean_ref.parent.parent / 'object/textured_mesh.obj')['metrics']}


def veto_class(rows):
    return 2 if any(r['catastrophe'] for r in rows) else 1 if any(r['orientation'] for r in rows) else 0


def summary(rows):
    k = len(rows)
    return {'k': k, 'AUC': float(np.mean([r['clean']['AUC'] for r in rows])),
            'MPPE': float(np.mean([r['clean']['MPPE'] for r in rows])),
            'SP': int(sum(r['clean']['SP-SR'] for r in rows) * 2 > k), 'MP': int(sum(r['clean']['MP-SR'] for r in rows) * 2 > k)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, action='append', required=True)
    p.add_argument('--pair', action='append', required=True, help='PRIMARY_ARM=CONTROL_ARM')
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    archives=[root/'frozen-screen' for root in a.root]
    if any(not (p/'manifest.json').is_file() for p in archives):
        raise SystemExit('missing frozen I1 screen; wait for a complete pull')
    manifests=[(p/'manifest.json').read_bytes() for p in archives]
    if len(set(manifests))!=1:
        raise SystemExit('different frozen I1 screens cannot share one decision')
    manifest = json.loads(manifests[0])
    archive=archives[0]
    result = []
    for pair in a.pair:
        primary, control = pair.split('=')
        for s in manifest['sequences']:
            arm = s['arms']['raw_repaired']
            own=frozen_reference(archive,arm['training_reference'])
            clean=frozen_reference(archive,arm['clean_scoring_reference'])
            for label in ('raw_repaired','clean'):
                for key in ('mesh','support'):
                    if key in s['arms'].get(label,{}):
                        frozen_reference(archive,s['arms'][label][key])
            p_rows = [assess(trained_on_screen(x, s['sequence'], arm), own, clean) for x in members(a.root, primary, s['sequence'])]
            c_rows = [assess(trained_on_screen(x, s['sequence'], arm), own, clean) for x in members(a.root, control, s['sequence'])]
            use_primary = not veto_class(c_rows) < veto_class(p_rows)
            chosen = summary(p_rows if use_primary else c_rows)
            result.append({'pair': pair, 'sequence': s['sequence'], 'rule_uses': primary if use_primary else control,
                           'R': chosen, 'C': summary(c_rows), 'primary_members': p_rows, 'control_members': c_rows})
            print(f"{pair:40s} {s['sequence'][20:45]:25s} R={'primary' if use_primary else 'fallback':8s} "
                  f"R AUC {chosen['AUC']:.3f} SP {chosen['SP']} MP {chosen['MP']} MPPE {chosen['MPPE']:6.2f} | "
                  f"C AUC {result[-1]['C']['AUC']:.3f} SP {result[-1]['C']['SP']} MP {result[-1]['C']['MP']} "
                  f"MPPE {result[-1]['C']['MPPE']:6.2f} | k {chosen['k']}/{result[-1]['C']['k']}")
    tot = lambda side, key: sum(r[side][key] for r in result)
    auc = lambda side: float(np.mean([r[side]['AUC'] for r in result]))
    checks = {'MP(R) >= MP(C)+2': tot('R', 'MP') >= tot('C', 'MP') + 2, 'SP(R) >= SP(C)': tot('R', 'SP') >= tot('C', 'SP'),
              'no clean catastrophe under R': all(m['clean']['MPPE'] <= 15 for r in result for m in
                  (r['primary_members'] if r['rule_uses']==r['pair'].split('=')[0] else r['control_members'])),
              'AUC(R) >= AUC(C)-.02': auc('R') >= auc('C') - .02}
    verdict = 'GO' if all(checks.values()) else 'NO-GO'
    print(f"MP {tot('R', 'MP')} vs {tot('C', 'MP')}; SP {tot('R', 'SP')} vs {tot('C', 'SP')}; AUC {auc('R'):.3f} vs {auc('C'):.3f}")
    print(json.dumps(checks), '->', verdict)
    if a.out:
        a.out.write_text(json.dumps({'verdict': verdict, 'checks': checks, 'rows': result}, indent=1))


if __name__ == '__main__':
    main()
