"""Sentinel scale gate CLI on qualification outputs shaped like production_repeats.py qualify writes them."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/sentinel_gate.py'
EARLY, FINAL = '330012672', '349999104'


def member(sp=1., mppe=2.):
    return {'AUC': .9, 'SP-SR': sp, 'MP-SR': 1., 'MPPE': mppe}


class SentinelGate(unittest.TestCase):
    def run_gate(self, rows, complete=True):
        with tempfile.TemporaryDirectory() as tmp:
            q = Path(tmp)
            (q / 'plan.json').write_text(json.dumps({'tier': 1, 'repeats': 3}))
            if complete:
                (q / 'RUN_COMPLETE').touch()
            (q / 'selection.json').write_text(json.dumps(rows))
            p = subprocess.run([sys.executable, '-B', str(SCRIPT), '--qualification', str(q)], capture_output=True, text=True)
            return p.returncode, json.loads(p.stdout)

    def row(self, name, final=None, early=None, selected=FINAL):
        return {'name': name, 'selected_step': int(selected),
                'member_metrics': {EARLY: early or [member()] * 3, FINAL: final or [member()] * 3}}

    def test_clean_sentinels_pass(self):
        self.assertEqual(self.run_gate([self.row(n) for n in 'abc']), (0, {'status': 'PASS', 'reasons': []}))

    def test_any_member_of_either_checkpoint_over_15cm_holds(self):
        rows = [self.row('a', early=[member(), member(mppe=15.5), member()]), self.row('b'), self.row('c')]
        self.assertEqual(self.run_gate(rows), (1, {'status': 'HOLD', 'reasons': [f'a {EARLY} member 2: MPPE 15.50 cm > 15']}))

    def test_single_phase_needs_two_of_three_sentinels(self):
        weak = [member(sp=0.), member(sp=0.), member()]
        self.assertEqual(self.run_gate([self.row('a', final=weak), self.row('b'), self.row('c')]),
                         (0, {'status': 'PASS', 'reasons': []}))
        self.assertEqual(self.run_gate([self.row('a', final=weak), self.row('b', final=weak), self.row('c')]),
                         (1, {'status': 'HOLD', 'reasons': ['a: mean SP-SR 0.333 < 2/3 on the selected checkpoint',
                                                            'b: mean SP-SR 0.333 < 2/3 on the selected checkpoint']}))

    def test_incomplete_cohort_holds(self):
        self.assertEqual(self.run_gate([self.row(n) for n in 'abc'], complete=False),
                         (1, {'status': 'HOLD', 'reasons': ['cohort incomplete or qualification missing']}))


if __name__ == '__main__':
    unittest.main()
