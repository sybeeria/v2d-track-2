"""screen_report.py applies the frozen PLAN.md rules to pilot_queue summaries; fixtures cover each rule's edge."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNACK, TRASH, CORN = ('2026-03-06_10-24-18_snack_box_pick_and_place_01', '2026-02-20_17-37-45_blue_trash_can_drag_007',
                      '2026-03-23_18-10-01_corn_can_right_left_handover_01')
HEADER = 'arm\tinput\tsequence\tstep\tAUC\tSP-SR\tMP-SR\tMPPE\n'


def write(root, job, rows):
    path = root / job / 'summary.tsv'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HEADER + ''.join('\t'.join(map(str, r)) + '\n' for r in rows))


def final(arm, input_name, scores):
    return [(arm, input_name, seq, 49999872, *values) for seq, values in zip((SNACK, TRASH, CORN), scores)]


def full(arm, best, last):
    return [(arm, 'clean', SNACK, 330012672, *best), (arm, 'clean', SNACK, 349999104, *last)]


class ScreenReportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as temp:
            pulled = Path(temp) / 'pulled'
            root = pulled / 'testday'  # test_day.sh layout: one directory per job below the run's out-root
            e1 = [(.70, 1, 1, 2.0), (.60, 1, 0, 4.0), (.50, 1, 0, 6.0)]
            write(root, 'E1-clean', final('E1-control', 'clean', e1))
            # E4: better on snack and trash (AUC up, MPPE down), equal successes everywhere -> PASS.
            write(root, 'E4-clean', final('E4-orientation', 'clean', [(.80, 1, 1, 1.0), (.70, 1, 0, 3.0), (.40, 1, 0, 7.0)]))
            # X5: best AUC everywhere but loses snack MP -> fail.
            write(root, 'X5', final('X5-finger', 'clean', [(.90, 1, 0, 1.0), (.90, 1, 1, 1.0), (.90, 1, 1, 1.0)]))
            write(root, 'seed43', final('E1-seed43', 'clean', [(.70, 1, 1, 2.0), (.60, 1, 1, 4.0), (.50, 1, 0, 6.0)]))
            # Noisy: corn input fails SP and MP, trash input fails MP (data/screen/tier2/reference_metrics.csv),
            # so E4 losing corn SP and trash MP is not counted; a new catastrophe on corn still fails the guard.
            write(root, 'E1-raw', final('E1-control', 'raw_repaired', [(.50, 1, 1, 3.0), (.40, 1, 1, 5.0), (.40, 1, 0, 6.0)]))
            write(root, 'T2C', final('T2C-contact', 'raw_repaired', [(.60, 1, 1, 2.0), (.50, 1, 0, 4.0), (.30, 0, 0, 8.0)]))
            write(root, 'E4-raw', final('E4-orientation', 'raw_repaired', [(.60, 1, 1, 2.0), (.50, 1, 0, 4.0), (.10, 0, 0, 50.0)]))
            write(root, 'reeval', [('E1-control', 'clean', SNACK, 49999872, .70, 1, 1, 2.0)])
            write(root, '.', final('E1-control', 'clean', [(0, 0, 0, 0)] * 3))  # merged copy: must be ignored
            write(root, 'F1', full('F1-control', (.46, 1, 0, 4.6), (.38, 1, 0, 5.6)))
            write(root, 'F4', full('F4-orientation', (.86, 1, 1, .8), (.835, 1, 1, .8)))
            # F6 needs final AUC >= F4's + .02 with no lost success: +.019 is not enough.
            write(root, 'F6', full('F6-rebalance-tail', (.90, 1, 1, .7), (.854, 1, 1, .7)))
            # Finger targets on E4 must clear .02 AUC on 2 of 3 sequences: snack +.019 does not count -> fail.
            write(root, 'E4F', final('E4-orientation', 'clean_fingers', [(.819, 1, 1, 1.0), (.73, 1, 0, 3.0), (.40, 1, 0, 7.0)]))
            # T2C-E4 seed 42 beats T2C-contact everywhere; seed 43 adds a corn catastrophe, so the group fails.
            write(root, 'T2CE4', final('T2C-E4', 'raw_repaired', [(.70, 1, 1, 1.5), (.60, 1, 0, 3.0), (.35, 0, 0, 7.0)]))
            write(root, 'T3', [('T2C-contact', 'tier3_repaired', 'ep18', 49999872, .5, 1, 0, 12.0)])
            write(root, 'T2CE4-43', final('T2C-E4-seed43', 'raw_repaired', [(.70, 1, 1, 1.5), (.60, 1, 0, 3.0), (.10, 0, 0, 40.0)]))
            cls.out = subprocess.run([sys.executable, '-I', str(ROOT / 'scripts/screen_report.py'), str(pulled)],
                                     capture_output=True, text=True, check=True).stdout

    def verdict(self, title):
        return next(line for line in self.out.splitlines() if line.startswith(f'## {title}')).split(': ', 1)[1]

    def test_two_of_three_rule(self):
        self.assertEqual(self.verdict('E4 vs E1, clean'), 'PASS')
        self.assertEqual(self.verdict('X5 vs its base E1'), 'fail')

    def test_seed_disagreement_is_reported(self):
        self.assertEqual(self.verdict('E1 seed 43 vs seed 42'), 'seeds disagree on 1 of 3 sequences')

    def test_noisy_rule_ignores_input_failed_gates_but_not_catastrophes(self):
        self.assertEqual(self.verdict('T2C contact vs E1, raw_repaired'), 'PASS')
        self.assertEqual(self.verdict('E4 vs E1, raw_repaired'), 'fail')
        self.assertIn('NEW CATASTROPHE', self.out)

    def test_missing_rows_give_no_verdict(self):
        self.assertEqual(self.verdict('E6 vs its base E4'), 'incomplete')
        self.assertEqual(self.verdict('R4-exactstarts, both seeds'), 'fail or incomplete')

    def test_full_length_rules(self):
        self.assertEqual(self.verdict('350M snack, F4 (E4 objective) vs F1 (stock)'), 'PASS')
        self.assertEqual(self.verdict('350M snack, F6 (E6 objective) vs F4'), 'fail')

    def test_fingers_on_e4_needs_the_margin_on_two_sequences(self):
        self.assertEqual(self.verdict('finger targets on E4 vs E4-clean'), 'fail')

    def test_t2c_e4_needs_both_seeds(self):
        self.assertEqual(self.verdict('T2C-E4 vs T2C-contact, raw_repaired, seed 42'), 'PASS')
        self.assertEqual(self.verdict('T2C-E4 vs T2C-contact, raw_repaired, seed 43'), 'fail')
        self.assertEqual(self.verdict('T2C-E4, both seeds'), 'fail or incomplete')

    def test_tier3_rows_are_listed_with_the_veto(self):
        self.assertIn('T2C-contact tier3_repaired   49999872 AUC 0.500 SP 1 MP 0 MPPE  12.00 VETO', self.out)

    def test_reeval_is_compared_not_merged(self):
        self.assertIn('step 49999872: AUC 0.700 SP 1 MP 1 MPPE   2.00 | first AUC 0.700', self.out)
        self.assertIn('identical', self.out)


if __name__ == '__main__':
    unittest.main()
