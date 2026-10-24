"""Tier 1 checkpoint choice: the final stays unless the earlier checkpoint is clearly better and no worse elsewhere."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from production_repeats import tier1_choice  # noqa: E402

EARLY, FINAL = 330012672, 349999104


def cohort(auc, sp=1., mp=1., mppe=2.):
    return [{'AUC': auc, 'SP-SR': sp, 'MP-SR': mp, 'MPPE': mppe}] * 3


class Tier1Choice(unittest.TestCase):
    def test_small_auc_lead_keeps_final(self):
        self.assertEqual(tier1_choice({EARLY: cohort(.94), FINAL: cohort(.90)}, FINAL), FINAL)

    def test_clear_auc_lead_with_no_other_loss_picks_earlier(self):
        self.assertEqual(tier1_choice({EARLY: cohort(.96, mppe=1.5), FINAL: cohort(.90)}, FINAL), EARLY)

    def test_lost_success_or_worse_mppe_keeps_final(self):
        lost = [{'AUC': .99, 'SP-SR': 1., 'MP-SR': 1., 'MPPE': 1.}] * 2 + [{'AUC': .99, 'SP-SR': 0., 'MP-SR': 1., 'MPPE': 1.}]
        self.assertEqual(tier1_choice({EARLY: lost, FINAL: cohort(.80)}, FINAL), FINAL)
        self.assertEqual(tier1_choice({EARLY: cohort(.99, mppe=2.1), FINAL: cohort(.80)}, FINAL), FINAL)


if __name__ == '__main__':
    unittest.main()
