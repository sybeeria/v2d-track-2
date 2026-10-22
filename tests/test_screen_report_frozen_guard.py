"""Public-CLI regressions for PLAN.md:187's E4 noisy-input guard.

Set SCREEN_REPORT to an isolated patched script to validate the proposal.
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent/'scripts/screen_report.py').is_file())
SCRIPT = Path(os.environ.get('SCREEN_REPORT', ROOT/'scripts/screen_report.py'))
SEQUENCES = ('2026-03-06_10-24-18_snack_box_pick_and_place_01',
             '2026-02-20_17-37-45_blue_trash_can_drag_007',
             '2026-03-23_18-10-01_corn_can_right_left_handover_01')
TITLE = 'E4 vs E1, raw_repaired (Tier 2/3 objective guard)'

class FrozenNoisyGuard(unittest.TestCase):
    def run_report(self, root):
        return subprocess.run([sys.executable,'-B','-I',str(SCRIPT),str(root)],
                              capture_output=True,text=True,check=True).stdout

    def fixture(self, candidate):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for arm,metrics in [('E1-control',[(.4,0,0,4.)]*3), ('E4-orientation',candidate)]:
                job = root/arm
                job.mkdir()
                text = 'arm\tinput\tsequence\tstep\tAUC\tSP-SR\tMP-SR\tMPPE\n'
                for sequence, score in zip(SEQUENCES,metrics):
                    text += '\t'.join(map(str,(arm,'raw_repaired',sequence,49999872,*score)))+'\n'
                (job/'summary.tsv').write_text(text)
            output = self.run_report(root)
            return next(line for line in output.splitlines() if line.startswith('## '+TITLE+':'))

    def test_success_gain_cannot_substitute_for_both_continuous_gains(self):
        # Snack/trash gain SP but worsen MPPE: PLAN:187 requires lower MPPE too.
        self.assertEqual(self.fixture([(.6,1,0,5.),(.6,1,0,5.),(.3,0,0,5.)]),
                         '## '+TITLE+': fail')

    def test_two_sequences_with_both_gains_pass(self):
        self.assertEqual(self.fixture([(.6,1,0,3.),(.6,1,0,3.),(.3,0,0,5.)]),
                         '## '+TITLE+': PASS')


if __name__ == '__main__':
    unittest.main()
