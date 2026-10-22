"""The alignment fixture lifecycle must never publish files into shared data/."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class SharedCheckoutTest(unittest.TestCase):
    def test_alignment_report_and_scratch_stay_in_temporary_directory(self):
        import test_tier3_align as alignment
        with tempfile.TemporaryDirectory(prefix='INTEGRATE-alignment-', dir='/tmp') as temp:
            root = Path(temp)
            (root / 'external').symlink_to(ROOT / 'external', target_is_directory=True)
            class Lifecycle(alignment.Tier3AlignmentTest):
                pass
            with patch.object(alignment, 'ROOT', root), patch(
                    'robotic_grounding.retarget.read_soma.SOMA', return_value=object()):
                Lifecycle.setUpClass()
                work = Lifecycle.work
                try:
                    Lifecycle.tearDownClass()
                    self.assertTrue((work / 'alignment_test_results.json').is_file())
                finally:
                    Lifecycle.doClassCleanups()
            self.assertFalse((root / 'data').exists(), 'test lifecycle wrote into shared data/')
            self.assertFalse(work.exists(), 'test lifecycle left its scratch directory behind')
            self.assertFalse(hasattr(Lifecycle, 'soma'), 'test lifecycle retained its body model')
