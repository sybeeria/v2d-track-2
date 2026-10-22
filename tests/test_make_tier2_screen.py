"""Archive I/O regression checks for the screen generator; no trajectory inputs."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from make_tier2_screen import save_params


class ScreenArchiveTest(unittest.TestCase):
    def test_preserves_array_names_that_collide_with_savez_options(self):
        params = {'allow_pickle': np.array(False), 'file': np.array([1, 2]),
                  'poses': np.arange(6, dtype=np.float32).reshape(2, 3), 'unit': np.array('meters')}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'soma_params.npz'
            save_params(path, params)
            with np.load(path, allow_pickle=False) as actual:
                self.assertEqual(set(actual.files), set(params))
                for key, value in params.items():
                    self.assertEqual(actual[key].dtype, value.dtype)
                    np.testing.assert_array_equal(actual[key], value)
            before = path.read_bytes()
            save_params(path, params)
            self.assertEqual(path.read_bytes(), before)

    def test_updates_existing_arrays(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'soma_params.npz'
            save_params(path, {'transl': np.zeros((2, 3))})
            save_params(path, {'transl': np.ones((2, 3))})
            with np.load(path, allow_pickle=False) as actual:
                np.testing.assert_array_equal(actual['transl'], np.ones((2, 3)))


if __name__ == '__main__':
    unittest.main()
