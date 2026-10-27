"""Oct 26 05_object failures: a fully hidden object recorded fp_centroid_distance=inf, which upstream's own validator rejects."""
import ast
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULES = ROOT / 'external/video_to_data/reconstruction/modules'
PREP = 'v2d_cari4d/lib/cari4d/prep'
EXPORT = PREP + '/run_foundationpose_mhr_export.py'
SAM2 = 'v2d_sam2/lib/sam2_utils.py'
APPLY = ROOT / 'scripts/tier3_native/apply_native_patches.py'

sys.path.insert(0, str(MODULES / 'v2d_cari4d/lib/cari4d'))
try:
    from prep.foundationpose_pose_selection import RejectionReason, filter_reacquisition_candidates, measure_visible_silhouettes
    from prep.mhr_foundationpose_diagnostics import FOUNDATIONPOSE_SELECTION_DTYPES, FOUNDATIONPOSE_SELECTION_FIELDS, validate_foundationpose_selection_arrays
finally:
    sys.path.pop(0)

H, W = 120, 160
OBJECT = np.zeros((H, W), bool); OBJECT[40:80, 60:100] = True


def box(rows, cols):
    mask = np.zeros((H, W), bool); mask[rows, cols] = True
    return mask


# run_foundationpose_mhr_export imports nvdiffrast and FoundationPose, so only this function is loaded from its source.
def selected_metric_values(path):
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == '_selected_metric_values')
    scope = {'np': np}
    exec(compile(ast.Module([node], []), str(path), 'exec'), scope)
    return scope['_selected_metric_values']


def tracked_frames(path):
    """A tracked pose rendered on a visible object, then on the same object fully behind the human (run_foundationpose_mhr_export.py:622-623)."""
    values = selected_metric_values(path)
    visible = measure_visible_silhouettes(OBJECT[None], box(slice(20, 100), slice(0, 40)), OBJECT)
    hidden = measure_visible_silhouettes(OBJECT[None], box(slice(20, 100), slice(40, 120)), OBJECT)
    return [values(visible, 0), values(hidden, 0)]


def validate(frames):
    arrays = {key: np.zeros(len(frames), FOUNDATIONPOSE_SELECTION_DTYPES[key]) for key in FOUNDATIONPOSE_SELECTION_FIELDS}
    for key, name in (('fp_observed_visible_pixels', 'observed_visible_pixels'), ('fp_rendered_visible_pixels', 'rendered_visible_pixels'),
                      ('fp_rendered_visibility', 'rendered_visibility'), ('fp_full_image_iou', 'iou'), ('fp_tolerant_precision', 'precision'),
                      ('fp_tolerant_recall', 'recall'), ('fp_centroid_distance', 'centroid_distance')):
        arrays[key] = np.asarray([frame[name] for frame in frames], FOUNDATIONPOSE_SELECTION_DTYPES[key])
    return validate_foundationpose_selection_arrays(arrays, (len(frames),), 'generated FoundationPose output')


def apply_patches(root):
    return subprocess.run([sys.executable, '-I', str(APPLY)], env={**os.environ, 'NATIVE_MODULES': str(root)}, capture_output=True, text=True)


class HiddenCentroidPatchTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        shutil.copytree(MODULES / PREP, self.root / PREP)
        (self.root / SAM2).parent.mkdir(parents=True)
        shutil.copyfile(MODULES / SAM2, self.root / SAM2)

    def test_unpatched_upstream_fails_on_hidden_frame(self):
        with self.assertRaisesRegex(ValueError, r'^generated FoundationPose output/fp_centroid_distance contains infinity$'):
            validate(tracked_frames(MODULES / EXPORT))

    def test_patched_upstream_records_nan_for_hidden_frame_only(self):
        for _ in range(2):
            result = apply_patches(self.root)
            self.assertEqual(result.returncode, 0, result.stderr)
        before, after = tracked_frames(MODULES / EXPORT), tracked_frames(self.root / EXPORT)
        validated = validate(after)
        self.assertEqual(validated['fp_centroid_distance'][0], 0.0)
        self.assertTrue(np.isnan(validated['fp_centroid_distance'][1]))
        self.assertEqual(after[1]['observed_visible_pixels'], 0)
        self.assertEqual(before[0], after[0])
        self.assertEqual({k: v for k, v in before[1].items() if k != 'centroid_distance'}, {k: v for k, v in after[1].items() if k != 'centroid_distance'})

    def test_patch_leaves_pose_selection_untouched(self):
        result = apply_patches(self.root)
        self.assertEqual(result.returncode, 0, result.stderr)
        changed = sorted(p.relative_to(self.root).as_posix() for p in (self.root / PREP).rglob('*.py') if p.read_bytes() != (MODULES / p.relative_to(self.root)).read_bytes())
        self.assertEqual(changed, [EXPORT])
        def statements(path):
            return [ast.dump(n) for n in ast.parse(path.read_text()).body if getattr(n, 'name', None) != '_selected_metric_values']
        self.assertEqual(statements(self.root / EXPORT), statements(MODULES / EXPORT))
        # Visible object; candidates: behind the human, barely visible, on target.
        human = box(slice(20, 120), slice(0, 60))
        candidates = np.stack([box(slice(40, 100), slice(10, 50)), box(slice(20, 120), slice(0, 60)) | box(slice(85, 90), slice(60, 90)), OBJECT])
        metrics = measure_visible_silhouettes(candidates, human, OBJECT)
        chosen = filter_reacquisition_candidates(metrics, (H, W))
        self.assertEqual(chosen.selected_index, 2)
        self.assertEqual(chosen.keep_mask.tolist(), [False, False, True])
        self.assertTrue(chosen.rejection_masks[0] & RejectionReason.CENTROID)
        before, after = selected_metric_values(MODULES / EXPORT), selected_metric_values(self.root / EXPORT)
        self.assertEqual(before(metrics, 0)['centroid_distance'], float('inf'))
        self.assertTrue(np.isnan(after(metrics, 0)['centroid_distance']))
        for index in (1, 2):
            self.assertEqual(before(metrics, index), after(metrics, index))

    def test_drifted_upstream_fails_setup_without_writing(self):
        target = self.root / EXPORT
        drifted = target.read_text().replace('"centroid_distance": float(metrics.centroid_distance[index])}', '"centroid_distance": float(metrics.centroid_distance[index]) }', 1)
        self.assertNotEqual(drifted, target.read_text())
        target.write_text(drifted)
        result = apply_patches(self.root)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(target.read_text(), drifted)


if __name__ == '__main__':
    unittest.main()
