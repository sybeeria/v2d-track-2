"""CPU check of Dex3 finger retargeting on an upstream example, never challenge trajectories."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SNACK = "2026-03-06_10-24-18_snack_box_pick_and_place_01"
SOURCE = ROOT / "data/screen/tier2" / SNACK / "clean"  # built by scripts/make_tier2_screen.py
URDF = ROOT / "external/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/urdfs/g1/main_with_hand.urdf"
REFERENCE = f"motion/whole_body/soma/sequence_id={SNACK}/robot_name=g1/data.parquet"


class FingerTargetsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name) / "work"
        subprocess.run([sys.executable, str(ROOT / "scripts/finger_targets.py"), "tree", str(SOURCE), str(cls.out), "--jobs", "1"],
                       check=True, capture_output=True)
        cls.old = pq.read_table(SOURCE / REFERENCE)
        cls.new = pq.read_table(cls.out / REFERENCE)
        cls.names = cls.old.column("robot_joint_names")[0].as_py()
        cls.fingers = [i for i, n in enumerate(cls.names) if "_hand_" in n]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def joints(self, table):
        return np.asarray(table.column("robot_joint_positions")[0].as_py())

    def test_only_finger_targets_change(self):
        for name in self.old.column_names:
            if name != "robot_joint_positions":
                self.assertTrue(self.old.column(name).equals(self.new.column(name)), name)
        old, new = self.joints(self.old), self.joints(self.new)
        body = [i for i in range(old.shape[1]) if i not in self.fingers]
        np.testing.assert_array_equal(new[:, body], old[:, body])
        np.testing.assert_array_equal(old[:, self.fingers], 0.0)
        self.assertGreater(np.abs(new[:, self.fingers]).max(), 0.5)

    def test_targets_stay_within_urdf_limits(self):
        limits = {j.get("name"): j.find("limit") for j in ET.parse(URDF).findall("joint")}
        new = self.joints(self.new)
        for i in self.fingers:
            limit = limits[self.names[i]]
            self.assertGreaterEqual(new[:, i].min(), float(limit.get("lower")) - 1e-6, self.names[i])
            self.assertLessEqual(new[:, i].max(), float(limit.get("upper")) + 1e-6, self.names[i])

    def test_right_hand_tips_move_toward_object_during_contact(self):
        report = json.loads((self.out / "finger_report.json").read_text())["episodes"][0]
        right = next(h for h in report["hands"] if h["side"] == "right")
        self.assertLess(right["retargeted"]["min_tip_gap_p50_m"], right["open"]["min_tip_gap_p50_m"] - 0.003)
        self.assertLess(right["retargeted"]["min_tip_gap_gt2cm_fraction"], right["open"]["min_tip_gap_gt2cm_fraction"])

    def test_support_stage_and_sequence_copied_beside_reference(self):
        self.assertTrue((self.out / "motion/whole_body/reconstructed_stage" / f"{SNACK}_support.usda").is_file())
        self.assertTrue((self.out / "sequences" / SNACK / "poses.npy").is_file())


if __name__ == "__main__":
    unittest.main()
