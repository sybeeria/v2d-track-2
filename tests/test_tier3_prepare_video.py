"""prepare_video.py names a challenge video for upstream's wild-export contract."""
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/tier3_native/prepare_video.py'


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'needs ffmpeg/ffprobe')
class PrepareVideo(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, self.tmp)
        self.source = self.tmp / 'videos/episode_000005.mp4'; self.source.parent.mkdir()
        subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=c=black:s=32x32:r=30', '-frames:v', '12',
                        '-pix_fmt', 'yuv420p', str(self.source)], check=True)

    def prepare(self, frames):
        output = self.tmp / 'episodes/ep05/input/ep05.0.color.mp4'
        return output, subprocess.run([sys.executable, '-I', str(SCRIPT), str(self.source), str(output), str(frames)],
                                      capture_output=True, text=True)

    def test_output_keeps_the_upstream_name_after_path_resolution(self):
        # Upstream pack_masks calls Path(video).resolve() and requires the .0.color.mp4 suffix; a symlink resolved
        # back to episode_000005.mp4 and failed the Oct 26 reconstruction canary.
        output, result = self.prepare(12)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(output.resolve().name, 'ep05.0.color.mp4')
        self.assertEqual(output.read_bytes(), self.source.read_bytes())

    def test_frame_count_mismatch_is_refused(self):
        output, result = self.prepare(13)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('frame count', result.stderr)
        self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
