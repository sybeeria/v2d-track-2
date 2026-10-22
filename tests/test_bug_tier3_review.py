"""Public-seam CPU regressions; every reference here is an owned synthetic fixture."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(os.environ.get('BUG_TIER3_REPO', Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0, str(ROOT / 'scripts'))
PYTHON = sys.executable


def record(path):
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


class IntegrityTests(unittest.TestCase):
    def manifest(self, work, mode='self'):
        own, clean, support = [work / n for n in ('tier3.parquet', 'clean.parquet', 'support.usda')]
        for p, value in ((own, b'reconstructed'), (clean, b'clean-fixture'), (support, b'#usda 1.0')):
            p.write_bytes(value)
        own_arm = {'training_reference': record(own), 'clean_scoring_reference': record(own),
                   'support': {**record(support), 'enabled': False}}
        entry = {'sequence': 'fixture', 'arms': {'tier3_raw': own_arm}}
        if mode == 'clean':
            own_arm['clean_scoring_reference'] = record(clean)
            entry['clean_scoring_reference'] = record(clean)
            entry['arms']['clean'] = {**own_arm, 'training_reference': record(clean)}
        manifest = work / 'manifest.json'
        manifest.write_text(json.dumps({'scoring_mode': mode, 'sequences': [entry]}))
        return manifest, own, clean

    def command(self, manifest, output, *args):
        return [PYTHON, '-B', str(ROOT / 'scripts/pilot_queue.py'), '--arms',
                str(ROOT / 'scripts/pilot_arms.txt'), '--arm', 'T2C-contact',
                '--manifest', str(manifest), '--inputs', 'tier3_raw', '--steps', '49999872',
                '--out-root', str(output), *args]

    def test_tier3_clean_mode_rejected_before_any_scoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp); manifest, _, clean = self.manifest(work, 'clean')
            result = subprocess.run(self.command(manifest, work/'out', '--dry-run'), capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0, 'Tier 3 dry-run scheduled clean scoring:\n'+result.stdout)
            self.assertNotIn('scripts/score_rollout.py', result.stdout)
            self.assertFalse((work/'out').exists())

    def test_self_mode_rejects_known_tier1_training_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp); manifest, own, _ = self.manifest(work)
            clean = work/'tier_1_multiview_caption/data.parquet'; clean.parent.mkdir(); clean.write_bytes(b'fixture')
            data = json.loads(manifest.read_text()); arm = data['sequences'][0]['arms']['tier3_raw']
            arm['training_reference'] = arm['clean_scoring_reference'] = record(clean)
            manifest.write_text(json.dumps(data))
            result = subprocess.run(self.command(manifest, work/'out', '--dry-run'), capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertFalse((work/'out').exists())

    def test_summarize_only_rejects_scores_from_another_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp); manifest, own, clean = self.manifest(work)
            out = work/'out'; arm = out/'T2C-contact'; result_dir = arm/'tier3_raw/evaluation/fixture/49999872'
            result_dir.mkdir(parents=True)
            from pilot_queue import read_inputs, read_arms
            refs = read_inputs(manifest, None, ['tier3_raw'])
            overrides, _ = read_arms(ROOT/'scripts/pilot_arms.txt', [49999872])['T2C-contact']
            (arm/'recipe.json').write_text(json.dumps({'schema_version':2, 'overrides':overrides, 'references':refs}))
            report = {'episodes':[{'reference':str(clean.resolve()), 'rollout':str((result_dir/'trajectory.parquet').resolve())}],
                      'mean':{'AUC':1., 'SP-SR':1., 'MP-SR':1., 'MPPE':0.}}
            scores = result_dir/'scores.json'; scores.write_text(json.dumps(report))
            result = subprocess.run(self.command(manifest, out, '--summarize-only'), capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0, 'clean scores entered Tier 3 summary:\n'+result.stdout)
            self.assertFalse((out/'summary.json').exists())
            report['episodes'][0]['reference'] = str(own.resolve())
            report['mean']['MPPE'] = 25.; scores.write_text(json.dumps(report))
            result = subprocess.run(self.command(manifest, out, '--summarize-only'), capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads((out/'summary.json').read_text())['candidates'][0]['MPPE'], 25.)

    def test_video_inspection_never_opens_tier1_parquet(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp); tier=work/'data/challenge/track_2/tier_1_multiview_caption'; (tier/'meta').mkdir(parents=True)
            (tier/'meta/info.json').write_text(json.dumps({'data_path':'episode_{episode_index}.parquet'}))
            (tier/'meta/episodes_metadata.jsonl').write_text(json.dumps({'episode_index':0,'sequence_id':'fixture'})+'\n')
            pq.write_table(pa.table({'frame_index':[0,1,2], 'timestamp':[0.,1/30,2/30]}), tier/'episode_0.parquet')
            video=work/'video.mp4'; video.write_bytes(b'video-fixture')
            sibling=lambda name: SimpleNamespace(rfilename=f'track_2/{name}/videos/chunk-000/observation.images.exo_camera/episode_000000.mp4', size=13, lfs=None)
            api=SimpleNamespace(dataset_info=lambda *a,**k:SimpleNamespace(sha='fixture',siblings=[sibling('tier_1_multiview_caption'),sibling('tier_2_synthetic_noise')]))
            probe={'streams':[{'nb_frames':'3','r_frame_rate':'30/1'}], 'frames':[{'best_effort_timestamp_time':str(i/30)} for i in range(3)]}
            argv=['inspect','--download','--episodes','0','--out',str(work/'out')]
            # The original root default is relative; run it against the owned synthetic layout.
            prior=Path.cwd();os.chdir(work)
            try:
                with patch('sys.argv',argv), patch('huggingface_hub.HfApi',return_value=api), \
                     patch('huggingface_hub.hf_hub_download',return_value=str(video)), \
                     patch('subprocess.check_output',return_value=json.dumps(probe)), \
                     patch('pyarrow.parquet.read_table',side_effect=AssertionError('forbidden all-frame Tier 1 parquet open')):
                    runpy.run_path(str(ROOT/'scripts/tier3_inspect.py'),run_name='__main__')
            finally:os.chdir(prior)
            result=json.loads((work/'out/video_inspection.json').read_text())
            self.assertEqual(result[0]['video_frames'],3)
            self.assertEqual(result[0]['sha256'],hashlib.sha256(video.read_bytes()).hexdigest())


class NativeCacheTests(unittest.TestCase):
    def test_changed_command_source_is_not_silently_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp);source=work/'worker.py';source.write_text('from pathlib import Path\nPath(__file__).with_name("artifact").write_text("A")\n')
            command=[PYTHON,'-B',str(ROOT/'scripts/tier3_native/stage.py'),'--root',str(work),'--name','fixture',
                     '--output',str(work/'artifact'),'--',PYTHON,'-B',str(source)]
            first=subprocess.run(command,capture_output=True,text=True);self.assertEqual(first.returncode,0,first.stderr)
            self.assertEqual((work/'artifact').read_text(),'A')
            stat=source.stat();source.write_text(source.read_text().replace('"A"','"B"'));os.utime(source,ns=(stat.st_atime_ns,stat.st_mtime_ns))
            second=subprocess.run(command,capture_output=True,text=True)
            self.assertNotEqual(second.returncode,0,'changed reconstruction code was SKIPped: '+second.stdout)
            self.assertEqual((work/'artifact').read_text(),'A')

    def test_same_size_and_mtime_input_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp);source=work/'input';source.write_bytes(b'RGB-A')
            command=[PYTHON,'-B',str(ROOT/'scripts/tier3_native/stage.py'),'--root',str(work),'--name','fixture',
                     '--input',str(source),'--output',str(work/'artifact'),'--',PYTHON,'-B','-c',
                     'from pathlib import Path; p=Path('+repr(str(work))+'); (p/"artifact").write_bytes((p/"input").read_bytes())']
            first=subprocess.run(command,capture_output=True,text=True);self.assertEqual(first.returncode,0,first.stderr)
            stat=source.stat();source.write_bytes(b'RGB-B');os.utime(source,ns=(stat.st_atime_ns,stat.st_mtime_ns))
            second=subprocess.run(command,capture_output=True,text=True)
            self.assertNotEqual(second.returncode,0,'different RGB input was SKIPped: '+second.stdout)
            self.assertEqual((work/'artifact').read_bytes(),b'RGB-A')


class CoverageTests(unittest.TestCase):
    def test_manifest_covers_30_sequences_and_empty_support(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as tmp:
            bridge=Path(tmp)
            names=['2026-05-06_10-00-19_blue_trash_can_drag_sideways_08']+[f'fixture_ep{i:02d}' for i in range(29)]
            for base in (bridge,bridge/'repaired'):
                for name in names:
                    motion=base/'assets/human_motion_data/whole_body'
                    path=motion/f'soma/sequence_id={name}/robot_name=g1/data.parquet';path.parent.mkdir(parents=True)
                    pq.write_table(pa.Table.from_pylist([{'coord_frame':'robot_base_z_up','object_body_position':[[[0.,0.,.5]],[[.1,0.,.5]]]}]),path)
                    support=motion/f'reconstructed_stage/{name}_support.usda';support.parent.mkdir(exist_ok=True)
                    support.write_text('#usda 1.0\ndef Xform "support_surfaces" {}\n')
            out=bridge/'manifest.json'
            result=subprocess.run([PYTHON,'-B',str(ROOT/'scripts/tier3_manifest.py'),'--bridge',str(bridge),'--out',str(out),'--check-queue'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            entries=json.loads(out.read_text())['sequences']
            self.assertEqual(sorted(e['sequence'] for e in entries),sorted(names))
            self.assertEqual([e['frames'] for e in entries],[2]*30)
            for entry in entries:
                for arm in entry['arms'].values():
                    self.assertEqual(arm['training_reference'],arm['clean_scoring_reference'])
                    self.assertIs(arm['support']['enabled'],False)


if __name__=='__main__':unittest.main(verbosity=2)
