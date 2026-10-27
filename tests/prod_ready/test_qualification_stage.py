"""Qualification-only staging through the real launch, box and watchdog, offline."""
import collections
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from .fixture import BASE, Fixture, put, sha, wait

READY = Path('data/brainstorm6/PROD-READY')


class QualificationStage(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='qualification-stage-', dir='/tmp')
        self.addCleanup(temporary.cleanup)
        self.f = Fixture(Path(temporary.name).resolve())
        self.addCleanup(self.f.stop_box)
        self.stage_script = self.f.controller/READY/'stage_qualification.py'
        self.stage_script.parent.mkdir(parents=True)
        shutil.copytree(BASE/READY/'generated/arms', self.f.controller/READY/'generated/arms')
        if (BASE/READY/'stage_qualification.py').exists():
            shutil.copyfile(BASE/READY/'stage_qualification.py', self.stage_script)

    def test_empty_outer_queue_finishes_without_training_or_evaluation(self):
        run = self.f.work/'empty'; run.mkdir()
        put(run/'jobs.txt', '# Qualification is owned by post.sh.\n')
        result = self.f.command(['bash', 'scripts/test_day.sh', '1', str(run/'output')],
                                {'JOBS_FILE': str(run/'jobs.txt')})
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertEqual((run/'output/summary.tsv').read_text(), '')
        dry = self.f.command(['bash', 'scripts/test_day.sh', '1', str(run/'dry'), '--dry-run'],
                            {'JOBS_FILE': str(run/'jobs.txt')})
        self.assertEqual(dry.returncode, 0, dry.stdout+dry.stderr)
        self.assertFalse((run/'dry').exists())
        self.assertFalse(self.f.events.exists())
        # Empty queues still honor the public runner lock and preserve another writer's output.
        import fcntl
        summary = run/'output/summary.tsv'; summary.write_text('active writer\n')
        with (run/'output/.runner.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            locked = self.f.command(['bash','scripts/test_day.sh','1',str(run/'output')],
                                    {'JOBS_FILE':str(run/'jobs.txt')})
        self.assertNotEqual(locked.returncode,0)
        self.assertEqual(summary.read_text(),'active writer\n')

    def trained_source(self):
        run, _, _ = self.f.run_spec(1, 'trained', episodes=(1,8,16))
        jobs = run/'jobs.txt'
        jobs.write_text(jobs.read_text().replace('t1-ep1|','t1-ep01|').replace('t1-ep8|','t1-ep08|'))
        # Include these fixture episodes in the fake public scoring grid.
        import pandas as pd
        kit = self.f.remote/'external/kit/v2d_submission_kit'
        sample = pd.read_csv(kit/'data/track_2_sample_submission.csv').iloc[:10]
        frames = []
        for episode in (1,8,16):
            frame = sample.copy()
            frame.row_id = [f'track_2/e{episode:06d}/w000/f{i:06d}/o0' for i in range(10)]
            frames.append(frame)
        pd.concat(frames).to_csv(kit/'data/track_2_sample_submission.csv',index=False)
        manifest = run/'payload/data/fixture/manifest1.json'
        manifest.write_text(json.dumps(json.loads(manifest.read_text()),indent=2)+'\n')
        self.f.launch(run); self.f.finished(run)
        return run, self.f.remote/'runs/trained', manifest

    def stage(self, original, source, template, name='t1q1', ok=True):
        result = self.f.command([sys.executable, '-B', str(self.stage_script), 'stage',
            '--source', str(source), '--jobs', str(original/'jobs.txt'),
            '--template', str(template), '--template-root', str(self.f.remote),
            '--box-root', str(self.f.remote), '--tier-root', str(self.f.remote/'data/challenge/track_2/tier_1_multiview_caption'),
            '--name', name])
        self.assertEqual(result.returncode == 0, ok, result.stdout+result.stderr)
        return self.f.controller/READY/'runs'/name

    def test_stage_launch_qualify_and_pull_exactly_eighteen_fresh_evaluations(self):
        original, source, template = self.trained_source()
        before = {str(p.relative_to(source)):sha(p) for p in source.rglob('*') if p.is_file()}
        prior_calls = len(self.f.calls())
        run = self.stage(original, source, template)
        self.stage(original, source, template, ok=False)
        self.assertFalse((run/'pulled/t1q1/qualification').exists())
        # Like LaunchCommit's tests, model a committed controller snapshot; never commit in this test.
        commit = subprocess.check_output(['git','-C',str(BASE),'rev-parse','HEAD'],text=True).strip()
        git = self.f.work/'bin/git'
        put(git, f'#!{sys.executable}\nimport subprocess,sys\n'
            f'if sys.argv[1:3]!=["-C",{str(self.f.controller)!r}]:sys.exit(subprocess.run([{shutil.which("git")!r},*sys.argv[1:]]).returncode)\n'
            f'elif sys.argv[-2:]==["rev-parse","HEAD"]:print({commit!r})\n'
            'elif sys.argv[-2:]==["status","--porcelain"]:pass\n'
            f'else:sys.exit(subprocess.run([{shutil.which("git")!r},"-C",{str(BASE)!r},*sys.argv[3:]]).returncode)\n')
        git.chmod(0o755)
        result = self.f.command(['bash','scripts/vast/start_run.sh',str(run),'124','7','1'])
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        pid = int((run/'controller-watchdog.pid').read_text())
        try:
            self.f.finished(run)
            wait(self.f.home/'periodic-pull',30)
            (self.f.home/'continue-cycle').touch()
            wait(run/'pull.success.json',30)
            pulled = run/'pulled/t1q1/qualification'
            remote = self.f.remote/'runs/t1q1/qualification'
            for name in ('selected.json','selection.json','export-binding.json'):
                self.assertEqual((pulled/name).read_bytes(), (remote/name).read_bytes())
            self.assertEqual(set(json.loads((pulled/'selected.json').read_text())), {'1','8','16'})
            self.assertEqual([r['members'] for r in json.loads((pulled/'selection.json').read_text())], [3]*3)
            self.assertEqual(len(list(pulled.rglob('trajectory.parquet'))), 18)
            calls = self.f.calls()[prior_calls:]
            self.assertEqual(collections.Counter(c['kind'] for c in calls), {'evaluate':18})
            self.assertEqual(collections.Counter(c['step'] for c in calls), {330012672:9,349999104:9})
            self.assertEqual((self.f.remote/'LAUNCH_COMMIT').read_text(), commit+' changed_files=0\n')
            self.assertEqual({str(p.relative_to(source)):sha(p) for p in source.rglob('*') if p.is_file()}, before)
            self.assertEqual((remote/'plan.json').read_text().count('"tier": 1'), 1)
            frozen_hash = json.loads((source/'frozen-inputs/manifest.json').read_text())['input_manifest_sha256']
            self.assertEqual(json.loads((remote/'plan.json').read_text())['manifest_sha256'], frozen_hash)
            rerun = subprocess.run(['bash',str(self.f.workspace/'t1q1.pre.sh')], cwd=self.f.remote,
                                   env=self.f.env,capture_output=True,text=True,timeout=10)
            self.assertNotEqual(rerun.returncode,0)
            self.assertEqual(len(self.f.calls()[prior_calls:]),18)
            print('QUALIFICATION CHAIN stage -> start_run -> box -> 18 fresh evals, 0 training -> qualify -> byte-identical watchdog pull; changed_files=0',flush=True)
        finally:
            import psutil
            try:
                parent = psutil.Process(pid)
                processes = [*parent.children(recursive=True), parent]
                for process in reversed(processes):
                    try: process.kill()
                    except psutil.NoSuchProcess: pass
                psutil.wait_procs(processes,timeout=2)
            except psutil.NoSuchProcess: pass
            try: os.killpg(pid,signal.SIGKILL)
            except ProcessLookupError: pass

    def test_missing_receipt_is_refused_and_changed_staged_policy_never_runs(self):
        original, source, template = self.trained_source()
        receipt = next(source.rglob('evaluation.done.json'))
        retained = receipt.read_bytes(); receipt.unlink()
        refused = self.stage(original, source, template, name='missing', ok=False)
        self.assertFalse(refused.exists())
        receipt.write_bytes(retained)
        run = self.stage(original, source, template)
        before = len(self.f.calls())
        policy = next((run/'pulled/t1q1').rglob('policy_330012672.safetensors'))
        policy.write_bytes(policy.read_bytes()+b'corrupt')
        verification = self.f.command([sys.executable,'-B',str(self.stage_script),'verify',str(run)])
        self.assertNotEqual(verification.returncode,0)
        self.assertIn('staged custody mismatch',verification.stderr)
        self.f.launch(run)
        with self.assertRaisesRegex(AssertionError,'staged custody mismatch'):
            self.f.finished(run)
        self.assertEqual(len(self.f.calls()),before)
        self.assertFalse((self.f.remote/'runs/t1q1/qualification/RUN_STARTED').exists())


if __name__ == '__main__': unittest.main()
