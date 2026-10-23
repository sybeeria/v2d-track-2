"""Public-process batch/atomic-stage regressions; CUDA/provider boundaries are fakes."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(os.environ.get('T3_BATCH_REPO', Path(__file__).resolve().parents[1])).resolve()
TMP = os.environ.get('T3_BATCH_TMP', '/tmp')
PYTHON = sys.executable


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


WORKER = '''import json,os,sys,time
from pathlib import Path
name,stage,out=sys.argv[1:]
with open(os.environ['EVENTS'],'a') as f: f.write(json.dumps([name,stage,os.environ.get('CUDA_VISIBLE_DEVICES')])+'\\n')
p=Path(out);p.parent.mkdir(parents=True,exist_ok=True);p.write_text('PARTIAL')
if os.environ.get('BLOCK') == name+':'+stage:
    Path(os.environ['READY']).write_text('ready')
    time.sleep(20)
if os.environ.get('FAIL') == name+':'+stage:
    print('injected GPU stage failure',flush=True);sys.exit(7)
time.sleep(float(os.environ.get('DELAY','.01')))
p.write_text(name+':'+stage+':'+os.environ.get('CUDA_VISIBLE_DEVICES','none'))
'''


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=TMP, prefix='batch-case-'); self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve(); self.project = self.base / 'project'
        (self.project / 'scripts').mkdir(parents=True)
        (self.project / 'scripts/frozen.py').write_text('source=1\n')
        modules = self.project / 'external/video_to_data/reconstruction/modules'
        modules.mkdir(parents=True); (modules / 'frozen.py').write_text('source=1\n')
        self.weights = self.base / 'weights'; self.weights.mkdir(); (self.weights / 'model.pt').write_text('weights-A')
        self.worker = self.base / 'worker.py'; self.worker.write_text(WORKER)
        self.runsh = self.base / 'run.sh'; self.runsh.write_text('''#!/bin/bash
set -euo pipefail
while [ $# -gt 0 ]; do
 case "$1" in
 --work-root) root=$2;shift 2;; --sequence) name=$2;shift 2;;
 --released-mesh-engineering-only) shift;; *) shift 2;;
 esac
done
for stage in first second; do
 output="$root/$name/$stage.txt"
 [ "$stage" != second ] || output="$root/$name/initializers.pth"
 "$BATCH_PYTHON" -B "$STAGE_SCRIPT" --root "$root/$name" --name "$stage" --input "$root/$name/inputs.json" --output "$output" -- "$BATCH_PYTHON" -B "$FAKE_WORKER" "$name" "$stage" "$output"
done
''')
        self.events = self.base / 'events.jsonl'; self.out = self.base / 'out'
        self.env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'BATCH_PYTHON': PYTHON,
                    'STAGE_SCRIPT': str(ROOT / 'scripts/tier3_native/stage.py'), 'FAKE_WORKER': str(self.worker),
                    'EVENTS': str(self.events), 'READY': str(self.base / 'ready')}
        self.plan = self.base / 'plan.json'
        self.rows = []
        for ep, frames in [(0, 30), (1, 90), (2, 60), (3, 45)]:
            row = {'episode': ep, 'sequence': f'ep{ep:02d}', 'frames': frames}
            for key in ('video', 'prompts', 'mesh'):
                file = self.base / f'{ep}.{key}'; file.write_text(key + str(ep))
                row[key] = str(file); row[key + '_sha256'] = sha(file)
            self.rows.append(row)
        self.plan.write_text(json.dumps({'episodes': self.rows}))

    def command(self, *extra):
        return [PYTHON, '-B', str(ROOT / 'scripts/tier3_native/batch.py'), '--plan', str(self.plan), '--root', str(self.out),
                '--weights', str(self.weights), '--project', str(self.project), '--gpus', '0', '--run-command', str(self.runsh), *extra]

    def execute(self, *extra, **env):
        return subprocess.run(self.command(*extra), env={**self.env, **env}, capture_output=True, text=True, timeout=20)

    def calls(self):
        return [json.loads(x) for x in self.events.read_text().splitlines()] if self.events.exists() else []

    def test_longest_first_failure_continues_and_restart_retries_only_failed_stage(self):
        before = self.execute(FAIL='ep02:second')
        self.assertEqual(before.returncode, 1, before.stderr)
        self.assertEqual([x[:2] for x in self.calls()], [['ep01','first'],['ep01','second'],['ep02','first'],['ep02','second'],['ep03','first'],['ep03','second'],['ep00','first'],['ep00','second']])
        failed = json.loads((self.out / 'ep02/result.json').read_text())
        self.assertEqual(failed['status'], 'FAIL'); self.assertIn('exited 7', failed['error'])
        self.assertIn('injected GPU stage failure', failed['stage_metrics'][-1]['error_tail'])
        self.assertFalse((self.out / 'ep02/initializers.pth').exists())
        after = self.execute()
        self.assertEqual(after.returncode, 0, after.stdout + after.stderr)
        self.assertEqual(self.calls()[-1], ['ep02', 'second', '0'])
        self.assertEqual(len(self.calls()), 9)
        self.assertEqual((self.out / 'ep02/initializers.pth').read_text(), 'ep02:second:0')
        self.assertEqual(json.loads((self.out / 'ep02/result.json').read_text())['status'], 'SUCCESS')

    def test_interruption_leaves_no_published_partial_and_resumes(self):
        process = subprocess.Popen(self.command(), env={**self.env, 'BLOCK': 'ep01:second'}, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        try:
            until = time.monotonic() + 10
            while not (self.base / 'ready').exists() and time.monotonic() < until: time.sleep(.02)
            self.assertTrue((self.base / 'ready').exists())
            self.assertFalse((self.out / 'ep01/initializers.pth').exists())
            self.assertEqual((self.out / 'ep01/first.txt').read_text(), 'ep01:first:0')
        finally:
            os.killpg(process.pid, signal.SIGTERM); process.wait(timeout=10); process.communicate()
        after = self.execute()
        self.assertEqual(after.returncode, 0, after.stdout + after.stderr)
        self.assertEqual([x for x in self.calls() if x[:2] == ['ep01','first']], [['ep01','first','0']])
        self.assertEqual(len([x for x in self.calls() if x[:2] == ['ep01','second']]), 2)
        self.assertEqual((self.out / 'ep01/initializers.pth').read_text(), 'ep01:second:0')

    def test_changed_video_and_weights_cannot_reuse_success(self):
        self.assertEqual(self.execute().returncode, 0)
        Path(self.rows[0]['video']).write_text('different')
        (self.weights / 'model.pt').write_text('weights-B')
        result = self.execute()
        self.assertEqual(result.returncode, 1)
        self.assertEqual(len(self.calls()), 8)
        self.assertEqual((self.out / 'ep00/initializers.pth').read_text(), 'ep00:second:0')
        for ep in range(4):
            self.assertEqual(json.loads((self.out / f'ep{ep:02d}/result.json').read_text())['status'], 'FAIL')

    def test_two_gpu_workers_never_overlap_on_one_gpu(self):
        result = self.execute('--gpus', '0,1', DELAY='.1')
        self.assertEqual(result.returncode, 0, result.stderr)
        first = [x for x in self.calls() if x[1] == 'first']
        self.assertEqual({x[0] for x in first[:2]}, {'ep01','ep02'})
        self.assertEqual({x[2] for x in first}, {'0','1'})
        for gpu in ('0','1'):
            entries = [x for x in self.calls() if x[2] == gpu]
            self.assertEqual([x[1] for x in entries], ['first','second'] * (len(entries)//2))
            for i in range(0,len(entries),2): self.assertEqual(entries[i][0],entries[i+1][0])

    def test_success_without_artifact_is_failure(self):
        self.runsh.write_text('exit 0\n')
        result = self.execute()
        self.assertEqual(result.returncode, 1)
        self.assertEqual(len(list(self.out.glob('*/result.json'))), 4)
        self.assertEqual({json.loads(x.read_text())['status'] for x in self.out.glob('*/result.json')}, {'FAIL'})

    def test_duplicate_batch_is_rejected_without_changing_input_binding(self):
        first = subprocess.Popen(self.command(), env={**self.env, 'BLOCK': 'ep01:second'}, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        try:
            until=time.monotonic()+10
            while not (self.base/'ready').exists() and time.monotonic()<until: time.sleep(.02)
            before=(self.out/'ep01/inputs.json').read_bytes()
            second=self.execute()
            self.assertNotEqual(second.returncode,0)
            self.assertEqual((self.out/'ep01/inputs.json').read_bytes(),before)
        finally:
            os.killpg(first.pid,signal.SIGTERM);first.wait(timeout=10);first.communicate()

    def test_replacement_remeasures_canary_and_requires_fresh_session_approval(self):
        session = self.base / 'session.json'
        prior_go = None
        for number in (1, 2):
            identity = str(number) * 32
            session.write_text(json.dumps({'plan_sha256': sha(self.plan), 'session_id': identity}))
            proc = subprocess.Popen(self.command('--canary', '--session', str(session), '--canary-poll-seconds', '.02'),
                                    env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
            try:
                ready = self.out / 'CANARY_READY.json'
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    if ready.exists() and json.loads(ready.read_text()).get('session_id') == identity:
                        break
                    time.sleep(.02)
                expected = json.loads(ready.read_text())
                self.assertEqual(expected['session_id'], identity)
                self.assertIsNone(proc.poll(), 'old approval released the replacement box')
                if number == 2:
                    self.assertEqual(len(self.calls()), 10, 'first episode was reused rather than measured again')
                    self.assertEqual(len(list((self.out / '.canary-prior').glob('ep01.*'))), 1)
                    (self.out / 'GO.json').write_text(json.dumps(prior_go))
                    time.sleep(.08)
                    self.assertIsNone(proc.poll(), 'wrong session approval released the batch')
                (self.out / 'GO.json').write_text(json.dumps(expected))
                stdout, stderr = proc.communicate(timeout=10)
                self.assertEqual(proc.returncode, 0, stdout + stderr)
                prior_go = expected
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM);proc.wait(timeout=10);proc.communicate()
        self.assertEqual(len(self.calls()), 10, 'other completed episodes must keep their stage markers')
        for path in self.out.glob('*/result.json'):
            self.assertEqual(json.loads(path.read_text())['session_id'], '2' * 32)


class AtomicStageTests(unittest.TestCase):
    def test_command_without_explicit_output_is_rejected_before_running(self):
        with tempfile.TemporaryDirectory(dir=TMP) as tmp:
            root=Path(tmp).resolve();out=root/'artifact';marker=root/'ran'
            code='from pathlib import Path; p=Path('+repr(str(root))+'); (p/"artifact").write_text("partial"); (p/"ran").touch()'
            result=subprocess.run([PYTHON,'-B',str(ROOT/'scripts/tier3_native/stage.py'),'--root',str(root),'--name','invalid','--output',str(out),'--',PYTHON,'-c',code],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0);self.assertFalse(out.exists());self.assertFalse(marker.exists())

    def test_interrupted_stage_never_exposes_partial_final_output(self):
        with tempfile.TemporaryDirectory(dir=TMP) as tmp:
            root=Path(tmp).resolve();worker=root/'worker.py';out=root/'published.txt';ready=root/'ready'
            worker.write_text("import os,sys,time\nfrom pathlib import Path\np=Path(sys.argv[1]);p.write_text('PARTIAL')\nif os.environ.get('BLOCK'):\n Path(os.environ['READY']).write_text('ready');time.sleep(20)\np.write_text('COMPLETE')\n")
            cmd=[PYTHON,'-B',str(ROOT/'scripts/tier3_native/stage.py'),'--root',str(root),'--name','publish','--output',str(out),'--',PYTHON,str(worker),str(out)]
            proc=subprocess.Popen(cmd,env={**os.environ,'BLOCK':'1','READY':str(ready)},stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
            try:
                until=time.monotonic()+10
                while not ready.exists() and time.monotonic()<until:time.sleep(.02)
                self.assertTrue(ready.exists());self.assertFalse(out.exists(),'incomplete stage output became publicly visible')
            finally:
                try:os.killpg(proc.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                proc.wait(timeout=10);proc.communicate()
            retry=subprocess.run(cmd,capture_output=True,text=True,timeout=10)
            self.assertEqual(retry.returncode,0,retry.stderr);self.assertEqual(out.read_text(),'COMPLETE')

    def test_directory_output_metadata_paths_are_rebased_and_restart_ignores_mtime(self):
        with tempfile.TemporaryDirectory(dir=TMP) as tmp:
            root=Path(tmp).resolve(); worker=root/'worker.py'
            worker.write_text("import sys,json\nfrom pathlib import Path\np=Path(sys.argv[1]);p.mkdir();(p/'mesh.glb').write_text('mesh');(p/'wild_export.json').write_text(json.dumps({'object_mesh_file':str(p/'mesh.glb')}))\n")
            out=root/'export';cmd=[PYTHON,'-B',str(ROOT/'scripts/tier3_native/stage.py'),'--root',str(root),'--name','export','--output',str(out),'--',PYTHON,str(worker),str(out)]
            first=subprocess.run(cmd,text=True,capture_output=True);self.assertEqual(first.returncode,0,first.stderr)
            self.assertEqual(json.loads((out/'wild_export.json').read_text()),{'object_mesh_file':str(out/'mesh.glb')})
            os.utime(out/'mesh.glb',(0,0))
            second=subprocess.run(cmd,text=True,capture_output=True);self.assertEqual(second.returncode,0,second.stderr);self.assertIn('SKIP export',second.stdout)


if __name__ == '__main__': unittest.main(verbosity=2)
