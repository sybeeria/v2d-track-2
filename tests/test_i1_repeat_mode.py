"""I1's exact post-hook consumer must remain checkpoint-only after preparation."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from tests.test_runbook import SOURCE, REPO, ENV, put

ENGINE=Path(os.environ.get('RUNBOOK_ENGINE',REPO))
sys.path[:0]=[str(ENGINE/'scripts'),str(ENGINE/'tests')]
import test_score_rollout as fixtures
from pilot_queue import read_arms, read_inputs


class I1RepeatMode(unittest.TestCase):
    def test_post_hook_never_requests_training_if_a_prepared_actor_is_lost(self):
        with tempfile.TemporaryDirectory(prefix='i1-mode-',dir=os.environ.get('TMPDIR','/tmp')) as tmp:
            work=Path(tmp);manifest=fixtures.PilotQueueTest().manifest(work)
            sequence=fixtures.SEQUENCES[0];arm='T2C-E4-seed44';name='I1-T2C-E4-s44-snack'
            source=work/'source';root=source/name/arm
            overrides=read_arms(ENGINE/'scripts/pilot_arms_i1.txt',[49999872])[arm][0]
            references=read_inputs(manifest,[sequence],['raw_repaired'])
            put(root/'recipe.json',json.dumps(dict(schema_version=2,overrides=overrides,references=references)))
            actor=root/f'raw_repaired/train/{sequence}/policy_49999872.safetensors';put(actor,'fixture checkpoint')
            stamp=root/f'raw_repaired/evaluation/{sequence}/49999872/evaluation.done.json'
            put(stamp,json.dumps({'checkpoint_sha256':hashlib.sha256(actor.read_bytes()).hexdigest()}))
            plan=work/'plan.json';put(plan,json.dumps({'repeats':3,'candidates':[dict(name=name,source=f'{name}/{arm}',
                group='i1',arm=arm,arms_file='pilot_arms_i1.txt',input='raw_repaired',sequence=sequence,
                steps=[49999872],final_step=49999872)]}))
            output=work/'repeats'
            prepared=subprocess.run([sys.executable,'-B',str(ENGINE/'scripts/prepare_repeats.py'),'--plan',str(plan),
                '--source-root',str(source),'--out-root',str(output)],cwd=ENGINE,env=ENV,capture_output=True,text=True)
            self.assertEqual(prepared.returncode,0,prepared.stderr)
            # The missing-file injection uses regular owned copies; no writes follow directory links.
            actors=[]
            for link in output.glob(f'i1/*/{arm}/raw_repaired/train/{sequence}'):
                original=link.resolve();link.unlink();shutil.copytree(original,link)
                actors.append(link/'policy_49999872.safetensors')
            self.assertEqual(len(actors),3)
            consumer=(REPO/'data/brainstorm7/runs/i1screen/post.sh').read_text().splitlines()[-1]
            consumer=consumer.replace('runs/i1screen/repeats',str(output)).replace('"$(nvidia-smi -L | wc -l)"','1')+' --dry-run'
            def run():
                return subprocess.run(['bash','-c',consumer],cwd=ENGINE,env={**ENV,'MANIFEST':str(manifest),
                    'PACK_PYTHON':sys.executable},capture_output=True,text=True)
            complete=run()
            self.assertEqual(complete.returncode,0,complete.stderr)
            self.assertEqual(complete.stdout.count('/evaluate_policy.py '),3)
            self.assertNotIn('/scripts/train_queue.sh ',complete.stdout)
            original_hash=hashlib.sha256(actor.read_bytes()).hexdigest()
            for checkpoint in actors:checkpoint.unlink()
            lost=run()
            self.assertEqual(lost.returncode,0,lost.stderr)  # dry run prints commands and never creates a policy
            self.assertEqual(lost.stdout.count('/evaluate_policy.py '),3)
            self.assertNotIn('/scripts/train_queue.sh ',lost.stdout)
            self.assertTrue(all(not p.exists() for p in actors))
            self.assertEqual(hashlib.sha256(actor.read_bytes()).hexdigest(),original_hash)
            self.assertFalse(list(output.rglob('trajectory.parquet')))


if __name__=='__main__':unittest.main(verbosity=2)
