"""Freeze all 30 prompts and rehearse replacement restore through the preparation CLI."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(os.environ.get('T3_BATCH_REPO',Path(__file__).resolve().parents[1])).resolve()


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=os.environ.get('T3_BATCH_TMP','/tmp'));self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name).resolve();self.project=self.base/'project'
        shutil.copytree(ROOT/'scripts/tier3_native',self.project/'scripts/tier3_native')
        self.scene=self.project/'data/challenge/track_2/tier_2_synthetic_noise';(self.scene/'meta').mkdir(parents=True)
        videos=self.scene/'videos/chunk-000/observation.images.exo_camera';videos.mkdir(parents=True)
        (self.scene/'mesh').mkdir();(self.scene/'mesh/fixture.glb').write_text('mesh')
        prompts=self.project/'data/tier3_bridge/prompts';prompts.mkdir(parents=True);self.prompts=prompts
        metadata=[];inventory=[]
        for ep in range(30):
            v=videos/f'episode_{ep:06d}.mp4';v.write_text('RGB'+str(ep))
            inventory.append({'episode':ep,'frames':30,'width':512,'height':384,'video_sha256':hashlib.sha256(v.read_bytes()).hexdigest()})
            metadata.append({'episode_index':ep,'sequence_id':f'fixture{ep}','mesh':'mesh/fixture.glb','ground_plane':'ground.json','support_surface':'support.usda'})
            (prompts/f'ep{ep:02d}.json').write_text(json.dumps({'prompts':[{'frame_index':0,'object_id':i,'role':role,'points':[],'point_labels':[],
                'box':{'x0':10+100*i,'y0':10,'x1':90+100*i,'y1':100}} for i,role in enumerate(('human','object'))]}))
        (prompts/'video_manifest.json').write_text(json.dumps(inventory))
        (self.scene/'meta/episodes_metadata.jsonl').write_text('\n'.join(map(json.dumps,metadata)))
        (self.scene/'ground.json').write_text('ground');(self.scene/'support.usda').write_text('support')
        modules=self.project/'external/video_to_data/reconstruction/modules';modules.mkdir(parents=True);(modules/'fixture.py').write_text('source=1')
        weights=self.project/'data/tier3_weights/sam-3d-body-dinov3';(weights/'assets').mkdir(parents=True)
        for name in ('model.ckpt','model_config.yaml','assets/mhr_model.pt'):(weights/name).write_text('weight')
        binary=self.base/'bin';binary.mkdir();probe=binary/'fake_probe'
        probe.write_text('#!'+sys.executable+'\nimport json\nprint(json.dumps({"streams":[{"nb_frames":"30","width":512,"height":384}]}))\n');probe.chmod(0o755)
        script=self.project/'scripts/tier3_native/prepare.py';script.write_text(script.read_text().replace("['ffprobe',","["+repr(str(probe))+','))
        self.run=self.base/'first/prepared'

    def invoke(self,run=None,*args):
        return subprocess.run([sys.executable,'-B',str(self.project/'scripts/tier3_native/prepare.py'),'prepare',str(run or self.run),
              '--project',str(self.project),*args],capture_output=True,text=True,timeout=20)

    def test_all30_schema_and_exact_remaining29_with_no_credentials(self):
        result=self.invoke();self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        plan=json.loads((self.run/'payload/plan.json').read_text())
        self.assertEqual({r['episode'] for r in plan['episodes']},set(range(30))-{18})
        self.assertEqual(len(json.loads((self.run/'prompt_sanity.json').read_text())),30)
        upload=(self.run/'upload.list').read_text().splitlines()
        self.assertEqual(len([p for p in upload if p.startswith('data/tier3_weights/')]),3)
        self.assertFalse(any('token' in p or 'tier_1' in p or 'data.parquet' in p for p in upload))

    def test_bad_prompt_aborts_before_prepared_output(self):
        p=self.prompts/'ep01.json';value=json.loads(p.read_text());value['prompts'][1]['box']['x1']=9999;p.write_text(json.dumps(value))
        result=self.invoke();self.assertNotEqual(result.returncode,0);self.assertFalse(self.run.exists())

    def test_subset_stages_exact_episodes_without_reusing_prior_batch(self):
        self.assertEqual(self.invoke().returncode,0)
        prior=self.run/'pulled/prepared/episodes'
        for ep in range(30):
            episode=prior/f'ep{ep:02d}';(episode/'.stages').mkdir(parents=True)
            (episode/'result.json').write_text('{"status":"SUCCESS"}')
            (episode/'.stages/05_object.done.json').write_text('prior stage')
            (episode/'initializers.pth').write_text('prior reconstruction')
        before={p.relative_to(self.run):(p.read_bytes(),p.stat().st_ino) for p in self.run.rglob('*') if p.is_file()}
        ids=['ep02','ep03','ep04','ep07','ep11','ep13','ep15','ep16','ep17','ep19','ep27']
        target=self.base/'rerun11'
        result=self.invoke(target,'--episodes',*reversed(ids));self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        plan=json.loads((target/'payload/plan.json').read_text())
        self.assertEqual([r['sequence'] for r in plan['episodes']],ids)
        self.assertEqual([r['episode'] for r in plan['episodes']],[2,3,4,7,11,13,15,16,17,19,27])
        self.assertEqual([r['native_sequence'] for r in json.loads((target/'bridge_plan.json').read_text())['episodes']],ids)
        self.assertEqual([r['episode'] for r in json.loads((target/'prompt_sanity.json').read_text())],[2,3,4,7,11,13,15,16,17,19,27])
        upload=(target/'upload.list').read_text().splitlines()
        self.assertEqual([Path(p).name for p in upload if p.endswith('.mp4')],[f'episode_{ep:06d}.mp4' for ep in (2,3,4,7,11,13,15,16,17,19,27)])
        self.assertEqual([Path(p).stem for p in upload if p.startswith('data/tier3_bridge/prompts/')],ids)
        self.assertEqual({p.relative_to(target).as_posix() for p in target.rglob('*') if p.is_file()},
                         {'entry.sh','upload.list','prompt_sanity.json','bridge_plan.json','payload/plan.json','payload/upload.json'})
        self.assertFalse(any(p.is_symlink() for p in target.rglob('*')))
        self.assertEqual({p.relative_to(self.run):(p.read_bytes(),p.stat().st_ino) for p in self.run.rglob('*') if p.is_file()},before)
        result=self.invoke(target,'--episodes',*ids);self.assertNotEqual(result.returncode,0)
        self.assertIn('choose a fresh run directory',result.stderr)

    def test_subset_refuses_unknown_duplicate_and_unapproved_ep18(self):
        result=self.invoke(self.base/'valid','--episodes','ep02');self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        for number,ids in enumerate((['ep02','ep30'],['ep2'],['ep02','ep02'],['ep18'])):
            with self.subTest(ids=ids):
                target=self.base/f'invalid{number}'
                result=self.invoke(target,'--episodes',*ids);self.assertNotEqual(result.returncode,0)
                self.assertIn('unknown/duplicate episode ids',result.stderr);self.assertFalse(target.exists())
        target=self.base/'include18'
        result=self.invoke(target,'--episodes','ep18','--include18');self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertEqual([r['episode'] for r in json.loads((target/'payload/plan.json').read_text())['episodes']],[18])

    def test_subset_refuses_restore_before_creating_output(self):
        self.assertEqual(self.invoke().returncode,0)
        (self.run/'STOPPED').write_text('finished');(self.run/'QUIESCED').write_text('writers stopped')
        artifact=self.run/'pulled/prepared/episodes/ep00/result.json';artifact.parent.mkdir(parents=True);artifact.write_text('{"status":"SUCCESS"}')
        target=self.base/'replacement/prepared'
        result=self.invoke(target,'--episodes','ep02','--restore-from',str(self.run))
        self.assertNotEqual(result.returncode,0);self.assertIn('--episodes requires fresh roots',result.stderr)
        self.assertFalse(target.exists());self.assertEqual(artifact.read_text(),'{"status":"SUCCESS"}')

    def test_replacement_preserves_remote_basename_and_hardlinks_only_completed_backup(self):
        self.assertEqual(self.invoke().returncode,0)
        (self.run/'STOPPED').write_text('finished');(self.run/'QUIESCED').write_text('writers stopped')
        artifact=self.run/'pulled/prepared/episodes/ep00/initializers.pth';artifact.parent.mkdir(parents=True);artifact.write_text('native complete')
        control=self.run/'pulled/prepared/control';control.mkdir();(control/'session.json').write_text('old mutable session')
        target=self.base/'replacement/prepared'
        result=self.invoke(target,'--restore-from',str(self.run));self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        copied=target/'pulled/prepared/episodes/ep00/initializers.pth'
        self.assertFalse((target/'pulled/prepared/control').exists())
        self.assertEqual(copied.read_text(),'native complete');self.assertEqual(copied.stat().st_ino,artifact.stat().st_ino)
        pending=copied.with_suffix('.tmp');pending.write_text('new generation');pending.replace(copied)
        self.assertEqual(artifact.read_text(),'native complete')
        self.assertEqual(copied.read_text(),'new generation')


if __name__=='__main__':unittest.main(verbosity=2)
