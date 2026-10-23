"""Actual bridge coordinator, real materializer/gates; only the model adapter is fake."""
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0,str(Path(__file__).resolve().parent))
import test_tier3_batch_gates as fixtures

ROOT=fixtures.ROOT
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


class BridgeCoordinatorTests(unittest.TestCase):
    def setUp(self):
        helper=fixtures.GateTests();helper.setUp();self.addCleanup(helper.doCleanups);self.fixture=helper
        self.base=helper.root;self.project=self.base/'project';self.cache=self.base
        shutil.copytree(ROOT/'scripts',self.project/'scripts')
        (self.project/'external').symlink_to(ROOT/'external',target_is_directory=True)
        # Fake only the model-side adapter boundary, not bridge/gate/status logic.
        (self.project/'scripts/tier3_initializers.py').write_text('''import sys
from pathlib import Path
args=sys.argv[1:]
for flag in ('--out','--mesh-out'):
 p=Path(args[args.index(flag)+1]);p.parent.mkdir(parents=True,exist_ok=True);p.write_text('adapter fixture')
''')
        self.native=self.base/'native/ep18'
        for rel,contents in [('mhr_init/ep18.pkl',b'human'),('foundationpose/ep18.pkl',b'object')]:
            p=self.native/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(contents)
        p=self.native/'export/ep18/wild_export.json';p.parent.mkdir(parents=True);p.write_bytes(helper.camera.read_bytes())
        raw=self.cache/'raw/ep18';raw.mkdir(parents=True)
        shutil.copyfile(helper.source/'soma_params.npz',raw/'soma_params.npz');shutil.copyfile(helper.source/'poses.npy',raw/'poses.npy')
        source_hashes={rel:sha(self.native/rel) for rel in ['mhr_init/ep18.pkl','foundationpose/ep18.pkl','export/ep18/wild_export.json']}
        (raw/'tier3_export.json').write_text(json.dumps({'source_metadata':{'ground_truth_used':False,'sources':source_hashes}}))
        self.motion=self.cache/'motion';shutil.copytree(self.cache/'assets/human_motion_data',self.motion)
        obj=self.motion/'whole_body/soma/sequence_id=synthetic/object/textured_mesh.obj'
        urdf=obj.parent/'textured_mesh.urdf';urdf.write_text('<robot name="fixture"/>')
        row=helper.row;row['object_mesh_paths']=[str(obj)];row['object_urdf_paths']=[str(urdf)]
        pq.write_table(pa.Table.from_pylist([row]),self.motion/'whole_body/soma/sequence_id=synthetic/robot_name=g1/data.parquet')
        video=self.base/'video.mp4';video.write_bytes(b'video fixture')
        prompts=self.base/'prompts.json';prompts.write_text('{}');(self.native/'prompts.json').write_text('{}')
        ground=self.base/'ground.json';ground.write_text('{"plane":[0,0,1,0]}')
        support=self.motion/'whole_body/reconstructed_stage/synthetic_support.usda'
        row={'episode':18,'sequence':'synthetic','native_sequence':'ep18','frames':30}
        for key,p in [('video',video),('prompts',prompts),('mesh',obj),('ground_plane',ground),('support_surface',support)]:
            row[key]=str(p);row[key+'_sha256']=sha(p)
        self.plan=self.base/'plan.json';self.plan.write_text(json.dumps({'episodes':[row]}))
        (self.native.parent/'input_manifest.json').write_text(json.dumps([{'path':'episode_000018.mp4','sha256':row['video_sha256']},{'path':'mesh.glb','sha256':row['mesh_sha256']}]))
        self.rgb=self.base/'rgb';self.rgb.mkdir()
        rgb=json.loads(helper.rgb.read_text());rgb['video_sha256']=row['video_sha256'];(self.rgb/'ep18.json').write_text(json.dumps(rgb))
        self.out=self.base/'output'

    def invoke(self,fresh=False):
        cmd=[sys.executable,'-B',str(self.project/'scripts/tier3_bridge_batch.py'),'--plan',str(self.plan),
             '--native-root',str(self.native.parent),'--out',str(self.out),'--tier1-root',str(self.base/'forbidden-unused-tier1'),
             '--protocol',str(fixtures.PROTOCOL),'--python',sys.executable,'--rgb-dir',str(self.rgb),
             '--episodes','18','--released-scene-assets-engineering-only']
        if not fresh:cmd+=['--ep18-cache',str(self.cache)]
        return subprocess.run(cmd,capture_output=True,text=True,timeout=30)

    def test_fresh_path_carries_staged_support_through_real_repair_and_materialization(self):
        # These three external model/IK boundaries are fakes. The coordinator,
        # repair, scene transfer, materializer and real G1 loader/gates execute.
        import trimesh
        source=self.fixture.source
        (source/'ground_plane.json').write_text('{"plane":[0,0,1,0]}')
        (source/'reconstructed_mesh').mkdir();trimesh.creation.box(extents=[.1,.1,.1]).export(source/'reconstructed_mesh/output_aligned.glb')
        (self.cache/'raw/ep18/reconstructed_mesh').mkdir();shutil.copyfile(source/'reconstructed_mesh/output_aligned.glb',self.cache/'raw/ep18/reconstructed_mesh/output_aligned.glb')
        worker=self.base/'model_boundary.py'
        worker.write_text("""import sys,shutil
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
args=sys.argv[2:];mode=sys.argv[1]
value=lambda k:Path(args[args.index(k)+1])
cache=Path("""+repr(str(self.cache))+""")
if mode=='export':shutil.copytree(cache/'raw/ep18',value('--out'))
elif mode=='align':
 shutil.copytree(cache/'sequences/synthetic',value('--out')/'synthetic')
 stage=value('--motion-root')/'whole_body/reconstructed_stage';stage.mkdir(parents=True)
 shutil.copyfile(value('--support-surface'),stage/'synthetic_support.usda')
elif mode=='retarget':
 target=Path(args[0])/'motion'
 assert (target/'whole_body/reconstructed_stage/synthetic_support.usda').exists(),'support lost before IK'
 shutil.copytree(cache/'motion',target,dirs_exist_ok=True)
 path=target/'whole_body/soma/sequence_id=synthetic/robot_name=g1/data.parquet'
 table=pq.ParquetFile(path).read()
 for key in ('object_mesh_paths','object_urdf_paths'):
  vals=[[p.replace(str(cache/'motion'),str(target)) for p in row] for row in table[key].to_pylist()]
  table=table.set_column(table.schema.get_field_index(key),table.schema.field(key),pa.array(vals,type=table[key].type))
 pq.write_table(table,path)
""")
        for script,mode in [('tier3_export.py','export'),('tier3_align.py','align')]:
            target=self.project/'scripts'/script
            original=target.read_text().rsplit('\nif __name__',1)[0]
            target.write_text(original+'\nif __name__ == "__main__":\n import runpy,sys\n sys.argv=['+repr(str(worker))+','+repr(mode)+',*sys.argv[1:]]\n runpy.run_path('+repr(str(worker))+',run_name="__main__")\n')
        (self.project/'scripts/retarget_tier.sh').write_text('#!/bin/bash\nexec "'+sys.executable+'" "'+str(worker)+'" retarget "$3"\n')
        result=self.invoke(fresh=True);self.assertEqual(result.returncode,0,result.stdout+result.stderr+(self.out/'status.tsv').read_text())
        support=self.out/'ep18/assets/human_motion_data/whole_body/reconstructed_stage/synthetic_support.usda'
        self.assertEqual(support.read_bytes(),(self.motion/'whole_body/reconstructed_stage/synthetic_support.usda').read_bytes())
        self.assertEqual(json.loads((self.out/'status.json').read_text())[0]['status'],'PASS')

    def test_exact_pass_manifest_and_no_manifest_after_a_failed_revalidation(self):
        first=self.invoke();self.assertEqual(first.returncode,0,first.stdout+first.stderr)
        manifest=json.loads((self.out/'manifest.json').read_text())
        self.assertEqual(manifest['tier'],3);self.assertEqual(manifest['scoring_mode'],'self')
        self.assertEqual([r['episode'] for r in manifest['sequences']],[18])
        arm=manifest['sequences'][0]['arms']['tier3_raw']
        self.assertEqual(arm['training_reference'],arm['clean_scoring_reference'])
        self.assertEqual(sha(Path(arm['training_reference']['path'])),arm['training_reference']['sha256'])
        self.assertIn('\tPASS\t',(self.out/'status.tsv').read_text())
        original_manifest=(self.out/'manifest.json').read_bytes()
        again=self.invoke();self.assertEqual(again.returncode,0,again.stdout+again.stderr)
        self.assertEqual((self.out/'manifest.json').read_bytes(),original_manifest)
        # User changes a video, retaining the same old plan and previous manifest.
        Path(json.loads(self.plan.read_text())['episodes'][0]['video']).write_bytes(b'changed video')
        second=self.invoke();self.assertEqual(second.returncode,1,second.stdout+second.stderr)
        self.assertFalse((self.out/'manifest.json').exists())
        self.assertIn('\tFAIL\tchanged bridge input: video',(self.out/'status.tsv').read_text().replace('ValueError: ',''))

    def test_hold_emits_status_without_training_manifest(self):
        alignment=self.fixture.source/'tier3_alignment.json';value=json.loads(alignment.read_text());value['scene_assets_mode']='released-engineering-only; ruling pending';alignment.write_text(json.dumps(value))
        result=self.invoke();self.assertEqual(result.returncode,1,result.stdout+result.stderr)
        self.assertFalse((self.out/'manifest.json').exists())
        statuses=json.loads((self.out/'status.json').read_text());self.assertEqual([r['status'] for r in statuses],['HOLD'])
        self.assertIn('released scene assets: organizer ruling pending',statuses[0]['reasons'])

    def test_changed_cached_G1_cannot_reuse_a_previous_pass_manifest(self):
        first=self.invoke();self.assertEqual(first.returncode,0,first.stdout+first.stderr)
        source=self.motion/'whole_body/soma/sequence_id=synthetic/robot_name=g1/data.parquet'
        row=pq.ParquetFile(source).read().to_pylist()[0]
        row['robot_joint_positions'][0][0]+=.01
        pq.write_table(pa.Table.from_pylist([row]),source)
        result=self.invoke();self.assertEqual(result.returncode,1,result.stdout+result.stderr)
        self.assertFalse((self.out/'manifest.json').exists())
        status=json.loads((self.out/'status.json').read_text())[0]
        self.assertEqual(status['status'],'FAIL');self.assertIn('binding changed',';'.join(status['reasons']))


if __name__=='__main__':unittest.main(verbosity=2)
