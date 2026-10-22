"""Public quality CLI: exact report/overlays and fail-closed cache provenance.

Only FlashCHORD/USD are fixture boundaries. Tier-1 frame-zero selection, all
quality math, video decoding, cache validation and report writing are real.
"""
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import trimesh

SCRIPT=Path(os.environ.get('T3_QUALITY_SCRIPT',Path(__file__).resolve().parents[1]/'scripts/tier3_quality.py'))


def arrays_hash(data):
    h=hashlib.sha256()
    for key,value in sorted(data.items()):
        value=np.ascontiguousarray(value)
        h.update(json.dumps([key,value.dtype.str,value.shape]).encode());h.update(value.tobytes())
    return h.hexdigest()


class QualityCacheCLI(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);root=self.root
        self.sequence=root/'sequence';self.raw=root/'raw'
        names=[f'joint{i}' for i in range(77)]
        # The real alignment report looks these semantic joints up by name.
        names[:16]=['Hips','Spine1','Chest','Head','LeftArm','RightArm','LeftForeArm','RightForeArm',
                    'LeftHand','RightHand','LeftLeg','RightLeg','LeftShin','RightShin','LeftFoot','RightFoot']
        self.data=dict(poses=np.zeros((3,77,3)),transl=np.tile([0.,0.,-3.],(3,1)),
                       bone_length_flexibles=np.zeros((3,6)),joint_names=np.array(names),
                       identity_model_type=np.array('mhr'),identity_coeffs=np.ones((1,45)),
                       scale_params=np.ones((1,68)),unit=np.array('meters'),keep_root=np.array(False))
        poses=np.tile(np.eye(4),(3,1,1));poses[:,0,3]=[0.,.1,.2];poses[:,2,3]=3.
        for source in (self.sequence,self.raw):
            (source/'reconstructed_mesh').mkdir(parents=True)
            np.savez(source/'soma_params.npz',**self.data);np.save(source/'poses.npy',poses)
            (source/'ground_plane.json').write_text('{"plane":[0,0,1,0]}')
            trimesh.creation.box().export(source/'reconstructed_mesh/output_aligned.glb')
        row=dict(object_body_position=poses[:,None,:3,3].tolist(),object_body_wxyz=np.tile([1.,0.,0.,0.],(3,1,1)).tolist(),
                 ee_pose_w=np.tile([0.,0.,3.,1.,0.,0.,0.],(3,1,1)).tolist(),hand_sides=['left'],
                 ee_link_names=['left_hand_palm_link'],hand_contact_active=[[1.,0.,1.]],
                 robot_root_position=np.zeros((3,3)).tolist(),robot_joint_positions=np.zeros((3,2)).tolist(),
                 source_payload=pickle.dumps({'ground_alignment':{'fixture':'ground'}}),ik_error_per_frame=[.1,.2,.3])
        self.parquet=root/'data.parquet';pq.write_table(pa.Table.from_pylist([row]),self.parquet)
        self.tier1=root/'frame0_only';(self.tier1/'meta').mkdir(parents=True)
        info=dict(features={'observation.human.pose':{'names':[f'{n}.{axis}' for n in names for axis in ('rx','ry','rz')]}},
                  data_path='episode_{episode_index:06d}.parquet',chunks_size=1000)
        (self.tier1/'meta/info.json').write_text(json.dumps(info))
        (self.tier1/'meta/episodes_metadata.jsonl').write_text(json.dumps(dict(episode_index=0,sequence_id='fixture',object='fixture'))+'\n')
        ref={'frame_index':0,'observation.human.pose':np.zeros(231).tolist(),'observation.human.translation':[0.,0.,-3.],
             'observation.human.identity_coeffs':np.ones(45).tolist(),'observation.human.scale_params':np.ones(68).tolist(),
             'observation.human.bone_length_flexibles':np.zeros(6).tolist(),'observation.object.pose':[0.,0.,3.,1.,0.,0.,0.],
             'observation.object.visible':True}
        # Later-frame poison must never enter the reference-frame digest/decoder.
        later={**ref,'frame_index':1,'observation.human.translation':[999.,999.,999.]}
        pq.write_table(pa.Table.from_pylist([ref,later]),self.tier1/'episode_000000.parquet')
        video=root/'video.mp4'
        subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=c=black:s=320x240:r=30',
                        '-frames:v','3','-threads','1',str(video)],check=True)
        camera=root/'camera.json';camera.write_text('{"intrinsics":[[100,0,160],[0,100,120],[0,0,1]]}')
        joints=np.tile([0.,0.,-3.],(3,77,1));target=np.tile([.01,0.,3.],(77,1))
        archive=root/'joints.npz';np.savez(archive,joints=joints,target=target,raw_joints=joints,raw_frames=[0,1,2])
        ref_data={**self.data,'poses':self.data['poses'][:1],'transl':self.data['transl'][:1],
                  'bone_length_flexibles':self.data['bone_length_flexibles'][:1]}
        self.record=dict(schema=1,inputs={key:{'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for key,path in
                         [('sequence',self.sequence/'soma_params.npz'),('raw',self.raw/'soma_params.npz'),('parquet',self.parquet)]},
                         frame0_arrays_sha256=arrays_hash(ref_data),archive={'path':archive.name,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()})
        self.cache=root/'cache.json';self.cache.write_text(json.dumps(self.record))
        stub=root/'stubs';stub.mkdir()
        (stub/'sitecustomize.py').write_text('''import sys,types
from pathlib import Path
class BlockModel:
 def __init__(self,*a,**k): raise RuntimeError('full SOMA decode forbidden with cache')
model=types.ModuleType('robotic_grounding.retarget.read_soma');model.SOMA=BlockModel
sys.modules[model.__name__]=model
for name in ('flash_chord','flash_chord.data','flash_chord.data.motion_v1','flash_chord.assets','flash_chord.assets.registry','pxr'):
 sys.modules[name]=types.ModuleType(name)
sys.modules['flash_chord.data.motion_v1'].load_motion_v1=lambda *a,**k: types.SimpleNamespace(num_frames=3)
sys.modules['flash_chord.assets.registry'].support_usda_for_parquet=lambda path,**k: Path(path).parent/'support.usda'
sys.modules['pxr'].Usd=types.SimpleNamespace(Stage=types.SimpleNamespace(Open=lambda p:types.SimpleNamespace(Traverse=lambda:[])))
''')
        (root/'support.usda').write_text('#usda 1.0\n')
        self.env={**os.environ,'PYTHONPATH':str(stub)+os.pathsep+os.environ.get('PYTHONPATH','')}
        self.command=[sys.executable,'-B',str(SCRIPT),'--sequence',str(self.sequence),'--raw',str(self.raw),
                      '--parquet',str(self.parquet),'--video',str(video),'--wild-export',str(camera),
                      '--tier1-root',str(self.tier1),'--episode','0','--out',str(root/'quality'),'--joint-cache',str(self.cache)]

    def run_cli(self):
        return subprocess.run(self.command,env=self.env,capture_output=True,text=True)

    def test_complete_report_and_all_three_overlays_use_native_joint_cache(self):
        result=self.run_cli();self.assertEqual(result.returncode,0,result.stderr)
        report=json.loads((self.root/'quality/quality.json').read_text())
        self.assertEqual(report['coverage'],{'video':3,'SOMA':3,'object':3,'G1_loader':3,'fps':30.})
        self.assertEqual(report['frame0']['reference_frames_used'],[0])
        self.assertAlmostEqual(report['frame0']['joint_rmse_m'],.01,places=14)
        self.assertEqual(report['frame0']['object_translation_error_m'],0.)
        self.assertEqual(report['contacts']['left']['labelled_frames'],[0,2])
        self.assertAlmostEqual(report['object_translation']['speed']['max'],3.,places=14)
        self.assertEqual(report['human_joint_translation']['speed']['max'],0.)
        self.assertEqual(report['scene_support_enabled'],False)
        self.assertEqual(sorted(p.name for p in (self.root/'quality').glob('overlay_*.png')),
                         ['overlay_000000.png','overlay_000001.png','overlay_000002.png'])

    def test_changed_source_and_changed_archive_are_rejected(self):
        for kind in ('sequence','archive'):
            with self.subTest(kind=kind):
                path=self.sequence/'soma_params.npz' if kind=='sequence' else self.root/'joints.npz'
                before=path.read_bytes();path.write_bytes(before+b'changed')
                result=self.run_cli();self.assertNotEqual(result.returncode,0)
                self.assertIn('hash mismatch',result.stderr)
                self.assertFalse((self.root/'quality/quality.json').exists());path.write_bytes(before)

    def test_changed_permitted_frame_zero_is_rejected(self):
        path=self.tier1/'episode_000000.parquet';table=pq.ParquetFile(path).read().to_pylist()
        table[0]['observation.human.translation']=[1.,0.,-3.];pq.write_table(pa.Table.from_pylist(table),path)
        result=self.run_cli();self.assertNotEqual(result.returncode,0)
        self.assertIn('frame-zero input mismatch',result.stderr)
        self.assertFalse((self.root/'quality/quality.json').exists())

    def test_hash_bound_invalid_joint_dimensions_are_rejected(self):
        archive=self.root/'joints.npz'
        np.savez(archive,joints=np.zeros((2,77,3)),target=np.zeros((77,3)),raw_joints=np.zeros((3,77,3)),raw_frames=[0,1,2])
        self.record['archive']['sha256']=hashlib.sha256(archive.read_bytes()).hexdigest();self.cache.write_text(json.dumps(self.record))
        result=self.run_cli();self.assertNotEqual(result.returncode,0)
        self.assertIn('dimensions, frame correspondence or values invalid',result.stderr)
        self.assertFalse((self.root/'quality/quality.json').exists())


if __name__=='__main__':
    unittest.main(verbosity=2)
