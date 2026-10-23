"""Frozen gates exercised through their CLI on generated motion and real USD/mesh I/O."""
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

ROOT=Path(os.environ.get('T3_BATCH_REPO',Path(__file__).resolve().parents[1])).resolve()
PROTOCOL=Path(os.environ.get('T3_BATCH_PROTOCOL',ROOT/'data/brainstorm5/T3-RECON/PROTOCOL.md'))


class GateTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=os.environ.get('T3_BATCH_TMP','/tmp'));self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.seq='synthetic';self.n=30
        self.source=self.root/'sequences'/self.seq;self.source.mkdir(parents=True)
        self.names=[f'{side}Hand{finger}End' for side in ('Left','Right') for finger in ('Thumb','Index','Middle','Ring','Little')]+[f'J{i}' for i in range(67)]
        np.savez(self.source/'soma_params.npz',poses=np.zeros((self.n,77,3)),transl=np.zeros((self.n,3)),joint_names=np.array(self.names),
                 unit=np.array('meters'),identity_model_type=np.array('mhr'),keep_root=np.array(False),identity_coeffs=np.zeros((1,45)),scale_params=np.zeros((1,68)))
        poses=np.tile(np.eye(4),(self.n,1,1));poses[:,2,3]=.0501;np.save(self.source/'poses.npy',poses)
        (self.source/'tier3_alignment.json').write_text(json.dumps({'rotation':np.eye(3).tolist(),'translation':[0,0,-3],'reference_frames_used':[0],'scene_assets_mode':'video-derived'}))
        self.ref=self.root/f'assets/human_motion_data/whole_body/soma/sequence_id={self.seq}/robot_name=g1/data.parquet'
        self.ref.parent.mkdir(parents=True);obj=self.ref.parent.parent/'object';obj.mkdir()
        trimesh.creation.box(extents=[.1,.1,.1]).export(obj/'textured_mesh.obj')
        (obj/'textured_mesh.urdf').write_text('<robot name="fixture"><link name="fixture_body"/></robot>')
        support=self.root/'assets/human_motion_data/whole_body/reconstructed_stage';support.mkdir(parents=True)
        (support/(self.seq+'_support.usda')).write_text('#usda 1.0\ndef Xform "support_surfaces" {}\n')
        payload={'soma_joints':np.zeros((self.n,77,3)), 'ground_alignment':{'object_ground_correction':{'applied_lift':0}}}
        self.row={'object_body_position':poses[:,:3,3][:,None].tolist(),'object_body_wxyz':np.tile([1.,0,0,0],(self.n,1,1)).tolist(),
                  'robot_joint_positions':np.zeros((self.n,29)).tolist(),'robot_root_position':np.zeros((self.n,3)).tolist(),'ee_pose_w':np.zeros((self.n,2,7)).tolist(),
                  'source_payload':pickle.dumps(payload)}
        self.row.update(schema_version='motion_v1',motion_kind='single_robot',coord_frame='world',fps=30.,source_dataset='synthetic',
            sequence_id=self.seq,robot_name='g1',raw_motion_file='generated fixture',robot_joint_names=[f'J{i}' for i in range(29)],
            robot_root_wxyz=np.tile([1.,0,0,0],(self.n,1)).tolist(),ee_link_names=['left_hand_palm_link','right_hand_palm_link'],
            hand_sides=['left','right'],hand_frame_names=[[],[]],hand_frames_w=[[],[]],object_name='fixture',object_body_names=['fixture_body'],
            object_mesh_paths=[str(obj/'textured_mesh.obj')],object_urdf_paths=[str(obj/'textured_mesh.urdf')],object_mesh_radius=[.1],object_articulation=np.zeros((self.n,0)).tolist(),
            hand_object_contact_positions=[[],[]],hand_object_contact_normals=[[],[]],hand_object_contact_part_ids=[[],[]],hand_contact_active=np.zeros((2,self.n)).tolist())
        ee=np.zeros((self.n,2,7));ee[:,:,3]=1.;self.row['ee_pose_w']=ee.tolist()
        pq.write_table(pa.Table.from_pylist([self.row]),self.ref)
        self.camera=self.root/'camera.json';self.camera.write_text(json.dumps({'intrinsics':[[100,0,50],[0,100,50],[0,0,1]]}))
        self.rgb=self.root/'rgb.json';self.rgb.write_text(json.dumps({'video_only':True,'independent_of_tracker':True,'image_width':100,'image_height':100,
            'polygons':{str(f):[[48,48],[51,48],[51,51],[48,51]] for f in range(self.n)}}))

    def call(self,*extra):
        out=self.root/'gates.json'
        result=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/tier3_gates.py'),'--base',str(self.root),'--sequence',self.seq,'--frames',str(self.n),
             '--camera',str(self.camera),'--protocol',str(PROTOCOL),'--out',str(out),*extra],capture_output=True,text=True,timeout=20)
        self.assertTrue(out.exists(),result.stdout+result.stderr)
        return result,json.loads(out.read_text())

    def test_missing_images_are_hold_and_full_independent_positive_control_passes(self):
        result,gate=self.call();self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(gate['status'],'HOLD');self.assertIn('independent RGB evidence missing',gate['reasons'])
        result,gate=self.call('--rgb',str(self.rgb));self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(gate['status'],'PASS',gate['reasons']);self.assertEqual(gate['coverage'],30)
        self.assertEqual(gate['penetration_frames'],[]);self.assertEqual(gate['physical']['detached_runs'],[])
        self.assertEqual(len(gate['rgb']),30)

    def move_object(self,frames,dx):
        position=np.asarray(self.row['object_body_position']);position[frames,0,0]+=dx;self.row['object_body_position']=position.tolist()
        pq.write_table(pa.Table.from_pylist([self.row]),self.ref)
        poses=np.load(self.source/'poses.npy');poses[frames,0,3]+=dx;np.save(self.source/'poses.npy',poses)

    def test_isolated_dynamics_flag_is_inspected_not_held(self):
        self.move_object(list(range(15,self.n)),.03)
        # The video agrees: 3 cm at the fixture's 3 m depth and 100 px focal length is 1 px.
        data=json.loads(self.rgb.read_text())
        for f in range(15,self.n): data['polygons'][str(f)]=[[x+1,y] for x,y in data['polygons'][str(f)]]
        self.rgb.write_text(json.dumps(data))
        result,gate=self.call('--rgb',str(self.rgb));self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(gate['status'],'PASS',gate['reasons'])
        self.assertIn('30Hz translation acceleration: isolated flags at frames [15, 16]',gate['inspected'])

    def test_sustained_dynamics_flag_still_holds(self):
        self.move_object(list(range(10,20,2)),.05)
        result,gate=self.call('--rgb',str(self.rgb));self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(gate['status'],'HOLD');self.assertIn('30Hz translation acceleration sustained',gate['reasons'])

    def test_invalid_coverage_is_fail_not_pass(self):
        self.row['robot_joint_positions']=self.row['robot_joint_positions'][:-1]
        pq.write_table(pa.Table.from_pylist([self.row]),self.ref)
        result,gate=self.call('--rgb',str(self.rgb))
        self.assertEqual(result.returncode,1);self.assertEqual(gate['status'],'FAIL');self.assertIn('shape',gate['reasons'][0])

    def test_released_assets_remain_hold_even_with_good_images(self):
        path=self.source/'tier3_alignment.json';d=json.loads(path.read_text());d['scene_assets_mode']='released-engineering-only; ruling pending';path.write_text(json.dumps(d))
        result,gate=self.call('--rgb',str(self.rgb));self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(gate['status'],'HOLD');self.assertIn('released scene assets: organizer ruling pending',gate['reasons'])

    def test_tracker_mask_cannot_masquerade_as_independent_evidence(self):
        data=json.loads(self.rgb.read_text());data['independent_of_tracker']=False;self.rgb.write_text(json.dumps(data))
        result,gate=self.call('--rgb',str(self.rgb));self.assertEqual(result.returncode,1)
        self.assertEqual(gate['status'],'FAIL');self.assertIn('independent of tracker',gate['reasons'][0])

    def test_independent_image_failure_remains_hold(self):
        data=json.loads(self.rgb.read_text());data['polygons']={str(f):[[1,1],[3,1],[3,3],[1,3]] for f in range(self.n)};self.rgb.write_text(json.dumps(data))
        result,gate=self.call('--rgb',str(self.rgb));self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(gate['status'],'HOLD');self.assertIn('independent RGB gate failed',gate['reasons'])


if __name__=='__main__':unittest.main(verbosity=2)
