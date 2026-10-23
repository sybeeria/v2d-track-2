"""Public CLI checks with synthetic initializers/videos; no challenge captions."""
import json
import os
from pathlib import Path
import pickle
import runpy
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from tier3_initializers import rigid
from tier3_prompts import make_prompts
from tier3_quality import continuity, mesh_plane_bounds


class InitializerTests(unittest.TestCase):
    def fixture(self, root):
        schema=ROOT/'external/video_to_data/reconstruction/modules/v2d_cari4d/lib/cari4d/lib_mhr/schema.py'
        dims=runpy.run_path(str(schema))['MHR_PARAM_DIMS']
        human={key:np.zeros((3,width),np.float32) for key,width in dims.items()}
        human.update(frames=['000000','000001','000002'],body_model='mhr',metadata={'source':'sam3d_body_rgb','alignment':'none'})
        poses=np.tile(np.eye(4),(3,1,1));poses[:,:3,3]=[[1,2,3],[2,3,4],[3,4,5]]
        obj={'frames':human['frames'],'camera_id':0,'obj_pose_world':poses,'fp_poses_cam':poses[:,None],
             'metadata':{key:False for key in ('first_usable_frame_gt_rotation_oracle','foundationpose_selection_uses_gt_pose','gt_object_pose_available','gt_translation_oracle')}}
        marker={'frame_count':3,'camera_id':0,'object_pose_frame':'centered_axis_aligned',
                'object_mesh_to_training_transform':np.eye(4).tolist(),'object_pose_storage_to_training_transform':np.eye(4).tolist()}
        paths=[root/'human.pkl',root/'object.pkl',root/'wild.json']
        paths[0].write_bytes(pickle.dumps(human));paths[1].write_bytes(pickle.dumps(obj));paths[2].write_text(json.dumps(marker))
        return human,obj,paths

    def run_adapter(self, root, paths):
        return subprocess.run([sys.executable,'-I',str(ROOT/'scripts/tier3_initializers.py'),
                               '--human',str(paths[0]),'--object',str(paths[1]),'--wild-export',str(paths[2]),
                               '--expected-frames','3','--out',str(root/'bundle.pth')],capture_output=True,text=True)

    def test_cli_preserves_all_frames_parameters_and_object_poses(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);human,obj,paths=self.fixture(root)
            result=self.run_adapter(root,paths);self.assertEqual(result.returncode,0,result.stderr)
            bundle=torch.load(root/'bundle.pth',weights_only=False)
            self.assertEqual(bundle['frames'],['000000','000001','000002'])
            self.assertIs(bundle['metadata']['ground_truth_used'],False)
            self.assertEqual(bundle['metadata']['rotation_so3_projection_max_element_change'],0.)
            for key in human:
                if key.startswith('mhr_'): np.testing.assert_array_equal(bundle['pr'][key],human[key])
            np.testing.assert_array_equal(bundle['pr']['pose_abs'],obj['obj_pose_world'])
            self.assertEqual(len(bundle['metadata']['sources']),3)

    def test_cli_rejects_gt_and_incomplete_timeline_without_output(self):
        for change in ['oracle','missing_flag','timeline','human_alignment']:
            with self.subTest(change=change),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);human,obj,paths=self.fixture(root)
                if change=='oracle':obj['metadata']['gt_translation_oracle']=True
                if change=='missing_flag':del obj['metadata']['gt_object_pose_available']
                if change=='timeline':obj['frames']=['000000','000002','000003']
                if change=='human_alignment':human['metadata']['alignment']='gt_depth'
                paths[0].write_bytes(pickle.dumps(human));paths[1].write_bytes(pickle.dumps(obj))
                self.assertNotEqual(self.run_adapter(root,paths).returncode,0)
                self.assertFalse((root/'bundle.pth').exists())

    def test_rotation_roundoff_is_repaired_but_deformation_is_rejected(self):
        a=np.eye(4)[None];a[0,0,0]+=1e-5;a[0,:3,3]=[1,2,3]
        fixed=rigid(a,1,repair_roundoff=True)
        assert_allclose(fixed[0,:3,:3],np.eye(3),atol=1e-14)
        np.testing.assert_array_equal(fixed[0,:3,3],[1,2,3])
        self.assertEqual(a[0,0,0],1.00001)
        a[0,0,0]=1.01
        with self.assertRaises(ValueError):rigid(a,1,repair_roundoff=True)
        a[0,0,0]=-1
        with self.assertRaises(ValueError):rigid(a,1,repair_roundoff=True)


class NativeStageTests(unittest.TestCase):
    def command(self,root,code):
        return [sys.executable,'-I',str(ROOT/'scripts/tier3_native/stage.py'),'--root',str(root),'--name','fixture',
                '--input',str(root/'input'),'--output',str(root/'artifact'),'--',sys.executable,'-I','-c',code,str(root/'artifact')]

    def test_real_stage_skips_success_and_rejects_changed_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'input').write_text('original')
            code=f"from pathlib import Path; import sys; p=Path({str(root)!r}); Path(sys.argv[1]).write_text('exact output'); (p/'counter').write_text((p/'counter').read_text()+'x' if (p/'counter').exists() else 'x')"
            cmd=self.command(root,code)
            for _ in range(2):
                r=subprocess.run(cmd,capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stderr)
            self.assertEqual((root/'artifact').read_text(),'exact output');self.assertEqual((root/'counter').read_text(),'x')
            (root/'input').write_text('changed')
            self.assertNotEqual(subprocess.run(cmd,capture_output=True).returncode,0)
            self.assertEqual((root/'artifact').read_text(),'exact output')

    def test_interrupted_stage_resumes_and_retains_failed_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'input').write_text('input')
            code=f"from pathlib import Path; import sys; p=Path({str(root)!r}); first=not (p/'attempt').exists(); (p/'attempt').write_text('yes'); Path(sys.argv[1]).write_text('partial' if first else 'complete'); sys.exit(7 if first else 0)"
            cmd=self.command(root,code)
            self.assertEqual(subprocess.run(cmd,capture_output=True).returncode,7)
            self.assertFalse((root/'artifact').exists())
            self.assertEqual(subprocess.run(cmd,capture_output=True).returncode,0)
            self.assertEqual((root/'artifact').read_text(),'complete')
            self.assertEqual(len(list((root/'logs').glob('*.log'))),2)

    def test_dry_run_has_all_stages_and_no_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'not-created'
            r=subprocess.run(['bash',str(ROOT/'scripts/tier3_native/run.sh'),'--work-root',str(root),
                              '--video','/video/ep.0.color.mp4','--prompts','/prompts.json','--mesh','/mesh.glb',
                              '--sequence','ep','--frames','3','--dry-run'],capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr)
            self.assertFalse(root.exists())
            for name in ['sam2','pack_masks','01_depth','02_export','03_human','04_align','05_object','validate']:
                self.assertIn('--name '+name,r.stdout)
            self.assertIn('--no-first-usable-frame-gt-rotation-oracle',r.stdout)
            self.assertNotIn('CoCoNet',r.stdout)


class QualityAndPromptTests(unittest.TestCase):
    def test_prompts_have_exact_native_coordinates_and_roles(self):
        a={'human':{'box':[10,20,30,40]},'object':{'box':[100,200,120,220],'points':[[110,210]],'point_labels':[1]}}
        p=make_prompts(a,1536,1152,10)['prompts']
        self.assertEqual(p[0],{'frame_index':0,'object_id':0,'role':'human','points':[],'point_labels':[],
                               'box':{'x0':30,'y0':60,'x1':90,'y1':120}})
        self.assertEqual(p[1]['object_id'],1);self.assertEqual(p[1]['points'],[{'x':330.,'y':630.}])
        schema=runpy.run_path(str(ROOT/'external/video_to_data/reconstruction/modules/v2d_common/datatypes.py'))
        parsed=schema['Sam2Prompts'].from_dict({'prompts':p})
        self.assertEqual(parsed.prompts[1].points[0].x,330.)
        self.assertEqual(parsed.prompts[1].points[0].y,630.)
        self.assertIsNone(p[1]['box'])
        self.assertEqual(sum([bool(p[1]['points']), p[1]['box'] is not None]),1)
        a['object']['frame']=10
        with self.assertRaises(ValueError):make_prompts(a,1536,1152,10)

    def test_continuity_flags_the_exact_jump_frames(self):
        x=np.array([[0,0,0],[0,0,0],[1,0,0],[1,0,0]],float)
        r=continuity(x,30.)
        self.assertEqual(r['speed_outlier_destination_frames'],[2])
        self.assertEqual(r['acceleration_outlier_destination_frames'],[2,3])
        self.assertEqual(r['speed']['max'],30.)
        self.assertEqual(r['acceleration']['max'],900.)

    def test_plane_bounds_use_full_mesh_and_normalized_plane(self):
        vertices=np.array([[0,0,-1],[0,0,2],[1,2,0]])
        poses=np.tile(np.eye(4),(2,1,1));poses[1,2,3]=3
        np.testing.assert_array_equal(mesh_plane_bounds(vertices,poses,[0,0,2,-2]),[[-2,1],[1,4]])



class PortableAssetsTest(unittest.TestCase):
    def test_cli_closure_resolves_after_relocation_and_preserves_motion(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from flash_chord.assets.registry import resolve_asset_path, support_usda_for_parquet
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);motion=root/'motion';seq=motion/'whole_body/soma/sequence_id=fixture'
            (seq/'object').mkdir(parents=True);(seq/'robot_name=g1').mkdir()
            mesh=seq/'object/mesh.obj';mesh.write_text('v 1 2 3\n')
            urdf=seq/'object/mesh.urdf';urdf.write_text('<robot/>')
            stage=motion/'whole_body/reconstructed_stage';stage.mkdir();(stage/'fixture_support.usda').write_text('#usda 1.0\n')
            path=seq/'robot_name=g1/data.parquet'
            pq.write_table(pa.Table.from_pylist([{'object_mesh_paths':[str(mesh)],'object_urdf_paths':[str(urdf)],
                                                 'robot_root_position':[[1.,2.,3.],[4.,5.,6.]]}]),path)
            out=root/'first/assets'
            result=subprocess.run([sys.executable,'-I',str(ROOT/'scripts/tier3_materialize.py'),'--motion-root',str(motion),'--out',str(out)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            (root/'second').mkdir();out.rename(root/'second/assets')
            relocated=root/'second/assets/human_motion_data/whole_body/soma/sequence_id=fixture/robot_name=g1/data.parquet'
            row=pq.read_table(relocated).to_pylist()[0]
            self.assertEqual(row['robot_root_position'],[[1.,2.,3.],[4.,5.,6.]])
            self.assertEqual(Path(resolve_asset_path(row['object_mesh_paths'][0],reference_path=relocated)).read_text(),'v 1 2 3\n')
            self.assertEqual(Path(resolve_asset_path(row['object_urdf_paths'][0],reference_path=relocated)).read_text(),'<robot/>')
            self.assertEqual(support_usda_for_parquet(str(relocated),robot_name='g1').read_text(),'#usda 1.0\n')

class MhrSomaExportTest(unittest.TestCase):
    def test_cpu_export_of_repeated_initializer_has_exact_coverage_and_objects(self):
        import torch
        from tier3_align import load_soma
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            human=pickle.loads((ROOT/'data/tier3_native/ep18/mhr_init/ep18.pkl').read_bytes())
            schema=ROOT/'external/video_to_data/reconstruction/modules/v2d_cari4d/lib/cari4d/lib_mhr/schema.py'
            dims=runpy.run_path(str(schema))['MHR_PARAM_DIMS']
            pr={key:np.repeat(human[key][:1],2,axis=0) for key in dims}
            poses=np.tile(np.eye(4),(2,1,1));poses[:,:3,3]=[[1,2,3],[4,5,6]];pr['pose_abs']=poses
            torch.save({'frames':['000000','000001'],'pr':pr,
                        'metadata':{'ground_truth_used':False,'object_mesh_to_training_transform':np.eye(4).tolist()}},root/'bundle.pth')
            mesh=ROOT/'data/tier3_bridge/ep18_aligned_mesh.glb'
            assets=ROOT/'external/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/body_models/soma'
            result=subprocess.run([sys.executable,'-I',str(ROOT/'scripts/tier3_export.py'),'--bundle',str(root/'bundle.pth'),
                                   '--mesh',str(mesh),'--weights',str(ROOT/'data/tier3_weights'),
                                   '--checkpoint',str(ROOT/'data/tier3_weights/sam-3d-body-dinov3/model.ckpt'),
                                   '--mhr-model',str(ROOT/'data/tier3_weights/sam-3d-body-dinov3/assets/mhr_model.pt'),
                                   '--soma-assets',str(assets),'--out',str(root/'export'),'--device','cpu','--batch-size','1',
                                   '--python-deps',str(ROOT/'data/tier3_bridge/deps')],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            data=load_soma(root/'export/soma_params.npz')
            self.assertEqual(data['poses'].shape,(2,77,3));self.assertEqual(data['transl'].shape,(2,3))
            self.assertEqual(data['identity_coeffs'].shape,(1,45));self.assertEqual(data['scale_params'].shape,(1,68))
            assert_allclose(data['poses'][0],data['poses'][1],atol=1e-7)
            assert_allclose(data['transl'][0],data['transl'][1],atol=1e-7)
            np.testing.assert_array_equal(np.load(root/'export/poses.npy'),poses)
            self.assertEqual(json.loads((root/'export/filled_frames.json').read_text()),{'frames':[]})
            self.assertEqual((root/'export/reconstructed_mesh/output_aligned.glb').read_bytes(),mesh.read_bytes())



class NativePatchTest(unittest.TestCase):
    def test_applies_once_to_unpatched_source_and_then_skips(self):
        import shutil
        source=ROOT/'external/video_to_data/reconstruction/modules/v2d_sam2/lib/sam2_utils.py'
        before=source.read_text()
        self.assertNotIn('from sam2 import _C as _sam2_cuda',before)
        expected=before.replace('import torch\n','import torch\n# Load before Decord codec libraries: the reverse order segfaults in connected components.\nfrom sam2 import _C as _sam2_cuda\n',1)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);dest=root/'v2d_sam2/lib';dest.mkdir(parents=True)
            shutil.copyfile(source,dest/'sam2_utils.py')
            for _ in range(2):
                result=subprocess.run([sys.executable,'-I',str(ROOT/'scripts/tier3_native/apply_native_patches.py')],env={**os.environ,'NATIVE_MODULES':str(root)},capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertEqual((dest/'sam2_utils.py').read_text(),expected)


if __name__=='__main__':unittest.main()
