"""Label-free diagnostics for generated Tier 3 inputs and a G1 reference.

Only the guarded loader may open Tier 1, and it exposes frame zero. Thresholds
flag discontinuities for inspection; they are not reconstruction accuracy scores.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import trimesh
from PIL import ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'external/video_to_data/robotic_grounding/flash_chord/src'))
from tier3_align import ALIGN_JOINTS, CV_TO_SOMA, load_soma, load_tier1_frame0, object_poses_matrix
from tier3_prompts import read_frame


def summary(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {'count': 0, 'min': None, 'median': None, 'p95': None, 'max': None}
    if not np.isfinite(values).all():
        raise ValueError('non-finite quality input')
    return {'count': int(values.size), 'min': float(values.min()), 'median': float(np.median(values)),
            'p95': float(np.percentile(values, 95)), 'max': float(values.max())}


def continuity(values, fps, *, rotation=False, speed_limit=5., acceleration_limit=50.):
    values = np.asarray(values)
    if rotation:
        velocity = (Rotation.from_matrix(values[1:]) * Rotation.from_matrix(values[:-1]).inv()).as_rotvec() * fps
    else:
        velocity = np.diff(values, axis=0) * fps
    acceleration = np.diff(velocity, axis=0) * fps
    speed = np.linalg.norm(velocity, axis=-1)
    accel = np.linalg.norm(acceleration, axis=-1)
    if speed.ndim > 1:
        speed = speed.max(axis=tuple(range(1, speed.ndim)))
        accel = accel.max(axis=tuple(range(1, accel.ndim)))
    return {'speed': summary(speed), 'acceleration': summary(accel),
            'speed_limit': speed_limit, 'acceleration_limit': acceleration_limit,
            'speed_outlier_destination_frames': (np.flatnonzero(speed > speed_limit)+1).tolist(),
            'acceleration_outlier_destination_frames': (np.flatnonzero(accel > acceleration_limit)+2).tolist()}


def mesh_plane_bounds(vertices, poses, plane):
    plane = np.asarray(plane, dtype=float)
    plane = plane / np.linalg.norm(plane[:3])
    heights = []
    for pose in poses:
        signed = vertices @ (pose[:3,:3].T @ plane[:3]) + pose[:3,3] @ plane[:3] + plane[3]
        heights.append([signed.min(), signed.max()])
    return np.asarray(heights)


def contact_gaps(vertices, poses, joints, row):
    """Same vertex distances and retargeter labels for challenge and external inputs."""
    from hoi import FINGERTIPS
    tree = cKDTree(vertices)
    frames = len(poses)
    object_pos = np.asarray(row['object_body_position'])[:, 0]
    object_rot = Rotation.from_quat(np.asarray(row['object_body_wxyz'])[:, 0], scalar_first=True).as_matrix()
    ee = np.asarray(row['ee_pose_w'])
    gaps = {}
    for k, side in enumerate(row['hand_sides']):
        source_hand = joints[:, FINGERTIPS[side.title()]] @ CV_TO_SOMA
        local = np.einsum('tkj,tji->tki', source_hand - poses[:, None, :3, 3], poses[:, :3, :3])
        human_gap = tree.query(local.reshape(-1, 3))[0].reshape(frames, -1).min(1)
        palm_id = row['ee_link_names'].index(side + '_hand_palm_link')
        palm_local = np.einsum('tj,tji->ti', ee[:, palm_id, :3] - object_pos, object_rot)
        robot_gap = tree.query(palm_local)[0]
        labelled = np.asarray(row['hand_contact_active'][k]) > .5
        gaps[side] = {'labelled_frames': np.flatnonzero(labelled).tolist(),
                      'human_fingertip_nearest_vertex_m_on_labels': summary(human_gap[labelled]),
                      'G1_palm_nearest_vertex_m_on_labels': summary(robot_gap[labelled]),
                      'human_fingertip_nearest_vertex_m_all_frames': summary(human_gap)}
    return gaps


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('sequence','raw','parquet','video','wild-export','tier1-root','out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--episode', type=int, required=True)
    a = p.parse_args()
    import torch
    torch.set_num_threads(2)
    from robotic_grounding.retarget.read_soma import SOMA
    from flash_chord.data.motion_v1 import load_motion_v1
    from flash_chord.assets.registry import support_usda_for_parquet
    from pxr import Usd
    import subprocess
    info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(a.video)]))['streams'][0]
    expected=int(info['nb_frames'])
    fps_num, fps_den=map(float,info['r_frame_rate'].split('/')); fps=fps_num/fps_den
    row=pq.read_table(a.parquet).to_pylist()[0]  # Generated reference only.
    motion=load_motion_v1(str(a.parquet),control_fps=50.,source_frame_playback=True)
    source=load_soma(a.sequence/'soma_params.npz')
    poses=np.load(a.sequence/'poses.npy',allow_pickle=False)
    if expected != len(poses) or expected != len(source['poses']) or expected != motion.num_frames:
        raise ValueError('video/human/object/G1 frame coverage differs')
    soma=SOMA(device=torch.device('cpu'))
    decoded=soma.load_motion(a.sequence/'soma_params.npz')
    joints=np.asarray(decoded['joints'])
    reference=load_tier1_frame0(a.tier1_root,a.episode)
    # Decode only the one permitted reference frame through the existing helper.
    from tier3_align import soma_frame0_joints
    import tempfile
    a.out.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=a.out) as tmp:
        ref=dict(poses=reference.poses_rotvec,transl=reference.transl,
                 joint_names=np.asarray(reference.joint_names),identity_model_type=np.array('mhr'),
                 identity_coeffs=reference.identity_coeffs[None],scale_params=reference.scale_params[None],
                 bone_length_flexibles=reference.bone_length_flexibles,unit=np.array('meters'),keep_root=np.array(False))
        target=soma_frame0_joints(ref,soma,Path(tmp)/'frame0.npz')
    ids=[list(source['joint_names']).index(n) for n in ALIGN_JOINTS]
    diff=joints[0,ids] @ CV_TO_SOMA - target[ids]
    ref_object,_=object_poses_matrix(reference.object_pose_wxyz,reference.object_visible)
    mesh=trimesh.load(a.sequence/'reconstructed_mesh/output_aligned.glb',force='mesh',process=False)
    plane=json.loads((a.sequence/'ground_plane.json').read_text())['plane']
    heights=mesh_plane_bounds(mesh.vertices,poses,plane)
    object_pos=np.asarray(row['object_body_position'])[:,0]
    object_rot=Rotation.from_quat(np.asarray(row['object_body_wxyz'])[:,0],scalar_first=True).as_matrix()
    robot_poses=np.tile(np.eye(4),(expected,1,1));robot_poses[:,:3,:3]=object_rot;robot_poses[:,:3,3]=object_pos
    runtime_heights=mesh_plane_bounds(mesh.vertices,robot_poses,[0,0,1,0])
    gaps=contact_gaps(mesh.vertices, poses, joints, row)
    support=support_usda_for_parquet(str(a.parquet),robot_name='g1')
    if support is None or not support.is_file():
        raise FileNotFoundError('support stage did not resolve')
    stage=Usd.Stage.Open(str(support))
    if stage is None:
        raise ValueError('support stage failed to open')
    enabled=any(x.GetTypeName() in ('Cylinder','Cube') for x in stage.Traverse())
    payload=pickle.loads(row['source_payload'])  # Our retargeter's trusted generated payload.
    report={'evidence':'measured diagnostics; no later Tier 1 reference was read',
            'coverage':{'video':expected,'SOMA':len(source['poses']),'object':len(poses),'G1_loader':motion.num_frames,'fps':fps},
            'threshold_policy':'fixed inspection flags: translation 5m/s,50m/s2; rotation 20rad/s,200rad/s2; no smoothing',
            'object_translation':continuity(poses[:,:3,3],fps),
            'human_root_translation':continuity(source['transl'],fps),
            'object_rotation_rad':continuity(poses[:,:3,:3],fps,rotation=True,speed_limit=20.,acceleration_limit=200.),
            'human_joint_translation':continuity(joints,fps),
            'G1_root_translation':continuity(np.asarray(row['robot_root_position']),fps),
            'G1_joint_angles_rad':continuity(np.asarray(row['robot_joint_positions'])[...,None],fps,speed_limit=20.,acceleration_limit=200.),
            'object_bottom_vs_released_plane_m':summary(heights[:,0]),
            'object_top_vs_released_plane_m':summary(heights[:,1]),
            'object_bottom_below_minus_2cm_frames':np.flatnonzero(heights[:,0]<-.02).tolist(),
            'G1_object_bottom_vs_z0_m':summary(runtime_heights[:,0]),
            'G1_object_bottom_below_minus_2cm_frames':np.flatnonzero(runtime_heights[:,0]<-.02).tolist(),
            'contacts':gaps,'contact_limitations':'labels derived from initializer geometry, not independent annotations; distances to vertices; a palm is not a fingertip',
            'frame0':{'reference_frames_used':[0],'joint_rmse_m':float(np.sqrt(np.square(diff).sum(1).mean())),
                      'object_translation_error_m':float(np.linalg.norm(poses[0,:3,3]-ref_object[0,:3,3])),
                      'object_rotation_error_rad':float(Rotation.from_matrix(poses[0,:3,:3]@ref_object[0,:3,:3].T).magnitude()),
                      'limitation':'frame zero is the alignment calibration, not held-out accuracy'},
            'support_stage':str(support),'scene_support_enabled':enabled,
            'ground_alignment':payload['ground_alignment'],'ik_error':summary(row['ik_error_per_frame'])}
    (a.out/'quality.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    # Render the raw reconstruction in the input camera for three video frames.
    raw=soma.load_motion(a.raw/'soma_params.npz')['joints'] @ CV_TO_SOMA
    raw_poses=np.load(a.raw/'poses.npy')
    raw_mesh=trimesh.load(a.raw/'reconstructed_mesh/output_aligned.glb',force='mesh',process=False)
    intrinsics=np.asarray(json.loads(a.wild_export.read_text())['intrinsics'])
    for frame in [0,expected//2,expected-1]:
        im=read_frame(a.video,frame)
        d=ImageDraw.Draw(im)
        project=lambda xyz: (xyz @ intrinsics.T)[:,:2] / xyz[:,2,None]
        uv=project(raw[frame])
        for x,y in uv:
            if np.isfinite([x,y]).all(): d.ellipse((x-3,y-3,x+3,y+3),fill='#00ff80')
        posed=raw_mesh.vertices @ raw_poses[frame,:3,:3].T+raw_poses[frame,:3,3]
        visible=posed[posed[:,2]>0]
        projected=project(visible)
        bounds=np.r_[projected.min(0),projected.max(0)]
        d.rectangle(bounds.tolist(),outline='#ff5555',width=4)
        d.text((20,20),f'ep{a.episode} frame {frame}: initializer SOMA joints / object projected bounds',fill='white')
        im.save(a.out/f'overlay_{frame:06d}.png')
    print(json.dumps({'frames':expected,'support':str(support),'support_enabled':enabled,'quality':str(a.out/'quality.json')}))


if __name__ == '__main__':
    main()
