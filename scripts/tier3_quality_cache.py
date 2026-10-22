"""Recover native joints from trusted retarget output; decode only sparse RGB/frame0.

Run with --quality-script pointing to the patched script in a disposable copy.
No all-frame native mesh decode is performed. All candidate joints are reused
from the actual retargeter, with the recorded ground/root transforms inverted.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import sys
import tempfile
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('quality-script','sequence','raw','parquet','tier1-root','out'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--episode',type=int,required=True)
    a=p.parse_args()
    sys.path.insert(0,str(a.quality_script.parent))
    spec=importlib.util.spec_from_file_location('quality_patched',a.quality_script)
    quality=importlib.util.module_from_spec(spec);spec.loader.exec_module(quality)
    from tier3_align import load_soma, load_tier1_frame0, soma_frame0_joints
    from robotic_grounding.retarget.robot_config import load_robot_config
    data=load_soma(a.sequence/'soma_params.npz')
    row=pq.ParquetFile(a.parquet).read().to_pylist()[0]
    if not Path(row['raw_motion_file']).samefile(a.sequence/'soma_params.npz'):
        raise ValueError('retarget payload belongs to another source')
    payload=pickle.loads(row['source_payload'])
    for name in ('identity_coeffs','scale_params'):
        np.testing.assert_array_equal(np.asarray(payload['soma_'+name],np.float32),data[name][0].astype(np.float32))
    world=np.asarray(load_robot_config('g1').r_world)
    g=payload['ground_alignment']
    rotation=world.T@np.asarray(g['ground_level_rotation_robot'])@world
    translation=world.T@np.asarray(g['ground_level_translation_robot'])
    normalized=(np.asarray(payload['soma_joints'])-translation)@rotation
    first_inverse=Rotation.from_rotvec(data['poses'][0,0].astype(np.float32).astype(float)).as_matrix().T
    joints=normalized@first_inverse+data['transl'][0].astype(np.float32).astype(float)
    if joints.shape != (len(data['poses']),77,3) or not np.isfinite(joints).all():
        raise ValueError('invalid retargeter joints')
    # Record all inputs before constructing the model and check again before publication.
    inputs={key:dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            for key,path in [('sequence',a.sequence/'soma_params.npz'),('raw',a.raw/'soma_params.npz'),('parquet',a.parquet)]}
    reference=load_tier1_frame0(a.tier1_root,a.episode)
    ref=quality.frame0_data(reference)
    a.out.mkdir(parents=True,exist_ok=False)
    import torch
    from robotic_grounding.retarget.read_soma import SOMA
    torch.set_num_threads(2)
    soma=SOMA(device=torch.device('cpu'))
    raw=load_soma(a.raw/'soma_params.npz')
    frames=[0,len(data['poses'])//2,len(data['poses'])-1]
    native_errors=[];raw_joints=[]
    with tempfile.TemporaryDirectory(dir=a.out) as temp:
        temp=Path(temp)
        target=soma_frame0_joints(ref,soma,temp/'reference_frame0.npz')
        for frame in frames:
            for name, source in [('candidate',data),('raw',raw)]:
                sliced={key:value[frame:frame+1] if key in ('poses','transl','bone_length_flexibles') else value[:1] if key in ('identity_coeffs','scale_params') else value for key,value in source.items()}
                path=temp/f'{name}.npz';np.savez(path,**sliced)
                decoded=np.asarray(soma.load_motion(path)['joints'])[0]
                if name=='candidate':
                    error=float(np.abs(decoded-joints[frame]).max());native_errors.append(error)
                    if error > 2e-6:
                        raise ValueError(f'native joint payload transform check failed at {frame}: {error}')
                else:
                    raw_joints.append(decoded)
        for value in inputs.values():
            if hashlib.sha256(Path(value['path']).read_bytes()).hexdigest()!=value['sha256']:
                raise ValueError('input changed during decode')
        archive=a.out/'joints.npz'
        np.savez(archive,joints=joints,target=target,raw_joints=np.asarray(raw_joints),raw_frames=np.asarray(frames))
    record=dict(schema=1,inputs=inputs,frame0_arrays_sha256=quality.arrays_sha256(ref),
                archive=dict(path=archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest()),
                evidence='native retargeter joints, no cached quality metrics; sparse native transform validation',
                native_validation_frames=frames,native_max_element_errors_m=native_errors,
                native_validation_tolerance_m=2e-6,reference_frames_used=[0],
                source_to_robot=world.tolist(),ground_rotation_source=rotation.tolist(),ground_translation_source=translation.tolist(),
                decoder_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (a.out/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    main()
