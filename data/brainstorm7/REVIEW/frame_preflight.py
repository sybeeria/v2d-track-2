#!/usr/bin/env python3
"""Pre-export check for Tier 2/3: pack_no_clean.convert_frame's rigid-transform test, run before any GPU export.
For each manifest sequence, the training reference's object poses must be ONE rigid transform (<=1 mm, <=1e-3 rad over
all frames) of WORK/sequences/<seq>/poses.npy, the file frame_references.py turns into the packer's input NPZ.
Usage: frame_preflight.py MANIFEST WORK_DIR [LABEL]. Exit 1 if any sequence would make pack_no_clean refuse.
Assumes the evaluator records the training reference's object pose up to one rigid transform [I]; the kit's
object_pose math is used unchanged. Reads no clean reference."""
import json,sys
from pathlib import Path
import numpy as np,pyarrow.parquet as pq
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'external/kit/v2d_submission_kit'))
from v2dlb import object_pose as op
def reference_xyzw(path):
    row=pq.ParquetFile(path).read(columns=['object_body_position','object_body_wxyz']).to_pylist()[0]
    pos=np.asarray(row['object_body_position'],float)[:,0];wxyz=np.asarray(row['object_body_wxyz'],float)[:,0]
    return np.c_[pos,wxyz[:,1:],wxyz[:,:1]]
def gaps(evaluator,poses):
    target=np.c_[poses[:,:3,3],Rotation.from_matrix(poses[:,:3,:3]).as_quat()]
    if evaluator.shape!=target.shape:return float('inf'),float('inf')
    evaluator,target=[op._normalise_quaternions(a,'pose',ValueError) for a in (evaluator,target)]
    q=op._quaternion_multiply(target[:1,3:],evaluator[:1,3:]*[-1.,-1.,-1.,1.])[0];q/=np.linalg.norm(q)
    rotation=op._rotation_matrix(q[None])[0];offset=target[0,:3]-rotation@evaluator[0,:3]
    return (float(np.linalg.norm(evaluator[:,:3]@rotation.T+offset-target[:,:3],axis=1).max()),
            float(op._quaternion_geodesic_angle(op._quaternion_multiply(q,evaluator[:,3:]),target[:,3:]).max()))
manifest,work=json.loads(Path(sys.argv[1]).read_text()),Path(sys.argv[2]);label=sys.argv[3] if len(sys.argv)>3 else None
bad=0
for entry in manifest['sequences']:
    arms=entry['arms'];arm=arms[label] if label else next(iter(arms.values()))
    p,a=gaps(reference_xyzw(Path(arm['training_reference']['path'])),np.load(work/'sequences'/entry['sequence']/'poses.npy',allow_pickle=False))
    ok=p<=1e-3 and a<=1e-3;bad+=not ok
    print(f"{'PASS' if ok else 'REFUSE'} {entry['sequence']} {p:.6f} m {a:.6f} rad")
print(f'{len(manifest["sequences"])-bad}/{len(manifest["sequences"])} sequences pass the packer frame check');sys.exit(1 if bad else 0)
