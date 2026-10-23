"""Video-confirmed rest medoids, ported from T3-REPAIR/repair_from_video.py.

The numerical method is unchanged: full-mesh support projection, maximum mean
RGB silhouette IoU, fixed-frame quintic translation/SO(3) blends. No scores.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]

def preflight(config, *, episode, frames, video_sha256, legacy_video_sha256=None):
    """T3-EVIDENCE's input gate; legacy evidence needs an explicit video binding."""
    result = copy.deepcopy(config)
    if config.get('video_sha256', legacy_video_sha256) != video_sha256:
        raise ValueError('rest evidence video hash differs or is missing')
    if legacy_video_sha256 is not None and legacy_video_sha256 != video_sha256:
        raise ValueError('legacy rest evidence video hash differs')
    if config.get('episode', episode) != episode or config['frame_count'] != frames:
        raise ValueError('rest evidence episode/frame count mismatch')
    extended = 'evidence_version' in config or 'automatic_repair_status' in config
    if not extended and legacy_video_sha256 is None:
        raise ValueError('legacy rest evidence requires explicit video hash binding')
    if any(type(config[k]) is not int or config[k] < 2 for k in ('image_width', 'image_height')):
        raise ValueError('positive integer image dimensions required')
    occupied = [False] * frames
    reasons = list(config.get('automatic_repair_holds', []))
    if not config['rests']:
        reasons.append('no confirmed long pose-rest; preserve motion, do not run a no-op repair')
    for phase in result['rests']:
        s, e = phase['rest']; l, r = phase['blend_extent']
        if any(type(x) is not int for x in (s, e, l, r)) or not 0 <= l <= s < e <= r <= frames or any(occupied[l:r]):
            raise ValueError('invalid or overlapping repair intervals')
        occupied[l:r] = [True] * (r-l)
        allowed = phase.get('automatic_fit_allowed', not extended)
        if allowed is not True:
            reasons.append(f'partial/thin contour for rest {s}:{e}')
        fit = phase['fit_frames']
        if not fit or any(type(f) is not int or not s <= f < e or str(f) not in config['polygons'] for f in fit):
            raise ValueError('missing in-rest RGB contour')
        for f in fit:
            polygon = config['polygons'][str(f)]
            if len(polygon) < 3 or any(len(p) != 2 or any(type(x) not in (int, float) or not math.isfinite(x) for x in p) for p in polygon):
                raise ValueError('finite RGB polygon with at least three points required')
            if config.get('contour_metadata', {}).get(str(f), {}).get('fit_eligible', True) is not True:
                reasons.append(f'partial/thin contour at fit frame {f}')
        support = phase['support'] if extended else phase.get('support', 'ground')
        if support != 'ground' and phase.get('support_plane') is None:
            reasons.append(f'explicit aligned plane missing for {support} rest {s}:{e}')
        elif support == 'ground' and allowed is True and extended:
            phase.pop('support_plane', None)
        if 'support_plane' in phase and phase['support_plane'] is not None:
            plane = phase['support_plane']
            if len(plane) != 4 or any(type(x) not in (int, float) or not math.isfinite(x) for x in plane):
                raise ValueError('finite four-element plane required')
            norm = math.sqrt(sum(x*x for x in plane[:3]))
            if norm < 1e-9:
                raise ValueError('nondegenerate support plane required')
            phase['support_plane'] = [x/norm for x in plane]
    protected = occupied.copy()
    for s, e in config['protected_motion']:
        if any(type(x) is not int for x in (s, e)) or not 0 <= s < e <= frames or any(protected[s:e]):
            raise ValueError('protected real/unknown motion overlaps a repair')
        protected[s:e] = [True] * (e-s)
    if extended and not all(protected):
        raise ValueError('repair/protected intervals must cover the source timeline')
    return result, list(dict.fromkeys(reasons))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def mask(vertices,pose,K,shape):
    import numpy as np
    import cv2
    xyz=vertices@pose[:3,:3].T+pose[:3,3]
    if np.any(xyz[:,2]<=0):raise ValueError('mesh crosses camera plane')
    uv=xyz@K.T;uv=uv[:,:2]/uv[:,2,None]
    out=np.zeros(shape,np.uint8)
    cv2.fillConvexPoly(out,np.rint(cv2.convexHull((uv*.5).astype(np.float32))).astype(np.int32),1)
    return out.astype(bool)

def polygon_mask(poly,shape):
    import numpy as np
    import cv2
    out=np.zeros(shape,np.uint8);cv2.fillConvexPoly(out,np.rint(np.asarray(poly)*.5).astype(np.int32),1);return out.astype(bool)

def metrics(pred,target):
    import numpy as np
    py,px=np.where(pred);ty,tx=np.where(target)
    if not len(px) or not len(tx):raise ValueError('empty image mask')
    delta=np.hypot(px.mean()-tx.mean(),py.mean()-ty.mean())
    return dict(iou=float((pred&target).sum()/(pred|target).sum()),centroid_error_px=float(delta*2),centroid_error_normalized=float(delta/np.hypot(np.ptp(tx),np.ptp(ty))))

def support_pose(pose,vertices,plane):
    out=pose.copy();bottom=(vertices@(out[:3,:3].T@plane[:3])+out[:3,3]@plane[:3]+plane[3]).min()
    out[:3,3]+=(.0001-bottom)*plane[:3];return out

def repair(poses,vertices,plane,camera_from_aligned,K,config):
    import numpy as np
    from scipy.spatial import ConvexHull
    from scipy.spatial.transform import Rotation
    out=poses.copy();hull=vertices[ConvexHull(vertices).vertices];shape=(config['image_height']//2,config['image_width']//2)
    evidence=[]
    for phase in config['rests']:
        start,end=phase['rest'];left,right=phase['blend_extent'];fit=phase['fit_frames']
        support=plane
        if 'support_plane' in phase:
            support=np.asarray(phase['support_plane'],float);support=support/np.linalg.norm(support[:3])
        targets=[polygon_mask(config['polygons'][str(f)],shape) for f in fit]
        scores=[]
        for frame in range(start,end):
            anchor=support_pose(poses[frame],vertices,support)
            pred=mask(hull,camera_from_aligned@anchor,K,shape)
            scores.append(float(np.mean([metrics(pred,t)['iou'] for t in targets])))
        selected=start+int(np.argmax(scores));anchor=support_pose(poses[selected],vertices,support)
        ids=np.arange(left,right);weight=np.ones(len(ids))
        if left<start:
            t=np.clip((ids-left)/(start-left),0,1);weight*=10*t**3-15*t**4+6*t**5
        if right>end:
            t=np.clip((right-1-ids)/(right-end),0,1);weight*=10*t**3-15*t**4+6*t**5
        delta=Rotation.from_matrix(anchor[:3,:3]@np.swapaxes(poses[ids,:3,:3],1,2)).as_rotvec()
        out[ids,:3,:3]=Rotation.from_rotvec(delta*weight[:,None]).as_matrix()@poses[ids,:3,:3]
        out[ids,:3,3]=(1-weight[:,None])*poses[ids,:3,3]+weight[:,None]*anchor[:3,3]
        # Exact rest and protected-array contracts, independent of rotation roundoff.
        out[start:end]=anchor
        for f in ids:
            bottom=(vertices@(out[f,:3,:3].T@support[:3])+out[f,:3,3]@support[:3]+support[3]).min()
            if bottom<.0001:out[f,:3,3]+=(.0001-bottom)*support[:3]
        evidence.append(dict(rest=[start,end],blend_extent=[left,right],fit_frames=fit,selected_source_frame=selected,mean_fit_iou=scores[selected-start],all_frame_candidate_iou=scores,anchor=anchor.tolist()))
    return out,evidence

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path)
    p.add_argument('--camera', type=Path)
    p.add_argument('--evidence', type=Path, required=True)
    p.add_argument('--evidence-sha256')
    p.add_argument('--episode', type=int, required=True)
    p.add_argument('--frames', type=int, required=True)
    p.add_argument('--video-sha256', required=True)
    p.add_argument('--legacy-video-sha256')
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--python-deps', type=Path, default=ROOT / 'data/tier3_bridge/deps')
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    if a.evidence_sha256 and sha(a.evidence) != a.evidence_sha256:
        raise ValueError('rest evidence bytes changed')
    config, reasons = preflight(json.loads(a.evidence.read_text()), episode=a.episode,
        frames=a.frames, video_sha256=a.video_sha256, legacy_video_sha256=a.legacy_video_sha256)
    if reasons or a.check_only:
        print(json.dumps({'episode': a.episode, 'status': 'HOLD' if reasons else 'INPUT_COMPATIBILITY_ONLY', 'reasons': reasons}))
        return 2 if reasons else 0
    if any(x is None for x in (a.source, a.camera, a.out)):
        p.error('--source, --camera and --out are required for repair')
    if a.out.exists():
        raise FileExistsError('choose a fresh destination; existing candidates are preserved')
    sys.path.insert(0, str(a.python_deps))
    import cv2
    cv2.setNumThreads(1)
    import numpy as np
    import trimesh
    source = a.source.resolve()
    poses = np.load(source / 'poses.npy', allow_pickle=False)
    if poses.shape != (a.frames, 4, 4) or not np.isfinite(poses).all():
        raise ValueError('finite Tx4x4 required on evidence frame grid')
    if not np.allclose(poses[:,3], [0,0,0,1]) or not np.allclose(poses[:,:3,:3] @ np.swapaxes(poses[:,:3,:3],1,2), np.eye(3), atol=1e-6) or not np.allclose(np.linalg.det(poses[:,:3,:3]),1,atol=1e-6):
        raise ValueError('rigid SE3 required')
    plane = np.asarray(json.loads((source / 'ground_plane.json').read_text())['plane'], float)
    if plane.shape != (4,) or not np.isfinite(plane).all() or np.linalg.norm(plane[:3]) < 1e-9:
        raise ValueError('finite nondegenerate ground plane required')
    plane /= np.linalg.norm(plane[:3])
    mesh = trimesh.load(source / 'reconstructed_mesh/output_aligned.glb', force='mesh', process=False)
    vertices = np.asarray(mesh.vertices)
    alignment = json.loads((source / 'tier3_alignment.json').read_text())
    if alignment['reference_frames_used'] != [0] or alignment['scale'] != 1:
        raise ValueError('expected retained frame-zero rigid alignment')
    A = np.eye(4); A[:3,:3] = alignment['rotation']; A[:3,3] = alignment['translation']
    K = np.asarray(json.loads(a.camera.read_text())['intrinsics'])
    repaired, evidence = repair(poses, vertices, plane, np.linalg.inv(A), K, config)
    for s, e in config['protected_motion']:
        np.testing.assert_array_equal(repaired[s:e], poses[s:e])
    files = ['poses.npy', 'soma_params.npz', 'ground_plane.json', 'filled_frames.json',
             'tier3_alignment.json', 'reconstructed_mesh/output_aligned.glb']
    records = {name: sha(source / name) for name in files}
    shutil.copytree(source, a.out)
    np.save(a.out / 'poses.npy', repaired)
    correction = np.linalg.norm(repaired[:,:3,3] - poses[:,:3,3], axis=1)
    report = dict(status='UNQUALIFIED; requires unchanged G1 retargeting and frozen gates',
        source=str(source), source_sha256=records, evidence_sha256=sha(a.evidence),
        camera_sha256=sha(a.camera), video_sha256=a.video_sha256,
        legacy_video_sha256=a.legacy_video_sha256, phases=evidence,
        protected_motion=config['protected_motion'],
        method='maximum mean RGB IoU supported pose medoid over every reconstruction frame of each reviewed rest; quintic SO3/translation blend at fixed source indices',
        object_correction_m=dict(max=float(correction.max()), p95=float(np.percentile(correction,95))),
        unchanged='SOMA human/identity, mesh, retained alignment, plane, frame IDs; object-only rest correction',
        scene_assets_mode='released-engineering-only; ruling pending')
    (a.out / 'video_rest_repair.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'phases'}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
