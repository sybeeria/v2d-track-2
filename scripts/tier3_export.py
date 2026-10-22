"""Decode a trusted CARI4D/initializer bundle to metric SOMA, on CPU or CUDA."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


def decoder_imports(project: Path) -> None:
    """Import only the MHR decoder; the parent package eagerly imports image models."""
    source = project / 'external/video_to_data/reconstruction/modules/v2d_sam3d_body/lib'
    if 'sam_3d_body' not in sys.modules:
        package = types.ModuleType('sam_3d_body')
        package.__path__ = [str(source / 'sam_3d_body')]
        sys.modules['sam_3d_body'] = package
    sys.path.insert(0, str(project / 'external/video_to_data/reconstruction/modules/v2d_cari4d/lib/cari4d'))
    os.environ.setdefault('MOMENTUM_ENABLED', '0')


def export_bundle(bundle_path: Path, mesh: Path, weights: Path, soma_assets: Path,
                  out: Path, *, device='cuda', batch_size=8, project=ROOT,
                  checkpoint=None, mhr_model=None) -> dict:
    import torch
    decoder_imports(project)
    from lib_mhr import MHRLayer, MHR_PARAM_DIMS
    from lib_mhr.mhr_layer import MHRParametricBackend
    from soma import SOMALayer
    from soma.pose_inversion import PoseInversion
    from soma.geometry.rig_utils import remove_joint_orient_local
    from soma.geometry.transforms import matrix_to_rotvec
    from soma.io import save_soma_npz
    from soma.units import Unit

    bundle = torch.load(bundle_path, map_location='cpu', weights_only=False)
    names = [str(f) for f in bundle['frames']]
    if not names or names != [f'{i:06d}' for i in range(len(names))]:
        raise ValueError('CARI4D frame IDs must be the full zero-based video timeline')
    if bundle['metadata'].get('ground_truth_used') is not False:
        raise ValueError('bundle must declare ground_truth_used=False')
    if not np.allclose(bundle['metadata']['object_mesh_to_training_transform'], np.eye(4), atol=1e-6):
        raise ValueError('expected output_aligned mesh with identity training transform')
    # Validate before creating output or loading the large models.
    from tier3_initializers import rigid
    poses = rigid(bundle['pr']['pose_abs'], len(names))
    pr = {}
    for key, width in MHR_PARAM_DIMS.items():
        values = np.asarray(bundle['pr'][key], dtype=np.float32)
        if values.shape != (len(names), width) or not np.isfinite(values).all():
            raise ValueError(f'Invalid {key}')
        pr[key] = values.copy()
    identity_drift = {}
    for key in ('mhr_shape', 'mhr_scale'):
        fixed = np.median(pr[key], axis=0, keepdims=True)
        identity_drift[key] = float(np.max(np.abs(pr[key] - fixed)))
        pr[key][:] = fixed
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    if out.exists():
        raise FileExistsError(out)
    if checkpoint is None and mhr_model is None:
        layer = MHRLayer.from_mhr_assets(mhr_assets_root=str(weights), device=device)
    else:
        layer = MHRLayer(MHRParametricBackend(mhr_assets_root=weights,
                         checkpoint_path=checkpoint, mhr_model_path=mhr_model, device=device))
    soma = SOMALayer(data_root=str(soma_assets), identity_model_type='mhr', device=device,
                     mode='warp', output_unit=Unit.CENTIMETERS, enable_procedural_transforms=False)
    inv = PoseInversion(soma, low_lod=False)
    rotations, translations, errors, resolved_params = [], [], [], []
    for start in range(0, len(names), batch_size):
        stop = min(start + batch_size, len(names))
        params = {key: torch.as_tensor(value[start:stop], dtype=torch.float32, device=device)
                  for key, value in pr.items()}
        with torch.no_grad():
            context = layer.backend._vertices_context(params, detach_fixed=True)
            trans, body, shape, scale = layer.backend._mutable_vertices_inputs(params, context)
            vertices, resolved = context.head.mhr_forward(
                global_trans=trans * context.flip, global_rot=context.global_rot, body_pose_params=body,
                hand_pose_params=context.hand, scale_params=scale, shape_params=shape, expr_params=context.face,
                return_keypoints=False, return_joint_coords=False, return_model_params=True, return_joint_rotations=False)
            if start == 0 and not torch.allclose(vertices * context.flip, layer(params).vertices, atol=1e-4, rtol=1e-5):
                raise ValueError('native/CV MHR bridge disagrees with CARI4D decoder')
        inv.prepare_identity(shape, resolved[:, 136:], kwargs={'bone_length_flexibles': resolved[:, 130:136]})
        result = inv.fit(vertices * 100, body_iters=2, full_iters=1, finger_iters=0,
                         lie_iters=3, lie_lambda=0.1, batch_size=None)
        rotations.append(result['rotations'].cpu())
        translations.append(result['root_translation'].cpu())
        errors.append(result['per_vertex_error'].cpu())
        resolved_params.append(resolved.cpu())
        print(f'SOMA {stop}/{len(names)}', flush=True)
    rotation = torch.cat(rotations)
    relative = remove_joint_orient_local(rotation.to(device), soma._t_pose_orient, soma._t_pose_orient_parent_T)
    rotvec = matrix_to_rotvec(relative.reshape(-1, 3, 3)).reshape(len(names), -1, 3).cpu()
    resolved = torch.cat(resolved_params)
    error_m = torch.cat(errors).numpy() / 100
    out.mkdir(parents=True)
    save_soma_npz(str(out / 'soma_params.npz'), rotvec, torch.cat(translations) / 100,
                  joint_names=list(soma.rig_data['joint_names']), identity_model_type='mhr',
                  identity_coeffs=pr['mhr_shape'][:1], scale_params=resolved[:1, 136:].numpy(),
                  joint_orient=soma._t_pose_orient, unit='meters', keep_root=False,
                  extra_arrays={'bone_length_flexibles': resolved[:, 130:136].numpy()})
    np.save(out / 'poses.npy', poses)
    (out / 'reconstructed_mesh').mkdir()
    shutil.copyfile(mesh, out / 'reconstructed_mesh/output_aligned.glb')
    (out / 'filled_frames.json').write_text(json.dumps({'frames': []}) + '\n')
    report = {'bundle': str(bundle_path), 'frames': len(names), 'device': device,
              'object_frame': 'OpenCV camera 0', 'human_frame': 'SOMA native Y-up', 'unit': 'meters',
              'identity': 'per-sequence median of reconstruction only',
              'identity_max_parameter_change': identity_drift,
              'inversion_vertex_error_m': {'mean': float(error_m.mean()), 'max': float(error_m.max())},
              'exporter': 'MHR decoder + SOMA PoseInversion; iterations 2/1/0/3, full LOD',
              'source_metadata': bundle['metadata']}
    (out / 'tier3_export.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'mesh', 'weights', 'soma-assets', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--project', type=Path, default=ROOT)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--mhr-model', type=Path)
    parser.add_argument('--python-deps', type=Path, help='Explicit isolated dependency directory')
    args = parser.parse_args()
    if args.python_deps:
        sys.path.insert(0, str(args.python_deps.resolve()))
    import torch
    torch.set_num_threads(2)
    print(json.dumps(export_bundle(args.bundle, args.mesh, args.weights, args.soma_assets, args.out,
                                  device=args.device, batch_size=args.batch_size, project=args.project,
                                  checkpoint=args.checkpoint, mhr_model=args.mhr_model), indent=2))


if __name__ == '__main__':
    main()
