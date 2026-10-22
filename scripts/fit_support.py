"""Fit released-layout support disks from fresh CLEAN G1 motion_v1 references only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import trimesh
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

PARAMS = dict(window_frames=9, linear_speed_m_s=0.03, angular_speed_rad_s=0.2,
              min_rest_frames=9, elevated_m=0.05, margin_m=0.01, thickness_m=0.04,
              rest_tolerance_m=0.005)


def load_reference(path):
    row = pq.read_table(path, columns=['object_root_position', 'object_root_axis_angle',
                                    'object_mesh_paths', 'hand_contact_active']).to_pylist()[0]
    positions = np.asarray(row['object_root_position'])
    rotations = Rotation.from_rotvec(row['object_root_axis_angle'])
    active = np.asarray(row['hand_contact_active']).any(axis=0)
    vertices = np.asarray(trimesh.load(row['object_mesh_paths'][0], force='mesh').vertices)
    if (positions.shape != (len(active), 3) or len(rotations) != len(active) or len(active) < 2
            or not all(np.isfinite(x).all() for x in (positions, vertices, rotations.as_quat()))):
        raise ValueError(f'{path}: invalid single-object reference')
    return positions, rotations, active, vertices


def world_frames(positions, rotations, vertices):
    return [vertices @ r.T + t for r, t in zip(rotations.as_matrix(), positions)]


def static_frames(positions, rotations, fps=30):
    # One-sided endpoint windows retain initial/final rests; path length prevents motion cancelling itself.
    half = PARAMS['window_frames'] // 2
    lo = np.maximum(np.arange(len(positions)) - half, 0)
    hi = np.minimum(np.arange(len(positions)) + half, len(positions) - 1)
    duration = (hi - lo) / fps
    distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))]
    angle = np.r_[0, np.cumsum((rotations[1:] * rotations[:-1].inv()).magnitude())]
    linear = (distance[hi] - distance[lo]) / duration
    angular = (angle[hi] - angle[lo]) / duration
    return (linear < PARAMS['linear_speed_m_s']) & (angular < PARAMS['angular_speed_rad_s'])


def fit_disks(positions, rotations, active, vertices, fps=30):
    world = world_frames(positions, rotations, vertices)
    bottom = np.array([w[:, 2].min() for w in world])
    rest = static_frames(positions, rotations, fps) & (bottom > PARAMS['elevated_m']) & ~active
    intervals = np.flatnonzero(np.diff(np.r_[False, rest, False])).reshape(-1, 2)
    disks = []
    for a, b in intervals:
        # The one-sided speed window already certifies qualifying endpoint frames.
        if b - a < PARAMS['min_rest_frames'] and a != 0 and b != len(world):
            continue
        points = np.concatenate(world[a:b])
        center = (points[:, :2].min(axis=0) + points[:, :2].max(axis=0)) / 2
        radius = float(np.linalg.norm(points[:, :2] - center, axis=1).max() + PARAMS['margin_m'])
        top = float(bottom[a:b].min())
        interval = [int(a), int(b)]
        for disk in disks:
            if (np.linalg.norm(center - disk['center_xy']) <= min(radius, disk['radius_m'])
                    and abs(top - disk['top_z_m']) <= 0.005):
                disk['radius_m'] = max(disk['radius_m'], float(np.linalg.norm(points[:, :2] - disk['center_xy'], axis=1).max() + PARAMS['margin_m']))
                disk['top_z_m'] = min(top, disk['top_z_m'])
                disk['intervals'].append(interval)
                break
        else:
            disks.append(dict(center_xy=center.tolist(), radius_m=radius, top_z_m=top,
                              thickness_m=PARAMS['thickness_m'], intervals=[interval]))
    return disks


def stage_text(disks):
    text = '#usda 1.0\n(\n    defaultPrim = "support_surfaces"\n    metersPerUnit = 1\n    upAxis = "Z"\n)\n\ndef Xform "support_surfaces"\n{\n'
    for i, disk in enumerate(disks):
        x, y = disk['center_xy']
        z = disk['top_z_m'] - disk['thickness_m'] / 2
        text += f'''    def Cylinder "object_{i}" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {{
        uniform token axis = "Z"
        double height = {disk['thickness_m']:.17g}
        color3f[] primvars:displayColor = [(0.91806465, 0.6436726, 0.8323664)]
        double radius = {disk['radius_m']:.17g}
        double3 xformOp:translate = ({x:.17g}, {y:.17g}, {z:.17g})
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }}
'''
    return text + '}\n'


def footprint_gap(points, center, radius):
    polygon = points[ConvexHull(points[:, :2]).vertices, :2] - center
    edges = np.roll(polygon, -1, axis=0) - polygon
    cross = edges[:, 0] * -polygon[:, 1] - edges[:, 1] * -polygon[:, 0]
    if (cross >= 0).all() or (cross <= 0).all():
        return -radius
    fraction = np.clip(-(polygon * edges).sum(axis=1) / (edges * edges).sum(axis=1), 0, 1)
    return float(np.linalg.norm(polygon + fraction[:, None] * edges, axis=1).min() - radius)


def audit_disks(positions, rotations, active, vertices, disks):
    world = world_frames(positions, rotations, vertices)
    result = []
    for disk in disks:
        top, thickness = disk['top_z_m'], disk['thickness_m']
        xy = np.array([footprint_gap(w, disk['center_xy'], disk['radius_m']) for w in world])
        vertical = np.array([max(w[:, 2].min() - top, top - thickness - w[:, 2].max()) for w in world])
        # A positive separating-axis gap proves separation for the entire mesh, not just sampled vertices.
        clearance = np.maximum(xy, vertical)
        intervals = []
        for a, b in disk['intervals']:
            intervals.append(dict(frames=[a, b], xy_overlap_frames=int((xy[a:b] <= 0).sum()),
                                  max_bottom_top_error_m=float(max(abs(w[:, 2].min() - top) for w in world[a:b]))))
        witnesses = []
        penetration = []
        for i in np.flatnonzero(active):
            w = world[i]
            inside = ((np.linalg.norm(w[:, :2] - disk['center_xy'], axis=1) < disk['radius_m'])
                      & (w[:, 2] < top) & (w[:, 2] > top - thickness))
            if inside.any():
                witnesses.append(int(i))
                penetration.append(float(top - w[inside, 2].min()))
        result.append(dict(**disk, rest_intervals=intervals,
                           carried_vertex_inside_frames=witnesses,
                           max_vertex_penetration_m=max(penetration, default=0.0),
                           frame0_bottom_top_error_m=float(world[0][:, 2].min() - top),
                           final_bottom_top_error_m=float(world[-1][:, 2].min() - top),
                           carried_frames=int(active.sum()),
                           carried_min_clearance_m=float(clearance[active].min()) if active.any() else None,
                           carried_intersection_candidates=int((clearance[active] < 0).sum())))
    return dict(frame0_mesh_bottom_z_m=float(world[0][:, 2].min()),
                final_mesh_bottom_z_m=float(world[-1][:, 2].min()), disks=result,
                # Lift-off and set-down touch the disk top; sinking deeper than the resting tolerance does not.
                carried_clearance_qualified=all(d['max_vertex_penetration_m'] <= PARAMS['rest_tolerance_m'] for d in result))


def fit_reference(reference, output):
    data = load_reference(reference)
    disks = fit_disks(*data)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = stage_text(disks)
    if not output.exists() or output.read_text() != text:
        output.write_text(text)
    return dict(parameters=PARAMS, **audit_disks(*data, disks))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('references', nargs='+', type=Path, help='fresh CLEAN G1 parquets')
    args = parser.parse_args()
    for reference in args.references:
        sequence = reference.parent.parent.name.removeprefix('sequence_id=')
        output = reference.parents[3] / 'reconstructed_stage' / f'{sequence}_support.usda'
        print(json.dumps(dict(reference=str(reference), output=str(output), **fit_reference(reference, output))))


if __name__ == '__main__':
    main()
