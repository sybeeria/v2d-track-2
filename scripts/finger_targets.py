"""Dex3 finger targets retargeted from the human fingertips, written into G1 references.

soma_to_g1.py leaves every Dex3 finger target at zero (open hand). FlashCHORD uses reference finger
targets as the action centre and the reset pose (the G1 recipe gives them no reward weight), so with
open targets every grasp has to be found by exploration. Here each hand's seven finger joints are
solved per frame so the robot thumb, index and middle tips move toward the human thumb, index and
middle tips, each weighted by how close that human tip is to the object; away from the object the
fingers relax to open. Frames where a hand link would penetrate the object (FlashCHORD's own CoACD
pieces) or a support more than PENETRATION deeper than the open hand already does are backed off toward
open; at frame 0, where evaluation starts, no link may go deeper at all. (Stock references already put
the open hand 2-4 cm inside thin objects on many contact frames.) Body joints, object poses and contact labels are unchanged.

Usage:
    python scripts/finger_targets.py tree SRC_WORK DST_WORK [--penetration M] [--jobs N] [--sequences ID ...]
    python scripts/finger_targets.py screen MANIFEST OUT_MANIFEST [--source clean] [--label clean_fingers] [...]
tree: SRC_WORK and DST_WORK are retarget work dirs such as data/work/tier1 (sequences/ plus
motion/whole_body/{soma/sequence_id=*/robot_name=g1/data.parquet, reconstructed_stage/}). DST_WORK gets
the rewritten parquets, copies of sequences/ and reconstructed_stage/ (FlashCHORD finds supports next
to the parquet) and finger_report.json with before/after fingertip gaps.
screen: adds input LABEL, derived from input SOURCE, to every sequence of a make_tier2_screen.py
manifest (written beside SOURCE's work dir) and saves the result as OUT_MANIFEST for pilot_queue.sh.
"""

from __future__ import annotations

import os

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_key, "1")

import argparse
import ast
import copy
import hashlib
import json
import pickle
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import coal
import numpy as np
import pinocchio as pin
import pyarrow as pa
import pyarrow.parquet as pq
import trimesh
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
RG = ROOT / "external/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding"
URDF = RG / "assets/urdfs/g1/main_with_hand.urdf"
SIDES = ("left", "right")
ROBOT_TIPS = ("thumb_2", "index_1", "middle_1")
HUMAN_TIPS = ("ThumbEnd", "IndexEnd", "MiddleEnd")
NEAR, FAR = 0.02, 0.06  # human tip-object gap (m): full weight below NEAR, none beyond FAR
TIP_UNIT = 0.01  # tip residuals in cm
OPEN_PULL, SMOOTH = 0.3, 1.0  # per-rad residual weights toward open and toward the previous frame
MAX_STEP = 0.2  # rad per reference frame (6 rad/s at 30 fps)
PENETRATION = 0.008  # m of extra link penetration over the open hand: a squeeze, not a reset blow-up
FLASH_PYTHON = next((p for p in (ROOT / ".venv-flash/bin/python", ROOT / ".venv-flash-cpu/bin/python") if p.exists()), None)
DECOMPOSE = """import sys, numpy as np, trimesh
from flash_chord.assets.mesh import convex_decompose
mesh = trimesh.load(sys.argv[1], force="mesh", process=False)
np.savez(sys.argv[2], *[h.vertices for h in convex_decompose(np.asarray(mesh.vertices), np.asarray(mesh.faces))])
"""


def bvh(mesh: trimesh.Trimesh) -> coal.BVHModelOBBRSS:
    vertices = coal.StdVec_Vec3s()
    vertices.extend(np.asarray(mesh.vertices, float))
    faces = coal.StdVec_Triangle()
    faces.extend([coal.Triangle(*map(int, f)) for f in mesh.faces])
    model = coal.BVHModelOBBRSS()
    model.beginModel(len(mesh.faces), len(mesh.vertices))
    model.addSubModel(vertices, faces)
    model.endModel()
    return model


def convex(points: np.ndarray) -> coal.ConvexBase:
    vertices = coal.StdVec_Vec3s()
    vertices.extend(np.asarray(points, float))
    return coal.Convex.convexHull(vertices, False, None)


def object_hulls(mesh_path: str) -> list:
    """The object's collision pieces exactly as FlashCHORD builds them (its CoACD call and cache)."""
    if FLASH_PYTHON is None:
        raise FileNotFoundError("need .venv-flash or .venv-flash-cpu for FlashCHORD's convex decomposition")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "hulls.npz"
        subprocess.run([str(FLASH_PYTHON), "-c", DECOMPOSE, mesh_path, str(out)], check=True, capture_output=True)
        with np.load(out) as hulls:
            return [convex(hulls[key]) for key in hulls.files]


def supports(path: Path) -> list:
    if not path.is_file():
        return []
    from pxr import Usd, UsdGeom

    shapes = []
    stage = Usd.Stage.Open(str(path))
    for prim in stage.Traverse():
        kind = prim.GetTypeName()
        if kind not in ("Cube", "Cylinder"):
            continue
        m = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)).T
        scale = np.linalg.norm(m[:3, :3], axis=0)
        if kind == "Cube":
            shape = coal.Box(scale * float(prim.GetAttribute("size").Get()))
        else:
            shape = coal.Cylinder(float(prim.GetAttribute("radius").Get()) * scale[0],
                                  float(prim.GetAttribute("height").Get()) * scale[2])
        shapes.append((shape, coal.Transform3s(m[:3, :3] / scale, m[:3, 3])))
    return shapes


def human_joints(row: dict) -> tuple[np.ndarray, list[str]]:
    """SOMA joint positions [T, J, 3] in the reference world frame."""
    params = (RG / "retarget/params.py").read_text()
    names = next(ast.literal_eval(node.value) for node in ast.parse(params).body
                 if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "SOMA_JOINTS_ORDER" for t in node.targets))
    swap = np.asarray(json.loads((RG / "retarget/configs/g1/frame_alignment.json").read_text())["world_axis_swap_matrix"])
    # source_payload is written by our own soma_to_g1.py run, not downloaded data.
    payload = pickle.loads(row["source_payload"])
    joints = np.asarray(payload["soma_joints"], float) @ swap.T
    np.testing.assert_allclose(joints[:, names.index("Hips")], payload["source_root_translation"], atol=1e-6)
    return joints, names


class Hands:
    """G1 + Dex3 kinematics and hand-link collision queries."""

    def __init__(self) -> None:
        self.model = pin.buildModelFromUrdf(str(URDF), pin.JointModelFreeFlyer())
        self.data = self.model.createData()
        self.geom = pin.buildGeomFromUrdf(self.model, str(URDF), pin.GeometryType.COLLISION, package_dirs=[str(URDF.parent)])
        self.geom_data = pin.GeometryData(self.geom)
        self.point = coal.Sphere(1e-7)
        self.request = coal.CollisionRequest()
        self.request.enable_contact = True
        self.tips, self.links, self.joints = [], [], []
        self.hulls = {i: convex(np.asarray(trimesh.load(g.meshPath, force="mesh", process=False).vertices) * np.asarray(g.meshScale))
                      for i, g in enumerate(self.geom.geometryObjects) if "_hand_" in g.name and g.meshPath}
        for side in SIDES:
            links = [i for i, g in enumerate(self.geom.geometryObjects) if f"{side}_hand_" in g.name]
            tips = []
            for finger in ROBOT_TIPS:
                name = f"{side}_hand_{finger}_link"
                g = self.geom.geometryObjects[next(i for i in links if self.geom.geometryObjects[i].name == name + "_0")]
                vertices = np.asarray(trimesh.load(g.meshPath, force="mesh", process=False).vertices) * np.asarray(g.meshScale)
                # Fingertip = mean of the 2% of link vertices furthest along the finger's distal axis.
                axis = 1 if finger.startswith("thumb") else 0
                direction = (-1 if side == "left" else 1) if axis == 1 else 1
                score = vertices[:, axis] * direction
                tips.append((self.model.getFrameId(name), vertices[score >= np.quantile(score, 0.98)].mean(axis=0)))
            self.tips.append(tips)
            self.links.append(links)
            names = [n for n in self.model.names if f"{side}_hand_" in n]
            self.joints.append([self.model.idx_qs[self.model.getJointId(n)] for n in names])
        self.lower = [self.model.lowerPositionLimit[ix] for ix in self.joints]
        self.upper = [self.model.upperPositionLimit[ix] for ix in self.joints]

    def configuration(self, row: dict, t: int) -> np.ndarray:
        q = pin.neutral(self.model)
        q[:3] = row["robot_root_position"][t]
        w, x, y, z = row["robot_root_wxyz"][t]
        q[3:7] = [x, y, z, w]
        for name, value in zip(row["robot_joint_names"], row["robot_joint_positions"][t]):
            q[self.model.idx_qs[self.model.getJointId(name)]] = value
        return q

    def tip_points(self, q: np.ndarray, hand: int) -> np.ndarray:
        pin.forwardKinematics(self.model, self.data, q)
        points = []
        for frame, local in self.tips[hand]:
            pin.updateFramePlacement(self.model, self.data, frame)
            points.append(self.data.oMf[frame].act(local))
        return np.array(points)

    def gaps(self, points: np.ndarray, obj, pose: coal.Transform3s) -> np.ndarray:
        identity = np.eye(3)
        return np.array([max(0.0, coal.distance(self.point, coal.Transform3s(identity, p), obj, pose,
                                                coal.DistanceRequest(), coal.DistanceResult())) for p in points])

    def penetration(self, q: np.ndarray, hand: int, hulls: list, pose, support: list) -> dict:
        """Deepest penetration (m) of each hand link into the object pieces or a support."""
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateGeometryPlacements(self.model, self.data, self.geom, self.geom_data)
        depth = {}
        for i in self.links[hand]:
            placement = self.geom_data.oMg[i]
            link = coal.Transform3s(placement.rotation, placement.translation)
            deepest = 0.0
            for shape, shape_pose in [(h, pose) for h in hulls] + support:
                result = coal.CollisionResult()
                if coal.collide(self.hulls[i], link, shape, shape_pose, self.request, result):
                    deepest = max(deepest, -min(c.penetration_depth for c in result.getContacts()))
            depth[i] = deepest
        return depth


def rate_limit(x: np.ndarray) -> np.ndarray:
    """Limit per-frame change to MAX_STEP: mean of a forward and a backward slew limiter (no net lag)."""
    def slew(y):
        y = y.copy()
        for t in range(1, len(y)):
            y[t] = np.clip(y[t], y[t - 1] - MAX_STEP, y[t - 1] + MAX_STEP)
        return y
    return 0.5 * (slew(x) + slew(x[::-1])[::-1])


def envelope(scale: np.ndarray, step: float = 0.1) -> np.ndarray:
    """Largest sequence <= scale whose frame-to-frame change is at most step."""
    s = scale.copy()
    for t in range(1, len(s)):
        s[t] = min(s[t], s[t - 1] + step)
    for t in range(len(s) - 2, -1, -1):
        s[t] = min(s[t], s[t + 1] + step)
    return s


def retarget(parquet: Path, support_path: Path, penetration: float = PENETRATION) -> tuple[np.ndarray, dict]:
    """New robot_joint_positions with retargeted finger columns, and a before/after report."""
    row = pq.read_table(parquet).to_pylist()[0]
    hands = Hands()
    human, names = human_joints(row)
    frames = len(row["robot_joint_positions"])
    mesh = trimesh.load(row["object_mesh_paths"][0], force="mesh", process=False)
    obj = bvh(mesh)
    hulls = object_hulls(row["object_mesh_paths"][0])
    position = np.asarray(row["object_body_position"])[:, 0]
    rotation = Rotation.from_quat(np.asarray(row["object_body_wxyz"])[:, 0][:, [1, 2, 3, 0]]).as_matrix()
    poses = [coal.Transform3s(rotation[t], position[t]) for t in range(frames)]
    support = supports(support_path)
    active = np.asarray(row["hand_contact_active"], bool).T  # [T, 2]
    joint_names = list(row["robot_joint_names"])
    original = np.asarray(row["robot_joint_positions"], float)
    result = original.copy()
    report = {"sequence": parquet.parent.parent.name.split("=", 1)[1], "frames": frames,
              "object_bodies": len(row["object_body_names"]), "hands": []}
    configurations = [hands.configuration(row, t) for t in range(frames)]
    for hand, side in enumerate(SIDES):
        prefix = side.title() + "Hand"
        targets = human[:, [names.index(prefix + tip) for tip in HUMAN_TIPS]]
        human_gap = np.array([hands.gaps(targets[t], obj, poses[t]) for t in range(frames)])
        weight = np.clip((FAR - human_gap) / (FAR - NEAR), 0.0, 1.0)
        ix = hands.joints[hand]
        lower, upper = np.asarray(hands.lower[hand]), np.asarray(hands.upper[hand])
        solved = np.zeros((frames, len(ix)))
        previous = np.zeros(len(ix))
        for t in range(frames):
            if not weight[t].any() and not previous.any():
                continue
            q = configurations[t].copy()

            def residual(x, t=t, q=q):
                q[ix] = x
                tip = (hands.tip_points(q, hand) - targets[t]) / TIP_UNIT
                return np.concatenate([(np.sqrt(weight[t])[:, None] * tip).ravel(), OPEN_PULL * x, SMOOTH * (x - previous)])

            start = np.clip(previous, lower + 1e-6, upper - 1e-6)
            previous = least_squares(residual, start, bounds=(lower, upper), max_nfev=60, x_scale=1.0).x
            solved[t] = previous
        solved = np.clip(rate_limit(solved), lower, upper)
        # Back off toward open wherever a link penetrates PENETRATION deeper than with the open hand. The open
        # hand passes by definition, so each pass resolves every bad frame; the envelope can only lower
        # neighbours, which the next pass rechecks.
        def clean(t, s):
            q = configurations[t].copy()
            base = hands.penetration(q, hand, hulls, poses[t], support)
            q[ix] = s * solved[t]
            allowed = 0.0 if t == 0 else penetration
            return all(d <= base[i] + allowed + 1e-6 for i, d in hands.penetration(q, hand, hulls, poses[t], support).items())

        scale = np.ones(frames)
        checked = np.full(frames, np.nan)
        rejected = 0
        for _ in range(50):
            bad = [t for t in np.flatnonzero(np.abs(solved).sum(axis=1) * scale > 1e-6)
                   if checked[t] != scale[t] and not clean(t, scale[t])]
            checked = scale.copy()
            if not bad:
                break
            rejected += len(bad)
            for t in bad:
                scale[t] = next((s for s in np.arange(scale[t] - 0.25, 0.0, -0.25) if clean(t, s)), 0.0)
            scale = envelope(scale)
            checked[bad] = scale[bad]
        else:
            raise RuntimeError(f"{parquet}: intersection back-off did not converge")
        final = scale[:, None] * solved
        q_names = [n for n in hands.model.names if f"{side}_hand_" in n]
        columns = [joint_names.index(n) for n in q_names]
        result[:, columns] = final
        before = np.array([hands.gaps(hands.tip_points(configurations[t], hand), obj, poses[t]) for t in range(frames)])
        after = []
        for t in range(frames):
            q = configurations[t].copy()
            q[ix] = final[t]
            after.append(hands.gaps(hands.tip_points(q, hand), obj, poses[t]))
        after = np.array(after)
        a = active[:, hand]

        def stats(gap):
            if not a.any():
                return None
            closest = gap[a].min(axis=1)
            return {"min_tip_gap_p50_m": float(np.median(closest)), "min_tip_gap_gt2cm_fraction": float(np.mean(closest > 0.02)),
                    "mean_tip_gap_p50_m": float(np.median(gap[a].mean(axis=1)))}

        report["hands"].append({
            "side": side, "active_frames": int(a.sum()),
            "human": stats(human_gap), "open": stats(before), "retargeted": stats(after),
            "max_abs_rad": float(np.abs(final).max()), "max_step_rad": float(np.abs(np.diff(final, axis=0)).max()) if frames > 1 else 0.0,
            "backoff_frames": int(np.sum(scale < 1)), "rejected_checks": rejected,
        })
    fingers = [i for i, n in enumerate(joint_names) if "_hand_" in n]
    body = [i for i in range(original.shape[1]) if i not in fingers]
    np.testing.assert_array_equal(result[:, body], original[:, body])
    assert np.isfinite(result).all()
    return result, report


def write(src: Path, dst: Path, joints: np.ndarray) -> None:
    table = pq.read_table(src)
    index = table.schema.get_field_index("robot_joint_positions")
    field = table.schema.field(index)
    column = pa.array([joints.astype(np.float32).tolist()], type=field.type)
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.set_column(index, field, column), dst)


def process(job: tuple[Path, Path, Path, float]) -> dict:
    src, dst, support_path, penetration = job
    joints, report = retarget(src, support_path, penetration)
    write(src, dst, joints)
    print(report["sequence"][:56], *(f"{h['side'][0].upper()} {h['open']['min_tip_gap_p50_m'] * 100:.1f}->{h['retargeted']['min_tip_gap_p50_m'] * 100:.1f}cm"
                                      for h in report["hands"] if h["open"]), flush=True)
    return report


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_inputs(src_work: Path, dst_work: Path) -> None:
    for sub in ("sequences", "motion/whole_body/reconstructed_stage"):
        if (src_work / sub).is_dir():
            shutil.copytree(src_work / sub, dst_work / sub, dirs_exist_ok=True)


def tree(args, pool) -> None:
    src_work, dst_work = args.src.resolve(), args.dst.resolve()
    if src_work == dst_work:
        raise SystemExit("DST_WORK must differ from SRC_WORK")
    src, dst = src_work / "motion/whole_body", dst_work / "motion/whole_body"
    parquets = sorted(src.glob("soma/sequence_id=*/robot_name=g1/data.parquet"))
    if args.sequences:
        parquets = [p for p in parquets if p.parent.parent.name.split("=", 1)[1] in args.sequences]
    if not parquets:
        raise SystemExit(f"no references under {src}/soma")
    copy_inputs(src_work, dst_work)
    jobs = []
    for p in parquets:
        sequence = p.parent.parent.name.split("=", 1)[1]
        jobs.append((p, dst / p.relative_to(src), src / "reconstructed_stage" / f"{sequence}_support.usda", args.penetration))
    reports = list(pool.map(process, jobs))
    (dst_work / "finger_report.json").write_text(json.dumps({"source": str(src), "penetration_m": args.penetration, "episodes": reports}, indent=2))


def screen(args, pool) -> None:
    manifest = json.loads(args.manifest.read_text())
    entries = [e for e in manifest["sequences"] if not args.sequences or e["sequence"] in args.sequences]
    jobs, arms = [], []
    for entry in entries:
        if args.label in entry["arms"]:
            raise SystemExit(f"{entry['sequence']}: input {args.label!r} already exists")
        arm = entry["arms"][args.source]
        src_work = Path(arm["work_dir"])
        dst_work = src_work.parent / args.label
        src_root, dst_root = src_work / "motion/whole_body", dst_work / "motion/whole_body"
        src_parquet = Path(arm["training_reference"]["path"])
        if sha256(src_parquet) != arm["training_reference"]["sha256"]:
            raise SystemExit(f"{src_parquet}: SHA-256 differs from the manifest; regenerate the screen")
        copy_inputs(src_work, dst_work)
        jobs.append((src_parquet, dst_root / src_parquet.relative_to(src_root),
                     src_root / "reconstructed_stage" / f"{entry['sequence']}_support.usda", args.penetration))
        new = copy.deepcopy(arm)
        new["work_dir"] = str(dst_work)
        new["support"]["path"] = str(dst_root / Path(arm["support"]["path"]).relative_to(src_root))
        arms.append((entry, new))
    for (entry, new), job, report in zip(arms, jobs, pool.map(process, jobs)):
        if sha256(Path(new["support"]["path"])) != new["support"]["sha256"]:
            raise SystemExit(f"{new['support']['path']}: support copy differs")
        new["training_reference"] = {"path": str(job[1]), "sha256": sha256(job[1])}
        new["finger_targets"] = {"source_input": args.source, "penetration_m": args.penetration, "report": report}
        entry["arms"][args.label] = new
    args.out_manifest.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {args.out_manifest}: input {args.label!r} on {len(arms)} sequences")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--penetration", type=float, default=PENETRATION, help="extra link penetration allowed (m)")
    common.add_argument("--jobs", type=int, default=4)
    common.add_argument("--sequences", nargs="+")
    commands = parser.add_subparsers(dest="command", required=True)
    t = commands.add_parser("tree", parents=[common])
    t.add_argument("src", type=Path)
    t.add_argument("dst", type=Path)
    m = commands.add_parser("screen", parents=[common])
    m.add_argument("manifest", type=Path)
    m.add_argument("out_manifest", type=Path)
    m.add_argument("--source", default="clean")
    m.add_argument("--label", default="clean_fingers")
    args = parser.parse_args()
    with ProcessPoolExecutor(args.jobs) as pool:
        (tree if args.command == "tree" else screen)(args, pool)


if __name__ == "__main__":
    main()
