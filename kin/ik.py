"""刃先が3D点列をなぞるPandaの姿勢をMuJoCoのIKで求め、Blender用に書き出す。

使い方:
    uv run python kin/ik.py <points.json> [--name NAME] [--key pattern_world] [--render]

入力JSON: {"points": [[x,y,z], ...], "closed": true}(--keyで別のキーの点列も読める)
出力:
    kin/out/<name>_poses.json  フレームごとの可視geomの位置・回転と刃先位置
    kin/out/meshes/<mesh>.obj  コンパイル後のメッシュ(geomのpos/quatをそのまま当てられる)
    kin/out/<name>_frame*.png  --render時の確認用画像

座標系: 机の天板z=0、Pandaの台座(link0)が原点、+xが前方、Z-up、単位m。
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PANDA_XML = ROOT / "assets" / "franka_emika_panda" / "panda.xml"
OUT_DIR = Path(__file__).resolve().parent / "out"

FPS = 24
TRACE_SPEED = 0.05  # m/s
HOVER = 0.05  # 始点の上空の高さ(m)
# 刃先: hand座標系のz軸上。閉じた指先(z=0.1123)から2cm先に刃が出ている想定
BLADE_TIP_LOCAL = (0.0, 0.0, 0.1323)
ARM_DOF = 7


def load_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    spec = mujoco.MjSpec.from_file(str(PANDA_XML))
    spec.body("hand").add_site(name="blade_tip", pos=BLADE_TIP_LOCAL)
    model = spec.compile()
    return model, mujoco.MjData(model)


def home_qpos(model: mujoco.MjModel) -> np.ndarray:
    q = model.key("home").qpos.copy()
    q[ARM_DOF:] = 0.0  # 指を閉じる
    return q


def resample_closed(points: np.ndarray, step: float, closed: bool) -> np.ndarray:
    """折れ線を弧長で等間隔に取り直す。closedなら始点に戻るまで含める。"""
    pts = np.vstack([points, points[:1]]) if closed else points
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(int(np.ceil(s[-1] / step)), 1)
    t = np.linspace(0.0, s[-1], n + 1)
    return np.stack([np.interp(t, s, pts[:, k]) for k in range(3)], axis=1)


def smooth_line(a: np.ndarray, b: np.ndarray, duration: float) -> np.ndarray:
    """aからbへ、最小躍度の速度プロファイルで動く点列(始点は含めない)。"""
    n = max(int(round(duration * FPS)), 1)
    tau = np.arange(1, n + 1) / n
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    return a + s[:, None] * (b - a)


def solve_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    q0: np.ndarray,
    target_pos: np.ndarray,
    target_mat: np.ndarray,
    q_rest: np.ndarray,
    iters: int = 100,
    damping: float = 1e-3,
    tol: float = 1e-6,
) -> np.ndarray:
    """刃先siteの位置と向きを合わせる減衰最小二乗IK。q0を初期値にする。"""
    sid = model.site("blade_tip").id
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    lo, hi = model.jnt_range[:ARM_DOF, 0], model.jnt_range[:ARM_DOF, 1]
    q = q0.copy()
    target_quat = np.zeros(4)
    mujoco.mju_mat2Quat(target_quat, target_mat.ravel())
    site_quat = np.zeros(4)
    err_rot = np.zeros(3)
    for _ in range(iters):
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        err_pos = target_pos - data.site_xpos[sid]
        mujoco.mju_mat2Quat(site_quat, data.site_xmat[sid])
        mujoco.mju_subQuat(err_rot, target_quat, site_quat)
        # subQuatはsite座標系の回転ベクトルを返す。jacrはworld座標系なので揃える
        err = np.concatenate([err_pos, data.site_xmat[sid].reshape(3, 3) @ err_rot])
        if np.linalg.norm(err) < tol:
            break
        mujoco.mj_jacSite(model, data, jacp, jacr, sid)
        J = np.vstack([jacp, jacr])[:, :ARM_DOF]
        JJt = J @ J.T + damping**2 * np.eye(6)
        dq = J.T @ np.linalg.solve(JJt, err)
        # 冗長な1自由度はホーム姿勢へ寄せる(肘が不自然に振れないように)
        null = np.eye(ARM_DOF) - J.T @ np.linalg.solve(JJt, J)
        dq += null @ (0.1 * (q_rest[:ARM_DOF] - q[:ARM_DOF]))
        q[:ARM_DOF] = np.clip(q[:ARM_DOF] + dq, lo, hi)
    return q


def build_targets(path: np.ndarray, home_tip: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ホーム→始点上空→下ろす→なぞる→持ち上げる→ホーム。(目標点列, なぞり中か)を返す。"""
    start, end = path[0], path[-1]
    above_start = start + [0, 0, HOVER]
    above_end = end + [0, 0, HOVER]
    hold = np.repeat(home_tip[None], FPS // 2, axis=0)
    parts = [
        (np.vstack([home_tip[None], hold]), False),
        (smooth_line(home_tip, above_start, 2.0), False),
        (smooth_line(above_start, start, 1.0), False),
        (path[1:], True),
        (smooth_line(end, above_end, 1.0), False),
        (smooth_line(above_end, home_tip, 2.0), False),
        (np.repeat(home_tip[None], FPS // 2, axis=0), False),
    ]
    targets = np.vstack([p for p, _ in parts])
    tracing = np.concatenate([np.full(len(p), t) for p, t in parts])
    return targets, tracing


def geom_names(model: mujoco.MjModel) -> dict[int, str]:
    """可視geom(group 2)のidと名前。無名なら<body>__<mesh>にする。"""
    names = {}
    for g in range(model.ngeom):
        if model.geom_group[g] != 2:
            continue
        name = model.geom(g).name or (
            f"{model.body(model.geom_bodyid[g]).name}__{model.mesh(model.geom_dataid[g]).name}"
        )
        names[g] = name
    return names


def geom_rgba(model: mujoco.MjModel, g: int) -> list[float]:
    mat = model.geom_matid[g]
    rgba = model.mat_rgba[mat] if mat >= 0 else model.geom_rgba[g]
    return [round(float(c), 6) for c in rgba]


def export_meshes(model: mujoco.MjModel, mesh_ids: set[int], out_dir: Path) -> None:
    """コンパイル後の頂点・面・法線をOBJで書く(MuJoCoが動かした後の座標)。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    for mid in sorted(mesh_ids):
        va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
        na, nn = model.mesh_normaladr[mid], model.mesh_normalnum[mid]
        verts = model.mesh_vert[va : va + vn]
        faces = model.mesh_face[fa : fa + fn] + 1
        normals = model.mesh_normal[na : na + nn]
        fnorms = model.mesh_facenormal[fa : fa + fn] + 1
        lines = [f"o {model.mesh(mid).name}"]
        lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in verts]
        lines += [f"vn {x:.6f} {y:.6f} {z:.6f}" for x, y, z in normals]
        lines += [
            f"f {a}//{na_} {b}//{nb} {c}//{nc}"
            for (a, b, c), (na_, nb, nc) in zip(faces, fnorms, strict=True)
        ]
        (out_dir / f"{model.mesh(mid).name}.obj").write_text("\n".join(lines) + "\n")


def frame_record(model: mujoco.MjModel, data: mujoco.MjData, names: dict[int, str]) -> dict:
    quat = np.zeros(4)
    geoms = {}
    for g, name in names.items():
        mujoco.mju_mat2Quat(quat, data.geom_xmat[g])
        geoms[name] = {
            "pos": [round(float(v), 6) for v in data.geom_xpos[g]],
            "quat": [round(float(v), 6) for v in quat],
        }
    sid = model.site("blade_tip").id
    mujoco.mju_mat2Quat(quat, data.site_xmat[sid])
    return {
        "geoms": geoms,
        "blade_tip": [round(float(v), 6) for v in data.site_xpos[sid]],
        "blade_quat": [round(float(v), 6) for v in quat],
        "qpos": [round(float(v), 6) for v in data.qpos[:ARM_DOF]],
    }


def render_frames(
    model: mujoco.MjModel, qposes: list[np.ndarray], path: np.ndarray, picks: list[int], stem: str
) -> list[Path]:
    """確認用。机と布と目標線を足した別モデルで、斜め上から数フレーム描く。"""
    from PIL import Image

    spec = mujoco.MjSpec.from_file(str(PANDA_XML))
    spec.body("hand").add_site(name="blade_tip", pos=BLADE_TIP_LOCAL)
    wb = spec.worldbody
    wb.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.01], pos=[0.4, 0, 0],
                rgba=[0.55, 0.45, 0.35, 1], contype=0, conaffinity=0)
    wb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.2, 0.2, 0.0005], pos=[0.5, 0, 0.001],
                rgba=[0.85, 0.85, 0.9, 1], contype=0, conaffinity=0)
    wb.add_light(pos=[0.5, -0.5, 1.5], dir=[0, 0.4, -1])
    vis = spec.compile()
    vdata = mujoco.MjData(vis)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.4, 0, 0.2]
    cam.distance, cam.azimuth, cam.elevation = 1.6, 135, -30
    out = []
    with mujoco.Renderer(vis, 480, 640) as r:
        for i in picks:
            vdata.qpos[:] = qposes[i]
            mujoco.mj_forward(vis, vdata)
            r.update_scene(vdata, cam)
            for p in path[::3]:
                if r.scene.ngeom >= r.scene.maxgeom:
                    break
                mujoco.mjv_initGeom(r.scene.geoms[r.scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                    [0.003, 0, 0], p, np.eye(3).ravel(), [1, 0, 0, 1])
                r.scene.ngeom += 1
            f = OUT_DIR / f"{stem}_frame{i:04d}.png"
            Image.fromarray(r.render()).save(f)
            out.append(f)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("points_json", type=Path)
    ap.add_argument("--key", default="points", help="点列のキー(renders/*.jsonならpattern_world)")
    ap.add_argument("--name", default=None, help="出力名(既定は入力ファイル名)")
    ap.add_argument("--render", action="store_true", help="確認用PNGを書く")
    args = ap.parse_args()

    src = json.loads(args.points_json.read_text())
    raw = np.asarray(src[args.key], dtype=float)
    closed = bool(src.get("closed", True))
    name = args.name or args.points_json.stem

    model, data = load_model()
    q_home = home_qpos(model)
    data.qpos[:] = q_home
    mujoco.mj_forward(model, data)
    sid = model.site("blade_tip").id
    home_tip = data.site_xpos[sid].copy()
    # ヨー固定: ホーム姿勢の刃の向き(刃先が鉛直下向き)をそのまま使う
    target_mat = data.site_xmat[sid].reshape(3, 3).copy()
    assert np.allclose(target_mat[:, 2], [0, 0, -1], atol=1e-3), "ホーム姿勢で刃が下向きでない"

    path = resample_closed(raw, TRACE_SPEED / FPS, closed)
    targets, tracing = build_targets(path, home_tip)

    names = geom_names(model)
    q = q_home.copy()
    qposes, frames, errs = [], [], []
    for tgt, tr in zip(targets, tracing, strict=True):
        q = solve_ik(model, data, q, tgt, target_mat, q_home)
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        qposes.append(q.copy())
        frames.append(frame_record(model, data, names))
        if tr:
            errs.append(np.linalg.norm(data.site_xpos[sid] - tgt))
        rot_dev = np.degrees(np.arccos(np.clip(-data.site_xmat[sid][8], -1, 1)))
        frames[-1]["tilt_deg"] = round(float(rot_dev), 4)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    export_meshes(model, {int(model.geom_dataid[g]) for g in names}, OUT_DIR / "meshes")
    out = {
        "fps": FPS,
        "blade_tip_local_in_hand": list(BLADE_TIP_LOCAL),
        "geom_info": {
            n: {"mesh": model.mesh(model.geom_dataid[g]).name, "rgba": geom_rgba(model, g)}
            for g, n in names.items()
        },
        "trace_frames": [int(np.argmax(tracing)), int(len(tracing) - 1 - np.argmax(tracing[::-1]))],
        "frames": frames,
    }
    pose_path = OUT_DIR / f"{name}_poses.json"
    pose_path.write_text(json.dumps(out))

    errs_mm = np.asarray(errs) * 1000
    dq = np.abs(np.diff(np.asarray(qposes)[:, :ARM_DOF], axis=0)).max()
    tilt = max(f["tilt_deg"] for f in frames)
    print(f"frames={len(frames)} ({len(frames) / FPS:.1f}s)  trace={len(errs)}フレーム")
    print(f"刃先誤差(なぞり中): 最大 {errs_mm.max():.4f} mm / 平均 {errs_mm.mean():.4f} mm")
    print(f"刃の傾き(鉛直から): 最大 {tilt:.4f}°  1フレームの関節変化: 最大 {np.degrees(dq):.3f}°")
    print(f"出力: {pose_path.relative_to(ROOT)}  meshes={len({model.geom_dataid[g] for g in names})}")

    if args.render:
        n = len(qposes)
        a, b = out["trace_frames"]
        picks = [0, a, a + (b - a) // 3, a + 2 * (b - a) // 3, b, n - 1]
        for f in render_frames(model, qposes, path, picks, name):
            print(f"画像: {f.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
