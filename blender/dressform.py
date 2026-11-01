"""1/2スケールの人台(ドレスフォーム)を作り、ERの線から前身頃の型紙を作る。

投影・展開・描画はbear2.pyの処理を使い、微調整だけを実際の型紙の基準線(脇線・肩線・袖ぐり・襟ぐり)に
吸着させるものに差し替える。

段階1(ERの前): Blender -b --factory-startup -P blender/dressform.py -- --shoot
    renders/dressform/front.png / front.json   人台の正面画像とカメラ情報
    renders/dressform/look_front.png / look_oblique.png   形の確認用
段階2(ERの後): Blender -b --factory-startup -P blender/dressform.py -- [--er er/out/dressform_front_er.json] [--anim]
    patterns/dressform_front.json と renders/dressform/ の画像(bear3と同じ構成)、
    renders/dressform/distortion.json  展開のゆがみ(面ごとの伸び縮み)

人台は-y方向を向いて立つ。腰の切り口がz=0、首の付け根がz≈29cm。単位m。
寸法は実物の約1/2(バスト42cm・ウエスト32cm・ヒップ45cm・肩幅19cm・首回り19cm)。
"""

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bear2  # noqa: E402

ROOT = bear2.ROOT
OUT = ROOT / "renders" / "dressform"
GRAY = (0.62, 0.62, 0.62, 1.0)

# 断面の表: (高さz, 半幅, 前の厚み, 後ろの厚み)[m]
SECTIONS = [
    (0.000, 0.080, 0.060, 0.068),  # 腰(ヒップ)
    (0.050, 0.072, 0.055, 0.062),
    (0.090, 0.056, 0.046, 0.050),  # ウエスト
    (0.130, 0.060, 0.050, 0.052),
    (0.170, 0.067, 0.054, 0.056),  # バスト(ふくらみは別に足す)
    (0.205, 0.075, 0.054, 0.057),  # 脇の下
    (0.235, 0.087, 0.050, 0.055),
    (0.258, 0.094, 0.044, 0.049),  # 肩先の高さ
    (0.272, 0.066, 0.036, 0.040),
    (0.284, 0.040, 0.030, 0.032),
    (0.292, 0.031, 0.028, 0.029),  # 首の付け根
    (0.315, 0.029, 0.027, 0.028),  # 首の上端
]
ARM_CUT_X = 0.075  # 腕の付け根の切り口(平面 x=±この値)
BUST = (0.172, 0.040, 0.013)  # バストポイントの(高さ, 中心からの横位置, ふくらみ)
SUPER = 2.5  # 断面の丸み(2で楕円、大きいほど四角い)
N_THETA, N_Z = 128, 160

# 型紙の基準線
Z_NECK = 0.292  # 首の付け根の高さ
Z_ARMPIT = 0.212  # 脇の下(切り口の下端)の高さの目安
Z_SHOULDER = 0.262  # 肩先の高さの目安
Y_SHOULDER = 0.0  # 肩線と脇線が乗る前後の境界の面
# 展開の方法。CONFORMAL(等角写像)は脇や肩で面積の縮みが大きいので、伸びを最小にするSLIMを使う
UNWRAP_METHOD = "MINIMUM_STRETCH"


# ---------- 人台 ----------

def profile(z: np.ndarray) -> tuple:
    """表を線形に補間してから高さ方向にならし、なめらかな断面の寸法にする。"""
    tab = np.array(SECTIONS)
    cols = [np.interp(z, tab[:, 0], tab[:, k]) for k in (1, 2, 3)]
    k = np.exp(-0.5 * (np.arange(-6, 7) / 2.5) ** 2)
    k /= k.sum()
    out = []
    for c in cols:
        padded = np.concatenate([np.full(6, c[0]), c, np.full(6, c[-1])])
        out.append(np.convolve(padded, k, mode="valid"))
    return out


def dressform_mesh(name: str) -> bpy.types.Mesh:
    """リングを積み上げた1つのつながったメッシュ(上下はふさぐ)。"""
    z = np.linspace(SECTIONS[0][0], SECTIONS[-1][0], N_Z)
    a, df, db = profile(z)
    th = np.linspace(-math.pi, math.pi, N_THETA, endpoint=False)
    s, c = np.sin(th), np.cos(th)
    e = 2 / SUPER
    verts = []
    for i in range(N_Z):
        x = a[i] * np.sign(s) * np.abs(s) ** e
        depth = np.where(c > 0, df[i], db[i])
        y = -depth * np.sign(c) * np.abs(c) ** e
        # 胸のふくらみ(前面だけ)
        bz, bx, bh = BUST
        g = np.exp(-(((z[i] - bz) / 0.032) ** 2)) * (
            np.exp(-(((x - bx) / 0.03) ** 2)) + np.exp(-(((x + bx) / 0.03) ** 2)))
        y = y - bh * g * (c > 0)
        x = np.clip(x, -ARM_CUT_X, ARM_CUT_X)  # 腕の付け根を平らに切る
        verts += [(float(px), float(py), float(z[i])) for px, py in zip(x, y)]
    faces = []
    for i in range(N_Z - 1):
        for j in range(N_THETA):
            j1 = (j + 1) % N_THETA
            faces.append((i * N_THETA + j, i * N_THETA + j1, (i + 1) * N_THETA + j1, (i + 1) * N_THETA + j))
    bottom, top = len(verts), len(verts) + 1
    verts += [(0.0, 0.0, float(z[0])), (0.0, 0.0, float(z[-1]))]
    last = (N_Z - 1) * N_THETA
    faces += [(bottom, (j + 1) % N_THETA, j) for j in range(N_THETA)]
    faces += [(top, last + j, last + (j + 1) % N_THETA) for j in range(N_THETA)]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.validate()
    for p in mesh.polygons:
        p.use_smooth = True
    mesh.set_sharp_from_angle(angle=math.radians(40))  # 切り口の縁は角を立てる
    return mesh


def build_dressform() -> bpy.types.Object:
    body = bpy.data.objects.new("DressForm", dressform_mesh("DressForm"))
    bpy.context.scene.collection.objects.link(body)
    body.data.materials.append(bear2.bear.material("FormGray", GRAY, 0.6))
    return body


# ---------- 微調整: 型紙の基準線へ吸着 ----------

NECK_R = 0.031  # 首の付け根の半径
LAST_ROLES: list[str] = []  # 直前の微調整で各点に割り当てた役割(製図の後処理で使う)


def assign_roles(pts: list[Vector]) -> list[str]:
    """線の各点の役割(裾・脇・袖ぐり・肩・首・襟ぐり)。左右対称にした後に一度だけ決める。"""
    zmin = min(p.z for p in pts)
    roles = []
    for p in pts:
        ax = abs(p.x)
        if p.z < zmin + 0.01:
            roles.append("hem")  # 裾: ERの高さのまま
        elif p.z > Z_NECK - 0.002:
            roles.append("neck")  # 首にかかった点 → 首の付け根の線
        elif p.z >= Z_SHOULDER and ax >= NECK_R:
            roles.append("shoulder")  # 肩の上 → 稜線
        elif Z_ARMPIT <= p.z < Z_SHOULDER and ax >= 0.04:
            roles.append("armhole")  # 腕の付け根の高さの外側 → 切り口の前の縁
        elif p.z < Z_ARMPIT and ax >= 0.03:
            roles.append("side")  # 脇の下より下の外側 → 体の真横
        else:
            roles.append("design")  # 襟ぐりのくぼみなど、ERのデザインのまま
    return roles


def reference_point(bvh: BVHTree, p: Vector, role: str) -> Vector | None:
    """役割に対応する基準線上の点(同じ高さ、または同じ横位置)。"""
    sgn = 1 if p.x >= 0 else -1
    if role == "neck":  # 首の付け根の前側の点(同じx)
        x = max(-(NECK_R - 0.001), min(NECK_R - 0.001, p.x))
        hit, *_ = bvh.ray_cast(Vector((x, -1.0, Z_NECK)), Vector((0, 1, 0)))
    elif role == "shoulder":  # 肩の上の稜線(前後の境界)
        hit, *_ = bvh.ray_cast(Vector((p.x, Y_SHOULDER, 1.0)), Vector((0, 0, -1)))
    elif role == "armhole":  # 切り口の前の縁
        hit, *_ = bvh.ray_cast(Vector((sgn * (ARM_CUT_X - 0.0015), -1.0, p.z)), Vector((0, 1, 0)))
    elif role == "side":  # 体の真横(前後の境界)
        hit, *_ = bvh.ray_cast(Vector((sgn * 0.5, Y_SHOULDER, p.z)), Vector((-sgn, 0, 0)))
    else:
        return None
    return hit


def snap_to_references(bvh: BVHTree, pts: list[Vector], roles: list[str]) -> dict:
    """役割のある点を基準線へ移す。裾と襟ぐりのデザインはERの線のまま。"""
    count: dict[str, int] = {}
    for i, (p, role) in enumerate(zip(pts, roles, strict=True)):
        target = reference_point(bvh, p, role)
        if target is not None:
            pts[i] = bear2.snap(bvh, target)[0]
            count[role] = count.get(role, 0) + 1
    return count


def adjust_dressform(bvh: BVHTree, projected: list[Vector]) -> tuple[list[Vector], dict]:
    """左右対称 → 役割ごとに基準線へ吸着 → ならす(ならすたびに吸着し直す)。bear2.adjust_lineの差し替え。"""
    n = bear2.ADJUST_POINTS
    pts = bear2.resample_closed3(projected, n)
    low = [i for i, p in enumerate(pts) if p.z < sum(q.z for q in pts) / n]
    start = min(low, key=lambda i: abs(pts[i].x))
    pts = pts[start:] + pts[:start]
    mirror = [Vector((-p.x, p.y, p.z)) for p in [pts[(n - i) % n] for i in range(n)]]
    pts = [bear2.snap(bvh, (a + b) / 2)[0] for a, b in zip(pts, mirror)]

    roles = assign_roles(pts)
    LAST_ROLES[:] = roles
    first = snap_to_references(bvh, pts, roles)
    for _ in range(bear2.SMOOTH_ITERS):
        pts = [bear2.snap(bvh, p.lerp((pts[i - 1] + pts[(i + 1) % n]) / 2, 0.5))[0] for i, p in enumerate(pts)]
        snap_to_references(bvh, pts, roles)

    dense = bear2.resample_closed3(projected, 2000)
    dist = [min((p - q).length for q in dense) for p in pts]
    return pts, {
        "side_points": first.get("side", 0),
        "pushed_points": sum(v for k, v in first.items() if k != "side"),
        "snapped": first,
        "moved_max_mm": max(dist) * 1000,
        "moved_mean_mm": sum(dist) / n * 1000,
        "rules": ["左右対称", "脇線=体の真横(前後の境界)", "肩線=肩の上の稜線(前後の境界)",
                  "袖ぐり=腕の付け根の切り口の前の縁", "首にかかった線=首の付け根の線(それより下の襟ぐりはERの深さのまま)",
                  "裾=ERの線の高さのまま", "表面に沿ってならす(ならすたびに基準線へ戻す)"],
    }


# ---------- 展開のゆがみ ----------

def distortion_report(garment, flat: list[Vector]) -> dict:
    """面ごとの伸び縮み(平面の辺の長さ / 3Dの辺の長さ、全体の平均に対する%)と、その大きい場所。"""
    mesh = garment.data
    rows = []
    for p in mesh.polygons:
        vs = list(p.vertices)
        a3 = p.area
        a2 = abs(bear2.bear.poly_area([tuple(flat[v]) for v in vs]))
        if a3 > 0:
            c = p.center
            rows.append((math.sqrt(a2 / a3), a3, c.x, c.z))
    total = sum(r[1] for r in rows)
    mean = sum(r[0] * r[1] for r in rows) / total
    stretch = [(100 * (r[0] / mean - 1), r[1], r[2], r[3]) for r in rows]
    bx, bz = BUST[1], BUST[0]
    near_bust = [s for s in stretch if (abs(s[2]) - bx) ** 2 + (s[3] - bz) ** 2 < 0.03 ** 2]
    worst = sorted(stretch, key=lambda s: -abs(s[0]))[: max(1, len(stretch) // 20)]

    def region(x: float, z: float) -> str:
        if (abs(x) - bx) ** 2 + (z - bz) ** 2 < 0.03 ** 2:
            return "胸"
        if z > Z_ARMPIT and abs(x) > 0.05:
            return "肩・袖ぐり"
        if z > Z_NECK - 0.03 and abs(x) < 0.05:
            return "襟ぐり"
        if abs(x) > 0.05:
            return "脇"
        return "前中心・腹"

    regions: dict[str, int] = {}
    for s in worst:
        regions[region(s[2], s[3])] = regions.get(region(s[2], s[3]), 0) + 1
    # 面積で重みを付けた分位点(潰れた小さな三角形の極端な値に引っ張られにくい)
    srt = sorted(stretch, key=lambda t: t[0])
    acc, pct = 0.0, {}
    for v, area, *_ in srt:
        acc += area
        for q in (5, 50, 95):
            if q not in pct and acc >= total * q / 100:
                pct[q] = round(v, 2)
    return {
        "stretch_pct_p5_p50_p95": [pct[5], pct[50], pct[95]],
        "stretch_pct_min": round(min(s[0] for s in stretch), 2),
        "stretch_pct_max": round(max(s[0] for s in stretch), 2),
        "near_bust_pct_mean_abs": round(sum(abs(s[0]) for s in near_bust) / max(1, len(near_bust)), 2),
        "elsewhere_pct_mean_abs": round(
            sum(abs(s[0]) for s in stretch if s not in near_bust) / max(1, len(stretch) - len(near_bust)), 2),
        "worst5pct_regions": regions,
        "unwrap_method": UNWRAP_METHOD,
        "note": "値は面ごとの長さの倍率(平面/3D、面積の平方根)の、全体平均からのずれ(%)。分位点は面積で重み付け",
    }


def write_flat_boundary(garment, flat: list[Vector]) -> None:
    """微調整後の線の各点の、平面での位置・役割・3Dの高さを書く(draft/draft.pyが製図の形に整える)。"""
    from mathutils.kdtree import KDTree

    lines = json.loads((OUT / "seam_lines.json").read_text())["adjusted"]
    loop = bear2.bear.boundary_loop(garment.data)
    kd = KDTree(len(loop))
    for k, v in enumerate(loop):
        kd.insert(garment.data.vertices[v].co, k)
    kd.balance()
    pts = []
    for p in lines:
        _, k, _ = kd.find(Vector(p))
        pts.append(flat[loop[k]])
    (OUT / "flat_boundary.json").write_text(json.dumps({
        "points": [[round(p.x, 6), round(p.y, 6)] for p in pts],
        "roles": LAST_ROLES,
        "z3d": [round(p[2], 5) for p in lines],
        "bust_z": BUST[0],
        "note": "平面は3Dの(x, z)に合わせた向き(首が+y)。単位m。点の順は微調整後の線(seam_lines.jsonのadjusted)と同じ",
    }, ensure_ascii=False, indent=1))


# ---------- 実行 ----------

def configure_bear2() -> None:
    bear2.OUT = OUT
    bear2.PATTERN_NAME = "dressform_front"
    bear2.adjust_line = adjust_dressform
    bear2.RELAX_ITERS = 40
    bear2.bear.UNWRAP_METHOD = UNWRAP_METHOD
    bear2.HEAD_C = Vector((0, 0, 0.17))  # 光線が体に当たらないときの深さの基準(胴の中心)
    bear2.OBLIQUE = ((0.6, -0.95, 0.42), (0, 0, 0.16), 40)
    bear2.UNFOLD_CAM = ((0.2, -1.2, 0.3), (0.2, 0, 0.16), 50)


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--shoot", action="store_true", help="段階1: 正面画像とカメラ情報を書く")
    ap.add_argument("--er", type=Path, default=None, help="ERの出力(既定: er/out/dressform_front_er.json)")
    ap.add_argument("--anim", action="store_true")
    args = ap.parse_args(argv)

    configure_bear2()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    body = build_dressform()
    bear2.setup_world()
    cam = bear2.add_camera("Cam_Front", (0, -1.5, 0.158), (0, 0, 0.158), 130)
    OUT.mkdir(parents=True, exist_ok=True)
    info = bear2.camera_info(cam)
    print(f"人台: 頂点{len(body.data.vertices)} 面{len(body.data.polygons)}")

    if args.shoot:
        bear2.render(cam, OUT / "front.png")
        (OUT / "front.json").write_text(json.dumps(info, indent=1))
        bear2.render(cam, OUT / "look_front.png", (1280, 720))
        bear2.render(bear2.add_camera("Cam_Oblique", *bear2.OBLIQUE), OUT / "look_oblique.png", (1280, 720))
        print("wrote renders/dressform/front.png, front.json, look_front.png, look_oblique.png")
        return

    er_path = args.er or ROOT / "er" / "out" / "dressform_front_er.json"
    if not er_path.is_absolute():
        er_path = ROOT / er_path
    poly_px = json.loads(er_path.read_text())["trajectory_px"]
    print(f"ERの線: {er_path.name}  {len(poly_px)}点")
    bvh = BVHTree.FromPolygons([v.co.copy() for v in body.data.vertices], [p.vertices for p in body.data.polygons])
    garment, flat = bear2.stage2(cam, bvh, info, poly_px, er_path.name, args.anim)
    write_flat_boundary(garment, flat)
    rep = distortion_report(garment, flat)
    (OUT / "distortion.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print(f"展開のゆがみ: 分位点5/50/95%={rep['stretch_pct_p5_p50_p95']}  範囲{rep['stretch_pct_min']}〜{rep['stretch_pct_max']}%  胸の周り平均{rep['near_bust_pct_mean_abs']}% "
          f"それ以外平均{rep['elsewhere_pct_mean_abs']}%  ゆがみ上位5%の場所{rep['worst5pct_regions']}")


if __name__ == "__main__":
    main()
