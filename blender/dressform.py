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
Z_WAIST = 0.090  # ウエストラインの高さ(SECTIONSのウエスト)
Z_HIP = 0.012  # ヒップラインの高さ(腰の切り口のすぐ上)
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


def remove_small_loops(bvh: BVHTree, pts: list[Vector], roles: list[str]) -> int:
    """正面(x, z)から見て線が交差してできた小さな輪(首の付け根と肩の稜線の継ぎ目などで起きる)を、
    間の点ごと取り除き、元の点数に取り直して表面へ戻す。取り除いた輪の数を返す。"""
    def cross(o, a, b):
        return (a.x - o.x) * (b.z - o.z) - (a.z - o.z) * (b.x - o.x)

    removed = 0
    while True:
        n, hit = len(pts), None
        for i in range(n):
            for j in range(i + 2, min(i + 20, n)):
                a, b, c, e = pts[i], pts[i + 1], pts[j], pts[(j + 1) % n]
                if cross(c, e, a) * cross(c, e, b) < 0 and cross(a, b, c) * cross(a, b, e) < 0:
                    hit = (i, j)
                    break
            if hit:
                break
        if hit is None:
            break
        i, j = hit
        del pts[i + 1 : j + 1], roles[i + 1 : j + 1]
        removed += 1
    if removed:
        dense = bear2.resample_closed3(pts, len(LAST_ROLES))
        kd_src = list(zip(pts, roles, strict=True))
        new_roles = [min(kd_src, key=lambda t: (t[0] - q).length)[1] for q in dense]
        pts[:] = [bear2.snap(bvh, q)[0] for q in dense]
        roles[:] = new_roles
    return removed


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
    loops = remove_small_loops(bvh, pts, roles)
    if loops:
        LAST_ROLES[:] = roles
        print(f"正面から見た線の小さな輪を{loops}個取り除いた")

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


def guide_contours(garment, flat: list[Vector]) -> dict:
    """布のガイドライン(展開の連番で布に描く線と同じ定義)を、平面の上の線分として求める。
    頂点の3D位置から各線の値を出し、三角形ごとに値が0になる区間を平面の座標で結ぶ。"""
    co = [v.co.copy() for v in garment.data.vertices]
    fields = {
        "center_front": [c.x for c in co],
        "princess": [abs(c.x) - princess_x(c.z) for c in co],
        "bust": [c.z - BUST[0] for c in co],
        "waist": [c.z - Z_WAIST for c in co],
    }
    out = {}
    for name, f in fields.items():
        segs = []
        for poly in garment.data.polygons:
            vs = list(poly.vertices)
            cross = []
            for a, b in zip(vs, vs[1:] + vs[:1]):
                if (f[a] < 0) != (f[b] < 0):
                    t = f[a] / (f[a] - f[b])
                    p = flat[a].lerp(flat[b], t)
                    cross.append([round(p.x, 6), round(p.y, 6)])
            if len(cross) == 2:
                segs.append(cross)
        out[name] = segs
    return out


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
        "waist_z": Z_WAIST,
        "guides": guide_contours(garment, flat),
        "note": "平面は3Dの(x, z)に合わせた向き(首が+y)。単位m。点の順は微調整後の線(seam_lines.jsonのadjusted)と同じ",
    }, ensure_ascii=False, indent=1))


# ---------- 平面に開く区間の見た目: 人台のガイドライン(赤いテープ)と半透明グレーの布 ----------

# プリンセスライン(前): 肩の中ほど → バストポイント → ウエスト → 腰。(高さz, 中心からの横位置x)
PRINCESS = ((Z_HIP, 0.042), (Z_WAIST, 0.034), (BUST[0], BUST[1]), (0.245, 0.050), (Z_SHOULDER, 0.053))
TAPE_RED = (0.62, 0.04, 0.03, 1.0)
TAPE_R = 0.0008  # テープの太さ(半径)
CLOTH_LINE_W = 0.0009  # 布に写したガイドラインの半幅
NECKLINE = (0.284, 0.012, 0.002)  # 襟ぐりのテープ: (横の高さ, 前中心での下がり, 後ろ中心での上がり)
ARM_LOW_Z = 0.15  # これより上の切り口だけを腕の付け根とみなす(腰の横も同じxで平らに切れているため)


def princess_x(z: float) -> float:
    return float(np.interp(z, [p[0] for p in PRINCESS], [p[1] for p in PRINCESS]))


def neckline_z(front: float) -> float:
    """襟ぐりのテープの高さ。front=光線の向きと前(-y)の内積。首の付け根より下で、前がやや深く下がる丸い線。"""
    side, drop, rise = NECKLINE
    return side - drop * max(front, 0.0) ** 1.5 + rise * max(-front, 0.0)


def tape(name: str, pts: list[Vector], cyclic: bool, mat: bpy.types.Material) -> None:
    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth, curve.bevel_resolution = TAPE_R, 2
    sp = curve.splines.new("POLY")
    sp.points.add(len(pts) - 1)
    for q, c in zip(sp.points, pts, strict=True):
        q.co = (*c, 1.0)
    sp.use_cyclic_u = cyclic
    obj = bpy.data.objects.new(name, curve)
    bpy.context.scene.collection.objects.link(obj)
    obj.data.materials.append(mat)


def add_guide_tapes(body: bpy.types.Object, bvh: BVHTree) -> int:
    """洋裁用ボディのように、人台へ赤い細いテープを貼る。前中心・後ろ中心・脇線・プリンセスライン・
    バスト/ウエスト/ヒップライン・襟ぐり・袖ぐり。テープは表面から半径だけ浮かせる。"""
    mat = bear2.bear.material("Tape", TAPE_RED, 0.7)
    lift = lambda hit, n: hit + n * TAPE_R  # noqa: E731
    count = 0

    def add(name, pts, cyclic=False):
        nonlocal count
        if len(pts) > 1:
            tape(name, pts, cyclic, mat)
            count += 1

    def ray(o, d):
        hit, n, *_ = bvh.ray_cast(Vector(o), Vector(d))
        return (hit, n) if hit is not None else None

    # 水平の線: 中心の軸から外へ光線を出す
    ts = np.linspace(0, 2 * math.pi, 241)[:-1]
    for name, z in (("Bust", BUST[0]), ("Waist", Z_WAIST), ("Hip", Z_HIP)):
        hits = [ray((0, 0, z), (math.cos(t), math.sin(t), 0)) for t in ts]
        add(f"Tape{name}", [lift(*h) for h in hits if h], cyclic=True)
    hits = [ray((0, 0, neckline_z(-math.sin(t))), (math.cos(t), math.sin(t), 0)) for t in ts]
    add("TapeNeck", [lift(*h) for h in hits if h], cyclic=True)
    zs = np.linspace(0.003, Z_NECK, 150)
    # 前中心・後ろ中心(襟ぐりまで)
    for name, sy in (("CF", -1), ("CB", 1)):
        top = neckline_z(-sy)
        add(f"Tape{name}", [lift(*h) for z in zs if z <= top and (h := ray((0, sy, z), (0, -sy, 0)))])
    # 脇線: 体の真横を、腕の付け根の切り口に当たるまで
    for sx in (-1, 1):
        pts = []
        for z in zs:
            h = ray((sx, Y_SHOULDER, z), (-sx, 0, 0))
            if h is None or (z > ARM_LOW_Z and abs(h[0].x) >= ARM_CUT_X - 1e-4):  # 腰も同じxで平らに切れている
                break
            pts.append(lift(*h))
        add(f"TapeSide{sx}", pts)
    # プリンセスライン(前): 正面からの光線で0.245mまで、その上は肩の上から下向きの光線で稜線まで
    for sx in (-1, 1):
        pts = [lift(*h) for z in np.linspace(Z_HIP, 0.245, 100) if (h := ray((sx * princess_x(z), -1, z), (0, 1, 0)))]
        y0 = pts[-1].y
        for t in np.linspace(0, 1, 20)[1:]:
            x = sx * (princess_x(0.245) + t * (princess_x(Z_SHOULDER) - princess_x(0.245)))
            if h := ray((x, y0 * (1 - t) + Y_SHOULDER * t, 1), (0, 0, -1)):
                pts.append(lift(*h))
        add(f"TapePrincess{sx}", pts)
    # 袖ぐり: 腕の付け根の平らな切り口の縁(切り口の頂点のうち、切られていない頂点と隣り合うもの)
    me = body.data
    cut = [abs(abs(v.co.x) - ARM_CUT_X) < 1e-6 and v.co.z > ARM_LOW_Z for v in me.vertices]
    for sx in (-1, 1):
        edge_v = set()
        for e in me.edges:
            a, b = e.vertices
            if cut[a] != cut[b]:
                k = a if cut[a] else b
                if me.vertices[k].co.x * sx > 0:
                    edge_v.add(k)
        pts = [me.vertices[k].co.copy() for k in edge_v]
        c = sum(pts, Vector()) / len(pts)
        pts.sort(key=lambda q: math.atan2(q.z - c.z, q.y - c.y))
        for _ in range(4):  # 格子の段差をならす
            pts = [(pts[i - 1] + pts[i] * 2 + pts[(i + 1) % len(pts)]) / 4 for i in range(len(pts))]
        add(f"TapeArmhole{sx}", [q + Vector((sx * TAPE_R * 0.5, 0, 0)) for q in pts], cyclic=True)
    return count


def cloth_material() -> bpy.types.Material:
    """半透明グレーの布。ふちはシーンで柔らかく光り、頂点の属性で描いたガイドラインは赤く不透明。"""
    mat = bpy.data.materials.new("ClothGray")
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.85
    bsdf.inputs["Sheen Weight"].default_value = 0.8
    bsdf.inputs["Sheen Roughness"].default_value = 0.45
    bsdf.inputs["Sheen Tint"].default_value = (1.0, 1.0, 1.0, 1.0)

    def math_node(op, a, b=None, c=None):
        n = nt.nodes.new("ShaderNodeMath")
        n.operation = op
        for i, v in enumerate((a, b, c)):
            if v is None:
                continue
            if isinstance(v, float):
                n.inputs[i].default_value = v
            else:
                nt.links.new(v, n.inputs[i])
        return n.outputs[0]

    mask = None
    for attr, center in (("guide_x", 0.0), ("guide_z", BUST[0]), ("guide_z", Z_WAIST), ("guide_p", 0.0)):
        at = nt.nodes.new("ShaderNodeAttribute")
        at.attribute_type, at.attribute_name = "GEOMETRY", attr
        m = math_node("LESS_THAN", math_node("ABSOLUTE", math_node("SUBTRACT", at.outputs["Fac"], center)),
                      CLOTH_LINE_W)
        mask = m if mask is None else math_node("MAXIMUM", mask, m)
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.inputs["A"].default_value = (0.40, 0.40, 0.42, 1.0)
    mix.inputs["B"].default_value = TAPE_RED
    nt.links.new(mask, mix.inputs["Factor"])
    nt.links.new(mix.outputs["Result"], bsdf.inputs["Base Color"])
    nt.links.new(math_node("MULTIPLY_ADD", mask, 0.5, 0.5), bsdf.inputs["Alpha"])  # 不透明度: 布0.5、線1.0
    mat.use_backface_culling = True  # 厚みの裏側の面は描かない(半透明が2重にならないように)
    return mat


def style_for_unfold(body: bpy.types.Object, bvh: BVHTree, garment: bpy.types.Object) -> None:
    """平面に開く画像の前に呼ぶ(bear2.BEFORE_UNFOLD)。服の頂点に3Dでの位置を属性として持たせるので、
    平面に開いてもガイドラインが布についてくる。"""
    n = add_guide_tapes(body, bvh)
    me = garment.data
    co = [v.co for v in me.vertices]
    for name, vals in (("guide_x", [c.x for c in co]), ("guide_z", [c.z for c in co]),
                       ("guide_p", [abs(c.x) - princess_x(c.z) for c in co])):
        me.attributes.new(name, "FLOAT", "POINT").data.foreach_set("value", vals)
    me.materials.clear()
    me.materials.append(cloth_material())
    garment.modifiers["Thick"].thickness = 0.003
    bev = garment.modifiers.new("SoftEdge", "BEVEL")  # 厚みの縁を丸める
    bev.width, bev.segments, bev.limit_method = 0.0012, 3, "ANGLE"
    print(f"人台のテープ{n}本、布のガイドライン(前中心・バスト・ウエスト・プリンセス)")


# ---------- 実行 ----------

def configure_bear2() -> None:
    bear2.OUT = OUT
    bear2.PATTERN_NAME = "dressform_front"
    bear2.adjust_line = adjust_dressform
    bear2.RELAX_ITERS = 40
    bear2.bear.UNWRAP_METHOD = UNWRAP_METHOD
    bear2.bear.MAX_SIZE = 0.35  # 1/2スケールの実寸を保つ(縫い代込みで50cm角の布に収まる)
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
    bear2.BEFORE_UNFOLD = lambda g: style_for_unfold(body, bvh, g)
    garment, flat = bear2.stage2(cam, bvh, info, poly_px, er_path.name, args.anim)
    write_flat_boundary(garment, flat)
    rep = distortion_report(garment, flat)
    (OUT / "distortion.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print(f"展開のゆがみ: 分位点5/50/95%={rep['stretch_pct_p5_p50_p95']}  範囲{rep['stretch_pct_min']}〜{rep['stretch_pct_max']}%  胸の周り平均{rep['near_bust_pct_mean_abs']}% "
          f"それ以外平均{rep['elsewhere_pct_mean_abs']}%  ゆがみ上位5%の場所{rep['worst5pct_regions']}")


if __name__ == "__main__":
    main()
