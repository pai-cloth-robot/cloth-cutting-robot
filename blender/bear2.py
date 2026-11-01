"""ぬいぐるみ風のクマを作り、ERが正面画像に描いた服の線を体に投影して前身頃の型紙にする。

段階1(ERの前): Blender -b --factory-startup -P blender/bear2.py -- --shoot
    renders/bear2/front.png / front.json   服なしのクマの正面画像とカメラ情報(scene.pyと同じ形式)
    renders/bear2/test_er.json             動作確認用の手作りの多角形(ERの代わり)
    renders/bear2/look_oblique.png   形の確認用(斜め)
段階2(ERの後): Blender -b --factory-startup -P blender/bear2.py -- [--er er/out/front_er.json] [--anim]
    patterns/bear2_front.json      前身頃の輪郭(bear_front.jsonと同じ形式)
    renders/bear2/seam_lines.json  投影直後と微調整後の線の3D点列・動いた量
    renders/bear2/er_line.png      ERの線を正面画像に重ねたもの
    renders/bear2/seam_front.png / seam_oblique.png / seam_morph_*.png  体の上の線(赤=投影直後、緑=微調整後)
    renders/bear2/dressed.png      体の上で前身頃の範囲を色付けしたもの(斜め)
    renders/bear2/unfold_*.png     前身頃が平面に開くまで(--animならanim/に6秒の連番)

クマは-y方向を向いて座る。床がz=0、単位m。画素は[x, y]で左上原点(er/to3d.pyと同じ)。
"""

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree
from mathutils.geometry import delaunay_2d_cdt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bear  # noqa: E402  (輪郭の処理を共用する)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "renders" / "bear2"
PATTERN_NAME = "bear2_front"  # bear3などから使うときは、OUTとあわせて差し替える
RES = 1024
MB_K = 0.574  # メタボール(しきい値0.6・硬さ2)の面の半径 / 要素の半径
OFFSET = 0.004  # 服を体から浮かせる距離(m)
MESH_STEP_PX = 8.0  # 服のメッシュの細かさ(画素)
ADJUST_POINTS = 200  # 微調整で線を取り直す点数
SMOOTH_ITERS = 12
CLEARANCE = 0.01  # 袖ぐり・襟ぐりを腕・首の付け根から離す距離(m)
SIDE_DEG = 90  # 脇線の位置(正面から左右へ何度か)
OBLIQUE = ((0.55, -0.85, 0.42), (0, 0, 0.14), 40)  # 斜めカメラ(位置, 注視点, レンズmm)
UNFOLD_CAM = ((0.2, -1.05, 0.3), (0.2, 0, 0.15), 50)  # 平面に開く場面のカメラ
# ---- モデルに合わせる値(モデルを差し替えたらここを直す) ----
# 頭と腕の形(build_bear2の要素と対応)。腕の半径と首の高さはmeasure_shapeで実際の面から測って上書きする
HEAD_C = Vector((0, 0, 0.215))
TORSO_TOP = (0.125, 0.05)  # 胴の上の球(中心z, 面の半径)
HEAD_BALL = (0.215, 0.074)  # 頭の球(中心z, 面の半径)
NECK_Z = 0.15
NECK_SQUASH_FROM = 0.09  # 首・裾に合わせて縦に縮めるときの基準の高さ
HEM_MIN_Z = 0.05  # 裾の下限。これより下では胴の面が内側へ回り込む
ARM_Z = 0.127
ARM_R = 0.032
ARM_AXES = [((s * 0.075, -0.008, 0.129), (s * 0.14, -0.015, 0.125)) for s in (-1, 1)]

FUR = (0.66, 0.45, 0.28, 1.0)
LIGHT = (0.95, 0.86, 0.72, 1.0)
DARK = (0.03, 0.025, 0.02, 1.0)


# ---------- クマ ----------

def ball(mb, co, r) -> None:
    e = mb.elements.new(type="BALL")
    e.co, e.radius = co, r / MB_K


def capsule(mb, a, b, r) -> None:
    """aからbへの太さrの棒(両端が丸い)。"""
    a, b = Vector(a), Vector(b)
    e = mb.elements.new(type="CAPSULE")
    e.co = (a + b) / 2
    e.radius = r / MB_K
    e.size_x = (b - a).length / 2
    e.rotation = Vector((1, 0, 0)).rotation_difference(b - a)


def smooth_part(name, co, radii, rgba, rot=(0, 0, 0)) -> bpy.types.Object:
    obj = bear.ellipsoid(name, co, radii, bear.material(name, rgba), rot=rot)
    obj.modifiers.new("Subsurf", "SUBSURF").levels = 1
    return obj


def metaball_mesh(name: str, legs: bool) -> bpy.types.Mesh:
    """胴・頭・腕(・脚)のメタボールをつないだなめらかなメッシュ。"""
    mb = bpy.data.metaballs.new(name)
    mb.resolution = mb.render_resolution = 0.004
    # 胴: 洋梨形(下が太い)
    ball(mb, (0, 0, 0.07), 0.068)
    ball(mb, (0, 0, TORSO_TOP[0]), TORSO_TOP[1])
    # 頭: 大きく丸い。少し横長
    ball(mb, (-0.012, 0, HEAD_BALL[0]), HEAD_BALL[1])
    ball(mb, (0.012, 0, HEAD_BALL[0]), HEAD_BALL[1])
    # 腕: 横に水平に広げる。太く短く、先が丸い
    for s in (-1, 1):
        capsule(mb, (s * 0.05, -0.005, 0.13), (s * 0.135, -0.015, 0.125), 0.03)
        ball(mb, (s * 0.14, -0.015, 0.125), 0.033)
    # 脚: 前に投げ出す
    for s in (-1, 1) if legs else ():
        capsule(mb, (s * 0.045, -0.02, 0.035), (s * 0.06, -0.105, 0.035), 0.035)
        ball(mb, (s * 0.062, -0.11, 0.038), 0.038)
    tmp = bpy.data.objects.new(name + "MB", mb)
    bpy.context.scene.collection.objects.link(tmp)
    dg = bpy.context.evaluated_depsgraph_get()
    mesh = bpy.data.meshes.new_from_object(tmp.evaluated_get(dg))
    bpy.data.objects.remove(tmp)
    mesh.name = name
    return mesh


def build_bear2() -> bpy.types.Object:
    """2頭身の座ったクマ(高さ約30cm)。胴・頭・腕・脚をつないだ体のメッシュを返す。"""
    mesh = metaball_mesh("BearBody", legs=True)
    for p in mesh.polygons:
        p.use_smooth = True
    body = bpy.data.objects.new("BearBody", mesh)
    bpy.context.scene.collection.objects.link(body)
    body.data.materials.append(bear.material("Fur", FUR, 0.9))

    # 顔のパーツと足裏は、正面からのレイキャストで体の表面に貼る(メタボールは要素がつながる分だけ膨らむため)
    def front_y(x: float, z: float) -> float:
        hit, *_ = bvh.ray_cast(Vector((x, -1.0, z)), Vector((0, 1, 0)))
        return hit.y

    bvh = BVHTree.FromPolygons([v.co for v in mesh.vertices], [p.vertices for p in mesh.polygons])
    # 耳: 頭の上の平たい円盤
    for s in (-1, 1):
        rot = (math.pi / 2, s * 0.35, 0)
        smooth_part(f"Ear_{s}", (s * 0.066, 0.005, 0.282), (0.034, 0.034, 0.011), FUR, rot)
        smooth_part(f"EarIn_{s}", (s * 0.066, -0.004, 0.281), (0.021, 0.021, 0.004), LIGHT, rot)
    # マズル: 前に突き出す
    my = front_y(0, 0.19)
    smooth_part("Muzzle", (0, my - 0.006, 0.19), (0.036, 0.026, 0.027), LIGHT)
    smooth_part("Nose", (0, my - 0.03, 0.202), (0.012, 0.008, 0.009), DARK)
    for s in (-1, 1):
        smooth_part(f"Eye_{s}", (s * 0.032, front_y(s * 0.032, 0.232) + 0.002, 0.232),
                    (0.0085, 0.0085, 0.0085), DARK)
        smooth_part(f"Sole_{s}", (s * 0.062, front_y(s * 0.062, 0.038) + 0.002, 0.038),
                    (0.025, 0.006, 0.025), LIGHT)
    return body


# ---------- カメラと照明 ----------

def setup_world(color=(0.93, 0.93, 0.95)) -> None:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.view_settings.view_transform = "Standard"
    world = bpy.data.worlds.new("World")
    world.color = color[:3]
    scene.world = world
    for loc, energy, size in (((0.4, -1.0, 0.8), 10, 1.0), ((-0.6, -0.6, 0.5), 4, 1.0), ((0, 0.6, 0.9), 4, 1.0)):
        bpy.ops.object.light_add(type="AREA", location=loc)
        lt = bpy.context.object
        lt.data.energy, lt.data.size = energy, size
        lt.rotation_euler = (Vector((0, 0, 0.15)) - lt.location).to_track_quat("-Z", "Y").to_euler()


def add_camera(name, loc, target, lens) -> bpy.types.Object:
    data = bpy.data.cameras.new(name)
    data.lens, data.sensor_width = lens, 36
    cam = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(cam)
    cam.location = loc
    cam.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    return cam


def render(cam, path: Path, res=(RES, RES)) -> None:
    scene = bpy.context.scene
    scene.camera = cam
    scene.render.resolution_x, scene.render.resolution_y = res
    scene.render.resolution_percentage = 100
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def camera_info(cam) -> dict:
    bpy.context.view_layer.update()
    return {
        "resolution": [RES, RES],
        "camera": {
            "location": list(cam.location),
            "matrix_world": [list(r) for r in cam.matrix_world],
            "lens_mm": cam.data.lens,
            "sensor_width_mm": cam.data.sensor_width,
        },
    }


def pixel_ray(info: dict, u: float, v: float) -> tuple[Vector, Vector]:
    """画素(左上原点)からの光線(始点, 向き)。er/to3d.pyと同じ式。"""
    w, h = info["resolution"]
    m = Matrix(info["camera"]["matrix_world"])
    f_px = info["camera"]["lens_mm"] / info["camera"]["sensor_width_mm"] * w
    d = m.to_3x3() @ Vector(((u - w / 2) / f_px, -(v - h / 2) / f_px, -1.0))
    return m.translation.copy(), d.normalized()


def world_to_pixel(info: dict, p: Vector) -> tuple[float, float]:
    w, h = info["resolution"]
    m = Matrix(info["camera"]["matrix_world"])
    f_px = info["camera"]["lens_mm"] / info["camera"]["sensor_width_mm"] * w
    c = m.inverted() @ Vector(p)
    return (w / 2 + f_px * c.x / -c.z, h / 2 - f_px * c.y / -c.z)


# ---------- ERの線 → 服 ----------

def test_polygon(info: dict) -> list[list[float]]:
    """動作確認用: 胴の前に手で置いたタンクトップの前身頃の輪郭を画素にしたもの。"""
    # (x, z) [m]。右脇の裾から反時計回り(正面から見て)
    half = [(0.062, 0.03), (0.064, 0.085), (0.052, 0.112), (0.036, 0.122), (0.033, 0.152),
            (0.022, 0.155), (0.014, 0.138), (0.0, 0.132)]
    left = [(-x, z) for x, z in reversed(half[:-1])]
    pts = half + left
    bottom = [(-0.062 + 0.124 * k / 8, 0.03 - 0.006 * math.sin(math.pi * k / 8)) for k in range(1, 8)]
    pts = pts + bottom
    return [list(world_to_pixel(info, Vector((x, -0.2, z)))) for x, z in pts]


def resample_px(poly: list, step: float) -> list:
    n = max(int(sum(math.dist(a, b) for a, b in zip(poly, poly[1:] + poly[:1])) / step), 12)
    return bear.resample([tuple(p) for p in poly], n)


def body_bvh(me: bpy.types.Mesh | None = None) -> BVHTree:
    """服を沿わせる面: 脚を除いた体(座ったクマの裾は脚の後ろに入る)。meを渡せばそのメッシュを使う。"""
    own = me is None
    if own:
        me = metaball_mesh("BearTorso", legs=False)
    bvh = BVHTree.FromPolygons([v.co.copy() for v in me.vertices], [p.vertices for p in me.polygons])
    if own:
        bpy.data.meshes.remove(me)
    return bvh


def snap(bvh: BVHTree, p: Vector) -> tuple[Vector, Vector]:
    """体の表面で最も近い点と法線。"""
    hit, normal, _, _ = bvh.find_nearest(p)
    return hit, normal


def resample_closed3(pts: list[Vector], n: int) -> list[Vector]:
    closed = pts + pts[:1]
    seg = [(b - a).length for a, b in zip(closed, closed[1:])]
    total = sum(seg)
    out, k, acc = [], 0, 0.0
    for m in range(n):
        s = total * m / n
        while acc + seg[k] < s:
            acc += seg[k]
            k += 1
        out.append(closed[k].lerp(closed[k + 1], (s - acc) / seg[k]))
    return out


def measure_shape(bvh: BVHTree) -> None:
    """頭と腕の実際の半径を、正面からの光線で測って上書きする。"""
    global NECK_Z, ARM_R
    # 首: 胴の上の球と頭の球が交わる高さ(くびれ)。頭が胴より前に出ているので正面の光線では測れない
    (z1, r1), (z2, r2) = TORSO_TOP, HEAD_BALL
    NECK_Z = ((r1**2 - r2**2) - (z1**2 - z2**2)) / (2 * (z2 - z1))
    a, b = (Vector(v) for v in ARM_AXES[1])
    mid = (a + b) / 2
    hit, *_ = bvh.ray_cast(Vector((mid.x, -1.0, mid.z)), Vector((0, 1, 0)))
    ARM_R = abs(hit.y - mid.y)
    print(f"首のくびれ z={NECK_Z * 100:.1f}cm 腕の半径{ARM_R * 100:.1f}cm")


def project_line(bvh: BVHTree, info: dict, poly_px: list) -> tuple[list[Vector], int]:
    """ERの線(画素)を、正面カメラからの光線で体の表面に写す。"""
    pts, miss = [], 0
    for u, v in poly_px:
        o, d = pixel_ray(info, u, v)
        hit, *_ = bvh.ray_cast(o, d)
        if hit is None:  # 体の外にはみ出た点: 光線上でクマの中心と同じ深さの点から最も近い表面へ
            miss += 1
            hit, _ = snap(bvh, o + d * (o - HEAD_C).length)
        pts.append(hit)
    return pts, miss


def adjust_line(bvh: BVHTree, projected: list[Vector]) -> tuple[list[Vector], dict]:
    """投影した線を体型に合わせて微調整する(左右対称 → 脇線 → 付け根から離す → ならす)。"""
    n = ADJUST_POINTS
    pts = resample_closed3(projected, n)
    # 始点を裾の前中心(体の中心面に近い点のうち最も低い点)にそろえる
    low = [i for i, p in enumerate(pts) if p.z < sum(q.z for q in pts) / n]
    start = min(low, key=lambda i: abs(pts[i].x))
    pts = pts[start:] + pts[:start]

    # 1. 左右対称: 中心面(x=0)で鏡映した線と平均する
    mirror = [Vector((-p.x, p.y, p.z)) for p in [pts[(n - i) % n] for i in range(n)]]
    pts = [snap(bvh, (a + b) / 2)[0] for a, b in zip(pts, mirror)]

    # 2. 脇線: 腕より下で横へ回り込んだ区間を、正面からSIDE_DEG度の線(体の中心軸から見た向き)に置く
    side_moved = 0
    for i, p in enumerate(pts):
        phi = math.degrees(math.atan2(abs(p.x), -p.y))  # 0=前中心、90=真横
        w = min(1.0, max(0.0, (phi - (SIDE_DEG - 20)) / 10))
        if w > 0 and p.z < ARM_Z - ARM_R:
            s = 1 if p.x > 0 else -1
            a = math.radians(SIDE_DEG)
            d = Vector((s * math.sin(a), -math.cos(a), 0))
            hit, *_ = bvh.ray_cast(Vector((0, 0, p.z)) + d * 0.5, -d)
            if hit is not None:
                pts[i] = snap(bvh, p.lerp(hit, w))[0]
                side_moved += 1

    # 3. 付け根から距離を空ける: 頭(首)と腕から一定距離以上離す
    def push(p: Vector) -> Vector:
        for _ in range(8):
            moved = False
            for a, b in ARM_AXES:
                a, b = Vector(a), Vector(b)
                t = max(0.0, min(1.0, (p - a).dot(b - a) / (b - a).length_squared))
                c = a + (b - a) * t
                if (p - c).length < ARM_R + CLEARANCE:
                    p = c + (p - c).normalized() * (ARM_R + CLEARANCE)
                    moved = True
            p = snap(bvh, p)[0]
            if not moved:
                break
        return p

    before = [p.copy() for p in pts]
    # 首: 線の上部(肩紐・襟ぐり)を縦に縮めて、くびれの高さから一定距離下に収める(形は保つ)
    top, limit, z0 = max(p.z for p in pts), NECK_Z - CLEARANCE, NECK_SQUASH_FROM
    if top > limit:
        k = (limit - z0) / (top - z0)
        pts = [snap(bvh, Vector((p.x, p.y, z0 + (p.z - z0) * k)))[0] if p.z > z0 else p for p in pts]
    # 裾: 線の下部を縦に縮めて、胴の面が下へ回り込む高さより上に収める(座面から離す)
    bottom = min(p.z for p in pts)
    if bottom < HEM_MIN_Z:
        k = (z0 - HEM_MIN_Z) / (z0 - bottom)
        pts = [snap(bvh, Vector((p.x, p.y, z0 - (z0 - p.z) * k)))[0] if p.z < z0 else p for p in pts]
    pts = [push(p) for p in pts]
    pushed = sum(1 for a, b in zip(before, pts) if (a - b).length > 1e-4)

    # 4. 表面に沿わせてなめらかに: 隣の点の平均へ寄せては表面に戻す(ラプラシアン平滑化)
    for _ in range(SMOOTH_ITERS):
        pts = [snap(bvh, p.lerp((pts[i - 1] + pts[(i + 1) % n]) / 2, 0.5))[0] for i, p in enumerate(pts)]
    pts = [push(p) for p in pts]

    # 調整で動いた量: 調整後の各点から投影直後の線(細かく取り直したもの)までの最短距離
    dense = resample_closed3(projected, 2000)
    dist = [min((p - q).length for q in dense) for p in pts]
    return pts, {"side_points": side_moved, "pushed_points": pushed,
                 "moved_max_mm": max(dist) * 1000, "moved_mean_mm": sum(dist) / n * 1000}


def garment_from_line(bvh: BVHTree, info: dict, line: list[Vector]) -> tuple[bpy.types.Object, dict]:
    """調整後の3Dの線を境界にした服の面。内部は正面からの光線で体に写し、境界は線の3D点そのものを使う。"""
    boundary = [world_to_pixel(info, p) for p in line]
    if bear.poly_area([(x, -y) for x, y in boundary]) < 0:  # 画素はy下向き。画面上で反時計回りにそろえる
        boundary.reverse()
        line = line[::-1]
    xs, ys = [p[0] for p in boundary], [p[1] for p in boundary]
    interior = []
    step = MESH_STEP_PX
    y = min(ys) + step / 2
    while y < max(ys):
        x = min(xs) + step / 2 + (step / 2 if int(y / step) % 2 else 0)
        while x < max(xs):
            if inside((x, y), boundary) and min(math.dist((x, y), b) for b in boundary) > step * 0.6:
                interior.append((x, y))
            x += step
        y += step * 0.866
    nb = len(boundary)
    if bear.self_intersects(boundary):
        print("警告: 調整後の線を正面から見ると自己交差している")
    pts2 = [Vector(p) for p in boundary + interior]
    edges = [(i, (i + 1) % nb) for i in range(nb)]
    verts, _, faces, orig, *_ = delaunay_2d_cdt(pts2, edges, [list(range(nb))], 1, 1e-6)
    co3, miss = [], 0
    for v, src in zip(verts, orig, strict=True):
        b = [k for k in src if k < nb]
        if b:
            hit, normal = snap(bvh, line[b[0]])
        else:
            o, d = pixel_ray(info, v.x, v.y)
            hit, normal, _, _ = bvh.ray_cast(o, d)
            if hit is None:
                miss += 1
                hit, normal = snap(bvh, o + d * (o - HEAD_C).length)
        co3.append(hit + normal * OFFSET)
    mesh = bpy.data.meshes.new("Garment2")
    mesh.from_pydata(co3, [], [list(f) for f in faces])
    for p in mesh.polygons:
        p.use_smooth = True
    obj = bpy.data.objects.new("Garment2", mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj, {"verts": len(co3), "faces": len(faces), "miss": miss}


def inside(p, poly) -> bool:
    x, y = p
    c = False
    for (x0, y0), (x1, y1) in zip(poly, poly[1:] + poly[:1]):
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            c = not c
    return c


def flatten2(panel) -> tuple[list[Vector], list[int], dict]:
    """UV展開 → 実寸 → 3Dの(x, z)に最も合うよう回転(必要なら左右反転)。"""
    uv = bear.unwrap(panel)
    mesh = panel.data
    area3d = sum(p.area for p in mesh.polygons)
    area_uv = sum(abs(bear.poly_area([tuple(uv[v]) for v in p.vertices])) for p in mesh.polygons)
    k = math.sqrt(area3d / area_uv)
    flat = [Vector((u.x * k, u.y * k)) for u in uv]
    tgt = [Vector((v.co.x, v.co.z)) for v in mesh.vertices]
    fc = sum(flat, Vector((0, 0))) / len(flat)
    tc = sum(tgt, Vector((0, 0))) / len(tgt)
    best = None
    for mirror in (1, -1):
        f = [Vector(((p.x - fc.x) * mirror, p.y - fc.y)) for p in flat]
        # 2Dの回転だけのプロクラステス
        sxx = sum(a.x * b.x + a.y * b.y for a, b in zip(f, (t - tc for t in tgt)))
        sxy = sum(a.x * b.y - a.y * b.x for a, b in zip(f, (t - tc for t in tgt)))
        th = math.atan2(sxy, sxx)
        rot = Matrix(((math.cos(th), -math.sin(th)), (math.sin(th), math.cos(th))))
        g = [rot @ p for p in f]
        err = sum((a - (b - tc)).length_squared for a, b in zip(g, tgt))
        if best is None or err < best[0]:
            best = (err, g)
    flat = best[1]
    loop = bear.boundary_loop(mesh)
    co = [v.co for v in mesh.vertices]
    l3 = sum((co[a] - co[b]).length for a, b in zip(loop, loop[1:] + loop[:1]))
    l2 = sum((flat[a] - flat[b]).length for a, b in zip(loop, loop[1:] + loop[:1]))
    return flat, loop, {"area_m2": area3d, "perimeter_3d": l3, "perimeter_flat": l2}


# ---------- 画像 ----------

def polyline(name, pts, rgba, width) -> bpy.types.Object:
    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = width
    sp = curve.splines.new("POLY")
    sp.points.add(len(pts) - 1)
    for p, c in zip(sp.points, pts):
        p.co = (*c, 1.0)
    sp.use_cyclic_u = True
    obj = bpy.data.objects.new(name, curve)
    bpy.context.scene.collection.objects.link(obj)
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = rgba
    em.inputs["Strength"].default_value = 1.0
    nt.links.new(em.outputs[0], nt.nodes.new("ShaderNodeOutputMaterial").inputs["Surface"])
    obj.data.materials.append(mat)
    return obj


def er_line_overlay(info: dict, poly_px: list) -> bpy.types.Object:
    """ERの線を正面カメラのすぐ前の面に置き、正面画像で画素どおりの位置に写るようにする。"""
    pts = []
    for u, v in poly_px:
        o, d = pixel_ray(info, u, v)
        pts.append(o + d * 0.9)  # クマより手前
    return polyline("ERLine", pts, (1.0, 0.1, 0.05, 1.0), 0.0012)


def pattern_from_er(bvh: BVHTree, info: dict, poly_px: list, source: str) -> tuple:
    """ERの線(画素) → 体の表面に投影 → 微調整 → 服の面 → UV展開 → patterns/bear2_front.json。

    モデルに依存しない処理。モデルを差し替えるときは、服を沿わせる面のBVHと正面カメラの情報を渡し、
    冒頭の「モデルに合わせる値」(首・腕・裾の位置)を新しいモデルに合わせる。
    戻り値: (服のオブジェクト, 平面の頂点座標, 投影直後の線, 微調整後の線)
    """
    projected, miss = project_line(bvh, info, poly_px)
    adjusted, adj = adjust_line(bvh, projected)
    print(f"投影: 体に当たらなかった点{miss}  微調整: 脇線{adj['side_points']}点 付け根から離した{adj['pushed_points']}点  "
          f"動いた量 最大{adj['moved_max_mm']:.1f}mm 平均{adj['moved_mean_mm']:.1f}mm")
    (OUT / "seam_lines.json").write_text(json.dumps({
        "projected": [[round(c, 5) for c in p] for p in projected],
        "adjusted": [[round(c, 5) for c in p] for p in adjusted],
        "moved_max_mm": round(adj["moved_max_mm"], 2),
        "moved_mean_mm": round(adj["moved_mean_mm"], 2),
        "rules": ["左右対称", f"脇線を正面から{SIDE_DEG}°へ", f"首・腕の付け根から{CLEARANCE * 100:.0f}cm以上",
                  f"裾をz={HEM_MIN_Z * 100:.0f}cm以上", "表面に沿ってならす"],
    }, indent=1))
    garment, st = garment_from_line(bvh, info, adjusted)
    print(f"服のメッシュ: 頂点{st['verts']} 面{st['faces']}  体に当たらなかった点{st['miss']}")
    garment.data.materials.append(bear.material("ShirtFront", (0.85, 0.3, 0.22, 1.0)))
    flat, loop, stats = flatten2(garment)
    out = bear.outline_json(flat, loop)
    out["name"] = PATTERN_NAME
    out["source"] = source
    pat = ROOT / "patterns" / f"{PATTERN_NAME}.json"
    pat.write_text(json.dumps(out, indent=1))
    print(f"outline: {len(out['outline'])}点  {out['size_m'][1] * 100:.1f}cm(縦) x "
          f"{out['size_m'][0] * 100:.1f}cm(横)  scale={out['scale']}")
    print(f"面積 {stats['area_m2'] * 1e4:.1f}cm2  外周 3D {stats['perimeter_3d'] * 100:.2f}cm / "
          f"平面 {stats['perimeter_flat'] * 100:.2f}cm")

    return garment, flat, projected, adjusted


def front_camera() -> bpy.types.Object:
    return add_camera("Cam_Front", (0, -1.4, 0.155), (0, 0, 0.155), 95)


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--shoot", action="store_true", help="段階1: 正面画像とカメラ情報を書く")
    ap.add_argument("--er", type=Path, default=None, help="ERの出力(既定: er/out/front_er.json、なければ手作りの多角形)")
    ap.add_argument("--anim", action="store_true")
    args = ap.parse_args(argv)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    body = build_bear2()
    setup_world()
    cam = front_camera()
    OUT.mkdir(parents=True, exist_ok=True)
    info = camera_info(cam)

    if args.shoot:
        render(cam, OUT / "front.png")
        (OUT / "front.json").write_text(json.dumps(info, indent=1))
        (OUT / "test_er.json").write_text(json.dumps({"trajectory_px": test_polygon(info)}, indent=1))
        oblique = add_camera("Cam_Oblique", *OBLIQUE)
        render(oblique, OUT / "look_oblique.png", (1280, 720))
        print("wrote front.png/front.json/test_er.json/look_*.png")
        return

    er_path = args.er or ROOT / "er" / "out" / "front_er.json"
    if not er_path.is_absolute():
        er_path = ROOT / er_path
    if not er_path.exists():
        er_path = OUT / "test_er.json"
    poly_px = json.loads(er_path.read_text())["trajectory_px"]
    print(f"ERの線: {er_path.relative_to(ROOT)}  {len(poly_px)}点")

    bvh = body_bvh()
    measure_shape(bvh)
    stage2(cam, bvh, info, poly_px, er_path.name, args.anim)


def stage2(cam, bvh: BVHTree, info: dict, poly_px: list, source: str, anim: bool) -> None:
    """型紙を作り、動画用の画像(ERの線・体の上の線・色付けした範囲・平面に開く流れ)を描く。"""
    garment, flat, projected, adjusted = pattern_from_er(bvh, info, poly_px, source)
    garment.modifiers.new("Thick", "SOLIDIFY").thickness = 0.002
    garment.hide_render = True
    line = er_line_overlay(info, poly_px)
    render(cam, OUT / "er_line.png")
    line.hide_render = True

    # 投影直後(赤)と微調整後(緑)の線を体の上に描く。法線方向へ少し浮かせる
    lift = lambda pts: [snap(bvh, p)[0] + snap(bvh, p)[1] * 0.002 for p in pts]  # noqa: E731
    oblique = add_camera("Cam_Oblique", *OBLIQUE)
    lines = [polyline("SeamER", lift(projected), (1.0, 0.1, 0.05, 1.0), 0.0012),
             polyline("SeamAdjusted", lift(adjusted), (0.1, 0.85, 0.2, 1.0), 0.0012)]
    render(cam, OUT / "seam_front.png")
    render(oblique, OUT / "seam_oblique.png", (1280, 720))
    for o in lines:
        bpy.data.objects.remove(o)
    # 投影直後 → 微調整後 へ変わる数コマ(線を弧長でそろえて補間)
    src = resample_closed3(projected, ADJUST_POINTS)
    start = min(range(ADJUST_POINTS), key=lambda i: (src[i] - adjusted[0]).length)
    src = src[start:] + src[:start]
    for k in range(5):
        t = k / 4
        pts = [snap(bvh, a.lerp(b, t))[0] for a, b in zip(src, adjusted)]
        o = polyline("SeamMorph", lift(pts), (1.0 - 0.9 * t, 0.1 + 0.75 * t, 0.05 + 0.15 * t, 1.0), 0.0012)
        render(oblique, OUT / f"seam_morph_{k}.png", (1280, 720))
        bpy.data.objects.remove(o)

    garment.hide_render = False
    render(oblique, OUT / "dressed.png", (1280, 720))

    # 平面に開く(bear.pyと同じ流れ)。カメラはクマの右隣まで入る構図
    side = add_camera("Cam_Unfold", *UNFOLD_CAM)
    bpy.context.scene.camera = side
    bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y = 1280, 720
    stills = [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (1.0, 0.5), (1.0, 1.0)]
    bear.render_unfold(garment, flat, stills, [OUT / f"unfold_{i}.png" for i in range(len(stills))])
    if anim:
        steps = bear.anim_steps()
        bear.render_unfold(garment, flat, steps, [OUT / "anim" / f"{i:04d}.png" for i in range(len(steps))])
        print(f"連番: {(OUT / 'anim').relative_to(ROOT)}/ {len(steps)}コマ")
    print(f"wrote patterns/{PATTERN_NAME}.json, {OUT.relative_to(ROOT)}/"
          "{er_line,seam_front,seam_oblique,seam_morph_*,dressed,unfold_*}.png, seam_lines.json")


if __name__ == "__main__":
    main()
