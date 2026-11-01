"""服を着たクマのぬいぐるみを作り、服の前身頃をUV展開して平面の型紙にする。

実行: Blender -b --factory-startup -P blender/bear.py [-- --anim]
出力:
    patterns/bear_front.json   前身頃の輪郭(m、重心が原点、首が+y、反時計回り)
    renders/bear/*.png         服を着たクマ → 前身頃が平面に開くまでの数コマ(1280x720)
    renders/bear/anim/*.png    --anim時: 同じ流れの6秒の連番(24fps)

クマは-y方向を向いて座る。床がz=0、単位m。
"""

import argparse
import json
import math
import sys
from pathlib import Path

import bmesh
import bpy
from mathutils import Matrix, Vector

ROOT = Path(__file__).resolve().parent.parent
N_THETA = 128  # 筒の周方向の分割(4の倍数: θ=±90°の脇線に列を置く)
N_ROWS = 32  # 裾から上端までの分割
HEM_Z = 0.04
SHOULDER_Z = 0.168
TUBE_R = 0.12
OUTLINE_POINTS = 160
UNWRAP_METHOD = "CONFORMAL"  # UV展開の方法(人台ではMINIMUM_STRETCHも使う)
MAX_SIZE = 0.30


# ---------- クマ ----------

def material(name: str, rgba: tuple, roughness: float = 0.8) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = rgba
    bsdf.inputs["Roughness"].default_value = roughness
    return mat


def ellipsoid(name: str, center, radii, mat, rot=(0, 0, 0)) -> bpy.types.Object:
    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=1.0,
                                         location=center, rotation=rot)
    obj = bpy.context.object
    obj.name = name
    obj.scale = radii
    bpy.ops.object.shade_smooth()
    obj.data.materials.append(mat)
    return obj


def build_bear() -> bpy.types.Object:
    """座ったクマ(高さ約30cm)。服を沿わせる胴体のオブジェクトを返す。"""
    fur = material("Fur", (0.45, 0.28, 0.14, 1.0))
    light = material("FurLight", (0.78, 0.62, 0.45, 1.0))
    dark = material("Eye", (0.02, 0.02, 0.02, 1.0), 0.3)
    torso = ellipsoid("Torso", (0, 0, 0.10), (0.08, 0.07, 0.10), fur)
    ellipsoid("Head", (0, 0, 0.24), (0.062, 0.058, 0.06), fur)
    for s in (-1, 1):
        ellipsoid(f"Ear_{s}", (s * 0.045, 0.005, 0.29), (0.022, 0.012, 0.022), fur)
        ellipsoid(f"EarIn_{s}", (s * 0.045, -0.006, 0.29), (0.013, 0.004, 0.013), light)
        ellipsoid(f"Eye_{s}", (s * 0.022, -0.052, 0.255), (0.007, 0.007, 0.007), dark)
        ellipsoid(f"Arm_{s}", (s * 0.095, -0.015, 0.11), (0.025, 0.025, 0.062), fur,
                  rot=(0.3, s * 0.35, 0))
        ellipsoid(f"Leg_{s}", (s * 0.045, -0.06, 0.03), (0.036, 0.062, 0.03), fur)
        ellipsoid(f"Sole_{s}", (s * 0.045, -0.118, 0.032), (0.026, 0.006, 0.022), light)
    ellipsoid("Muzzle", (0, -0.052, 0.226), (0.03, 0.02, 0.022), light)
    ellipsoid("Nose", (0, -0.073, 0.236), (0.01, 0.007, 0.007), dark)
    return torso


# ---------- 服 ----------

def top_edge(theta: float) -> float:
    """服の上端の高さ。θ=0が前中心、±π/2が脇、±πが後ろ中心。

    前後とも 襟ぐり(くぼみ)→肩紐(平ら)→袖ぐり(脇へ向かって深いくぼみ) の形。
    """
    phi = abs(theta)
    front = phi <= math.pi / 2
    p = phi if front else math.pi - phi  # 前中心または後ろ中心からの角度
    neck_w, strap_w = math.radians(36), math.radians(18)
    neck_d = 0.045 if front else 0.018
    arm_d = 0.06
    if p < neck_w:
        return SHOULDER_Z - neck_d * math.sqrt(1 - (p / neck_w) ** 2)
    if p < neck_w + strap_w:
        return SHOULDER_Z
    a = (p - neck_w - strap_w) / (math.pi / 2 - neck_w - strap_w)  # 0..1(1が脇)
    return SHOULDER_Z - arm_d * math.sqrt(1 - (1 - a) ** 2)


def build_garment(torso: bpy.types.Object) -> tuple[bpy.types.Object, list, list]:
    """筒を作ってShrinkwrapで胴に沿わせる。(服, 各頂点の(列,行), 前身頃の面)を返す。"""
    verts, grid = [], []
    for j in range(N_THETA):
        th = -math.pi + 2 * math.pi * j / N_THETA
        h = top_edge(th)
        for i in range(N_ROWS + 1):
            z = HEM_Z + (h - HEM_Z) * i / N_ROWS
            verts.append((TUBE_R * math.sin(th), -TUBE_R * math.cos(th), z))
            grid.append((j, i))
    idx = lambda j, i: (j % N_THETA) * (N_ROWS + 1) + i  # noqa: E731
    faces, front = [], []
    for j in range(N_THETA):
        th_mid = -math.pi + 2 * math.pi * (j + 0.5) / N_THETA
        for i in range(N_ROWS):
            faces.append((idx(j, i), idx(j + 1, i), idx(j + 1, i + 1), idx(j, i + 1)))
            front.append(abs(th_mid) < math.pi / 2)
    mesh = bpy.data.meshes.new("Garment")
    mesh.from_pydata(verts, [], faces)
    obj = bpy.data.objects.new("Garment", mesh)
    bpy.context.scene.collection.objects.link(obj)

    mod = obj.modifiers.new("Fit", "SHRINKWRAP")
    mod.target = torso
    # 最近点だと上下の端が胴の赤道側へ寄って縦に潰れるので、筒の法線(水平・内向き)に沿って投影する
    mod.wrap_method = "PROJECT"
    mod.use_negative_direction = True
    mod.offset = 0.004
    dg = bpy.context.evaluated_depsgraph_get()
    fitted = bpy.data.meshes.new_from_object(obj.evaluated_get(dg))
    obj.modifiers.clear()
    obj.data = fitted
    fitted.name = "Garment"
    return obj, grid, front


def split_front(garment: bpy.types.Object, grid: list, front: list) -> tuple[bpy.types.Object, list]:
    """前身頃の面を別オブジェクトに分ける。(前身頃, 前身頃の各頂点の(列,行))を返す。"""
    src = garment.data
    keep = [p for p, f in zip(src.polygons, front, strict=True) if f]
    vids = sorted({v for p in keep for v in p.vertices})
    remap = {v: k for k, v in enumerate(vids)}
    mesh = bpy.data.meshes.new("FrontPanel")
    mesh.from_pydata([src.vertices[v].co.copy() for v in vids], [],
                     [[remap[v] for v in p.vertices] for p in keep])
    obj = bpy.data.objects.new("FrontPanel", mesh)
    bpy.context.scene.collection.objects.link(obj)
    # 服本体からは前身頃の面を消し、後ろ身頃だけ残す
    bm = bmesh.new()
    bm.from_mesh(src)
    bm.faces.ensure_lookup_table()
    bmesh.ops.delete(bm, geom=[bm.faces[p.index] for p in keep], context="FACES_ONLY")
    bm.to_mesh(src)
    bm.free()
    return obj, [grid[v] for v in vids]


def unwrap(obj: bpy.types.Object) -> list[Vector]:
    """UV展開して頂点ごとのUVを返す(シームなし=1頂点1UV)。"""
    for o in bpy.context.selected_objects:
        o.select_set(False)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.unwrap(method=UNWRAP_METHOD, margin=0.0)
    bpy.ops.object.mode_set(mode="OBJECT")
    uv = [None] * len(obj.data.vertices)
    layer = obj.data.uv_layers.active.data
    for loop in obj.data.loops:
        uv[loop.vertex_index] = layer[loop.index].uv.copy()
    return uv


# ---------- 輪郭 ----------

def boundary_loop(mesh: bpy.types.Mesh) -> list[int]:
    """外周の境界辺を順にたどった頂点番号の列。"""
    count: dict[tuple, int] = {}
    for p in mesh.polygons:
        for e in p.edge_keys:
            count[e] = count.get(e, 0) + 1
    nbr: dict[int, list] = {}
    for (a, b), c in count.items():
        if c == 1:
            nbr.setdefault(a, []).append(b)
            nbr.setdefault(b, []).append(a)
    assert all(len(v) == 2 for v in nbr.values()), "境界が単純な1本のループでない"
    start = min(nbr)
    loop, prev, cur = [start], None, start
    while True:
        nxt = nbr[cur][0] if nbr[cur][0] != prev else nbr[cur][1]
        if nxt == start:
            break
        loop.append(nxt)
        prev, cur = cur, nxt
    assert len(loop) == len(nbr), "境界が複数のループに分かれている"
    return loop


def poly_area(pts: list) -> float:
    return 0.5 * sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]))


def poly_centroid(pts: list) -> tuple[float, float]:
    a = poly_area(pts)
    cx = cy = 0.0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]):
        c = x0 * y1 - x1 * y0
        cx += (x0 + x1) * c
        cy += (y0 + y1) * c
    return cx / (6 * a), cy / (6 * a)


def resample(pts: list, n: int) -> list:
    closed = pts + pts[:1]
    seg = [math.dist(a, b) for a, b in zip(closed, closed[1:])]
    total = sum(seg)
    out, k, acc = [], 0, 0.0
    for m in range(n):
        s = total * m / n
        while acc + seg[k] < s:
            acc += seg[k]
            k += 1
        t = (s - acc) / seg[k]
        (x0, y0), (x1, y1) = closed[k], closed[k + 1]
        out.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
    return out


def self_intersects(pts: list) -> bool:
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    n = len(pts)
    segs = [(pts[i], pts[(i + 1) % n]) for i in range(n)]
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            (p1, p2), (p3, p4) = segs[i], segs[j]
            d1, d2 = cross(p3, p4, p1), cross(p3, p4, p2)
            d3, d4 = cross(p1, p2, p3), cross(p1, p2, p4)
            if d1 * d2 < 0 and d3 * d4 < 0:
                return True
    return False


def flatten(panel: bpy.types.Object, cells: list, uv: list) -> tuple[list, dict]:
    """UVを実寸(m)に戻し、首が+y・3Dの+xが平面の+xになるよう回す。頂点ごとの平面座標を返す。"""
    mesh = panel.data
    area3d = sum(p.area for p in mesh.polygons)
    area_uv = sum(abs(poly_area([tuple(uv[v]) for v in p.vertices])) for p in mesh.polygons)
    k = math.sqrt(area3d / area_uv)
    pts = [Vector((u.x * k, u.y * k)) for u in uv]
    hem = sum((p for p, (_, i) in zip(pts, cells) if i == 0), Vector((0.0, 0.0))) / sum(1 for _, i in cells if i == 0)
    top = sum((p for p, (_, i) in zip(pts, cells) if i == N_ROWS), Vector((0.0, 0.0))) / sum(1 for _, i in cells if i == N_ROWS)
    up = (top - hem).normalized()
    rot = Matrix(((up.y, -up.x), (up.x, up.y)))  # upを+yへ回す
    pts = [rot @ p for p in pts]
    # 3Dで+x側(脇θ=+90°)の頂点が平面でも+x側になるよう、必要なら左右反転
    def side_x(col: int) -> float:
        xs = [p.x for p, (j, _) in zip(pts, cells) if j == col]
        return sum(xs) / len(xs)

    if side_x(3 * N_THETA // 4) < side_x(N_THETA // 4):
        pts = [Vector((-p.x, p.y)) for p in pts]
    # 境界の長さが3Dと平面でどれだけ違うか(展開のゆがみの目安)
    loop = boundary_loop(mesh)
    co = [v.co for v in mesh.vertices]
    l3 = sum((co[a] - co[b]).length for a, b in zip(loop, loop[1:] + loop[:1]))
    l2 = sum((pts[a] - pts[b]).length for a, b in zip(loop, loop[1:] + loop[:1]))
    return pts, {"area_m2": area3d, "perimeter_3d": l3, "perimeter_flat": l2, "loop": loop}


def outline_json(pts: list, loop: list) -> dict:
    ring = [(pts[v].x, pts[v].y) for v in loop]
    if poly_area(ring) < 0:
        ring.reverse()
    ring = resample(ring, OUTLINE_POINTS)
    cx, cy = poly_centroid(ring)
    ring = [(x - cx, y - cy) for x, y in ring]
    w = max(x for x, _ in ring) - min(x for x, _ in ring)
    h = max(y for _, y in ring) - min(y for _, y in ring)
    scale = min(1.0, MAX_SIZE / max(w, h))
    ring = [(x * scale, y * scale) for x, y in ring]
    assert poly_area(ring) > 0, "反時計回りでない"
    assert not self_intersects(ring), "輪郭が自己交差している"
    return {
        "name": "bear_front",
        "outline": [[round(x, 6), round(y, 6)] for x, y in ring],
        "closed": True,
        "scale": round(scale, 6),
        "size_m": [round(w * scale, 4), round(h * scale, 4)],
    }


# ---------- 動画用の画像 ----------

def setup_render() -> None:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.view_settings.view_transform = "Standard"
    scene.render.resolution_x, scene.render.resolution_y = 1280, 720
    world = bpy.data.worlds.new("World")
    world.color = (0.18, 0.18, 0.2)
    scene.world = world
    bpy.ops.mesh.primitive_plane_add(size=3, location=(0, 0, 0))
    bpy.context.object.data.materials.append(material("Floor", (0.75, 0.72, 0.68, 1.0)))
    bpy.ops.object.light_add(type="AREA", location=(0.3, -0.8, 1.0))
    key = bpy.context.object
    key.data.energy, key.data.size = 12, 0.8
    key.rotation_euler = (Vector((0.15, 0, 0.15)) - key.location).to_track_quat("-Z", "Y").to_euler()
    bpy.ops.object.light_add(type="AREA", location=(-0.6, -0.3, 0.8))
    fill = bpy.context.object
    fill.data.energy, fill.data.size = 4, 1.0
    fill.rotation_euler = (Vector((0, 0, 0.15)) - fill.location).to_track_quat("-Z", "Y").to_euler()
    cam_data = bpy.data.cameras.new("Cam_Bear")
    cam_data.lens = 50
    cam = bpy.data.objects.new("Cam_Bear", cam_data)
    scene.collection.objects.link(cam)
    cam.location = (0.2, -1.05, 0.32)
    cam.rotation_euler = (Vector((0.2, 0, 0.15)) - cam.location).to_track_quat("-Z", "Y").to_euler()
    scene.camera = cam


def render_unfold(panel: bpy.types.Object, flat: list, steps: list, paths: list[Path]) -> None:
    """前身頃をクマの前でその場で平らにし、そのまま右隣へ滑らせる。

    steps: コマごとの(平らにする度合い, 横へ動かす度合い)、どちらも0..1。
    """
    paths[0].parent.mkdir(parents=True, exist_ok=True)
    co3 = [v.co.copy() for v in panel.data.vertices]
    cx = sum(p.x for p in flat) / len(flat)
    cy = sum(p.y for p in flat) / len(flat)
    front_y = min(c.y for c in co3) - 0.01
    z0 = sum(c.z for c in co3) / len(co3)
    co_flat = [Vector((p.x - cx, front_y, z0 + p.y - cy)) for p in flat]
    shift = Vector((0.42, front_y - 0.02 - front_y, 0.17 - z0))
    for (tf, tm), f in zip(steps, paths, strict=True):
        tf, tm = tf * tf * (3 - 2 * tf), tm * tm * (3 - 2 * tm)
        for v, a, b in zip(panel.data.vertices, co3, co_flat, strict=True):
            v.co = a.lerp(b, tf) + shift * tm
        panel.data.update()
        bpy.context.scene.render.filepath = str(f)
        bpy.ops.render.render(write_still=True)
    for v, a in zip(panel.data.vertices, co3, strict=True):
        v.co = a
    panel.data.update()


def anim_steps() -> list[tuple[float, float]]:
    """24fpsで 静止1.5秒 → 平らに開く2秒 → 右隣へ移動1.5秒 → 静止1秒。"""
    return (
        [(0.0, 0.0)] * 36
        + [(k / 48, 0.0) for k in range(1, 49)]
        + [(1.0, k / 36) for k in range(1, 37)]
        + [(1.0, 1.0)] * 24
    )


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--anim", action="store_true", help="renders/bear/anim/に6秒の連番PNGも描く")
    args = ap.parse_args(argv)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    torso = build_bear()
    garment, grid, front = build_garment(torso)
    panel, cells = split_front(garment, grid, front)
    garment.data.materials.append(material("ShirtBack", (0.16, 0.36, 0.62, 1.0)))
    panel.data.materials.append(material("ShirtFront", (0.85, 0.3, 0.22, 1.0)))
    for o in (garment, panel):
        o.modifiers.new("Thick", "SOLIDIFY").thickness = 0.002
        for p in o.data.polygons:
            p.use_smooth = True

    uv = unwrap(panel)
    flat, stats = flatten(panel, cells, uv)
    out = outline_json(flat, stats["loop"])
    pat_dir = ROOT / "patterns"
    pat_dir.mkdir(exist_ok=True)
    (pat_dir / "bear_front.json").write_text(json.dumps(out, indent=1))
    print(f"outline: {len(out['outline'])}点  {out['size_m'][1] * 100:.1f}cm(縦) x "
          f"{out['size_m'][0] * 100:.1f}cm(横)  scale={out['scale']}")
    print(f"面積 {stats['area_m2'] * 1e4:.1f}cm2  外周 3D {stats['perimeter_3d'] * 100:.2f}cm / "
          f"平面 {stats['perimeter_flat'] * 100:.2f}cm")

    setup_render()
    img_dir = ROOT / "renders" / "bear"
    stills = [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (1.0, 0.5), (1.0, 1.0)]
    render_unfold(panel, flat, stills, [img_dir / f"unfold_{i}.png" for i in range(len(stills))])
    print(f"画像: renders/bear/unfold_0〜{len(stills) - 1}.png")
    if args.anim:
        steps = anim_steps()
        render_unfold(panel, flat, steps, [img_dir / "anim" / f"{i:04d}.png" for i in range(len(steps))])
        print(f"連番: renders/bear/anim/ {len(steps)}コマ")


if __name__ == "__main__":
    main()
