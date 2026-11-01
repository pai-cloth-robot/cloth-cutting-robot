"""シーン・Panda・切れ目の演出をまとめ、斜め上のカメラ(Cam_Wide)でレンダリングする。

実行: Blender -b --factory-startup -P blender/animate.py -- --pattern dressform_front_drafted --marks --poses kin/out/er_dressform_front_drafted_marks_poses.json [--frames 0 300 607 | --all]
出力: renders/anim_<name>/<frame>.png
切れ目は型紙の正解でなく、刃先が実際に通った軌跡(=ERの点列をIKでなぞった結果)から作る。
"""

import argparse
import json
import math
import sys
from pathlib import Path

import bmesh
import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from robot import add_robot, add_wide_camera  # noqa: E402
from scene import build_scene, material  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CUT_Z = 0.0025  # 型紙の線(0.0015)より少し上に描き、切れ目が線を覆うようにする


def add_cut(blade_tip: list[list[float]], trace: tuple[int, int]) -> bpy.types.Object:
    """刃先の軌跡を暗い溝のカーブにし、刃の通過に合わせて伸ばす。"""
    t0, t1 = trace
    pts = blade_tip[t0 : t1 + 1]
    curve = bpy.data.curves.new("Cut", "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = 0.0018
    curve.bevel_factor_mapping_end = "SPLINE"
    spline = curve.splines.new("POLY")
    spline.points.add(len(pts) - 1)
    for p, (x, y, _) in zip(spline.points, pts):
        p.co = (x, y, CUT_Z, 1.0)
    cut = bpy.data.objects.new("Cut", curve)
    bpy.context.scene.collection.objects.link(cut)
    cut.data.materials.append(material("CutGap", (0.01, 0.01, 0.015), 1.0))

    # 2点だけのキーフレームはBlender既定の補間で中盤が速くなり、切れ目が刃を追い越す。
    # 毎フレーム、刃先がそこまでに進んだ道のりの割合を打つ。
    seg = [math.dist(a[:2], b[:2]) for a, b in zip(pts, pts[1:])]
    total = sum(seg)
    done = 0.0
    for i, f in enumerate(range(t0, t1 + 1)):
        if i > 0:
            done += seg[i - 1]
        curve.bevel_factor_end = done / total
        curve.keyframe_insert("bevel_factor_end", frame=f)
    return cut


def align_blade(frames: list[dict], trace: tuple[int, int]) -> None:
    """刃(円盤)を、刃先を中心に鉛直軸まわりに回し、常に進行方向へ向ける(ドラッグナイフ方式)。
    手首(joint7)は約332°しか回らず一周を追えないので、手首の向きは固定のまま刃だけを回す。
    円盤の軸はローカルy、刃の向き(切り進む方向)はローカルx。円盤は前後対称なので180°ごとに同じ向き。"""
    from mathutils import Quaternion, Vector
    t0, t1 = trace
    tips = [Vector(f["blade_tip"]) for f in frames]
    blade = bpy.data.objects["Blade"]
    blade.rotation_mode = "QUATERNION"
    prev = None
    targets = {}
    for f in range(t0, t1 + 1):
        a, b = tips[max(f - 3, t0)], tips[min(f + 3, t1)]
        d = b - a
        if d.xy.length < 1e-6:
            continue
        targets[f] = math.atan2(d.y, d.x)
    for f in range(len(frames)):
        key = min(max(f, min(targets)), max(targets))  # なぞる前後は、始まり・終わりの向きのまま
        key = min(targets, key=lambda k: abs(k - key))
        q = Quaternion(frames[f]["blade_quat"])
        x_world = q @ Vector((1.0, 0.0, 0.0))
        delta = targets[key] - math.atan2(x_world.y, x_world.x)
        delta = (delta + math.pi / 2) % math.pi - math.pi / 2  # 前後対称なので±90°に収める
        if prev is not None:  # 前のフレームと180°ずれた解に飛ばないよう、近い方を選ぶ
            delta += math.pi * round((prev - delta) / math.pi)
        prev = delta
        blade.rotation_quaternion = Quaternion((0.0, 0.0, 1.0), delta) @ q
        blade.keyframe_insert("rotation_quaternion", frame=f)


def add_piece(blade_tip: list[list[float]], trace: tuple[int, int], grasp: tuple[int, int]) -> bpy.types.Object:
    """刃先の軌跡で囲まれた布片を作り、切り終えたら布に穴を開けて置き換え、つかんだら手先と一緒に動かす。
    刃の向き(ヨー)は固定なので、布片は刃先の移動量だけ平行移動させればよい。"""
    t0, t1 = trace
    g0, g_end = grasp
    ring = [(x, y) for x, y, _ in blade_tip[t0 : t1 + 1]]
    if math.dist(ring[0], ring[-1]) < 1e-4:
        ring = ring[:-1]
    cloth = bpy.data.objects["Cloth"]
    z = cloth.location.z

    def ngon(name: str, zs: list[float]) -> bpy.types.Object:
        bm = bmesh.new()
        layers = [[bm.verts.new((x, y, zz)) for x, y in ring] for zz in zs]
        for vs in layers:
            bm.faces.new(vs)
        if len(layers) == 2:  # 角柱(ブーリアン用)の側面
            a, b = layers
            for i in range(len(ring)):
                j = (i + 1) % len(ring)
                bm.faces.new((a[i], a[j], b[j], b[i]))
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        mesh = bpy.data.meshes.new(name)
        bm.to_mesh(mesh)
        bm.free()
        obj = bpy.data.objects.new(name, mesh)
        bpy.context.scene.collection.objects.link(obj)
        return obj

    cutter = ngon("PieceCutter", [z - 0.01, z + 0.01])
    cutter.hide_render = True
    cutter.hide_viewport = True
    hole = cloth.modifiers.new("Hole", "BOOLEAN")
    hole.operation, hole.solver, hole.object = "DIFFERENCE", "EXACT", cutter

    # 布片: 細かい格子のうち、切った線の内側の面だけを残し、縁の頂点を線の上へ寄せる
    # (ブーリアンでくり抜くと細い三角形が残り、陰影が崩れる)
    from mathutils.geometry import intersect_point_line
    xs, ys = [x for x, _ in ring], [y for _, y in ring]
    step = 0.004
    nx, ny = int((max(xs) - min(xs)) / step) + 3, int((max(ys) - min(ys)) / step) + 3
    x0, y0 = min(xs) - step, min(ys) - step

    def inside(px: float, py: float) -> bool:
        hit = False
        for (ax, ay), (bx, by) in zip(ring, ring[1:] + ring[:1]):
            if (ay > py) != (by > py) and px < ax + (py - ay) * (bx - ax) / (by - ay):
                hit = not hit
        return hit

    bm = bmesh.new()
    verts = [[bm.verts.new((x0 + i * step, y0 + j * step, z + 0.0005)) for j in range(ny)] for i in range(nx)]
    for i in range(nx - 1):
        for j in range(ny - 1):
            if inside(x0 + (i + 0.5) * step, y0 + (j + 0.5) * step):
                bm.faces.new((verts[i][j], verts[i + 1][j], verts[i + 1][j + 1], verts[i][j + 1]))
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
    edges = list(zip(ring, ring[1:] + ring[:1]))
    for v in bm.verts:
        if v.is_boundary:
            best = None
            for a, b in edges:
                q, t = intersect_point_line(v.co.xy, a, b)
                t = min(max(t, 0.0), 1.0)
                q = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
                d = math.dist(q, v.co.xy)
                if best is None or d < best[0]:
                    best = (d, q)
            v.co.x, v.co.y = best[1]
    mesh = bpy.data.meshes.new("Piece")
    bm.to_mesh(mesh)
    bm.free()
    piece = bpy.data.objects.new("Piece", mesh)
    bpy.context.scene.collection.objects.link(piece)
    piece.data.materials.append(cloth.data.materials[0])

    # つかむ点の近く(半径3cm、指の幅ほど)を手先に固定する
    gx, gy, _ = blade_tip[g0]
    pin = piece.vertex_groups.new(name="pin")
    pin.add([v.index for v in mesh.vertices if math.dist(v.co[:2], (gx, gy)) < 0.03], 1.0, "REPLACE")
    hook_target = bpy.data.objects.new("GraspHook", None)
    bpy.context.scene.collection.objects.link(hook_target)
    hook_target.location = (gx, gy, z)
    hook = piece.modifiers.new("Grasp", "HOOK")
    hook.object, hook.vertex_group = hook_target, "pin"
    hook.center = (gx, gy, z)
    bpy.context.view_layer.update()
    hook.matrix_inverse = hook_target.matrix_world.inverted()
    base = blade_tip[g0]
    for f in range(g0, g_end + 1):
        hook_target.location = [gx, gy, z] if f == g0 else [(gx, gy, z)[k] + blade_tip[f][k] - base[k] for k in range(3)]
        hook_target.keyframe_insert("location", frame=f)

    sim = piece.modifiers.new("Cloth", "CLOTH")
    cs = sim.settings
    cs.quality = 10
    cs.mass = 0.2
    cs.tension_stiffness = cs.compression_stiffness = 40
    cs.shear_stiffness = 20
    cs.bending_stiffness = 150  # 綿のシーチングのハリ(垂れても広がりを保つ)
    cs.air_damping = 1.0
    cs.vertex_group_mass = "pin"
    sim.collision_settings.distance_min = 0.0015
    sim.collision_settings.use_self_collision = False
    sim.point_cache.frame_start, sim.point_cache.frame_end = g0, g_end
    for name in ("CuttingMat", "Table"):  # 机に触れた部分は引きずられる
        col = bpy.data.objects[name].modifiers.new("Collision", "COLLISION")
        bpy.data.objects[name].collision.thickness_outer = 0.001
    piece.modifiers.new("Thickness", "SOLIDIFY").thickness = 0.001
    piece.modifiers.new("Smooth", "SUBSURF").levels = 1

    # 切り終えるまでは布片を隠し、穴も開けない(刃より先に切れ目が見えないように)
    for f, cut in ((t0, False), (t1, True)):
        piece.hide_render = not cut
        piece.keyframe_insert("hide_render", frame=f)
        hole.show_render = cut
        hole.keyframe_insert("show_render", frame=f)

    # 刃先は指先より2cm下に出ているので、つかむときは刃を引っ込めた見せ方にする(刃を上げ終えたら隠す)
    blade = bpy.data.objects["Blade"]
    for f, shown in ((t1, True), (t1 + 24, False)):
        blade.hide_render = not shown
        blade.keyframe_insert("hide_render", frame=f)

    # 布片の上の型紙の線(出来上がり線・布目線)は、手先が布片に触れた時点で消す。
    # 外側の裁ち切り線は切れ目に重なっているので、切り終えたら隠す
    for name in ("FinishedLine", "Grainline"):
        if name in bpy.data.objects:
            line = bpy.data.objects[name]
            for f, shown in ((t0, True), (g0, False)):
                line.hide_render = not shown
                line.keyframe_insert("hide_render", frame=f)
    outer = bpy.data.objects.get("PatternLine")
    if outer:
        for f, shown in ((t0, True), (t1, False)):
            outer.hide_render = not shown
            outer.keyframe_insert("hide_render", frame=f)
    return piece


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="dressform_front_drafted")
    ap.add_argument("--marks", action="store_true")
    ap.add_argument("--poses", type=Path, required=True)
    ap.add_argument("--frames", type=int, nargs="*", default=[0, 200, 400, 607])
    ap.add_argument("--all", action="store_true", help="全フレームを書き出す")
    ap.add_argument("--start", type=int, default=0, help="--allのとき、このフレームから書き出す(布片の場面だけ描き直す用)")
    ap.add_argument("--res", type=int, nargs=2, default=[1280, 720])
    args = ap.parse_args(argv)

    build_scene(args.pattern, args.marks)
    info = add_robot(ROOT / args.poses)
    add_cut(info["blade_tip"], info["trace_frames"])
    poses = json.loads((ROOT / args.poses).read_text())
    align_blade(poses["frames"], info["trace_frames"])
    if "grasp_frames" in poses:  # kin/ik.py --graspの出力: 切り抜いた布片をつかんで持ち上げる
        add_piece(info["blade_tip"], info["trace_frames"], poses["grasp_frames"])

    scene = bpy.context.scene
    scene.camera = add_wide_camera()
    scene.render.resolution_x, scene.render.resolution_y = args.res
    out = ROOT / "renders" / f"anim_{args.poses.stem.removesuffix('_poses')}"
    out.mkdir(parents=True, exist_ok=True)
    frames = range(args.start, info["frame_count"]) if args.all else args.frames
    for f in frames:
        scene.frame_set(f)
        scene.render.filepath = str(out / f"{f:04d}.png")
        bpy.ops.render.render(write_still=True)
    print(f"wrote {len(frames)} frames to {out.name}/")


if __name__ == "__main__":
    main()
