"""机・布・型紙の線・照明・俯瞰カメラを作り、俯瞰画像とカメラ情報を出力する。

実行: Blender -b --factory-startup -P blender/scene.py -- --pattern star [--marks]
      --patternはcircle/star/wave/bodice(平面の前身頃)か、patterns/<名前>.jsonの名前
出力: renders/overhead_<pattern>[_marks].png / .json
--marks: 服の型紙らしく、裁ち切り線(実線・切る線)の内側に出来上がり線(破線)と布目線(矢印)を描く。
         切るべきは外側の実線で、内側の破線は切ってはいけない紛らわしい線になる

座標系(MuJoCoと共通): 机の天板がz=0、Pandaの台座が原点、+xが前方。単位はm。
"""

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

ROOT = Path(__file__).resolve().parent.parent
CLOTH_CENTER = (0.5, 0.0)  # Pandaの前方50cm
CLOTH_SIZE = 0.5
CLOTH_Z = 0.001
LINE_Z = 0.0015
RES = (1024, 1024)
SEAM_ALLOWANCE = 0.01  # 縫い代1cm


def pattern_points(name: str, n: int = 120) -> list[tuple[float, float]]:
    """布中心からの相対座標(m)で、閉じた型紙の線を返す。"""
    if name == "bodice":
        return resample(bodice_outline(), 160)
    src = ROOT / "patterns" / f"{name}.json"
    if src.exists():
        return resample([tuple(p) for p in json.loads(src.read_text())["outline"]], 160)
    pts = []
    for i in range(n):
        t = 2 * math.pi * i / n
        match name:
            case "circle":
                r = 0.11
            case "star":
                return star_points(n)
            case "wave":
                r = 0.10 + 0.025 * math.sin(6 * t)
            case _:
                raise ValueError(name)
        pts.append((r * math.cos(t), r * math.sin(t)))
    return pts


def star_points(n: int) -> list[tuple[float, float]]:
    """辺が直線の五芒星。外側と内側の頂点を交互に結び、各辺を等分する。"""
    verts = []
    for j in range(10):
        r = 0.13 if j % 2 == 0 else 0.055
        a = math.pi / 2 + j * math.pi / 5
        verts.append((r * math.cos(a), r * math.sin(a)))
    per = n // 10
    pts = []
    for j in range(10):
        (x0, y0), (x1, y1) = verts[j], verts[(j + 1) % 10]
        pts += [(x0 + (x1 - x0) * k / per, y0 + (y1 - y0) * k / per) for k in range(per)]
    return pts


def quad(p0, c, p1, n=20) -> list[tuple[float, float]]:
    return [((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * c[0] + t * t * p1[0],
             (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * c[1] + t * t * p1[1]) for t in (i / n for i in range(n))]


def bodice_outline() -> list[tuple[float, float]]:
    """平面の前身頃(縮小版・縦約24cm)。x右・y上(首が上)。右半分を描いて前中心で反転する。"""
    cf_neck, hnp, shoulder = (0.0, 0.085), (0.04, 0.115), (0.095, 0.10)
    underarm, side_hem, cf_hem = (0.115, 0.03), (0.11, -0.12), (0.0, -0.125)
    right = (quad(cf_neck, (0.04, 0.085), hnp) + quad(hnp, hnp, shoulder, 6)
             + quad(shoulder, (0.08, 0.04), underarm) + quad(underarm, underarm, side_hem, 10)
             + quad(side_hem, side_hem, cf_hem, 8))
    left = [(-x, y) for x, y in reversed(right[1:])]  # cf_hemから前中心の首へ戻る側
    return right + left[:-1] if left[-1] == cf_neck else right + left


def ccw(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    area = sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]))
    return pts if area > 0 else pts[::-1]


def resample(pts: list[tuple[float, float]], n: int) -> list[tuple[float, float]]:
    """閉じた折れ線を、道のりで等間隔なn点に取り直す。"""
    pts = ccw(pts)
    ring = pts + pts[:1]
    seg = [math.dist(a, b) for a, b in zip(ring, ring[1:])]
    total, out, k, acc = sum(seg), [], 0, 0.0
    for i in range(n):
        s = total * i / n
        while acc + seg[k] < s:
            acc += seg[k]
            k += 1
        t = (s - acc) / seg[k] if seg[k] else 0.0
        (x0, y0), (x1, y1) = ring[k], ring[k + 1]
        out.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
    return out


def offset(pts: list[tuple[float, float]], d: float) -> list[tuple[float, float]]:
    """反時計回りの閉じた線を外側へdだけずらす(頂点ごとに前後の辺の法線を平均)。"""
    out = []
    for i, (x, y) in enumerate(pts):
        (xa, ya), (xb, yb) = pts[i - 1], pts[(i + 1) % len(pts)]
        nx, ny = yb - ya, -(xb - xa)
        L = math.hypot(nx, ny) or 1.0
        out.append((x + d * nx / L, y + d * ny / L))
    return out


def add_curve(name: str, splines: list[list[tuple[float, float]]], cyclic: bool, mat, depth=0.002):
    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = depth
    for pts in splines:
        sp = curve.splines.new("POLY")
        sp.points.add(len(pts) - 1)
        for p, (x, y) in zip(sp.points, pts):
            p.co = (x, y, LINE_Z, 1.0)
        sp.use_cyclic_u = cyclic
    obj = bpy.data.objects.new(name, curve)
    bpy.context.scene.collection.objects.link(obj)
    obj.data.materials.append(mat)
    return obj


def dashed(pts: list[tuple[float, float]], on: float = 0.008, off: float = 0.006) -> list[list[tuple[float, float]]]:
    ring = resample(pts, 400)
    step = math.dist(ring[0], ring[1])
    per_on, per = max(2, round(on / step)), round((on + off) / step)
    return [ring[i : i + per_on] for i in range(0, len(ring) - per_on, per)]


def fabric_material(color: tuple[float, float, float]) -> bpy.types.Material:
    """綿の平織りらしい布: 縦横の糸の凹凸(バンプ)と、わずかな色むら・光沢(シーン)。"""
    mat = bpy.data.materials.new("Fabric")
    nt = mat.node_tree
    n, link = nt.nodes, nt.links.new
    bsdf = n["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.95
    bsdf.inputs["Sheen Weight"].default_value = 0.4
    coord = n.new("ShaderNodeTexCoord")
    waves = []
    for axis in ("X", "Y"):
        w = n.new("ShaderNodeTexWave")
        w.wave_type, w.bands_direction = "BANDS", axis
        w.inputs["Scale"].default_value = 900.0  # 糸の間隔 約1mm(座標はオブジェクト基準・布は0.5m)
        w.inputs["Distortion"].default_value = 2.0
        link(coord.outputs["Object"], w.inputs["Vector"])
        waves.append(w)
    weave = n.new("ShaderNodeMath")
    weave.operation = "MULTIPLY"
    link(waves[0].outputs["Fac"], weave.inputs[0])
    link(waves[1].outputs["Fac"], weave.inputs[1])
    bump = n.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.25
    link(weave.outputs["Value"], bump.inputs["Height"])
    link(bump.outputs["Normal"], bsdf.inputs["Normal"])
    noise = n.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 6.0
    noise.inputs["Detail"].default_value = 6.0
    link(coord.outputs["Object"], noise.inputs["Vector"])
    ramp = n.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (*[c * 0.8 for c in color], 1.0)
    ramp.color_ramp.elements[1].color = (*color, 1.0)
    link(noise.outputs["Fac"], ramp.inputs["Fac"])
    link(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def cutting_mat_material() -> bpy.types.Material:
    """裁断用のカッティングマット: 深緑に1cm方眼の線。"""
    mat = bpy.data.materials.new("CuttingMat")
    nt = mat.node_tree
    n, link = nt.nodes, nt.links.new
    bsdf = n["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.7
    coord = n.new("ShaderNodeTexCoord")
    sep = n.new("ShaderNodeSeparateXYZ")
    link(coord.outputs["Object"], sep.inputs["Vector"])
    lines = []
    for axis in ("X", "Y"):
        mul = n.new("ShaderNodeMath"); mul.operation = "MULTIPLY"; mul.inputs[1].default_value = 100.0
        link(sep.outputs[axis], mul.inputs[0])
        fr = n.new("ShaderNodeMath"); fr.operation = "FRACT"
        link(mul.outputs["Value"], fr.inputs[0])
        lt = n.new("ShaderNodeMath"); lt.operation = "LESS_THAN"; lt.inputs[1].default_value = 0.035
        link(fr.outputs["Value"], lt.inputs[0])
        lines.append(lt)
    mx = n.new("ShaderNodeMath"); mx.operation = "MAXIMUM"
    link(lines[0].outputs["Value"], mx.inputs[0]); link(lines[1].outputs["Value"], mx.inputs[1])
    mix = n.new("ShaderNodeMix"); mix.data_type = "RGBA"
    mix.inputs["A"].default_value = (0.05, 0.16, 0.10, 1.0)
    mix.inputs["B"].default_value = (0.25, 0.36, 0.28, 1.0)
    link(mx.outputs["Value"], mix.inputs["Factor"])
    link(mix.outputs["Result"], bsdf.inputs["Base Color"])
    return mat


def material(name: str, color: tuple[float, float, float], roughness: float) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    return mat


def build_scene(pattern: str, marks: bool = False) -> dict:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    bpy.ops.mesh.primitive_plane_add(size=1.6, location=(0.4, 0.0, 0.0))
    table = bpy.context.object
    table.name = "Table"
    table.data.materials.append(material("Wood", (0.35, 0.22, 0.12), 0.6))
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.5, 0.0, 0.0004))
    mat_obj = bpy.context.object
    mat_obj.name = "CuttingMat"
    mat_obj.scale = (0.7, 0.9, 1.0)
    mat_obj.data.materials.append(cutting_mat_material())

    cx, cy = CLOTH_CENTER
    bpy.ops.mesh.primitive_plane_add(size=CLOTH_SIZE, location=(cx, cy, CLOTH_Z))
    cloth = bpy.context.object
    cloth.name = "Cloth"
    cloth.data.materials.append(fabric_material((0.16, 0.24, 0.42)))  # 紺の綿布
    thick = cloth.modifiers.new("Thickness", "SOLIDIFY")  # 布の厚み1mmで端に影を出す
    thick.thickness = 0.001

    pts = pattern_points(pattern)
    garment = pattern not in ("circle", "star", "wave")
    if marks:
        finished, pts = pts, offset(pts, SEAM_ALLOWANCE)  # 外側の裁ち切り線を切る
    # 服の型紙は首が画像の上(ロボットから遠い側=+x)に来るよう回す。星などは従来どおり
    to_world = (lambda x, y: (cx + y, cy - x)) if garment else (lambda x, y: (cx + x, cy + y))
    chalk = material("Chalk", (0.85, 0.85, 0.80), 1.0)
    add_curve("PatternLine", [[to_world(x, y) for x, y in pts]], True, chalk, 0.0013)
    if marks:
        add_curve("FinishedLine", [[to_world(x, y) for x, y in d] for d in dashed(finished)], False, chalk, 0.0008)
        ys = [y for _, y in finished]
        mid, half = (max(ys) + min(ys)) / 2, (max(ys) - min(ys)) * 0.3  # 型紙の高さの6割の長さ
        top, bot = mid + half, mid - half
        arrow = [[(0, bot), (0, top)], [(-0.008, top - 0.012), (0, top), (0.008, top - 0.012)],
                 [(-0.008, bot + 0.012), (0, bot), (0.008, bot + 0.012)]]
        add_curve("Grainline", [[to_world(x, y) for x, y in a] for a in arrow], False, chalk, 0.0008)
    world_pts = [to_world(x, y) for x, y in pts]

    bpy.ops.object.light_add(type="AREA", location=(cx, cy, 1.5))
    light = bpy.context.object
    light.data.energy = 18
    light.data.size = 1.0
    bpy.ops.object.light_add(type="AREA", location=(cx - 0.6, cy + 0.6, 1.0))
    fill = bpy.context.object  # 斜めからの弱い光で布の凹凸を見せる
    fill.data.energy = 5
    fill.data.size = 0.8
    fill.rotation_euler = (math.radians(45), 0.0, math.radians(-135))

    world = bpy.data.worlds.new("World")
    world.color = (0.04, 0.04, 0.04)
    scene.world = world

    bpy.ops.object.camera_add(location=(cx, cy, 1.0), rotation=(0.0, 0.0, -math.pi / 2))
    cam = bpy.context.object
    cam.data.lens = 50
    cam.data.sensor_width = 36
    scene.camera = cam

    scene.render.engine = "BLENDER_EEVEE"
    scene.view_settings.view_transform = "Standard"
    scene.render.resolution_x, scene.render.resolution_y = RES
    scene.render.resolution_percentage = 100

    bpy.context.view_layer.update()
    gt_px = []
    for wx, wy in world_pts:
        v = world_to_camera_view(scene, cam, Vector((wx, wy, LINE_Z)))
        gt_px.append([round(v.x * RES[0], 1), round((1 - v.y) * RES[1], 1)])

    return {
        "pattern": pattern,
        "marks": marks,
        "resolution": list(RES),
        "camera": {
            "location": list(cam.location),
            "matrix_world": [list(r) for r in cam.matrix_world],
            "lens_mm": cam.data.lens,
            "sensor_width_mm": cam.data.sensor_width,
        },
        "cloth": {"center": [cx, cy, CLOTH_Z], "size": CLOTH_SIZE},
        "table_z": 0.0,
        "pattern_world": [[wx, wy, LINE_Z] for wx, wy in world_pts],  # 切るべき線
        "pattern_px": gt_px,  # [x, y] 画素、左上原点
    }


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="star")
    ap.add_argument("--marks", action="store_true")
    args = ap.parse_args(argv)

    info = build_scene(args.pattern, args.marks)
    name = args.pattern + ("_marks" if args.marks else "")
    out = ROOT / "renders"
    out.mkdir(exist_ok=True)
    scene = bpy.context.scene
    scene.render.filepath = str(out / f"overhead_{name}.png")
    bpy.ops.render.render(write_still=True)
    (out / f"overhead_{name}.json").write_text(json.dumps(info, indent=1))
    print(f"wrote overhead_{name}.png/.json")


if __name__ == "__main__":
    main()
