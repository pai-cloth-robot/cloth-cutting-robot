"""机・布・型紙の線・照明・俯瞰カメラを作り、俯瞰画像とカメラ情報を出力する。

実行: Blender -b --factory-startup -P blender/scene.py -- --pattern star
出力: renders/overhead_<pattern>.png / renders/overhead_<pattern>.json

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
CLOTH_SIZE = 0.4
CLOTH_Z = 0.001
LINE_Z = 0.0015
RES = (1024, 1024)


def pattern_points(name: str, n: int = 120) -> list[tuple[float, float]]:
    """布中心からの相対座標(m)で、閉じた型紙の線を返す。"""
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


def material(name: str, color: tuple[float, float, float], roughness: float) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    return mat


def build_scene(pattern: str) -> dict:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    bpy.ops.mesh.primitive_plane_add(size=1.6, location=(0.4, 0.0, 0.0))
    table = bpy.context.object
    table.name = "Table"
    table.data.materials.append(material("Wood", (0.35, 0.22, 0.12), 0.6))

    cx, cy = CLOTH_CENTER
    bpy.ops.mesh.primitive_plane_add(size=CLOTH_SIZE, location=(cx, cy, CLOTH_Z))
    cloth = bpy.context.object
    cloth.name = "Cloth"
    cloth.data.materials.append(material("Denim", (0.08, 0.14, 0.32), 0.9))

    pts = pattern_points(pattern)
    curve = bpy.data.curves.new("PatternLine", "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = 0.002
    spline = curve.splines.new("POLY")
    spline.points.add(len(pts) - 1)
    for p, (x, y) in zip(spline.points, pts):
        p.co = (cx + x, cy + y, LINE_Z, 1.0)
    spline.use_cyclic_u = True
    line = bpy.data.objects.new("PatternLine", curve)
    scene.collection.objects.link(line)
    line.data.materials.append(material("Chalk", (0.92, 0.92, 0.88), 1.0))

    bpy.ops.object.light_add(type="AREA", location=(cx, cy, 1.5))
    light = bpy.context.object
    light.data.energy = 40
    light.data.size = 1.0

    world = bpy.data.worlds.new("World")
    world.color = (0.05, 0.05, 0.05)
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
    for x, y in pts:
        v = world_to_camera_view(scene, cam, Vector((cx + x, cy + y, LINE_Z)))
        gt_px.append([round(v.x * RES[0], 1), round((1 - v.y) * RES[1], 1)])

    return {
        "pattern": pattern,
        "resolution": list(RES),
        "camera": {
            "location": list(cam.location),
            "matrix_world": [list(r) for r in cam.matrix_world],
            "lens_mm": cam.data.lens,
            "sensor_width_mm": cam.data.sensor_width,
        },
        "cloth": {"center": [cx, cy, CLOTH_Z], "size": CLOTH_SIZE},
        "table_z": 0.0,
        "pattern_world": [[cx + x, cy + y, LINE_Z] for x, y in pts],
        "pattern_px": gt_px,  # [x, y] 画素、左上原点
    }


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="star", choices=["circle", "star", "wave"])
    args = ap.parse_args(argv)

    info = build_scene(args.pattern)
    out = ROOT / "renders"
    out.mkdir(exist_ok=True)
    scene = bpy.context.scene
    scene.render.filepath = str(out / f"overhead_{args.pattern}.png")
    bpy.ops.render.render(write_still=True)
    (out / f"overhead_{args.pattern}.json").write_text(json.dumps(info, indent=1))
    print(f"wrote overhead_{args.pattern}.png/.json")


if __name__ == "__main__":
    main()
