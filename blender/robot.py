"""kin/ik.pyが書き出したPandaの姿勢をBlenderのシーンに組み込む。

scene.pyのbuild_scene(pattern)の直後に、同じBlenderプロセスでadd_robot(poses_json)を呼ぶ。

動作確認: Blender -b --factory-startup -P blender/robot.py -- --pattern star [--frames 0 85 231]
出力: renders/robot_<pattern>_<frame>.png

座標系(MuJoCoと共通): 机の天板がz=0、Pandaの台座が原点、+xが前方。単位はm。
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parent.parent
MESH_DIR = ROOT / "kin" / "out" / "meshes"
BLADE_RADIUS = 0.02
BLADE_THICKNESS = 0.001


def load_mesh(name: str) -> bpy.types.Mesh:
    """コンパイル後のOBJを座標変換なしで読み、meshデータだけ残す。"""
    bpy.ops.wm.obj_import(
        filepath=str(MESH_DIR / f"{name}.obj"), forward_axis="Y", up_axis="Z"
    )
    obj = bpy.context.selected_objects[0]
    mesh = obj.data
    mesh.name = name
    mesh.materials.clear()
    mesh.materials.append(None)  # 空のスロットを1つ。中身はオブジェクト側で割り当てる
    bpy.data.objects.remove(obj)
    return mesh


def rgba_material(name: str, rgba: list[float]) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = rgba
    bsdf.inputs["Roughness"].default_value = 0.4
    return mat


def blade_mesh() -> bpy.types.Mesh:
    """ロータリーカッターの円盤。原点が刃先(円盤の最下点)、軸はローカルy、円盤はローカル-z側(上)へ伸びる。

    刃先siteの座標系はz軸が鉛直下向きなので、円盤の中心はローカル(0, 0, -半径)。
    """
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=64, radius=BLADE_RADIUS, depth=BLADE_THICKNESS,
        location=(0, 0, -BLADE_RADIUS), rotation=(math.pi / 2, 0, 0),
    )
    obj = bpy.context.object
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    mesh = obj.data
    mesh.name = "Blade"
    bpy.data.objects.remove(obj)
    return mesh


def keyframe_object(obj: bpy.types.Object, positions: list, quats: list) -> None:
    """全フレームの位置とクォータニオンをまとめてキーフレームにする(フレーム番号=配列の添字)。"""
    obj.rotation_mode = "QUATERNION"
    # 隣のフレームと符号をそろえ、補間で裏回りしないようにする
    fixed = [quats[0]]
    for q in quats[1:]:
        fixed.append([-c for c in q] if sum(a * b for a, b in zip(q, fixed[-1])) < 0 else q)
    action = bpy.data.actions.new(f"{obj.name}_Action")
    obj.animation_data_create().action = action
    n = len(positions)
    for path, values, dim in (("location", positions, 3), ("rotation_quaternion", fixed, 4)):
        for i in range(dim):
            fc = action.fcurve_ensure_for_datablock(obj, path, index=i)
            fc.keyframe_points.add(n)
            co = [v for f in range(n) for v in (f, values[f][i])]
            fc.keyframe_points.foreach_set("co", co)
            fc.keyframe_points.foreach_set("interpolation", [0] * n)  # CONSTANT: 毎フレーム打つので補間不要
            fc.update()


def add_robot(poses_json: Path) -> dict:
    """Pandaの各部品と刃を作り、全フレームのキーフレームを打つ。"""
    poses = json.loads(Path(poses_json).read_text())
    frames = poses["frames"]
    scene = bpy.context.scene
    coll = bpy.data.collections.new("Robot")
    scene.collection.children.link(coll)

    meshes: dict[str, bpy.types.Mesh] = {}
    materials: dict[tuple, bpy.types.Material] = {}
    for gname, info in poses["geom_info"].items():
        mname = info["mesh"]
        if mname not in meshes:
            meshes[mname] = load_mesh(mname)
        key = tuple(info["rgba"])
        if key not in materials:
            materials[key] = rgba_material(f"Panda_{len(materials)}", info["rgba"])
        obj = bpy.data.objects.new(gname, meshes[mname])
        coll.objects.link(obj)
        # 同じmeshを色違いで使うことがあるので、色はオブジェクト側に持たせる
        obj.material_slots[0].link = "OBJECT"
        obj.material_slots[0].material = materials[key]
        keyframe_object(
            obj,
            [f["geoms"][gname]["pos"] for f in frames],
            [f["geoms"][gname]["quat"] for f in frames],
        )

    blade = bpy.data.objects.new("Blade", blade_mesh())
    coll.objects.link(blade)
    blade.data.materials.append(rgba_material("BladeSteel", [0.8, 0.82, 0.85, 1.0]))
    steel = blade.data.materials[0].node_tree.nodes["Principled BSDF"]
    steel.inputs["Metallic"].default_value = 1.0
    steel.inputs["Roughness"].default_value = 0.2
    keyframe_object(blade, [f["blade_tip"] for f in frames], [f["blade_quat"] for f in frames])

    scene.render.fps = poses["fps"]
    scene.frame_start = 0
    scene.frame_end = len(frames) - 1
    return {
        "frame_count": len(frames),
        "fps": poses["fps"],
        "trace_frames": poses["trace_frames"],
        "blade": blade.name,
        "blade_tip": [f["blade_tip"] for f in frames],
        "collection": coll.name,
    }


def add_wide_camera() -> bpy.types.Object:
    """斜め上から机・布・ロボット全体を映す動画用カメラ「Cam_Wide」。"""
    target = Vector((0.4, 0.0, 0.28))
    loc = Vector((1.55, -1.15, 1.0))
    cam_data = bpy.data.cameras.new("Cam_Wide")
    cam_data.lens = 35
    cam_data.sensor_width = 36
    cam = bpy.data.objects.new("Cam_Wide", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    cam.location = loc
    cam.rotation_euler = (target - loc).to_track_quat("-Z", "Y").to_euler()
    return cam


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="star", choices=["circle", "star", "wave"])
    ap.add_argument("--frames", type=int, nargs="*", default=[0, 85, 231, 377, 523, 607])
    args = ap.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from scene import build_scene

    build_scene(args.pattern)
    t0 = time.perf_counter()
    info = add_robot(ROOT / "kin" / "out" / f"{args.pattern}_poses.json")
    print(f"add_robot: {time.perf_counter() - t0:.1f}s  frames={info['frame_count']}")

    scene = bpy.context.scene
    scene.camera = add_wide_camera()
    scene.render.resolution_x, scene.render.resolution_y = 640, 480
    out = ROOT / "renders"
    for f in args.frames:
        scene.frame_set(f)
        scene.render.filepath = str(out / f"robot_{args.pattern}_{f:04d}.png")
        t0 = time.perf_counter()
        bpy.ops.render.render(write_still=True)
        print(f"frame {f}: {time.perf_counter() - t0:.2f}s  blade_tip={info['blade_tip'][f]}")


if __name__ == "__main__":
    main()
