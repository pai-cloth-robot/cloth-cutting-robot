"""グレー単色・ローポリのクマ(箱モデリング風)を作り、ERの線から前身頃の型紙を作る。

投影・微調整・展開はbear2.pyの処理をそのまま使い、モデルと出力先だけ差し替える。
胴は縦長(袖ぐりの下端から裾まで約10cm)、脇線は正面から左右60°に置く。

段階1(ERの前): Blender -b --factory-startup -P blender/bear3.py -- --shoot
    renders/bear3/front.png / front.json   服なしのクマの正面画像とカメラ情報
    renders/bear3/look_front.png / look_oblique.png   形の確認用
段階2(ERの後): Blender -b --factory-startup -P blender/bear3.py -- [--er er/out/bear3_front_er.json] [--anim]
    patterns/bear3_front.json と renders/bear3/ の画像(bear2.pyと同じ構成)

クマは-y方向を向いて座る。床がz=0、単位m。
"""

import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bear2  # noqa: E402

ROOT = bear2.ROOT
OUT = ROOT / "renders" / "bear3"
GRAY = (0.6, 0.6, 0.6, 1.0)
LOWPOLY_RES = 0.016  # メタボールを面にするときの格子の粗さ(m)。大きいほど面が粗い

# 形(面の半径で指定。bear2.ball/capsuleが要素の半径に直す)
TORSO = [((0, 0, 0.06), 0.06), ((0, 0, 0.11), 0.057), ((0, 0, 0.16), 0.052)]
HEAD = ((0, 0, 0.27), 0.075)
MUZZLE = ((0, -0.066, 0.252), 0.032)
EARS = [((s * 0.055, 0.0, 0.335), 0.021) for s in (-1, 1)]
TORSO_SCALE = 1.25  # 胴の横幅と厚みの倍率(縦はそのまま)
ARMS = [((s * 0.045, 0, 0.175), (s * 0.126, 0, 0.175), 0.028) for s in (-1, 1)]
HANDS = [((s * 0.13, 0, 0.175), 0.031) for s in (-1, 1)]
LEGS = [((s * 0.045, -0.01, 0.032), (s * 0.09, -0.07, 0.032), 0.032) for s in (-1, 1)]
FEET = [((s * 0.093, -0.075, 0.034), 0.034) for s in (-1, 1)]


def lowpoly_mesh(name: str, legs: bool) -> bpy.types.Mesh:
    """1つにつながったローポリの体。"""
    mb = bpy.data.metaballs.new(name)
    mb.resolution = mb.render_resolution = LOWPOLY_RES
    for co, r in TORSO:  # 胴: 縦はそのままで横幅と厚みだけ太らせた楕円体
        e = mb.elements.new(type="ELLIPSOID")
        e.co, e.radius = co, r / bear2.MB_K
        e.size_x, e.size_y, e.size_z = TORSO_SCALE, TORSO_SCALE, 1.0
    for co, r in [HEAD, MUZZLE] + EARS + HANDS + (FEET if legs else []):
        bear2.ball(mb, co, r)
    for a, b, r in ARMS + (LEGS if legs else []):
        bear2.capsule(mb, a, b, r)
    tmp = bpy.data.objects.new(name + "MB", mb)
    bpy.context.scene.collection.objects.link(tmp)
    dg = bpy.context.evaluated_depsgraph_get()
    mesh = bpy.data.meshes.new_from_object(tmp.evaluated_get(dg))
    bpy.data.objects.remove(tmp)
    mesh.name = name
    for p in mesh.polygons:
        p.use_smooth = False  # 面の分割が見えるフラットシェーディング
    return mesh


def build_bear3() -> bpy.types.Object:
    body = bpy.data.objects.new("Bear3", lowpoly_mesh("Bear3", legs=True))
    bpy.context.scene.collection.objects.link(body)
    body.data.materials.append(bear2.bear.material("ClayGray", GRAY, 0.9))
    return body


def configure_bear2() -> None:
    """bear2の処理(投影・微調整・展開・描画)を、このモデルと出力先に合わせる。"""
    bear2.OUT = OUT
    bear2.PATTERN_NAME = "bear3_front"
    bear2.SIDE_DEG = 60
    bear2.HEAD_C = Vector(HEAD[0])
    bear2.TORSO_TOP = (TORSO[-1][0][2], TORSO[-1][1])
    bear2.HEAD_BALL = (HEAD[0][2], HEAD[1])
    bear2.ARM_Z = ARMS[1][0][2]
    bear2.ARM_R = ARMS[1][2]
    bear2.ARM_AXES = [((s * 0.085, 0, 0.175), (s * 0.126, 0, 0.175)) for s in (-1, 1)]
    bear2.HEM_MIN_Z = 0.05
    bear2.NECK_SQUASH_FROM = 0.10
    bear2.OBLIQUE = ((0.6, -0.95, 0.5), (0, 0, 0.18), 40)
    bear2.UNFOLD_CAM = ((0.2, -1.3, 0.34), (0.2, 0, 0.19), 50)


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--shoot", action="store_true", help="段階1: 正面画像とカメラ情報を書く")
    ap.add_argument("--er", type=Path, default=None, help="ERの出力(既定: er/out/bear3_front_er.json)")
    ap.add_argument("--anim", action="store_true")
    args = ap.parse_args(argv)

    configure_bear2()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    body = build_bear3()
    bear2.setup_world()
    cam = bear2.add_camera("Cam_Front", (0, -1.5, 0.185), (0, 0, 0.185), 95)
    OUT.mkdir(parents=True, exist_ok=True)
    info = bear2.camera_info(cam)
    print(f"体: 頂点{len(body.data.vertices)} 面{len(body.data.polygons)}")

    if args.shoot:
        bear2.render(cam, OUT / "front.png")
        (OUT / "front.json").write_text(json.dumps(info, indent=1))
        bear2.render(cam, OUT / "look_front.png", (1280, 720))
        bear2.render(bear2.add_camera("Cam_Oblique", *bear2.OBLIQUE), OUT / "look_oblique.png", (1280, 720))
        print("wrote renders/bear3/front.png, front.json, look_front.png, look_oblique.png")
        return

    er_path = args.er or ROOT / "er" / "out" / "bear3_front_er.json"
    if not er_path.is_absolute():
        er_path = ROOT / er_path
    poly_px = json.loads(er_path.read_text())["trajectory_px"]
    print(f"ERの線: {er_path.name}  {len(poly_px)}点")
    torso = lowpoly_mesh("Bear3Torso", legs=False)
    bvh = bear2.body_bvh(torso)
    bear2.measure_shape(bvh)
    bear2.stage2(cam, bvh, info, poly_px, er_path.name, args.anim)


if __name__ == "__main__":
    main()
