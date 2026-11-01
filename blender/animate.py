"""シーン・Panda・切れ目の演出をまとめ、斜め上のカメラ(Cam_Wide)でレンダリングする。

実行: Blender -b --factory-startup -P blender/animate.py -- --pattern star --poses kin/out/star_poses.json [--frames 0 300 607 | --all]
出力: renders/anim_<name>/<frame>.png
切れ目は型紙の正解でなく、刃先が実際に通った軌跡(=ERの点列をIKでなぞった結果)から作る。
"""

import argparse
import math
import sys
from pathlib import Path

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


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="star")
    ap.add_argument("--marks", action="store_true")
    ap.add_argument("--poses", type=Path, required=True)
    ap.add_argument("--frames", type=int, nargs="*", default=[0, 200, 400, 607])
    ap.add_argument("--all", action="store_true", help="全フレームを書き出す")
    ap.add_argument("--res", type=int, nargs=2, default=[1280, 720])
    args = ap.parse_args(argv)

    build_scene(args.pattern, args.marks)
    info = add_robot(ROOT / args.poses)
    add_cut(info["blade_tip"], info["trace_frames"])

    scene = bpy.context.scene
    scene.camera = add_wide_camera()
    scene.render.resolution_x, scene.render.resolution_y = args.res
    out = ROOT / "renders" / f"anim_{args.poses.stem.removesuffix('_poses')}"
    out.mkdir(parents=True, exist_ok=True)
    frames = range(info["frame_count"]) if args.all else args.frames
    for f in frames:
        scene.frame_set(f)
        scene.render.filepath = str(out / f"{f:04d}.png")
        bpy.ops.render.render(write_still=True)
    print(f"wrote {len(frames)} frames to {out.name}/")


if __name__ == "__main__":
    main()
