"""ERが返した画素座標を、カメラからの光線と布の面(z=一定)の交点で3D点に直す。

実行: uv run python er/to3d.py er/out/overhead_star_er.json renders/overhead_star.json
出力: er/out/<stem>_points3d.json  {"points": [[x,y,z], ...], "closed": true, "grasp": [x,y,z]}
      (kin/ik.pyの入力形式)
"""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BLADE_Z = 0.0015  # 布の上面(scene.pyのLINE_Zと同じ)


def pixel_to_world(px: list[list[float]], cam: dict, res: list[int], z: float) -> np.ndarray:
    """Blenderのカメラ(-zが視線、+yが画像の上)の透視投影を逆にたどる。"""
    w, h = res
    m = np.array(cam["matrix_world"])
    origin, rot = m[:3, 3], m[:3, :3]
    f_px = cam["lens_mm"] / cam["sensor_width_mm"] * w  # センサー幅をwに合わせる(Blenderの既定sensor_fit=AUTO・正方形画像)
    pts = []
    for u, v in px:
        d_cam = np.array([(u - w / 2) / f_px, -(v - h / 2) / f_px, -1.0])
        d = rot @ d_cam
        t = (z - origin[2]) / d[2]
        pts.append(origin + t * d)
    return np.array(pts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("er_json", type=Path)
    ap.add_argument("scene_json", type=Path)
    ap.add_argument("--px-key", default="trajectory_px", help="検証用: pattern_pxを渡すと正解の3D点が戻るか確かめられる")
    args = ap.parse_args()

    er = json.loads(args.er_json.read_text())
    sc = json.loads(args.scene_json.read_text())
    pts = pixel_to_world(er[args.px_key], sc["camera"], sc["resolution"], BLADE_Z)
    out = {"points": pts.round(5).tolist(), "closed": True}
    if "grasp_px" in er:
        out["grasp"] = pixel_to_world([er["grasp_px"]], sc["camera"], sc["resolution"], BLADE_Z)[0].round(5).tolist()

    if "pattern_world" in er:  # 検証モード: scene_jsonを両方に渡したとき
        err = np.linalg.norm(pts - np.array(er["pattern_world"]), axis=1) * 1000
        print(f"pattern_px → 3D の誤差: 最大{err.max():.3f}mm 平均{err.mean():.3f}mm")
        return

    dst = ROOT / "er" / "out" / f"{args.er_json.stem.removesuffix('_er')}_points3d.json"
    dst.write_text(json.dumps(out, indent=1))
    print(f"wrote {dst.name}: {len(pts)} points")


if __name__ == "__main__":
    main()
