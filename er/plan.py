"""俯瞰画像をGemini Robotics-ERに渡し、型紙の線に沿って切る軌道とつかむ点を得る。

実行: uv run python er/plan.py renders/overhead_star.png
出力: er/out/<stem>_er.json(ERの生出力と画素座標) / er/out/<stem>_overlay.png(重ね描き)
APIキーはmacOSキーチェーン(service=pai-final-cloth-gemini)から読み、表示しない。
"""

import argparse
import json
import math
import re
import subprocess
import time
from pathlib import Path

from google import genai
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
MODEL = "gemini-robotics-er-2-preview"
PROMPT = """
The image is a top-down view of a blue cloth on a table. A white chalk line
is drawn on the cloth as a closed outline. A robot will cut the cloth along it.

1. Plan the cutting trajectory: trace the white line as a closed loop, with
   40 to 60 points ordered along the line, starting anywhere on it.
2. Point to one grasp point inside the outline, where a gripper should pick up
   the cut-out piece.

Answer in this JSON format only:
{"trajectory": [{"step": 1, "point": [y, x]}, ...],
 "grasp": {"point": [y, x], "label": "grasp"}}
The points are in [y, x] format normalized to 0-1000.
"""


def api_key() -> str:
    return subprocess.run(
        ["security", "find-generic-password", "-s", "pai-final-cloth-gemini", "-a", "gemini-api-key", "-w"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def parse_json(text: str) -> dict:
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    return json.loads(m.group(1) if m else text)


def to_px(pt: list[float], w: int, h: int) -> tuple[float, float]:
    y, x = pt
    return x / 1000 * w, y / 1000 * h


def line_error_mm(er_px: list[tuple[float, float]], info: dict) -> dict:
    """ERの各点から正解の線(折れ線の頂点)までの最短距離。画素→mmは正解の3D点と画素の対応から求める。"""
    gt_px = info["pattern_px"]
    gw = info["pattern_world"]
    span_px = math.dist(gt_px[0], gt_px[len(gt_px) // 2])
    span_m = math.dist(gw[0][:2], gw[len(gw) // 2][:2])
    mm_per_px = span_m * 1000 / span_px
    d = [min(math.dist(p, g) for g in gt_px) * mm_per_px for p in er_px]
    return {"mean_mm": round(sum(d) / len(d), 2), "max_mm": round(max(d), 2), "n": len(d)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--thinking", default="high", choices=["low", "medium", "high"])
    args = ap.parse_args()

    client = genai.Client(api_key=api_key())
    uploaded = client.files.upload(file=str(args.image))
    t0 = time.time()
    res = client.interactions.create(
        model=MODEL,
        input=[
            {"type": "image", "uri": uploaded.uri, "mime_type": uploaded.mime_type},
            {"type": "text", "text": PROMPT},
        ],
        generation_config={"thinking_level": args.thinking},
    )
    elapsed = time.time() - t0
    out = parse_json(res.output_text)

    im = Image.open(args.image).convert("RGB")
    w, h = im.size
    traj_px = [to_px(s["point"], w, h) for s in sorted(out["trajectory"], key=lambda s: s["step"])]
    grasp_px = to_px(out["grasp"]["point"], w, h)

    d = ImageDraw.Draw(im)
    d.line(traj_px + traj_px[:1], fill=(255, 60, 0), width=3)
    for x, y in traj_px:
        d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(255, 60, 0))
    gx, gy = grasp_px
    d.ellipse([gx - 9, gy - 9, gx + 9, gy + 9], outline=(0, 255, 80), width=4)

    result = {"model": MODEL, "thinking": args.thinking, "elapsed_s": round(elapsed, 1),
              "raw": res.output_text, "trajectory_px": traj_px, "grasp_px": grasp_px}
    gt = args.image.with_suffix(".json")
    if gt.exists():
        result["error_vs_pattern"] = line_error_mm(traj_px, json.loads(gt.read_text()))

    out_dir = ROOT / "er" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem
    im.save(out_dir / f"{stem}_overlay.png")
    (out_dir / f"{stem}_er.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(f"{stem}: {len(traj_px)} points, {elapsed:.1f}s, error={result.get('error_vs_pattern')}")


if __name__ == "__main__":
    main()
