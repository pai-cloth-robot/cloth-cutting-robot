"""俯瞰画像をGemini Robotics-ERに渡し、型紙の線に沿って切る軌道とつかむ点を得る。

実行: uv run python er/plan.py renders/dressform/front.png --task design_form --name dressform_front
      uv run python er/plan.py renders/overhead_dressform_front_drafted_marks.png --task garment
--task line/designは初期の試作(星型・クマ)用
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


# 服の型紙用: 線の種類(実線・破線・矢印)は伝えず、どの線で切るかをERに判断させる
PROMPT_GARMENT = """
The image is a top-down view of a cloth on a table. A sewing pattern for a
garment piece is drawn on the cloth. A robot will cut out this
garment piece from the cloth, the way a tailor would.

1. Plan the cutting trajectory as a closed loop, with 50 to 80 points ordered
   along the line the robot should cut, starting anywhere on it.
2. Point to one grasp point inside the piece, where a gripper should pick up
   the cut-out piece.

Answer in this JSON format only:
{"trajectory": [{"step": 1, "point": [y, x]}, ...],
 "grasp": {"point": [y, x], "label": "grasp"},
 "reason": "<one sentence: which line you chose to cut and why>"}
The points are in [y, x] format normalized to 0-1000.
"""


# クマの正面画像に、体に合う服(タンクトップ)の前身頃の輪郭をERに設計させる
PROMPT_DESIGN = """
The image is a front view of a 3D model of a plush teddy bear.
We want to sew a sleeveless tank top that fits this bear.

1. Design the front panel of the tank top on this image: draw its outline as
   a closed loop of 60 to 90 points ordered along the outline. It must have a
   clear tank-top shape, not a rectangle or trapezoid:
   - Two narrow shoulder straps at the base of the neck. Each strap is about
     one sixth of the panel width.
   - A deep U-shaped neckline between the straps. Its lowest point is about one
     third of the way down from the strap tops to the hem.
   - Two deep, rounded armholes. Each one curves from the outer edge of a strap,
     under the arm, to the side seam, reaching about 40% of the way down.
   - Straight side seams down the torso, and a hem just above the legs.
2. Point to one grasp point inside the panel.

Answer in this JSON format only:
{"trajectory": [{"step": 1, "point": [y, x]}, ...],
 "grasp": {"point": [y, x], "label": "grasp"},
 "reason": "<one sentence: how you fitted the panel to the bear's body>"}
The points are in [y, x] format normalized to 0-1000.
"""


# 1/2スケールの人台(ドレスフォーム)の正面画像に、袖なしトップスの前身頃を設計させる
PROMPT_DESIGN_FORM = """
The image is a front view of a half-scale dress form (a tailor's mannequin
torso without head and arms), used for pattern making.
We want to sew a sleeveless top that fits this dress form.

1. Design the front bodice of the top on this image: draw its outline as a
   closed loop of 60 to 90 points ordered along the outline. Follow how a
   pattern maker would place the seams:
   - Shoulder seams along the top of each shoulder, from the base of the neck
     out to near the shoulder tip.
   - A neckline around the base of the neck, scooped lower at the front center.
   - Armholes that follow the edge of the arm openings down to the underarm.
   - Side seams that run exactly along the left and right outer edges
     (silhouette) of the torso, from the underarm down to the hem, following
     the curve of the waist. The bodice must cover the full width of the torso
     at every height; do not draw the side seams inside the silhouette.
   - A hem at about hip level, spanning the full width of the torso.
2. Point to one grasp point inside the bodice.

Answer in this JSON format only:
{"trajectory": [{"step": 1, "point": [y, x]}, ...],
 "grasp": {"point": [y, x], "label": "grasp"},
 "reason": "<one sentence: how you fitted the bodice to the dress form>"}
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
    ap.add_argument("--task", default="line", choices=["line", "garment", "design", "design_form"])
    ap.add_argument("--name", help="出力名(既定は画像のファイル名)")
    args = ap.parse_args()

    prompt = {"line": PROMPT, "garment": PROMPT_GARMENT, "design": PROMPT_DESIGN, "design_form": PROMPT_DESIGN_FORM}[args.task]
    client = genai.Client(api_key=api_key())
    uploaded = client.files.upload(file=str(args.image))
    t0 = time.time()
    res = client.interactions.create(
        model=MODEL,
        input=[
            {"type": "image", "uri": uploaded.uri, "mime_type": uploaded.mime_type},
            {"type": "text", "text": prompt},
        ],
        generation_config={"thinking_level": args.thinking},
    )
    elapsed = time.time() - t0
    out_dir = ROOT / "er" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.name or args.image.stem
    (out_dir / f"{stem}_raw.txt").write_text(res.output_text)  # 解析に失敗しても応答を残す
    out = parse_json(res.output_text)

    im = Image.open(args.image).convert("RGB")
    w, h = im.size
    # ERは指示どおりの{"step","point"}でなく、[y, x]の並びだけを返すことがある
    steps = out["trajectory"]
    if steps and isinstance(steps[0], dict):
        steps = [s["point"] for s in sorted(steps, key=lambda s: s.get("step", 0))]
    traj_px = [to_px(p, w, h) for p in steps]
    grasp = out["grasp"]
    grasp_px = to_px(grasp["point"] if isinstance(grasp, dict) else grasp, w, h)

    d = ImageDraw.Draw(im)
    d.line(traj_px + traj_px[:1], fill=(255, 60, 0), width=3)
    for x, y in traj_px:
        d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(255, 60, 0))
    gx, gy = grasp_px
    d.ellipse([gx - 9, gy - 9, gx + 9, gy + 9], outline=(0, 255, 80), width=4)

    result = {"model": MODEL, "task": args.task, "prompt": prompt.strip(), "reason": out.get("reason"), "thinking": args.thinking, "elapsed_s": round(elapsed, 1),
              "raw": res.output_text, "trajectory_px": traj_px, "grasp_px": grasp_px}
    gt = args.image.with_suffix(".json")
    if gt.exists() and "pattern_px" in (gt_info := json.loads(gt.read_text())):
        result["error_vs_pattern"] = line_error_mm(traj_px, gt_info)

    im.save(out_dir / f"{stem}_overlay.png")
    (out_dir / f"{stem}_er.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(f"{stem}: {len(traj_px)} points, {elapsed:.1f}s, error={result.get('error_vs_pattern')}")
    if out.get("reason"):
        print(f"reason: {out['reason']}")


if __name__ == "__main__":
    main()
