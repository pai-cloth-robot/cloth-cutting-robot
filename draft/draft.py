"""平面に開いた人台の前身頃を、製図の形(直線の脇・裾・肩、なめらかな襟ぐり・袖ぐり)に整え、
CAD風の型紙画面を描く。

実行: uv run python draft/draft.py
入力: renders/dressform/flat_boundary.json(blender/dressform.pyの段階2が書く。点・役割・3Dの高さ)
出力: patterns/dressform_front_drafted.json  整えた型紙(patterns/*.jsonと同じ形式+vertices・segments)
      renders/dressform/pattern_view.png    CAD風の型紙画面(1280x720)
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "renders" / "dressform" / "flat_boundary.json"
OUT_JSON = ROOT / "patterns" / "dressform_front_drafted.json"
OUT_PNG = ROOT / "renders" / "dressform" / "pattern_view.png"
N_POINTS = 160
FONT = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
FONT_BOLD = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"


# ---------- 製図の形に整える ----------

def smooth_curve(pts: np.ndarray, n: int, iters: int = 30) -> np.ndarray:
    """両端を固定してならし、弧長でn点に取り直したなめらかな曲線。"""
    p = pts.copy()
    for _ in range(iters):
        p[1:-1] = 0.5 * p[1:-1] + 0.25 * (p[:-2] + p[2:])
    return resample(p, n)


def resample(p: np.ndarray, n: int) -> np.ndarray:
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, s[-1], n)
    return np.stack([np.interp(t, s, p[:, k]) for k in (0, 1)], axis=1)


def line(a, b, n: int) -> np.ndarray:
    return np.linspace(a, b, n)


def length(p: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def draft(src: dict) -> tuple[np.ndarray, dict, dict, np.ndarray]:
    pts = np.array(src["points"])
    roles = src["roles"]
    # 前中心をx=0にそろえ、右半分(x>0)を製図して左へ鏡映する
    pts[:, 0] -= pts[:, 0].mean()
    role = np.array(roles)
    right = pts[:, 0] > 0
    get = {r: pts[(role == r) & right] for r in ("hem", "side", "armhole", "shoulder")}

    def mirror_x(a):
        a = np.array(a, dtype=float)
        a[..., 0] = np.abs(a[..., 0])
        return a

    hem_pts, side, arm, shoulder = (mirror_x(get[k]) for k in ("hem", "side", "armhole", "shoulder"))
    neck_all = pts[[i for i, r in enumerate(roles) if r == "design"]]
    # 頂点
    y_hem = float(np.mean(hem_pts[:, 1]))
    side_sorted = side[np.argsort(side[:, 1])]
    hem_corner = np.array([side_sorted[0, 0], y_hem])
    underarm = side_sorted[-1]
    arm_sorted = arm[np.argsort(arm[:, 1])]
    sh_sorted = shoulder[np.argsort(shoulder[:, 0])]
    shoulder_tip = sh_sorted[-1]
    side_neck = sh_sorted[0]
    cf_neck = np.array([0.0, float(neck_all[np.argmin(np.abs(neck_all[:, 0])), 1])])
    cf_hem = np.array([0.0, y_hem])
    # 脇: 脇の下・ウエスト(いちばん内側)・裾の角を結ぶ折れ線(くびれが5mm未満なら直線)
    inner = side_sorted[np.argmin(side_sorted[:, 0])]
    chord_x = np.interp(inner[1], [hem_corner[1], underarm[1]], [hem_corner[0], underarm[0]])
    waist = inner if chord_x - inner[0] > 0.005 else None
    # 襟ぐり: 右半分の点を首の付け根→前中心で並べ、前中心で水平になるよう鏡映した点も使ってならす
    neck_r = neck_all[neck_all[:, 0] >= 0]
    neck_r = neck_r[np.argsort(-neck_r[:, 0])]
    neck_pts = np.vstack([side_neck, neck_r, cf_neck])
    sym = np.vstack([neck_pts, neck_pts[-2::-1] * [-1, 1]])
    neck_curve = smooth_curve(sym, 81)[:41]
    neck_curve[0], neck_curve[-1] = side_neck, cf_neck
    # 袖ぐり: 肩先→脇の下
    arm_pts = np.vstack([shoulder_tip, arm_sorted[::-1], underarm])
    arm_curve = smooth_curve(arm_pts, 40)
    arm_curve[0], arm_curve[-1] = shoulder_tip, underarm

    # 裾の角 → (ウエスト) → 脇の下
    side_line = np.vstack([line(hem_corner, waist, 20), line(waist, underarm, 20)[1:]]) if waist is not None \
        else line(hem_corner, underarm, 30)
    segs_right = {
        "hem": line(cf_hem, hem_corner, 20),
        "side": side_line,
        "armhole": arm_curve[::-1],
        "shoulder": line(shoulder_tip, side_neck, 15),
        "neckline": neck_curve,
    }
    # 前中心の裾から反時計回り: 右半分(裾→脇→袖ぐり→肩→襟ぐり)→左半分(鏡映、逆順)
    right = np.vstack([segs_right["hem"], segs_right["side"][1:], segs_right["armhole"][1:],
                       segs_right["shoulder"][1:], segs_right["neckline"][1:]])
    left = right[::-1][1:-1] * [-1, 1]
    ring = np.vstack([right, left])
    vertices = {"shoulder_tip": shoulder_tip, "side_neck": side_neck, "cf_neck": cf_neck,
                "underarm": underarm, "hem_corner": hem_corner, "cf_hem": cf_hem}
    if waist is not None:
        vertices["waist_side"] = waist
    lengths = {
        "shoulder": length(segs_right["shoulder"]),
        "neckline_half": length(segs_right["neckline"]),
        "armhole": length(segs_right["armhole"]),
        "side": length(segs_right["side"]),
        "hem_half": length(segs_right["hem"]),
        "center_front": float(cf_neck[1] - cf_hem[1]),
    }
    return ring, vertices, lengths, pts


def poly_area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def centroid(p: np.ndarray) -> np.ndarray:
    x, y = p[:, 0], p[:, 1]
    c = x * np.roll(y, -1) - np.roll(x, -1) * y
    a = c.sum() / 2
    return np.array([((x + np.roll(x, -1)) * c).sum(), ((y + np.roll(y, -1)) * c).sum()]) / (6 * a)


def resample_closed(p: np.ndarray, n: int) -> np.ndarray:
    return resample(np.vstack([p, p[:1]]), n + 1)[:-1]


def dist_to_polyline(q: np.ndarray, poly: np.ndarray) -> np.ndarray:
    a, b = poly, np.roll(poly, -1, axis=0)
    ab = b - a
    t = np.clip(((q[:, None, :] - a) * ab).sum(-1) / (ab * ab).sum(-1), 0, 1)
    proj = a + t[..., None] * ab
    return np.linalg.norm(q[:, None, :] - proj, axis=-1).min(axis=1)


def self_intersects(p: np.ndarray) -> bool:
    n = len(p)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            p1, p2, p3, p4 = p[i], p[(i + 1) % n], p[j], p[(j + 1) % n]
            if cross(p3, p4, p1) * cross(p3, p4, p2) < 0 and cross(p1, p2, p3) * cross(p1, p2, p4) < 0:
                return True
    return False


# ---------- CAD風の型紙画面 ----------

def render_view(ring: np.ndarray, vertices: dict, lengths: dict, bust_y: float, bust_w: float) -> None:
    W, H = 1280, 720
    scale = 1900  # px/m(1cm=19px)。下端に動画のテロップが入る余白を残す
    cx, cy = W / 2 + 40, H / 2 - 8
    to_px = lambda p: (cx + p[0] * scale, cy - p[1] * scale)  # noqa: E731
    img = Image.new("RGBA", (W, H), (43, 43, 46, 255))
    grid = ImageDraw.Draw(img)
    for gx in range(0, W, 20):
        grid.line([(gx, 0), (gx, H)], fill=(52, 52, 56, 255))
    for gy in range(0, H, 20):
        grid.line([(0, gy), (W, gy)], fill=(52, 52, 56, 255))
    over = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    poly = [to_px(p) for p in ring]
    d.polygon(poly, fill=(240, 140, 50, 70))
    d.line(poly + poly[:1], fill=(245, 150, 60, 255), width=3)
    img = Image.alpha_composite(img, over)
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype(FONT, 17)
    fb = ImageFont.truetype(FONT_BOLD, 24)
    # 前中心(破線)
    top, bot = to_px(vertices["cf_neck"]), to_px(vertices["cf_hem"])
    for y in np.arange(top[1] - 30, bot[1] + 30, 14):
        d.line([(cx, y), (cx, y + 7)], fill=(200, 200, 210, 255), width=1)
    d.text((cx + 6, bot[1] + 12), "前中心", font=f, fill=(200, 200, 210, 255))
    # 胸囲線・裾の幅
    for y, w, label in ((bust_y, bust_w, "胸幅"), (vertices["cf_hem"][1], 2 * lengths["hem_half"], "裾幅")):
        a, b = to_px((-w / 2, y)), to_px((w / 2, y))
        d.line([a, b], fill=(120, 190, 255, 255), width=1)
        d.text((b[0] + 10, b[1] - 11), f"{label} {w * 100:.1f} cm", font=f, fill=(150, 205, 255, 255))
    # 頂点(左右)
    for name, p in vertices.items():
        for s in ((1, -1) if p[0] > 1e-6 else (1,)):
            x, y = to_px((p[0] * s, p[1]))
            d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(255, 255, 255, 255))
    # 区間の長さ(右側の辺の外側に)
    v = vertices
    mid = lambda a, b: (np.array(a) + np.array(b)) / 2  # noqa: E731
    labels = [
        (mid(v["shoulder_tip"], v["side_neck"]) + [0.004, 0.012], f"肩線 {lengths['shoulder'] * 100:.1f} cm"),
        (v["cf_neck"] + [0.012, -0.022], f"襟ぐり(片側) {lengths['neckline_half'] * 100:.1f} cm"),
        (mid(v["shoulder_tip"], v["underarm"]) + [0.018, 0.0], f"袖ぐり {lengths['armhole'] * 100:.1f} cm"),
        (mid(v["underarm"], v["hem_corner"]) + [0.016, 0.0], f"脇線 {lengths['side'] * 100:.1f} cm"),
        (mid(v["cf_hem"], v["hem_corner"]) + [0.0, -0.016], f"裾(片側) {lengths['hem_half'] * 100:.1f} cm"),
        (mid(v["cf_neck"], v["cf_hem"]) + [-0.072, 0.03], f"前丈 {lengths['center_front'] * 100:.1f} cm"),
    ]
    for p, text in labels:
        x, y = to_px(p)
        d.text((x, y), text, font=f, fill=(235, 235, 240, 255), anchor="lm")
    d.text((28, 22), "前身頃(1/2スケール)", font=fb, fill=(245, 245, 250, 255))
    d.text((28, 58), "人台の体に合わせて展開し、製図の形に整えた型紙  単位: cm", font=f, fill=(190, 190, 200, 255))
    img.convert("RGB").save(OUT_PNG)


def main() -> None:
    src = json.loads(SRC.read_text())
    ring, vertices, lengths, orig = draft(src)
    if poly_area(ring) < 0:
        ring = ring[::-1]
    # 整形で動いた量: 整形後の各点から整形前の輪郭までの最短距離
    moved = dist_to_polyline(ring, orig - [0, 0]) * 1000
    c = centroid(ring)
    ring_c = resample_closed(ring - c, N_POINTS)
    assert poly_area(ring_c) > 0 and not self_intersects(ring_c), "整えた輪郭が反時計回りでないか自己交差している"
    w = float(ring_c[:, 0].max() - ring_c[:, 0].min())
    h = float(ring_c[:, 1].max() - ring_c[:, 1].min())
    scale = min(1.0, 0.30 / max(w, h))
    verts_c = {k: [round(float(x), 5) for x in (np.array(p) - c) * scale] for k, p in vertices.items()}
    out = {
        "name": "dressform_front_drafted",
        "outline": [[round(float(x) * scale, 6), round(float(y) * scale, 6)] for x, y in ring_c],
        "closed": True,
        "scale": round(scale, 6),
        "size_m": [round(w * scale, 4), round(h * scale, 4)],
        "source": "dressform_front.json(flat_boundary.json)を製図の形に整えたもの",
        "vertices": verts_c,
        "segments_cm": {k: round(v * scale * 100, 2) for k, v in lengths.items()},
        "moved_mm": {"max": round(float(moved.max()), 2), "mean": round(float(moved.mean()), 2)},
    }
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    # 胸囲線: 3Dのバストの高さに当たる脇の点の高さ
    z3d = np.array(src["z3d"])
    side_idx = [i for i, r in enumerate(src["roles"]) if r == "side"]
    j = side_idx[int(np.argmin(np.abs(z3d[side_idx] - src["bust_z"])))]
    bust_y = orig[j, 1]
    xs = ring[np.abs(ring[:, 1] - bust_y) < 0.004][:, 0]
    bust_w = float(xs.max() - xs.min())
    render_view(ring - [0, c[1]], {k: np.array(v) - [0, c[1]] for k, v in vertices.items()}, lengths,
                bust_y - c[1], bust_w)
    print(f"整えた型紙: {len(out['outline'])}点  {h * 100:.1f}cm(縦) x {w * 100:.1f}cm(横)  scale={scale}")
    print(f"区間(cm): {out['segments_cm']}")
    print(f"整形で動いた量: 最大{out['moved_mm']['max']}mm 平均{out['moved_mm']['mean']}mm  胸幅{bust_w * 100:.1f}cm")
    print(f"wrote {OUT_JSON.relative_to(ROOT)}, {OUT_PNG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
