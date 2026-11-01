"""平面に開いた人台の前身頃をもとに、縫い合わせる辺(肩線・脇線・裾)を頂点どうしの直線にした型紙を作り、
CAD風の型紙画面を描く。袖ぐり・襟ぐりは平面に開いた曲線のまま。

実行: uv run python draft/draft.py
入力: renders/dressform/flat_boundary.json(blender/dressform.pyの段階2が書く。点・役割・3Dの高さ)
出力: patterns/dressform_front_drafted.json  型紙(patterns/*.jsonと同じ形式+vertices・segments)
      renders/dressform/pattern_view.png    CAD風の型紙画面(1280x720)。出来上がり線(破線)、縫い代1cmの
                                            裁ち切り線(実線、角はマイター)、布と同じガイドライン(前中心・
                                            プリンセスライン・バストライン・ウエストライン)、寸法の測り方
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
SEAM = 0.01  # 縫い代(m)
MAX_SIZE = 0.35  # これより大きい型紙は縮小する(m)。1/2スケールの実寸を保てるよう、縫い代込みで50cm角の布に収まる大きさ
WAIST_Z = 0.09  # 人台のウエストの高さ(blender/dressform.pyのSECTIONS)。flat_boundary.jsonにwaist_zがあればそれを使う
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
    del segs_right  # 曲線のならしはしない(平面に開いた形をもとにする)
    # 輪郭は平面に開いた形そのもの。画素の段差だけを軽くならす
    ring = pts.copy()
    for _ in range(2):
        ring = 0.5 * ring + 0.25 * (np.roll(ring, 1, axis=0) + np.roll(ring, -1, axis=0))
    snap_i = lambda q: int(np.argmin(np.linalg.norm(ring - q, axis=1)))  # noqa: E731
    vertices = {"shoulder_tip": shoulder_tip, "side_neck": side_neck, "cf_neck": cf_neck,
                "underarm": underarm, "hem_corner": hem_corner, "cf_hem": cf_hem}
    if waist is not None:
        vertices["waist_side"] = waist
    idx = {k: snap_i(v) for k, v in vertices.items()}
    vertices = {k: ring[i].copy() for k, i in idx.items()}
    # 他のパーツと縫い合わせる辺(肩線・脇線・裾)は、頂点どうしを結ぶ直線にする(左右とも)。
    # 袖ぐり・襟ぐりは平面に開いた曲線のまま
    order = ["underarm", "waist_side", "hem_corner"] if "waist_side" in idx else ["underarm", "hem_corner"]
    pairs = [("side_neck", "shoulder_tip")] + list(zip(order, order[1:]))
    n = len(ring)
    for sgn in (1, -1):
        side_idx = {k: (i if sgn > 0 else snap_i(vertices[k] * [-1, 1])) for k, i in idx.items()}
        for a, b in pairs:
            i, j = side_idx[a], side_idx[b]
            fwd = (j - i) % n
            if fwd > n // 2:  # 短い側を通る
                i, j, fwd = j, i, n - fwd
            pa, pb = ring[i].copy(), ring[j].copy()
            for k in range(1, fwd):
                ring[(i + k) % n] = pa + (pb - pa) * k / fwd
    # 裾: 左右の裾の角を結ぶ1本の直線(前中心の裾もこの線の上に置く)
    i, j = idx["hem_corner"], snap_i(vertices["hem_corner"] * [-1, 1])
    fwd = (j - i) % n
    if fwd > n // 2:
        i, j, fwd = j, i, n - fwd
    pa, pb = ring[i].copy(), ring[j].copy()
    for k in range(1, fwd):
        ring[(i + k) % n] = pa + (pb - pa) * k / fwd
    vertices["cf_hem"] = ring[idx["cf_hem"]].copy()

    def arc(a: str, b: str) -> float:
        """輪郭に沿った2つの頂点の間の長さ(短い側)。"""
        i, j = sorted((idx[a], idx[b]))
        seg = np.linalg.norm(np.diff(np.vstack([ring, ring[:1]]), axis=0), axis=1)
        inner = float(seg[i:j].sum())
        return min(inner, float(seg.sum()) - inner)

    lengths = {
        "shoulder": arc("shoulder_tip", "side_neck"),
        "neckline_half": arc("side_neck", "cf_neck"),
        "armhole": arc("shoulder_tip", "underarm"),
        "side": arc("underarm", "hem_corner"),
        "hem_half": arc("cf_hem", "hem_corner"),
        "center_front": float(vertices["cf_neck"][1] - vertices["cf_hem"][1]),
    }
    keep = np.linalg.norm(ring - np.roll(ring, 1, axis=0), axis=1) > 1e-5  # 重なった点(縫い代の角が乱れる)を除く
    return ring[keep], vertices, lengths, pts


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


def offset_miter(ring: np.ndarray, d: float) -> np.ndarray:
    """反時計回りの輪郭を外側へdずらす。角は丸めずに辺の延長が交わる点(マイター)にする。"""
    e = np.roll(ring, -1, axis=0) - ring
    n = np.stack([e[:, 1], -e[:, 0]], axis=1) / np.linalg.norm(e, axis=1, keepdims=True)  # 辺の外向き法線
    n_prev = np.roll(n, 1, axis=0)
    m = n_prev + n
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    out = ring + m * (d / (m * n).sum(axis=1, keepdims=True))
    return remove_loops(out)


def remove_loops(p: np.ndarray) -> np.ndarray:
    """くぼんだ曲線を外へずらしてできる小さなループを、交わった2辺の交点で切り取る。"""
    while True:
        n, best = len(p), None
        for i in range(n):
            for j in range(i + 2, min(i + n // 4, n)):
                a, b, c, e = p[i], p[i + 1], p[j], p[(j + 1) % n]
                den = (b[0] - a[0]) * (e[1] - c[1]) - (b[1] - a[1]) * (e[0] - c[0])
                if abs(den) < 1e-15:
                    continue
                t = ((c[0] - a[0]) * (e[1] - c[1]) - (c[1] - a[1]) * (e[0] - c[0])) / den
                u = ((c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])) / den
                if 0 < t < 1 and 0 < u < 1 and (best is None or j - i > best[1] - best[0]):
                    best = (i, j, a + t * (b - a))
            if best is not None:
                break
        if best is None:
            return p
        i, j, x = best
        p = np.vstack([p[: i + 1], x[None], p[j + 1 :]])


def dashed(d: ImageDraw.ImageDraw, pts: list, fill, width: int, dash: float = 10, gap: float = 7) -> None:
    """閉じた折れ線を弧長で破線にして描く。"""
    p = np.array(pts + pts[:1])
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    for a in np.arange(0, s[-1], dash + gap):
        t = np.linspace(a, min(a + dash, s[-1]), 6)
        q = np.stack([np.interp(t, s, p[:, k]) for k in (0, 1)], axis=1)
        d.line([tuple(x) for x in q], fill=fill, width=width)


def chain(segs: list) -> list[np.ndarray]:
    """三角形ごとの線分を端点でつないで折れ線にし、軽くならす(両端は固定)。"""
    key = lambda p: (round(p[0], 6), round(p[1], 6))  # noqa: E731
    nbr: dict = {}
    for a, b in segs:
        nbr.setdefault(key(a), []).append(key(b))
        nbr.setdefault(key(b), []).append(key(a))
    seen, lines = set(), []
    for start in sorted(nbr, key=lambda k: len(nbr[k])):
        if start in seen:
            continue
        line, cur = [start], start
        seen.add(start)
        while nxt := [n for n in nbr[cur] if n not in seen]:
            cur = nxt[0]
            seen.add(cur)
            line.append(cur)
        if len(line) > 2:
            lines.append(smooth_curve(np.array(line), len(line), iters=6))
    return lines


def inside(q: np.ndarray, poly: np.ndarray) -> bool:
    a, b = poly, np.roll(poly, -1, axis=0)
    c = (a[:, 1] > q[1]) != (b[:, 1] > q[1])
    x = a[:, 0] + (q[1] - a[:, 1]) * (b[:, 0] - a[:, 0]) / np.where(c, b[:, 1] - a[:, 1], 1)
    return bool(np.sum(c & (q[0] < x)) % 2)


# ---------- CAD風の型紙画面 ----------

def render_view(ring: np.ndarray, vertices: dict, lengths: dict, bust_y: float, bust_w: float,
                waist_y: float, guides: dict) -> None:
    W, H = 1280, 720
    scale = 1600  # px/m(1cm=16px)。下端に動画のテロップの帯が入る余白を残す
    cut = offset_miter(ring, SEAM)
    assert not self_intersects(cut), "縫い代の線が自己交差している"
    cx, cy = W / 2 + 60, 92 + cut[:, 1].max() * scale  # 裁ち切り線の上端をy=92pxに置く
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
    cut_px = [to_px(p) for p in cut]
    d.polygon(cut_px, fill=(240, 140, 50, 70))
    d.line(cut_px + cut_px[:1], fill=(245, 150, 60, 255), width=3, joint="curve")  # 裁ち切り線(実線)
    img = Image.alpha_composite(img, over)
    d = ImageDraw.Draw(img)
    dashed(d, poly, (250, 200, 150, 255), 2)  # 出来上がり線(破線)
    f = ImageFont.truetype(FONT, 17)
    fb = ImageFont.truetype(FONT_BOLD, 24)
    # ガイドライン(人台のテープと同じ赤): 展開の連番で布に描いた線と同じもの(flat_boundary.jsonのguides)。
    # 出来上がり線の内側の部分だけ描く
    red, red_t = (225, 60, 55, 255), (255, 120, 110, 255)
    for segs in guides.values():
        for line in chain(segs):
            keep = [inside(q, ring) for q in line]
            for a, b, ka, kb in zip(line[:-1], line[1:], keep[:-1], keep[1:], strict=True):
                if ka and kb:
                    d.line([to_px(a), to_px(b)], fill=red, width=2)
    bot = to_px(vertices["cf_hem"])
    d.text((cx + 6, bot[1] + 26), "前中心", font=f, fill=red_t)
    right = to_px(cut.max(axis=0))[0] + 14
    for y, label in ((bust_y, "バストライン"), (waist_y, "ウエストライン")):
        d.text((right, to_px((0, y))[1]), label, font=f, fill=red_t, anchor="lm")
    left = to_px(cut.min(axis=0))[0] - 14
    d.text((left, to_px((0, (bust_y + waist_y) / 2))[1]), "プリンセスライン(左右の縦の線)", font=f, fill=red_t,
           anchor="rm")
    # 胸幅: バストラインでの出来上がり線の幅
    d.text((right, to_px((0, bust_y))[1] + 22), f"胸幅 {bust_w * 100:.1f} cm", font=f,
           fill=(235, 235, 240, 255), anchor="lm")
    # 頂点(左右)
    for name, p in vertices.items():
        for s in ((1, -1) if p[0] > 1e-6 else (1,)):
            x, y = to_px((p[0] * s, p[1]))
            d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(255, 255, 255, 255))
    # 区間の長さ(右側の辺の外側に)
    v = vertices
    mid = lambda a, b: (np.array(a) + np.array(b)) / 2  # noqa: E731
    labels = [
        (mid(v["shoulder_tip"], v["side_neck"]) + [0.014, 0.012], f"肩線 {lengths['shoulder'] * 100:.1f} cm"),
        (v["cf_neck"] + [0.012, -0.022], f"襟ぐり(片側) {lengths['neckline_half'] * 100:.1f} cm"),
        (mid(v["shoulder_tip"], v["underarm"]) + [0.038, -0.012], f"袖ぐり {lengths['armhole'] * 100:.1f} cm"),
        (mid(v["underarm"], v["hem_corner"]) + [0.016, 0.0], f"脇線 {lengths['side'] * 100:.1f} cm"),
        (mid(v["cf_hem"], v["hem_corner"]) + [0.0, -0.02], f"裾(片側) {lengths['hem_half'] * 100:.1f} cm"),
        (mid(v["cf_neck"], v["cf_hem"]) + [-0.072, 0.03], f"前丈 {lengths['center_front'] * 100:.1f} cm"),
    ]
    for p, text in labels:
        x, y = to_px(p)
        d.text((x, y), text, font=f, fill=(235, 235, 240, 255), anchor="lm")
    d.text((28, 22), "前身頃(1/2スケール)", font=fb, fill=(245, 245, 250, 255))
    d.text((28, 58), "平面に展開し、縫い合わせる辺(肩・脇・裾)を直線にした型紙  単位: cm", font=f, fill=(190, 190, 200, 255))
    # 凡例
    lx, ly = 30, 100
    d.line([(lx, ly), (lx + 40, ly)], fill=(245, 150, 60, 255), width=3)
    d.text((lx + 52, ly), "裁ち切り線(縫い代 1 cm)", font=f, fill=(235, 235, 240, 255), anchor="lm")
    dashed(d, [(lx, ly + 28), (lx + 40, ly + 28)][:1] + [(lx + 40, ly + 28)], (250, 200, 150, 255), 2)
    d.text((lx + 52, ly + 28), "出来上がり線", font=f, fill=(235, 235, 240, 255), anchor="lm")
    d.line([(lx, ly + 56), (lx + 40, ly + 56)], fill=red, width=2)
    d.text((lx + 52, ly + 56), "ガイドライン(人台と同じ位置)", font=f, fill=(235, 235, 240, 255), anchor="lm")
    # 寸法の測り方
    fs = ImageFont.truetype(FONT, 15)
    d.text((lx, ly + 100), "寸法の測り方(出来上がり線の上の長さ)", font=f, fill=(245, 245, 250, 255), anchor="lm")
    for k, (name, how) in enumerate((
        ("肩線", "首の付け根〜肩先"),
        ("襟ぐり(片側)", "首の付け根〜前中心のカーブ"),
        ("袖ぐり", "肩先〜脇の下のカーブ"),
        ("脇線", "脇の下〜裾"),
        ("裾(片側)", "前中心〜脇の裾"),
        ("前丈", "前中心の首〜裾"),
        ("胸幅", "バストラインでの左右の幅"),
    )):
        y = ly + 130 + k * 25
        d.text((lx + 6, y), name, font=fs, fill=(235, 235, 240, 255), anchor="lm")
        d.text((lx + 110, y), how, font=fs, fill=(190, 190, 200, 255), anchor="lm")
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
    scale = min(1.0, MAX_SIZE / max(w, h))
    verts_c = {k: [round(float(x), 5) for x in (np.array(p) - c) * scale] for k, p in vertices.items()}
    out = {
        "name": "dressform_front_drafted",
        "outline": [[round(float(x) * scale, 6), round(float(y) * scale, 6)] for x, y in ring_c],
        "closed": True,
        "scale": round(scale, 6),
        "size_m": [round(w * scale, 4), round(h * scale, 4)],
        "source": "flat_boundary.json(平面に開いた形)の肩線・脇線・裾を直線にしたもの",
        "vertices": verts_c,
        "segments_cm": {k: round(v * scale * 100, 2) for k, v in lengths.items()},
        "moved_mm": {"max": round(float(moved.max()), 2), "mean": round(float(moved.mean()), 2)},
    }
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    # バストライン・ウエストライン: 3Dでその高さに当たる脇の点の、平面での高さ
    z3d = np.array(src["z3d"])
    side_idx = [i for i, r in enumerate(src["roles"]) if r == "side"]

    def level_y(z: float) -> float:
        return float(orig[side_idx[int(np.argmin(np.abs(z3d[side_idx] - z)))], 1])

    bust_y = level_y(src["bust_z"])
    waist_y = level_y(src.get("waist_z", WAIST_Z))
    xs = ring[np.abs(ring[:, 1] - bust_y) < 0.004][:, 0]
    bust_w = float(xs.max() - xs.min())
    # 布のガイドライン: draft()と同じく前中心をx=0にそろえる
    shift = np.array([np.array(src["points"])[:, 0].mean(), c[1]])
    guides = {k: [[list(np.array(a) - shift), list(np.array(b) - shift)] for a, b in segs]
              for k, segs in src["guides"].items()}
    render_view(ring - [0, c[1]], {k: np.array(v) - [0, c[1]] for k, v in vertices.items()}, lengths,
                bust_y - c[1], bust_w, waist_y - c[1], guides)
    print(f"整えた型紙: {len(out['outline'])}点  {h * 100:.1f}cm(縦) x {w * 100:.1f}cm(横)  scale={scale}")
    print(f"区間(cm): {out['segments_cm']}")
    print(f"整形で動いた量: 最大{out['moved_mm']['max']}mm 平均{out['moved_mm']['mean']}mm  胸幅{bust_w * 100:.1f}cm")
    print(f"wrote {OUT_JSON.relative_to(ROOT)}, {OUT_PNG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
