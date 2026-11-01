"""最終動画をBlenderのシーケンサー(VSE)で組み立て、MP4(H.264・1280x720・24fps)にする。

実行: Blender -b --factory-startup -P blender/compose.py -- [--robot renders/anim_er_star]
      [--overlay er/out/overhead_star_overlay.png] [--out renders/video/final.mp4]

区間: タイトル → 仕組み → クマの展開 → 布の俯瞰 → ERの推論 → ロボットの切断 → エンディング
秒数は下のSECONDSで変える。課題の規定で全体を1〜3分に収める(外れたら警告を出す)。
"""

import argparse
import json
import sys
from pathlib import Path

import bpy

ROOT = Path(__file__).resolve().parent.parent
FPS = 24
W, H = 1280, 720
# 日本語を出すためmacOS標準のフォントを使う(リポジトリには入れない)
FONT_BOLD = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"
FONT_REG = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"

SECONDS = {"title": 4, "pipeline": 7, "overhead": 5, "er": 9, "robot_hold": 2, "ending": 6}
BEAR_DIR = ROOT / "renders" / "bear" / "anim"
OVERHEAD = ROOT / "renders" / "overhead_bear_front_marks.png"
OVERLAY_MAIN = ROOT / "er" / "out" / "overhead_bear_front_marks_overlay.png"
OVERLAY_TEMP = ROOT / "er" / "out" / "overhead_star_overlay.png"
ROBOT_TEMP = ROOT / "renders" / "anim_er_star"

# ERが服の線を設計する版(bear2)ができたら「Gemini Robotics-ERが服の線を設計 → プログラムで3Dに投影し平面に展開」に変える
BEAR_CAPTION = "3Dのクマに服を合わせ、前身頃をプログラムで平面の型紙に展開"

PIPELINE_TEXT = (
    "① Blender: 型紙を描いた布の俯瞰画像を作る\n"
    "② Gemini Robotics-ER: 画像から切る線とつかむ点を推論\n"
    "③ 画素 → 机の上の3D座標(m)に変換\n"
    "④ MuJoCo: 刃先が線をなぞるPandaの関節角を計算(IK)\n"
    "⑤ Blender: 関節角どおりにロボットを動かして描画"
)


class Timeline:
    """区間を前から順に並べる。チャンネル1=背景・映像、2〜=テロップ。"""

    def __init__(self) -> None:
        scene = bpy.context.scene
        self.ed = scene.sequence_editor_create()
        self.frame = 1
        self.fonts = {k: bpy.data.fonts.load(p) for k, p in (("bold", FONT_BOLD), ("reg", FONT_REG))}
        self.sections: list[tuple[str, float]] = []

    def background(self, length: int, color=(0.08, 0.09, 0.11)) -> None:
        s = self.ed.strips.new_effect("bg", type="COLOR", channel=1, frame_start=self.frame, length=length)
        s.color = color

    def still(self, path: Path, length: int) -> None:
        self.background(length)
        s = self.ed.strips.new_image(path.stem, str(path), channel=2, frame_start=self.frame, fit_method="FIT")
        s.frame_final_duration = length

    def sequence(self, folder: Path, hold: int = 0) -> int:
        frames = sorted(folder.glob("*.png"))
        s = self.ed.strips.new_image(folder.name, str(frames[0]), channel=2, frame_start=self.frame, fit_method="FIT")
        for f in frames[1:]:
            s.elements.append(f.name)
        length = len(frames)
        if hold:
            # 最後のコマを止めて見せる
            last = self.ed.strips.new_image(f"{folder.name}_hold", str(frames[-1]), channel=2,
                                            frame_start=self.frame + length, fit_method="FIT")
            last.frame_final_duration = hold
            length += hold
        self.background(length)
        return length

    def text(self, body: str, length: int, *, y: float = 0.06, size: int = 34, bold: bool = True,
             box: bool = True, channel: int = 3, align: str = "CENTER") -> None:
        s = self.ed.strips.new_effect("text", type="TEXT", channel=channel, frame_start=self.frame, length=length)
        s.text = body
        s.font = self.fonts["bold" if bold else "reg"]
        s.font_size = size
        s.location = (0.5, y)
        s.anchor_x, s.anchor_y = "CENTER", "BOTTOM"
        s.alignment_x = align
        s.wrap_width = 0.92
        s.color = (1, 1, 1, 1)
        s.use_shadow = not box
        s.use_box = box
        s.box_color = (0, 0, 0, 0.6)
        s.box_margin = 0.012

    def advance(self, name: str, length: int) -> None:
        self.sections.append((name, length / FPS))
        self.frame += length


def er_reason(overlay: Path) -> str | None:
    """overlay画像と同じ名前のER出力(<名前>_er.json)から理由の1文を取る。"""
    er_json = overlay.with_name(overlay.name.replace("_overlay.png", "_er.json"))
    if not er_json.exists():
        return None
    return json.loads(er_json.read_text()).get("reason")


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", type=Path, default=ROBOT_TEMP, help="ロボットの切断の連番PNGのフォルダ")
    ap.add_argument("--overlay", type=Path, default=OVERLAY_MAIN if OVERLAY_MAIN.exists() else OVERLAY_TEMP)
    ap.add_argument("--out", type=Path, default=ROOT / "renders" / "video" / "final.mp4")
    args = ap.parse_args(argv)
    robot = args.robot if args.robot.is_absolute() else ROOT / args.robot
    overlay = args.overlay if args.overlay.is_absolute() else ROOT / args.overlay
    out = args.out if args.out.is_absolute() else ROOT / args.out

    scene = bpy.context.scene
    r = scene.render
    # FITの拡大率はストリップを作った時点の解像度で決まるので、先に設定する
    r.resolution_x, r.resolution_y, r.resolution_percentage = W, H, 100
    r.fps = FPS

    tl = Timeline()
    sec = {k: v * FPS for k, v in SECONDS.items()}

    # a. タイトル
    tl.background(sec["title"])
    tl.text("AIが型紙を読み、ロボットが布を切る", sec["title"], y=0.50, size=64, box=False)
    tl.text("Gemini Robotics-ER × Franka Panda(シミュレーション)", sec["title"], y=0.40, size=30,
            bold=False, box=False, channel=4)
    tl.advance("タイトル", sec["title"])

    # 仕組み(1分に届かせるための説明スライド)
    tl.background(sec["pipeline"])
    tl.text("仕組み", sec["pipeline"], y=0.78, size=48, box=False)
    tl.text(PIPELINE_TEXT, sec["pipeline"], y=0.22, size=32, bold=False, box=False, channel=4, align="LEFT")
    tl.advance("仕組み", sec["pipeline"])

    # b. クマの展開
    n = tl.sequence(BEAR_DIR)
    tl.text(BEAR_CAPTION, n)
    tl.advance("クマの展開", n)

    # c. 布の俯瞰
    tl.still(OVERHEAD, sec["overhead"])
    tl.text("型紙を布に描く(実線=裁つ線、破線=縫う線)", sec["overhead"])
    tl.advance("布の俯瞰", sec["overhead"])

    # d. ERの推論
    tl.still(overlay, sec["er"])
    tl.text("Gemini Robotics-ERが切る線とつかむ点を推論", sec["er"], y=0.13)
    reason = er_reason(overlay)
    if reason:
        tl.text(f"ERの説明: {reason}", sec["er"], y=0.03, size=24, bold=False, channel=4)
    tl.advance("ERの推論", sec["er"])

    # e. ロボットの切断(等倍。最後のコマを少し止める)
    n = tl.sequence(robot, hold=sec["robot_hold"])
    tl.text("ERが推論した線に沿って、Pandaが刃でなぞって切る(等倍)", n)
    tl.advance("ロボットの切断", n)

    # f. エンディング
    tl.background(sec["ending"])
    tl.text("開ループの限界と今後", sec["ending"], y=0.55, size=48, box=False)
    tl.text("最初の画像1枚で軌道を決めるため、布がずれても追従できない。\n"
            "手足をVLAに置き換え、カメラを見ながら動く閉ループにしたい",
            sec["ending"], y=0.33, size=30, bold=False, box=False, channel=4)
    tl.advance("エンディング", sec["ending"])

    scene.frame_start, scene.frame_end = 1, tl.frame - 1
    r.image_settings.media_type = "VIDEO"
    r.image_settings.file_format = "FFMPEG"
    r.ffmpeg.format = "MPEG4"
    r.ffmpeg.codec = "H264"
    r.ffmpeg.constant_rate_factor = "HIGH"
    r.use_stamp = False
    out.parent.mkdir(parents=True, exist_ok=True)
    r.filepath = str(out)

    total = (tl.frame - 1) / FPS
    for name, s in tl.sections:
        print(f"  {name}: {s:.1f}s")
    print(f"合計 {total:.1f}s  ER画像={overlay.relative_to(ROOT)}  理由={'あり' if reason else 'なし'}  "
          f"ロボット={robot.relative_to(ROOT)}")
    if not 60 <= total <= 180:
        print(f"警告: 尺{total:.1f}sが課題の規定(1〜3分)から外れている")
    bpy.ops.render.render(animation=True)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
