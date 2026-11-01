"""最終動画(bear3版)をBlenderのシーケンサー(VSE)で組み立て、MP4(H.264・1280x720・24fps)にする。

実行: Blender -b --factory-startup -P blender/compose.py -- [--version bear3|dressform]
      [--robot renders/anim_er_<版>_front_marks] [--out renders/video/final_<版>.mp4]
保険の版(final.mp4を作った流れ)は blender/compose_v1.py。

区間: タイトル → 仕組み → ERが服を設計 → 投影と微調整 → 平面に展開 → 布の型紙 → ERが切る線を選ぶ
      → ロボットの切断 → エンディング
秒数はSECONDS、素材と呼び名はVERSIONSで変える。課題の規定で全体を1〜3分に収める(外れたら警告を出す)。
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

FADE = 12  # 展開の最後のコマから型紙画面へのクロスフェード(フレーム)
SECONDS = {"pattern": 6, "title": 4, "pipeline": 9, "bear_front": 2.5, "design": 7, "morph_each": 1.2, "overhead": 5,
           "select": 8, "robot_hold": 2, "ending": 6}
def assets_for(model: str, front_dir: str, garment: str) -> dict:
    """版ごとの素材のパス。命名はbear3と同じ規則(<model>_front・overhead_<model>_front_marks)。"""
    out, er = ROOT / "renders", ROOT / "er" / "out"
    return {
        "bear_front": out / front_dir / "front.png",
        "design_overlay": er / f"{model}_front_overlay.png",
        "design_er": er / f"{model}_front_er.json",
        "morph": [out / front_dir / f"seam_morph_{k}.png" for k in range(5)],
        "unfold": out / front_dir / "anim",
        "overhead": out / f"overhead_{model}_front_marks.png",
        "select_overlay": er / f"overhead_{model}_front_marks_overlay.png",
        "select_er": er / f"overhead_{model}_front_marks_er.json",
        "robot": out / f"anim_er_{model}_front_marks",
        "out": out / "video" / f"final_{model}.mp4",
        "garment": garment,
        "pattern_view": None,  # 製図の形の型紙画面(あれば「平面に展開」の後に入れる)
        "skip": set(),  # 外す区間
    }


# 版: --versionで選ぶ。subjectはテロップでの呼び名
VERSIONS = {
    "bear3": {
        "subject": "クマ",
        "short": "クマ",
        "assets": assets_for("bear3", "bear3", "タンクトップの前身頃"),
        # ERの理由(英語)の日本語訳。原文はER出力の"reason"から読んで小さく添える
        "reason_ja": {
            "design_er": "首と腕の付け根を余裕をもって回り込む切り欠きを付け、胸とお腹に沿う前身頃にした",
            "select_er": "外側の実線が服のパーツの裁断境界だから",
        },
    },
    "dressform": {
        "subject": "人台(1/2スケールのマネキン)",
        "short": "人台",  # 長いと仕組みと見出しが折り返すので、短い呼び名も持つ
        "assets": assets_for("dressform", "dressform", "服の前身頃") | {
            "pattern_view": ROOT / "renders" / "dressform" / "pattern_view.png",
            # 布に描いたのは製図の形に整えた型紙(drafted)
            "select_overlay": ROOT / "er" / "out" / "overhead_dressform_front_drafted_marks_overlay.png",
            "select_er": ROOT / "er" / "out" / "overhead_dressform_front_drafted_marks_er.json",
            "robot": ROOT / "renders" / "anim_er_dressform_front_drafted_marks",
            "skip": {"overhead"},  # 布の俯瞰は「おもちゃみたい」なので外す(ユーザーの評価)
        },
        "reason_ja": {
            "design_er": "人台の胴の輪郭に沿って、くぼんだ襟ぐり・肩線・脇の線・脇の下・まっすぐな裾を持つ前身頃の輪郭を描いた",
            "select_er": "布からパーツを切り離すには外側の白い実線に沿って切り、そのあと中心をつかんで持ち上げるべき",
        },
    },
}
ASSETS: dict = VERSIONS["bear3"]["assets"]
REASON_JA: dict = VERSIONS["bear3"]["reason_ja"]
SUBJECT = VERSIONS["bear3"]["subject"]

PIPELINE_TEXT = (
    "① Gemini Robotics-ER: {subject}の正面画像を見て、服(前身頃)の線を設計\n"
    "② プログラム: 線を3Dの体に投影し、体型に合わせて微調整し、平面の型紙に展開\n"
    "③ Blender: 型紙を布に描く(実線=裁つ線、破線=縫う線)\n"
    "④ Gemini Robotics-ER: 線の意味を教えずに、布の画像から切る線を選ぶ\n"
    "⑤ MuJoCo: 刃先が線をなぞるPandaの関節角を計算(IK)\n"
    "⑥ Blender: 関節角どおりにPandaを動かして、布を切る様子を描画"
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

    def slides(self, paths: list[Path], each: int) -> int:
        """静止画を1枚ずつeachフレームずつ並べる。"""
        self.background(each * len(paths))
        for i, p in enumerate(paths):
            s = self.ed.strips.new_image(p.stem, str(p), channel=2, frame_start=self.frame + i * each,
                                         fit_method="FIT")
            s.frame_final_duration = each
        return each * len(paths)

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


def reason_lines(key: str) -> str | None:
    """ERの理由のテロップ: 日本語訳と原文。"""
    path = ASSETS[key]
    en = json.loads(path.read_text()).get("reason") if path.exists() else None
    ja = REASON_JA.get(key)
    if not (en or ja):
        return None
    return "\n".join(t for t in (f"ERの説明: {ja}" if ja else None, f"(原文) {en}" if en else None) if t)


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="bear3", choices=list(VERSIONS), help="素材の版(bear3・dressform)")
    ap.add_argument("--robot", type=Path, default=None, help="ロボットの切断の連番PNGのフォルダ(既定は版の素材)")
    ap.add_argument("--out", type=Path, default=None, help="出力(既定はrenders/video/final_<版>.mp4)")
    args = ap.parse_args(argv)
    global ASSETS, REASON_JA, SUBJECT
    ver = VERSIONS[args.version]
    ASSETS, REASON_JA, SUBJECT = ver["assets"], ver["reason_ja"], ver["subject"]
    short = ver["short"]
    args.robot = args.robot or ASSETS["robot"]
    args.out = args.out or ASSETS["out"]
    robot = args.robot if args.robot.is_absolute() else ROOT / args.robot
    out = args.out if args.out.is_absolute() else ROOT / args.out

    scene = bpy.context.scene
    r = scene.render
    # FITの拡大率はストリップを作った時点の解像度で決まるので、先に設定する
    r.resolution_x, r.resolution_y, r.resolution_percentage = W, H, 100
    r.fps = FPS

    tl = Timeline()
    sec = {k: round(v * FPS) for k, v in SECONDS.items()}

    # 1. タイトル
    tl.background(sec["title"])
    tl.text("AIが服を設計し、ロボットが布を切る", sec["title"], y=0.50, size=64, box=False)
    tl.text("Gemini Robotics-ER × Franka Panda(シミュレーション)", sec["title"], y=0.40, size=30,
            bold=False, box=False, channel=4)
    tl.advance("タイトル", sec["title"])

    # 2. 仕組み
    tl.background(sec["pipeline"])
    tl.text("仕組み", sec["pipeline"], y=0.80, size=48, box=False)
    tl.text(PIPELINE_TEXT.format(subject=short), sec["pipeline"], y=0.18, size=28, bold=False, box=False, channel=4, align="LEFT")
    tl.advance("仕組み", sec["pipeline"])

    # 3. モデルの正面 → ERの設計線
    tl.still(ASSETS["bear_front"], sec["bear_front"])
    tl.text(f"この{SUBJECT}に合う服を、AIに設計させる", sec["bear_front"])
    tl.advance("モデルの正面", sec["bear_front"])
    tl.still(ASSETS["design_overlay"], sec["design"])
    tl.text(f"Gemini Robotics-ERが{short}に合う{ASSETS['garment']}を設計", sec["design"], y=0.89)
    design_reason = reason_lines("design_er")
    if design_reason:
        tl.text(design_reason, sec["design"], y=0.02, size=22, bold=False, channel=4)
    tl.advance("ERが服を設計", sec["design"])

    # 4. 投影と微調整
    n = tl.slides(ASSETS["morph"], sec["morph_each"])
    tl.text("プログラムで3Dに投影し、体型に合わせて微調整(赤→緑)", n)
    tl.advance("投影と微調整", n)

    # 5. 平面に展開
    n = tl.sequence(ASSETS["unfold"])
    tl.text("前身頃を平面の型紙に展開", n)
    tl.advance("平面に展開", n)

    # 5b. 製図の形に整えた型紙画面(展開の最後のコマからクロスフェード)
    if ASSETS["pattern_view"]:
        n = sec["pattern"]
        tl.background(n)
        v = tl.ed.strips.new_image("pattern_view", str(ASSETS["pattern_view"]), channel=6,
                                   frame_start=tl.frame - FADE, fit_method="FIT")
        v.frame_final_duration = n + FADE
        v.blend_alpha = 0.0
        v.keyframe_insert("blend_alpha", frame=tl.frame - FADE)
        v.blend_alpha = 1.0
        v.keyframe_insert("blend_alpha", frame=tl.frame)
        tl.text("平面に開き、製図の形に整えた前身頃の型紙(1/2スケール)", n, y=0.02, channel=7)
        tl.advance("型紙画面", n)

    # 6. 布の型紙
    if "overhead" not in ASSETS["skip"]:
        tl.still(ASSETS["overhead"], sec["overhead"])
        tl.text("型紙を布に描く(実線=裁つ線、破線=縫う線)", sec["overhead"])
        tl.advance("布の型紙", sec["overhead"])

    # 7. ERが切る線を選ぶ
    tl.still(ASSETS["select_overlay"], sec["select"])
    tl.text("線の意味は教えずに切り抜きを指示 → ERは外側の実線を選んだ", sec["select"], y=0.89)
    select_reason = reason_lines("select_er")
    if select_reason:
        tl.text(select_reason, sec["select"], y=0.02, size=22, bold=False, channel=4)
    tl.advance("ERが切る線を選ぶ", sec["select"])

    # 8. ロボットの切断(等倍。最後のコマを少し止める)
    n = tl.sequence(robot, hold=sec["robot_hold"])
    tl.text("ERが選んだ線に沿って、Pandaが刃でなぞって切る(等倍)", n)
    tl.advance("ロボットの切断", n)

    # 9. エンディング
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
    print(f"合計 {total:.1f}s  理由(設計/選択)={'あり' if design_reason else 'なし'}/"
          f"{'あり' if select_reason else 'なし'}  ロボット={robot.relative_to(ROOT)}")
    if not 60 <= total <= 180:
        print(f"警告: 尺{total:.1f}sが課題の規定(1〜3分)から外れている")
    bpy.ops.render.render(animation=True)
    print(f"wrote {out.name}")


if __name__ == "__main__":
    main()
