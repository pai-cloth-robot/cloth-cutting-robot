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
        # ERの区間(設計・切る線の選択)は、理由の代わりにERへ与えたプロンプトの全文(_er.jsonの"prompt")を右に出す
        "show_prompt": True,
        "seconds": {"design": 11, "select": 11, "robot_hold": 2, "ending": 9},
        "texts": {
            "front": "この{subject}に合う服を、AIが設計する",
            "design": "Gemini Robotics-ERが、\n人台に合う服の前身頃を設計",
            "pipeline": (
                "① Gemini Robotics-ER: {subject}の正面画像を見て、服(前身頃)の線を設計\n"
                "② Python(Blender API): ERの線を3Dの人台に投影し、洋裁の基準線(脇・肩・袖ぐり)に合わせて補正し、"
                "平面の型紙に展開\n"
                "③ Blender: 型紙を布に描く(実線=裁つ線、破線=縫う線)\n"
                "④ Gemini Robotics-ER: 『型紙のパーツを切り抜いて』とだけ指示し、布の画像から切る線を判断\n"
                "⑤ MuJoCo: 刃先が線をなぞるPandaの関節角を計算(IK)\n"
                "⑥ Blender: 関節角どおりにPandaを動かして、布を切る様子を描画"
            ),
            "select": "『型紙のパーツを切り抜いて』とだけ指示 → ERは外側の実線(裁ち切り線)を選んだ",
            "robot": "ERが選んだ線に沿ってPandaが切り、ERが示した点をつかんで持ち上げる(等倍)",
            "ending_title": "今回の限界と今後",
            # テキストのストリップは空白でしか折り返さないので、日本語は手で改行する
            "ending_body": (
                "ロボットは、切る前に撮った画像1枚から決めた軌道を、\n"
                "作業中に布を見直さずにそのまま実行している(開ループ制御)。\n\n"
                "実際の布は刃に引かれてずれたり、しわが寄ったりするため、\n"
                "作業中もカメラで布を見て軌道を修正し続ける仕組み\n"
                "(閉ループ制御。例: VLAモデル)が今後の課題"
            ),
        },
        # テロップ(箱付き)は区間の前半6割だけ出して0.5秒で消す。箱は薄く
        "caption": {"show": 0.6, "fade": 12, "box_alpha": 0.35},
        "robot_offset_y": 60,  # ロボットの映像を上へずらす(px)。下端のテロップと布が重ならないように
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
        self.caption: dict = {}  # 箱付きテロップの出し方(show: 区間の何割で消すか、fade: フレーム、box_alpha)

    def background(self, length: int, color=(0.08, 0.09, 0.11)) -> None:
        s = self.ed.strips.new_effect("bg", type="COLOR", channel=1, frame_start=self.frame, length=length)
        s.color = color

    def still(self, path: Path, length: int, offset_x: int = 0) -> None:
        self.background(length)
        s = self.ed.strips.new_image(path.stem, str(path), channel=2, frame_start=self.frame, fit_method="FIT")
        s.frame_final_duration = length
        s.transform.offset_x = offset_x

    def slides(self, paths: list[Path], each: int) -> int:
        """静止画を1枚ずつeachフレームずつ並べる。"""
        self.background(each * len(paths))
        for i, p in enumerate(paths):
            s = self.ed.strips.new_image(p.stem, str(p), channel=2, frame_start=self.frame + i * each,
                                         fit_method="FIT")
            s.frame_final_duration = each
        return each * len(paths)

    def sequence(self, folder: Path, hold: int = 0, offset_y: int = 0) -> int:
        frames = sorted(folder.glob("*.png"))
        s = self.ed.strips.new_image(folder.name, str(frames[0]), channel=2, frame_start=self.frame, fit_method="FIT")
        for f in frames[1:]:
            s.elements.append(f.name)
        s.transform.offset_y = offset_y
        length = len(frames)
        if hold:
            # 最後のコマを止めて見せる
            last = self.ed.strips.new_image(f"{folder.name}_hold", str(frames[-1]), channel=2,
                                            frame_start=self.frame + length, fit_method="FIT")
            last.frame_final_duration = hold
            last.transform.offset_y = offset_y
            length += hold
        self.background(length)
        return length

    def text(self, body: str, length: int, *, y: float = 0.06, size: int = 34, bold: bool = True,
             box: bool = True, channel: int = 3, align: str = "CENTER", x: float = 0.5,
             anchor: tuple[str, str] = ("CENTER", "BOTTOM"), wrap: float = 0.92, start: int = 0) -> None:
        fade = box and bool(self.caption)
        if fade:  # 区間の前半だけ出して消す(最短2秒)
            length = min(length, max(round(length * self.caption["show"]), 2 * FPS))
        s = self.ed.strips.new_effect("text", type="TEXT", channel=channel, frame_start=self.frame + start,
                                      length=length)
        s.text = body
        s.font = self.fonts["bold" if bold else "reg"]
        s.font_size = size
        s.location = (x, y)
        s.anchor_x, s.anchor_y = anchor
        s.alignment_x = align
        s.wrap_width = wrap
        s.color = (1, 1, 1, 1)
        s.use_shadow = not box
        s.use_box = box
        s.box_color = (0, 0, 0, self.caption.get("box_alpha", 0.6))
        s.box_margin = 0.012
        if fade:
            end = self.frame + start + length
            s.blend_alpha = 1.0
            s.keyframe_insert("blend_alpha", frame=end - self.caption["fade"])
            s.blend_alpha = 0.0
            s.keyframe_insert("blend_alpha", frame=end)

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


PROMPT_X = 0.565  # プロンプトを出す区間: 画像を左へ寄せ、この位置から右にテキストを置く
IMAGE_LEFT = -(W - H) // 2  # 正方形の画像を左端に寄せるずれ(px)


def er_section(tl: "Timeline", image: Path, er_json: Path, heading: str, length: int) -> None:
    """ERの区間(プロンプト表示版): 左に画像、右に見出しとERへ与えたプロンプトの全文(英文のまま)。"""
    tl.still(image, length, offset_x=IMAGE_LEFT)
    wrap = 0.98 - PROMPT_X
    tl.text(heading, length, x=PROMPT_X, y=0.95, size=24, align="LEFT", anchor=("LEFT", "TOP"), wrap=wrap,
            box=False)
    prompt = json.loads(er_json.read_text()).get("prompt")
    if not prompt:
        print(f"警告: {er_json.name}に\"prompt\"がない")
        return
    tl.text("ERに与えたプロンプト(全文)", length, x=PROMPT_X, y=0.76, size=18, align="LEFT",
            anchor=("LEFT", "TOP"), wrap=wrap, box=False, channel=4)
    tl.text(prompt, length, x=PROMPT_X, y=0.71, size=13, bold=False, align="LEFT", anchor=("LEFT", "TOP"),
            wrap=wrap, box=False, channel=5)


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
    texts, show_prompt = ver.get("texts", {}), ver.get("show_prompt", False)
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
    tl.caption = ver.get("caption", {})
    sec = {k: round(v * FPS) for k, v in (SECONDS | ver.get("seconds", {})).items()}

    # 1. タイトル
    tl.background(sec["title"])
    tl.text("AIが服を設計し、ロボットが布を切る", sec["title"], y=0.50, size=64, box=False)
    tl.text("Gemini Robotics-ER × Franka Panda(シミュレーション)", sec["title"], y=0.40, size=30,
            bold=False, box=False, channel=4)
    tl.advance("タイトル", sec["title"])

    # 2. 仕組み
    tl.background(sec["pipeline"])
    tl.text("仕組み", sec["pipeline"], y=0.80, size=48, box=False)
    tl.text(texts.get("pipeline", PIPELINE_TEXT).format(subject=short), sec["pipeline"], y=0.18, size=28, bold=False, box=False, channel=4, align="LEFT")
    tl.advance("仕組み", sec["pipeline"])

    # 3. モデルの正面 → ERの設計線
    tl.still(ASSETS["bear_front"], sec["bear_front"])
    tl.text(texts.get("front", "この{subject}に合う服を、AIに設計させる").format(subject=SUBJECT), sec["bear_front"])
    tl.advance("モデルの正面", sec["bear_front"])
    design_heading = texts.get("design", f"Gemini Robotics-ERが{short}に合う{ASSETS['garment']}を設計")
    design_reason = None
    if show_prompt:
        er_section(tl, ASSETS["design_overlay"], ASSETS["design_er"], design_heading, sec["design"])
    else:
        tl.still(ASSETS["design_overlay"], sec["design"])
        tl.text(design_heading, sec["design"], y=0.89)
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
    select_heading = texts.get("select", "線の意味は教えずに切り抜きを指示 → ERは外側の実線を選んだ")
    select_reason = None
    if show_prompt:
        er_section(tl, ASSETS["select_overlay"], ASSETS["select_er"], select_heading, sec["select"])
    else:
        tl.still(ASSETS["select_overlay"], sec["select"])
        tl.text(select_heading, sec["select"], y=0.89)
        select_reason = reason_lines("select_er")
        if select_reason:
            tl.text(select_reason, sec["select"], y=0.02, size=22, bold=False, channel=4)
    tl.advance("ERが切る線を選ぶ", sec["select"])

    # 8. ロボットの切断(等倍。最後のコマを少し止める)
    n = tl.sequence(robot, hold=sec["robot_hold"], offset_y=ver.get("robot_offset_y", 0))
    robot_text = texts.get("robot", "ERが選んだ線に沿って、Pandaが刃でなぞって切る(等倍)")
    tl.text(robot_text, n)
    tl.advance("ロボットの切断", n)

    # 9. エンディング
    tl.background(sec["ending"])
    if "ending_body" in texts:
        tl.text(texts["ending_title"], sec["ending"], y=0.66, size=48, box=False)
        tl.text(texts["ending_body"], sec["ending"], y=0.18, size=28, bold=False, box=False, channel=4,
                align="LEFT")
    else:
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
