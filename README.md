# 布を裁断するロボット(Gemini Robotics-ER + Franka Panda + Blender)

> 講座の規定により、匿名で提出しています。

Gemini Robotics-ERが1/2スケールの人台の正面画像から服の前身頃を設計します。
その線をBlenderのPython APIで人台に投影し、平面に展開して型紙にします。
次に、型紙を描いた布の画像から、ERが「どの線を切るか」を判断します。
最後に、Franka PandaがMuJoCoのIKで計算した動きで布を切り抜き、つかんで持ち上げます。
ロボットと布はすべてシミュレーション(CG)です。

## 環境

- macOS(Apple M2で確認)
- Blender 5.2.3(`blender -b --factory-startup -P <script> -- <引数>`で実行)
- Python 3.13 + [uv](https://docs.astral.sh/uv/)。`uv sync`で依存(google-genai・mujoco・numpy・pillow)を入れます
- Gemini APIキー: macOSのキーチェーンから読みます。次のコマンドで登録します(`-w`に値を書かずに対話入力にすると、履歴に残りません)

  ```
  security add-generic-password -s pai-final-cloth-gemini -a gemini-api-key -w
  ```

## 実行手順(人台版)

`blender`はBlenderの実行ファイル(macOSでは`/Applications/Blender.app/Contents/MacOS/Blender`)です。

| # | コマンド | 出力 |
|---|---|---|
| 1 | `blender -b --factory-startup -P blender/dressform.py -- --shoot` | 人台の正面画像 `renders/dressform/front.png` |
| 2 | `uv run python er/plan.py renders/dressform/front.png --task design_form --name dressform_front` | ERが設計した輪郭 `er/out/dressform_front_er.json` |
| 3 | `blender -b --factory-startup -P blender/dressform.py -- --anim` | 投影・補正・平面に展開した輪郭と画像 `renders/dressform/` |
| 4 | `uv run python draft/draft.py` | 型紙 `patterns/dressform_front_drafted.json`(縫い合わせる辺を直線化、縫い代1cm)と型紙画面 |
| 5 | `blender -b --factory-startup -P blender/scene.py -- --pattern dressform_front_drafted --marks` | 型紙を描いた布の俯瞰画像 `renders/overhead_dressform_front_drafted_marks.png` |
| 6 | `uv run python er/plan.py renders/overhead_dressform_front_drafted_marks.png --task garment` | ERが選んだ切断線とつかむ点 `er/out/overhead_dressform_front_drafted_marks_er.json` |
| 7 | `uv run python er/to3d.py er/out/overhead_dressform_front_drafted_marks_er.json renders/overhead_dressform_front_drafted_marks.json` | 布の上の3D点列 `er/out/..._points3d.json` |
| 8 | `uv run python kin/ik.py er/out/overhead_dressform_front_drafted_marks_points3d.json --name er_dressform_front_drafted_marks --grasp` | Pandaの関節角 `kin/out/er_dressform_front_drafted_marks_poses.json` |
| 9 | `blender -b --factory-startup -P blender/animate.py -- --pattern dressform_front_drafted --marks --poses kin/out/er_dressform_front_drafted_marks_poses.json --all` | 切断と持ち上げの連番画像 `renders/anim_er_dressform_front_drafted_marks/` |
| 10 | `blender -b --factory-startup -P blender/compose.py -- --version dressform` | 動画 `renders/video/final_dressform.mp4` |

ERの結果(`er/out/dressform_front_*`、`er/out/overhead_dressform_front_drafted_marks_*`)と型紙(`patterns/dressform_*.json`)は同梱しています。
APIキーがなくても、手順2と6を飛ばせば動画を作り直せます。`renders/`と`kin/out/`は実行で作られるので、リポジトリには入れていません。

## フォルダ構成

- `er/`: Gemini Robotics-ERの呼び出し(`plan.py`)と、画素から3D点への変換(`to3d.py`)
- `blender/`: 人台と型紙(`dressform.py`。`bear.py`・`bear2.py`の輪郭処理を部品として使う)、布のシーン(`scene.py`)、ロボットと切断(`robot.py`・`animate.py`)、動画の合成(`compose.py`)
- `draft/`: 平面に展開した輪郭を型紙に整え、型紙画面を描く
- `kin/`: MuJoCoによるPandaの逆運動学(IK)
- `patterns/`: 型紙の輪郭(JSON)
- `assets/`: Pandaのモデル

## クレジット

- Franka Emika Pandaのモデル(`assets/franka_emika_panda/`): [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)(Google DeepMind)、Apache License 2.0。ライセンス本文は同フォルダの`LICENSE`
- 使用したもの: Gemini Robotics-ER(`gemini-robotics-er-2-preview`、Google)、[MuJoCo](https://mujoco.org/)、[Blender](https://www.blender.org/)
