"""連番PNGをBlenderのシーケンサーでMP4(H.264)にまとめる。ffmpeg不要。

実行: Blender -b --factory-startup -P blender/encode.py -- renders/anim_er_star renders/video/er_star.mp4
"""

import sys
from pathlib import Path

import bpy

ROOT = Path(__file__).resolve().parent.parent
FPS = 24


def main() -> None:
    src, dst = (ROOT / a for a in sys.argv[sys.argv.index("--") + 1 :][:2])
    frames = sorted(src.glob("*.png"))
    scene = bpy.context.scene
    ed = scene.sequence_editor_create()
    strip = ed.strips.new_image("frames", str(frames[0]), channel=1, frame_start=1)
    for f in frames[1:]:
        strip.elements.append(f.name)

    w, h = bpy.data.images.load(str(frames[0])).size
    r = scene.render
    r.resolution_x, r.resolution_y, r.resolution_percentage = w, h, 100
    r.fps = FPS
    scene.frame_start, scene.frame_end = 1, len(frames)
    r.image_settings.media_type = "VIDEO"
    r.image_settings.file_format = "FFMPEG"
    r.ffmpeg.format = "MPEG4"
    r.ffmpeg.codec = "H264"
    r.ffmpeg.constant_rate_factor = "HIGH"
    r.use_stamp = False
    dst.parent.mkdir(parents=True, exist_ok=True)
    r.filepath = str(dst)
    bpy.ops.render.render(animation=True)
    print(f"wrote {dst.name}: {len(frames)} frames")


if __name__ == "__main__":
    main()
