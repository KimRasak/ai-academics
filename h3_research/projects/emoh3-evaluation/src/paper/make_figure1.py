#!/usr/bin/env python3
"""Compose the paper's Figure 1: reference | base | v3 comparison collages.

Per key case, stack [reference portrait, 1 base frame, 1 v3 frame] side by side,
one row per case, with a title row. Uses frames already extracted with ffmpeg.
Run on the container (PIL available) then pull the PNG.
"""
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

H3 = "/gluster_hdd_cv/user/jinzili/code/anime_style/model_demo/minimax_h3"
PILOT = "/gluster_hdd_cv/user/jinzili/tmp/pilot"
REF = H3 + "/inputs/expressions_0825"

# case -> (ref_portrait, base_mp4, v3_mp4, base_verdict, v3_verdict)
CASES = [
    ("x01_neutral_to_joyful", REF + "/a15_neutral_1537882489_v1.png",
     PILOT + "/outputs_base16/x01_neutral_to_joyful_base.mp4",
     PILOT + "/outputs_lora3/x01_neutral_to_joyful_lora.mp4",
     "base: Happy:90 (16s s42)", "v3: happy:80 (16s s42)"),
    ("x02_neutral_to_sad", REF + "/a15_neutral_1537882489_v1.png",
     PILOT + "/outputs_base16x02/x02_neutral_to_sad_base.mp4",
     PILOT + "/outputs_lora3_s1234/x02_neutral_to_sad_lora.mp4",
     "base: neutral:100 (16s s1234)", "v3: sad:80 (16s s1234)"),
    ("x03_angry_to_happy", REF + "/a02_angry_1534927518_v1.png",
     PILOT + "/outputs_base16/x03_angry_to_happy_base.mp4",
     PILOT + "/outputs_lora3/x03_angry_to_happy_lora.mp4",
     "base: Happy:90 (16s s42)", "v3: happy:80 (16s s42)"),
    ("x07_sad_to_cheerful", REF + "/a18_sad_1538269564_v1.png",
     PILOT + "/outputs_base16b/x07_sad_to_cheerful_base.mp4",
     PILOT + "/outputs_lora3/x07_sad_to_cheerful_lora.mp4",
     "base: neutral:50 (16s s1234)", "v3: happy:80 (16s s42)"),
]

FRAME_T = "0.50"  # middle frame
OUT = Path(PILOT + "/paper_figure1.png")
FF = "/gluster_hdd_cv/user/jinzili/env/sglang310/bin/ffmpeg"

def grab(mp4, dst):
    import subprocess
    subprocess.run([FF, "-v", "error", "-ss", FRAME_T, "-i", mp4,
                    "-frames:v", "1", "-q:v", "2", "-y", str(dst)],
                   capture_output=True, check=True)
    return Image.open(dst).convert("RGB")

def main():
    tmp = Path("/tmp/f1")
    tmp.mkdir(exist_ok=True)
    # per-case cell images [ref, base, v3]
    cells = []
    for (name, ref, bmp4, v3mp4, bv, v3v) in CASES:
        refimg = Image.open(ref).convert("RGB")
        baseimg = grab(bmp4, tmp / (name + "_b.jpg"))
        v3img = grab(v3mp4, tmp / (name + "_v3.jpg"))
        cells.append([refimg, baseimg, v3img])
    # normalize sizes
    w = 256; h = 256
    norm = lambda im: im.resize((w, h), Image.LANCZOS)
    cells = [[norm(im) for im in row] for row in cells]
    pad = 6
    hdr = 34
    ftr = 40
    # LANDSCAPE: rows = method (ref/base/v3), cols = task
    W = pad + len(cells) * (w + pad)
    H = hdr + 3 * (h + pad) + ftr
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    for c in range(len(cells)):
        name, _, _, _, bv, v3v = CASES[c]
        draw.text((pad + c * (w + pad), 6), name.replace("_", " "), fill="black", font=font)
    for r, label in enumerate(["Reference", "Base (16 steps)", "LoRA-v3 (ref-cond)"]):
        y = hdr + r * (h + pad)
        draw.text((pad, y), label, fill="black", font=font)
        for c in range(len(cells)):
            x = pad + c * (w + pad)
            canvas.paste(cells[c][r], (x, y))
    # verdict footer
    ftxt = "   |   ".join(f"{CASES[c][0]}: {CASES[c][4]} vs {CASES[c][5]}" for c in range(len(CASES)))
    draw.text((pad, hdr + 3 * (h + pad)), ftxt, fill="black", font=font)
    canvas.save(OUT)
    print("saved", OUT, canvas.size)

if __name__ == "__main__":
    main()
