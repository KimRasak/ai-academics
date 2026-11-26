#!/usr/bin/env python3
"""Draw the method schematic (Figure 2): reference-conditioned LoRA training layout for EmoH3."""
from PIL import Image, ImageDraw, ImageFont

W, H = 1200, 620
im = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(im)
try:
    f = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    fs = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    ft = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
except Exception:
    f = fs = ft = ImageFont.load_default()

BLUE = (79, 129, 189); GREEN = (106, 168, 79); ORANGE = (237, 148, 68)
RED = (196, 78, 82); PURPLE = (150, 96, 176); GRAY = (120, 120, 120)

def box(x, y, w, h, text, color, font=None, text_color="white"):
    d.rounded_rectangle([x, y, x + w, y + h], radius=10, fill=color)
    f2 = font or f
    lines = text.split("\n")
    lh = f2.size + 4
    total = len(lines) * lh
    ty = y + (h - total) / 2
    for ln in lines:
        tw = d.textlength(ln, font=f2)
        d.text((x + (w - tw) / 2, ty), ln, fill=text_color, font=f2)
        ty += lh

def arrow(x1, y1, x2, y2):
    d.line([x1, y1, x2, y2], fill="black", width=3)
    # arrowhead
    import math
    ang = math.atan2(y2 - y1, x2 - x1)
    L = 12
    for da in (2.6, -2.6):
        d.line([x2, y2, x2 - L * math.cos(ang + da), y2 - L * math.sin(ang + da)],
               fill="black", width=3)

def tlabel(x, y, text, color=GRAY, font=None):
    d.text((x, y), text, fill=color, font=font or fs)

# Title
d.text((W/2 - d.textlength("Reference-Conditioned LoRA Training (v3)", font=ft)/2, 14),
       "Reference-Conditioned LoRA Training (v3)", fill="black", font=ft)

# --- Row 1: reference portrait
box(40, 60, 220, 60, "Reference portrait\n(emotion A)", PURPLE, fs)
tlabel(50, 126, "Video VAE encode (spatial)\n+ sample, manual_seed=42", GRAY, fs)
box(40, 190, 220, 60, "ref latent rows  (64, 96)\npatchified (1,2,2), CLEAN", PURPLE, fs)
arrow(150, 120, 150, 188)
tlabel(250, 95, "rows concatenated into the", GRAY, fs)

# --- Row 2: target video
box(300, 60, 220, 60, "Target video\n(emotion B)", GREEN, fs)
tlabel(310, 126, "Video VAE encode", GRAY, fs)
box(300, 190, 220, 60, "target latent rows\nnoised @ t = 0.999", GREEN, fs)
arrow(410, 120, 410, 188)

# --- Row 3: text
box(560, 60, 280, 60, "Text: \"<Picture 1> <target>\nKeep the same character.\"", ORANGE, fs)
tlabel(570, 126, "Qwen3-VL (layer 50)", GRAY, fs)
box(560, 190, 280, 60, "text embed rows", ORANGE, fs)
arrow(700, 120, 700, 188)

# --- Row 4: audio
box(880, 60, 280, 60, "Audio track\n(from target video)", BLUE, fs)
tlabel(890, 126, "Audio VAE encode", GRAY, fs)
box(880, 190, 280, 60, "audio latent rows", BLUE, fs)
arrow(1020, 120, 1020, 188)

# --- conditioning concat
box(240, 290, 720, 56, "conditioning = concat(ref rows, audio rows, text rows)", GRAY, fs)
for cx, cy, cw in [(150, 250, 0), (410, 250, 0), (700, 250, 0), (1020, 250, 0)]:
    arrow(cx, 250, cx, 288)

# --- noised target joins too
arrow(410, 250, 410, 288)  # target (dup arrow ok)
tlabel(170, 330, "target rows", GRAY, fs)
arrow(410, 352, 470, 346)

# --- denoiser
box(280, 380, 640, 80, "MiniMax H3 Ref2VA denoiser  +  LoRA (r=16, alpha=32)\n(FSDP2, 8xH100, 900 steps)", RED, ft)
arrow(600, 346, 600, 378)

# --- output
box(380, 500, 440, 60, "denoised video latents  ->  decode  ->  emotion-B video", BLUE, ft)
arrow(600, 460, 600, 498)

# --- loss
tlabel(360, 575, "Loss on target rows only (pred[:, num_cond:] vs clean target)", GRAY, fs)
arrow(600, 560, 600, 573)

im.save("paper_figure2.png")
print("saved paper_figure2.png", im.size)
