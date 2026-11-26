#!/usr/bin/env python3
"""
Exp 31A (Video DiT Track — Controlled Paired Dataset Builder):
Generates 16 matched photorealistic human studio portraits using open-weight `FLUX.1-dev`,
paired 1-to-1 in gender, framing, lighting, and gentle smile with our first 16 clean 2D flat
cel-shaded anime portraits (`fc_00..fc_15` from `manifest_scale48_9var.jsonl`).

Allows strict controlled comparison of Video DiT behavior (`Wan2.2-TI2V-5B` & `CogVideoX-5b-I2V`)
on 2D Anime vs. Photorealistic Humans under IDENTICAL motion prompts, seeds, and resolutions.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from diffusers import FluxPipeline
from PIL import Image, ImageDraw


def _save_sheet(pairs: list[tuple[str, Path]], out_path: Path, cols: int = 4, thumb: int = 256) -> None:
    rows = math.ceil(len(pairs) / cols)
    sheet = Image.new("RGB", (cols * thumb, rows * (thumb + 22)), (22, 22, 26))
    draw = ImageDraw.Draw(sheet)
    for idx, (lbl, p) in enumerate(pairs):
        r, c = divmod(idx, cols)
        im = Image.open(p).convert("RGB").resize((thumb, thumb), Image.LANCZOS)
        sheet.paste(im, (c * thumb, r * (thumb + 22) + 22))
        draw.text((c * thumb + 6, r * (thumb + 22) + 3), lbl, fill=(240, 240, 240))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest48", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp17a_scale48/manifest_scale48_9var.jsonl"))
    ap.add_argument("--flux-dir", type=str, default="/gluster_hdd_cv/user/jinzili/weights/flux/FLUX.1-dev")
    ap.add_argument("--out-dir", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp31_video_dit/real_portraits"))
    ap.add_argument("--n-chars", type=int, default=16)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    recs = [json.loads(l) for l in args.manifest48.read_text(encoding="utf-8").splitlines() if l.strip()]
    orig = sorted([r for r in recs if r["variant"] == "orig"], key=lambda x: x["char_id"])[: args.n_chars]

    hair_styles = [
        "shoulder-length straight hair", "short wavy bob hair", "neat layered short hair", "long flowing hair",
        "sleek medium-length hair", "fluffy tousled short hair", "side-parted wavy hair", "messy layered hair",
    ]
    outfits = [
        "clean white cotton shirt", "casual beige knit sweater", "navy blue cardigan", "light grey hoodie",
        "soft pastel denim jacket", "simple dark collared shirt", "cream turtleneck sweater", "light blue button-up shirt",
    ]

    todo = []
    for idx, r in enumerate(orig):
        out_p = args.out_dir / f"real_{idx:02d}_{r['gender']}.png"
        if not out_p.exists() and (idx % args.num_shards == args.shard):
            todo.append((idx, r, out_p))

    if todo:
        pipe = FluxPipeline.from_pretrained(args.flux_dir, torch_dtype=torch.bfloat16)
        pipe.to("cuda")
        for idx, r, out_p in todo:
            subj = "young East Asian woman in her early 20s" if r["gender"] == "girl" else "young East Asian man in his early 20s"
            hs = hair_styles[idx % len(hair_styles)]
            of = outfits[idx % len(outfits)]
            prompt = (
                f"Clean editorial studio portrait photograph of a {subj} with {hs}, wearing a {of}, "
                "facing straight toward the camera with a gentle natural warm smile, chest-up framing, "
                "soft even diffused studio lighting, plain soft light-grey studio backdrop, sharp focus, 85mm lens, DSLR photo."
            )
            gen = torch.Generator(device="cuda").manual_seed(3100 + idx)
            img = pipe(
                prompt=prompt,
                height=768,
                width=768,
                num_inference_steps=24,
                guidance_scale=3.5,
                generator=gen,
            ).images[0]
            img.save(out_p)
            print(f"[real_portraits shard {args.shard}] saved {out_p.name}", flush=True)

    # Write manifest & QC sheet once all n_chars exist
    done = []
    for idx, r in enumerate(orig):
        p = args.out_dir / f"real_{idx:02d}_{r['gender']}.png"
        if p.exists():
            done.append((f"real_{idx:02d}_{r['gender']}", p, r))
    if len(done) == args.n_chars:
        _save_sheet([(lbl, p) for lbl, p, _ in done], args.out_dir / "real16_qc_sheet.png", cols=4, thumb=256)
        man = []
        for idx, (lbl, p, r) in enumerate(done):
            man.append({
                "pair_idx": idx,
                "gender": r["gender"],
                "anime_id": r["char_id"],
                "anime_path": r["image_path"],
                "real_id": lbl,
                "real_path": str(p),
            })
        (args.out_dir / "paired_anime_real_16.jsonl").write_text(
            "\n".join(json.dumps(x) for x in man) + "\n", encoding="utf-8"
        )
        print(f"Saved paired_anime_real_16.jsonl and real16_qc_sheet.png!", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
