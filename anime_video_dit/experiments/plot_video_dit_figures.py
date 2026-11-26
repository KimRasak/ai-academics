#!/usr/bin/env python3
"""
Render publication-grade Video DiT figure (`fig7_video_dit_mechanism_and_surgery.png`)
directly from verified H100 JSON outputs (`Exp 32A`, `Exp 33`, `Exp 34A`, `Exp 34B`, `Exp 35`).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT_ROOT = Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs")
PLOT_DIR = OUT_ROOT / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> int:
    d32a = json.loads((OUT_ROOT / "exp32a_wan22_mechanistic/exp32a_mechanistic_summary.json").read_text())
    d33 = json.loads((OUT_ROOT / "exp33_wan22_decoupled/exp33_decoupled_summary.json").read_text())
    d34b_lyr = json.loads((OUT_ROOT / "exp34b_wan22_scale16_layers/layer_ablation/summary_layer_ablation.json").read_text())
    d34b_j = json.loads((OUT_ROOT / "exp34b_wan22_scale16_layers/scale16/exp34b_vlm_judge_summary.json").read_text())
    d35_j_p = OUT_ROOT / "exp35_wan22_safety_unseen/exp35_vlm_judge_summary.json"
    d35_j = json.loads(d35_j_p.read_text()) if d35_j_p.exists() else None

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), dpi=220)

    # ------------------------------------------------------------------
    # Panel (a): Frame-0 Anchor Attention Starvation across Latent Frames t=1..12
    # ------------------------------------------------------------------
    ax = axes[0]
    ts = np.arange(1, 13)
    ax.plot(ts, np.array(d32a["R1_HeadTurn"]["real"]["f0_attn_by_t_cond_s20"]) * 100, "o-", color="#2b6cb0", lw=2.2, label="Real Human (HeadTurn)")
    ax.plot(ts, np.array(d32a["R1_HeadTurn"]["anime"]["f0_attn_by_t_cond_s20"]) * 100, "s-", color="#e53e3e", lw=2.2, label="2D Anime (HeadTurn)")
    ax.plot(ts, np.array(d32a["R2_DynamicWind"]["real"]["f0_attn_by_t_cond_s20"]) * 100, "o--", color="#63b3ed", lw=1.8, label="Real Human (DynamicWind)")
    ax.plot(ts, np.array(d32a["R2_DynamicWind"]["anime"]["f0_attn_by_t_cond_s20"]) * 100, "s--", color="#fc8181", lw=1.8, label="2D Anime (DynamicWind)")
    ax.fill_between(
        ts,
        np.array(d32a["R1_HeadTurn"]["anime"]["f0_attn_by_t_cond_s20"]) * 100,
        np.array(d32a["R1_HeadTurn"]["real"]["f0_attn_by_t_cond_s20"]) * 100,
        color="#feb2b2", alpha=0.28,
    )
    ax.set_xlabel("Generated Latent Frame Index $t$")
    ax.set_ylabel("3D Self-Attn Mass to Frame-0 Anchor (%)")
    ax.set_title("(a) Frame-0 Anchor Starvation in Wan2.2-5B\n(+40–81% CFG Push $\\rightarrow$ -30% Anchor Drop)", fontweight="bold")
    ax.set_xticks([1, 3, 5, 7, 9, 12])
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8.2, frameon=True, loc="upper right")

    # ------------------------------------------------------------------
    # Panel (b): Double Causal Localization (Denoising Time x DiT Depth)
    # ------------------------------------------------------------------
    ax = axes[1]
    labels = [
        "Base\n(Wan2.2)",
        "Mid-Late\nSteps\n(s=8..29)",
        "Early\nSteps\n(s=0..9)",
        "Late\nBlocks\n(L20–29)",
        "Mid\nBlocks\n(L10–19)",
        "Early\nBlocks\n(L0–9)",
        "OURS\nEarlyDual\n(1.0x)",
    ]
    r1_vals = [
        d34b_lyr["R1_HeadTurn"]["L_Base"]["luma_id"],
        d33["R1_HeadTurn"]["C2_MidLateAttn"]["luma_clip_mean"],
        d33["R1_HeadTurn"]["C1_EarlyAttn"]["luma_clip_mean"],
        d34b_lyr["R1_HeadTurn"]["L_Late_20_30"]["luma_id"],
        d34b_lyr["R1_HeadTurn"]["L_Mid_10_20"]["luma_id"],
        d34b_lyr["R1_HeadTurn"]["L_Early_0_10"]["luma_id"],
        d34b_lyr["R1_HeadTurn"]["L_OURS_EarlyDual"]["luma_id"],
    ]
    r2_vals = [
        d34b_lyr["R2_DynamicWind"]["L_Base"]["luma_id"],
        d33["R2_DynamicWind"]["C2_MidLateAttn"]["luma_clip_mean"],
        d33["R2_DynamicWind"]["C1_EarlyAttn"]["luma_clip_mean"],
        d34b_lyr["R2_DynamicWind"]["L_Late_20_30"]["luma_id"],
        d34b_lyr["R2_DynamicWind"]["L_Mid_10_20"]["luma_id"],
        d34b_lyr["R2_DynamicWind"]["L_Early_0_10"]["luma_id"],
        d34b_lyr["R2_DynamicWind"]["L_OURS_EarlyDual"]["luma_id"],
    ]
    x = np.arange(len(labels))
    w = 0.36
    b1 = ax.bar(x - w / 2, r1_vals, w, color="#3182ce", label="R1: HeadTurn")
    b2 = ax.bar(x + w / 2, r2_vals, w, color="#dd6b20", label="R2: DynamicWind")
    ax.axvline(2.5, color="#888888", ls=":", lw=1.0)
    ax.set_ylim(0.76, 0.98)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.6)
    ax.set_ylabel("Luma-CLIP Structural Identity")
    ax.set_title("(b) Early-Step x Early-Layer Anchor Funnel\n(Causal Window & Depth Localization)", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8.5, loc="upper left")

    # ------------------------------------------------------------------
    # Panel (c): Order-Balanced Blind Qwen3-VL-32B Pairwise Win Rates
    # ------------------------------------------------------------------
    ax = axes[2]
    cats = [
        "HeadTurn\nOverall\n(n=16)",
        "HeadTurn\nIdentity\n(n=16)",
        "Wind\nOverall\n(n=16)",
        "Wind\nMotion\n(n=16)",
    ]
    rates = [
        d34b_j["R1_HeadTurn"]["Overall_Quality"]["ours_balanced_win_rate"] * 100,
        d34b_j["R1_HeadTurn"]["Identity_and_Artifact_Free"]["ours_balanced_win_rate"] * 100,
        d34b_j["R2_DynamicWind"]["Overall_Quality"]["ours_balanced_win_rate"] * 100,
        d34b_j["R2_DynamicWind"]["Motion_Compliance"]["ours_balanced_win_rate"] * 100,
    ]
    colors = ["#2f855a", "#2f855a", "#2b6cb0", "#2b6cb0"]
    if d35_j is not None:
        cats.extend(["3 Unseen\nAnime\n(n=24)", "Real Human\nSafety Ctrl\n(n=16)"])
        rates.extend([
            d35_j["unseen_anime"]["OVERALL"]["win_rate"] * 100,
            d35_j["real_ctrl"]["OVERALL"]["win_rate"] * 100,
        ])
        colors.extend(["#805ad5", "#718096"])

    xc = np.arange(len(cats))
    bars = ax.bar(xc, rates, width=0.58, color=colors, edgecolor="#222222", linewidth=0.7)
    ax.axhline(50.0, color="#c53030", ls="--", lw=1.3, label="Parity (50%)")
    for bar, val in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.8, f"{val:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.set_ylim(0, 102)
    ax.set_xticks(xc)
    ax.set_xticklabels(cats, fontsize=8.0)
    ax.set_ylabel("Blind Qwen3-VL-32B Win Rate (%)")
    ax.set_title("(c) Zero-Training 1.0x Surgery Win Rate\n(Order-Balanced Blind Pairwise Judge)", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8.5, loc="lower right")

    plt.tight_layout()
    out_p = PLOT_DIR / "fig7_video_dit_mechanism_and_surgery.png"
    fig.savefig(out_p, bbox_inches="tight")
    print(f"Saved {out_p}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
