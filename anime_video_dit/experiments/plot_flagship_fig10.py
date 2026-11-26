#!/usr/bin/env python3
"""
Publication Figure 10 (Ultimate 3-Architecture & 3-Axis Mechanistic Master Synthesis):
  (A) Universal sigma >= 0.80 Noise-Space Law Across ALL THREE Independent Flagship Video DiT Architectures
      (Wan2.2-27B MoE shift=3.0 | LTX-Video-13B Non-MoE linear_quadratic | HunyuanVideo-I2V 13B MM-DiT shift=17.0)
  (B) 40-Layer Depth Gradient & Combined 3-Axis Micro-Subspace Compression (Exp 51 & Exp 52: down to 0.017% of 27B!)
  (C) Training-Free Baselines Double-Bind vs Weight-Level High-Noise Adaptation (Exp 49: 38W/12L Pooled, p=0.0003)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/plots/fig10_three_axis_localization_and_3arch_master.png"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 10.8,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 220,
    })

    fig = plt.figure(figsize=(18.8, 5.5), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.22, 1.18, 1.15], wspace=0.34, left=0.045, right=0.985, top=0.83, bottom=0.18)

    # -------------------------------------------------------------------------
    # PANEL A: Universal sigma >= 0.80 Noise-Space Law Across 3 Architectures
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    archs = [
        "Wan2.2-27B MoE\n(14B+14B, shift=3.0)",
        "LTX-Video-13B\n(48L Non-MoE, lin-quad)",
        "HunyuanVideo-13B\n(60L MM-DiT, shift=17.0)",
    ]
    x = np.arange(len(archs)) * 1.18
    w = 0.31

    low_vals = [50.0, 50.0, 52.5]        # sigma < 0.80 (Low-Only): 0% cure, 0W/0L/16T, 1W/0L/19T (ALL NULL!)
    high_vals = [71.2, 75.0, 85.0]       # sigma >= 0.80 / High-Only: p=0.0005, p=0.0063, p=0.0005!
    full_vals = [71.2, 73.2, 90.0]       # Full LoRA Grammar WR: 71.2%, 73.2%, 90.0% (24W/0L overall!)

    b1 = ax1.bar(x - w, low_vals, width=w, color="#CBD5E1", edgecolor="#475569", linewidth=0.8, label="Low-Noise Only (σ < 0.80) — Null")
    b2 = ax1.bar(x, high_vals, width=w, color="#2563EB", edgecolor="#1E3A8A", linewidth=0.8, label="High-Noise Only (σ ≥ 0.80) — Cure")
    b3 = ax1.bar(x + w, full_vals, width=w, color="#0D9488", edgecolor="#134E4A", linewidth=0.8, label="Full Adaptation (All Steps)")

    ax1.axhline(50.0, color="#64748B", linestyle="--", linewidth=1.1)

    low_tags = ["0%\nNull", "0W/0L\np=1.0", "1W/0L\np=1.0"]
    hi_tags = ["20W/3L\np=.0005", "11W/1L\np=.0063", "15W/1L\np=.0005"]
    full_tags = ["20W/3L\np=.0005", "15W/2L\np=.0023", "16W/0L\np<.0001"]

    for bars, tags, col, dx in [(b1, low_tags, "#475569", -0.02), (b2, hi_tags, "#1E40AF", -0.02), (b3, full_tags, "#115E59", 0.04)]:
        for bar, tg in zip(bars, tags):
            ax1.text(bar.get_x() + bar.get_width() / 2.0 + dx, bar.get_height() + 1.4, tg,
                     ha="center", va="bottom", fontsize=6.7, fontweight="bold", color=col)

    ax1.set_xticks(x)
    ax1.set_xticklabels(archs, fontsize=8.3)
    ax1.set_ylabel("All-Grammar-5 Blind Win Rate vs Base (%)")
    ax1.set_ylim(38, 106)
    ax1.set_title("(A) Universal σ ≥ 0.80 Law Across 3 Architectures")
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.legend(loc="upper left", fontsize=7.6, frameon=True, facecolor="white", framealpha=0.92)

    # -------------------------------------------------------------------------
    # PANEL B: 40-Layer Depth Gradient & 3-Axis 0.017% Micro-Subspace (Exp 51 & 52)
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    depth_rows = [
        ("Blocks 0–13\n(r=16, 0.068%)", 78.1, "10W/1L\np=0.0117", "#2563EB"),
        ("Blocks 13–26\n(r=16, 0.068%)", 68.8, "7W/1L\np=0.070", "#60A5FA"),
        ("Blocks 26–39\n(r=16, 0.068%)", 56.2, "3W/1L\nn.s.", "#CBD5E1"),
        ("3-Axis 0–19\n(r=4, 0.024%!)", 78.1, "11W/2L\np=0.0225", "#7C3AED"),
        ("3-Axis 0–13\n(r=4, 0.017%!)", 75.0, "10W/2L\np=0.0386", "#9333EA"),
    ]
    xd = np.arange(len(depth_rows))
    bars_d = ax2.bar(xd, [r[1] for r in depth_rows], width=0.58, color=[r[3] for r in depth_rows],
                     edgecolor="#1E293B", linewidth=0.75, alpha=0.92)
    ax2.axvline(2.5, color="#64748B", linestyle=":", linewidth=1.3)
    ax2.axhline(50.0, color="#64748B", linestyle="--", linewidth=1.1)

    ax2.text(1.0, 94.5, "40-Layer Depth Slice (Zero Retrain)", ha="center", va="top",
             fontsize=7.9, fontweight="bold", color="#1E40AF",
             bbox=dict(boxstyle="round,pad=0.20", fc="#EFF6FF", ec="#93C5FD"))
    ax2.text(3.5, 94.5, "3-Axis Micro-LoRA", ha="center", va="top",
             fontsize=7.9, fontweight="bold", color="#6B21A8",
             bbox=dict(boxstyle="round,pad=0.20", fc="#FAF5FF", ec="#D8B4FE"))

    for bar, r in zip(bars_d, depth_rows):
        ax2.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 1.3, r[2],
                 ha="center", va="bottom", fontsize=7.6, fontweight="bold", color="#0F172A")

    ax2.set_xticks(xd)
    ax2.set_xticklabels([r[0] for r in depth_rows], fontsize=7.9)
    ax2.set_ylabel("Blind Grammar Win Rate vs Base 27B (%)")
    ax2.set_ylim(38, 99)
    ax2.set_title("(B) 40-Layer Depth Law & 0.017% Subspace")
    ax2.grid(axis="y", linestyle="--", alpha=0.35)

    # -------------------------------------------------------------------------
    # PANEL C: Training-Free Baselines vs OURS Clean v2 (Exp 49)
    # -------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    tf_rows = [
        ("OURS vs High CFG 6.0\n(Literalism Amplified)", 72.5, "11W/2L/7T\np=0.0225", "#DC2626"),
        ("OURS vs Low CFG 2.0\n(Compliance Collapse)", 70.0, "10W/2L/8T\np=0.0386", "#EA580C"),
        ("OURS vs Strong NegPrompt\n(Bleach & Refusal)", 65.0, "10W/4L/6T\nG3: 4–0", "#D97706"),
        ("OURS vs HyperExplicit+Neg\n(Suppression Collapse)", 57.5, "7W/4L/9T\nS3/G4: 6–0", "#CA8A04"),
        ("POOLED All 4 Baselines\n(All 5 Regimes, n=80)", 66.2, "38W/12L/30T\np=0.0003", "#2563EB"),
        ("POOLED (S3, G2–G4)\n(Non-G1, n=64)", 75.8, "38W/5L/21T\np=1.6e-7", "#059669"),
    ]
    tf_rows = tf_rows[::-1]
    yt = np.arange(len(tf_rows))
    bars_t = ax3.barh(yt, [r[1] for r in tf_rows], height=0.60, color=[r[3] for r in tf_rows],
                      edgecolor="#1E293B", linewidth=0.7, alpha=0.90)
    ax3.axvline(50.0, color="#334155", linestyle="--", linewidth=1.2)
    ax3.set_yticks(yt)
    ax3.set_yticklabels([r[0] for r in tf_rows], fontsize=7.9)
    ax3.set_xlabel("OURS Clean v2 Blind Win Rate vs Training-Free (%)")
    ax3.set_xlim(40, 102)
    ax3.set_title("(C) Weight Adaptation vs Training-Free Baselines")
    ax3.grid(axis="x", linestyle="--", alpha=0.35)

    for bar, r in zip(bars_t, tf_rows):
        ax3.text(bar.get_width() + 1.0, bar.get_y() + bar.get_height() / 2.0, r[2],
                 va="center", ha="left", fontsize=7.5, fontweight="bold", color="#0F172A")

    fig.suptitle(
        "Three-Architecture Proof of the Universal σ ≥ 0.80 Noise-Space Law, 40-Layer Depth Localization, and Training-Free Baseline Dominance",
        fontsize=12.4, fontweight="bold", y=0.955, color="#0F172A",
    )
    fig.savefig(args.out, dpi=220)
    plt.close(fig)
    print(f"Saved publication Figure 10 -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
