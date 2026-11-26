#!/usr/bin/env python3
"""
Publication Figure 9:
  (A) Monolithic Non-MoE 13B DiT (`LTX-Video-0.9.7`) Critical Timestep & Causal Inverse Curve (`Exp 47` / `Exp 47B`)
  (B) Cross-Architecture & Efficiency Blind `Qwen3-VL-32B` Win Rates (`Exp 44, 45B, 46, 47`)
  (C) Parameter-Budget Frontier on 27B MoE (`0.048% Rank-4` vs `0.19% High-Only` vs `0.39% Dual` vs `Low-Only`)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s47", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp47_ltx13b_timestep_curve/summary_exp47.json"))
    ap.add_argument("--s47b", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp47_ltx13b_timestep_curve/summary_exp47b_full.json"))
    ap.add_argument("--out", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/plots/fig9_cross_arch_timestep_and_efficiency.png"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    OUT = args.out
    p47b, p47 = args.s47b, args.s47

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 11.0,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 220,
    })

    # Load Exp 47 (or 47B full if available)
    s47 = json.loads((p47b if p47b.exists() else p47).read_text(encoding="utf-8"))

    fig = plt.figure(figsize=(18.6, 5.4), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.12, 1.32, 1.04], wspace=0.52, left=0.045, right=0.985, top=0.84, bottom=0.17)

    # -------------------------------------------------------------------------
    # PANEL A: Monolithic Non-MoE 13B LTX-Video Critical Timestep Localization
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    if p47b.exists():
        keys = [
            ("W0_Base", "Base\n(0/36)"),
            ("W1_LowOnly_lt550", "Low\n<0.55\n(6/36)"),
            ("W1c_LowOnly_lt800", "Low\n<0.80\n(10/36)"),
            ("W1b_LowOnly_lt950", "Low\n<0.95\n(15/36)"),
            ("W2a_High_ge980", "High\n>=0.98\n(15/36)"),
            ("W2b_High_ge950", "High\n>=0.95\n(21/36)"),
            ("W2_High_ge800", "High\n>=0.80\n(26/36)"),
            ("W4_High_ge550", "High\n>=0.55\n(30/36)"),
            ("W5_Full", "Full\n(36/36)"),
        ]
    else:
        keys = [
            ("W0_Base", "Base\n(0/36)"),
            ("W1_LowOnly_lt550", "Low-Only\nσ<0.55\n(6/36)"),
            ("W2_High_ge800", "High\nσ≥0.80\n(26/36)"),
            ("W3_High_ge650", "High\nσ≥0.65\n(29/36)"),
            ("W4_High_ge550", "High\nσ≥0.55\n(30/36)"),
            ("W5_Full", "Full\nAll σ\n(36/36)"),
        ]

    xs = np.arange(len(keys))
    s3_low_cure = [-s47["S3_NonBioAffect"][k]["dL_low"] for k, _ in keys]
    g2_up_effect = [-s47["G2_ForeheadShadow_Clean"][k]["dL_up"] for k, _ in keys]

    ax1.plot(xs, g2_up_effect, marker="o", linewidth=2.4, markersize=7, color="#2563EB", label="G2 Shadow |ΔL_upper| (↑)")
    ax1.plot(xs, s3_low_cure, marker="s", linewidth=2.4, markersize=7, color="#DC2626", linestyle="--", label="S3 Smile Leak |ΔL_lower| (↓)")

    split_x = 3.5 if p47b.exists() else 1.5
    ax1.axvspan(-0.4, split_x, color="#FEE2E2", alpha=0.45, zorder=0)
    ax1.axvspan(split_x, len(keys) - 0.6, color="#DBEAFE", alpha=0.35, zorder=0)
    ax1.axvline(split_x, color="#64748B", linestyle=":", linewidth=1.3)

    ax1.text(split_x * 0.5 - 0.2, 0.292, "Low-Noise Only\n0% Cure (0W/0L/16T)", ha="center", va="top",
             fontsize=8.0, fontweight="bold", color="#991B1B",
             bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#FCA5A5", alpha=0.92))
    ax1.text((split_x + len(keys) - 0.6) * 0.5, 0.292, "High-Noise Activated\n~100% Cure (12W/1L, p=0.0034)", ha="center", va="top",
             fontsize=8.0, fontweight="bold", color="#1E40AF",
             bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#93C5FD", alpha=0.92))

    ax1.set_xticks(xs)
    ax1.set_xticklabels([lbl for _, lbl in keys], fontsize=8.0)
    ax1.set_ylabel("Luma Magnitude |ΔL| (n=4 Unseen Chars)")
    ax1.set_ylim(-0.015, 0.312)
    ax1.set_title("(A) Non-MoE 13B LTX-Video Critical Timestep Law")
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.legend(loc="center right", bbox_to_anchor=(0.98, 0.50), fontsize=7.9, frameon=True, facecolor="white", framealpha=0.94)

    # -------------------------------------------------------------------------
    # PANEL B: Cross-Architecture Blind High-Res CoT Qwen3-VL-32B Win Rates
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    rows = [
        ("LTX-13B High vs LowOnly\n(Causal Inverse, n=16)", 84.4, "12W/1L/3T (p=0.0034)", "#0D9488"),
        ("LTX-13B High-Only vs Base\n(Non-MoE Unseen-4, n=28)", 73.2, "14W/1L/13T (p=0.0010)", "#0284C7"),
        ("Wan2.2-27B Dual v2\n(Unseen-8 Grammar, n=40)", 71.2, "20W/3L/17T (p=0.0005)", "#2563EB"),
        ("Wan2.2-27B Dual v2\n(Unseen-8 Grand, n=72)", 67.4, "28W/3L/41T (p<1e-5)", "#1D4ED8"),
        ("Wan2.2-27B High-Only v2\n(52M Low Frozen, n=72)", 66.0, "29W/6L/37T (p=0.0001)", "#4F46E5"),
        ("Wan2.2-27B Rank-4\n(13.1M = 0.048%, n=36)", 65.3, "12W/1L/23T (p=0.0034)", "#7C3AED"),
        ("Wan2.2-27B Dual v2 OOD\n(6 Styles/Gen, n=24)", 64.6, "13W/6L/5T (G4:6-0)", "#9333EA"),
        ("LTX-13B Low-Only vs Base\n(Causal Control, n=16)", 50.0, "0W/0L/16T (Null)", "#9CA3AF"),
    ]
    rows = rows[::-1]
    ys = np.arange(len(rows))
    vals = [r[1] for r in rows]
    cols = [r[3] for r in rows]

    bars = ax2.barh(ys, vals, height=0.62, color=cols, edgecolor="#1E293B", linewidth=0.6, alpha=0.92)
    ax2.axvline(50.0, color="#334155", linestyle="--", linewidth=1.2)
    ax2.set_yticks(ys)
    ax2.set_yticklabels([r[0] for r in rows], fontsize=8.1)
    ax2.set_xlabel("Balanced Blind Win Rate (%) — Qwen3-VL-32B (320px CoT, AB+BA)")
    ax2.set_xlim(40, 112)
    ax2.set_title("(B) Order-Balanced Blind VLM Judge Across Architectures")
    ax2.grid(axis="x", linestyle="--", alpha=0.35)

    for b, r in zip(bars, rows):
        ax2.text(b.get_width() + 1.0, b.get_y() + b.get_height() / 2.0, r[2],
                 va="center", ha="left", fontsize=7.8, fontweight="bold", color="#0F172A")

    # -------------------------------------------------------------------------
    # PANEL C: Parameter-Budget & Expert Frontier on Wan2.2-27B MoE
    # -------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    pts = [
        ("Base 27B (0M)", 0.0, 50.0, "#64748B", "o", (10, -4)),
        ("Low-Noise Only\n(52.4M, 0.19%)", 52.4, 50.0, "#DC2626", "X", (-12, -28)),
        ("High-Noise Rank-4\n(13.1M, 0.048%!)", 13.1, 67.5, "#7C3AED", "D", (12, -10)),
        ("High-Noise Rank-16\n(52.4M, 0.19%)", 52.4, 71.2, "#2563EB", "s", (-10, 12)),
        ("Dual-MoE Rank-16\n(104.8M, 0.39%)", 104.8, 71.2, "#059669", "^", (-72, -32)),
    ]
    ax3.plot([0.0, 13.1, 52.4, 104.8], [50.0, 67.5, 71.2, 71.2], color="#2563EB", linewidth=2.0, alpha=0.55, zorder=1)
    for lbl, x_m, y_wr, c, mk, (dx, dy) in pts:
        ax3.scatter([x_m], [y_wr], s=110, color=c, marker=mk, edgecolor="#0F172A", linewidth=0.9, zorder=3)
        ax3.annotate(lbl, (x_m, y_wr), xytext=(dx, dy), textcoords="offset points",
                     fontsize=7.9, fontweight="bold", color=c,
                     bbox=dict(boxstyle="round,pad=0.20", fc="white", ec=c, alpha=0.90))

    ax3.axhline(50.0, color="#94A3B8", linestyle=":", linewidth=1.1)
    ax3.set_xlabel("Trainable LoRA Parameters (Millions, out of 27,000M)")
    ax3.set_ylabel("All-Grammar-5 Blind Win Rate (%)")
    ax3.set_xlim(-6, 118)
    ax3.set_ylim(43, 77.5)
    ax3.set_title("(C) 27B MoE Parameter-Budget Frontier")
    ax3.grid(True, linestyle="--", alpha=0.35)

    fig.suptitle(
        "Universal High-Noise Semantic Grammar Localization Across Flagship Video DiT Architectures (13B Non-MoE & 27B MoE)",
        fontsize=12.8, fontweight="bold", y=0.96, color="#0F172A",
    )
    fig.savefig(OUT, dpi=220)
    plt.close(fig)
    print(f"Saved publication Figure 9 -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
