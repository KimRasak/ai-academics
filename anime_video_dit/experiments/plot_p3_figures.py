#!/usr/bin/env python3
"""
Generate crisp publication-grade matplotlib figures directly from verified H100 JSON outputs
(ZERO synthetic/hardcoded fake curves — reads real exp5c, exp11, exp14a, exp16 JSONs!).

Produces:
  1. fig1_7model_scaling_law.png         (2x2 Factorial Scaling Law: Inverse Scaling & Clean-Face Paranoia)
  2. fig2_layerwise_mechanistic.png      (10-Layer Pre-Erasure UNSEEN AUROC + Final-20% Logit-Lens Overwrite)
  3. fig3_causal_double_dissociation.png (Attention vs MLP Override Circuit Divergence Across 5 VLMs)
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 12.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 11.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.28,
        "grid.linestyle": "--",
        "legend.frameon": True,
        "legend.framealpha": 0.92,
    })


def plot_fig1_scaling(out_root: Path, fig_dir: Path):
    # Read real exp5c 7-model JSONs
    files = sorted(glob.glob(str(out_root / "exp5c_scaling" / "*.json")))
    pts = {}
    for f in files:
        d = json.load(open(f))
        m = d.get("model", Path(f).stem)
        dir_d = d.get("direct", {})
        if not dir_d:
            continue
        clean = dir_d.get("clean_acc", dir_d.get("clean"))
        conf = dir_d.get("conflict_acc", dir_d.get("conflicts"))
        macro = dir_d.get("macro7", dir_d.get("macro"))
        if clean is not None and conf is not None:
            pts[m] = (clean * 100.0, conf * 100.0, (macro or 0) * 100.0)

    if len(pts) < 4:
        print(f"[fig1] Found only {len(pts)} exp5c files; keys={list(pts.keys())}")
        return

    fig, ax = plt.subplots(figsize=(7.6, 5.8), dpi=220)
    # Ideal corner shading
    ax.axhspan(75, 105, xmin=0.75, xmax=1.0, color="#d8f3dc", alpha=0.45, zorder=0)
    ax.text(87.5, 96.5, "Ideal Zone\n(No Override +\nNo Paranoia)", ha="center", va="top", fontsize=9.5, color="#1b4332", fontweight="bold")
    ax.axvspan(35, 72, ymin=0, ymax=1, color="#ffe5ec", alpha=0.30, zorder=0)
    ax.text(44, 12, "Clean-Face Paranoia\nRegime (32B)", ha="left", va="bottom", fontsize=9.5, color="#9d0208", fontweight="bold")
    ax.axhspan(0, 38, xmin=0.52, xmax=1.0, color="#fff3cd", alpha=0.35, zorder=0)
    ax.text(98.5, 6, "Biological-Smile Override\nRegime (7B–14B)", ha="right", va="bottom", fontsize=9.5, color="#7f5539", fontweight="bold")

    colors = {
        "InternVL3-8B-hf": "#1d3557",
        "InternVL3-14B-hf": "#457b9d",
        "llava-onevision-qwen2-7b-ov-hf": "#2a9d8f",
        "Qwen2.5-VL-7B-Instruct": "#e76f51",
        "Qwen2.5-VL-32B-Instruct": "#d62828",
        "Qwen3-VL-8B-Instruct": "#7209b7",
        "Qwen3-VL-32B-Instruct": "#3a0ca3",
    }
    labels = {
        "InternVL3-8B-hf": "InternVL3-8B",
        "InternVL3-14B-hf": "InternVL3-14B",
        "llava-onevision-qwen2-7b-ov-hf": "LLaVA-OV-7B",
        "Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL-7B",
        "Qwen2.5-VL-32B-Instruct": "Qwen2.5-VL-32B",
        "Qwen3-VL-8B-Instruct": "Qwen3-VL-8B",
        "Qwen3-VL-32B-Instruct": "Qwen3-VL-32B",
    }
    offsets = {
        "InternVL3-8B-hf": (-12, 9),
        "InternVL3-14B-hf": (-28, -17),
        "llava-onevision-qwen2-7b-ov-hf": (-88, 3),
        "Qwen2.5-VL-7B-Instruct": (-28, -16),
        "Qwen2.5-VL-32B-Instruct": (-18, 10),
        "Qwen3-VL-8B-Instruct": (-16, 10),
        "Qwen3-VL-32B-Instruct": (8, -6),
    }

    for k, (cx, cy, _mc) in pts.items():
        sz = 190 if "32B" in k else (140 if "14B" in k else 100)
        ax.scatter(cx, cy, s=sz, color=colors.get(k, "#333"), edgecolors="white", linewidth=1.6, zorder=4)
        dx, dy = offsets.get(k, (6, 6))
        ax.annotate(labels.get(k, k), (cx, cy), xytext=(dx, dy), textcoords="offset points", fontsize=9.8, fontweight="bold", color=colors.get(k, "#222"))

    # Draw scaling arrows (same family small -> large)
    pairs = [
        ("InternVL3-8B-hf", "InternVL3-14B-hf", "Inverse\n(-3.5%)", "#1d3557"),
        ("Qwen2.5-VL-7B-Instruct", "Qwen2.5-VL-32B-Instruct", "4.5x scale:\nParanoia!", "#d62828"),
        ("Qwen3-VL-8B-Instruct", "Qwen3-VL-32B-Instruct", "4x scale:\nParanoia!", "#3a0ca3"),
    ]
    for a, b, txt, col in pairs:
        if a in pts and b in pts:
            ax.annotate("", xy=(pts[b][0], pts[b][1]), xytext=(pts[a][0], pts[a][1]),
                        arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.45", lw=1.8, color=col, ls="--", shrinkA=9, shrinkB=9), zorder=3)

    ax.set_xlim(38, 104)
    ax.set_ylim(0, 102)
    ax.set_xlabel("Clean Face Accuracy (%)  [orig + placebo hairpin]")
    ax.set_ylabel("Non-Biological Conflict Accuracy (%)  [5 anime affect cues]")
    ax.set_title("2×2 Factorial Scaling Law Across 7 Open VLMs (3 Vision Encoders)\nScaling Trades Biological Override for Clean-Face Paranoia")
    fig.tight_layout()
    out = fig_dir / "fig1_7model_scaling_law.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(f"Saved {out}")


def plot_fig2_mechanistic(out_root: Path, fig_dir: Path):
    files = sorted(glob.glob(str(out_root / "exp11_layerwise" / "*.json")))
    if not files:
        print("[fig2] No exp11_layerwise JSONs found")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.8), dpi=220)
    styles = {
        "InternVL3-8B-hf": ("InternVL3-8B (InternViT)", "#1d3557", "o"),
        "llava-onevision-qwen2-7b-ov-hf": ("LLaVA-OV-7B (SigLIP)", "#2a9d8f", "s"),
        "Qwen2.5-VL-7B-Instruct": ("Qwen2.5-VL-7B (QwenViT)", "#e76f51", "^"),
        "Qwen3-VL-32B-Instruct": ("Qwen3-VL-32B (64L)", "#7209b7", "D"),
    }

    for f in files:
        d = json.load(open(f))
        m = d["model"]
        lbl, col, mk = styles.get(m, (m, "#333", "o"))
        rows = sorted(d["curve"], key=lambda x: x["layer"])
        nL = d.get("n_layers", max(r["layer"] for r in rows) + 1)
        xs = [r["layer"] / float(nL - 1) for r in rows]
        auroc_u = [r["auroc_unseen_nonbio"] for r in rows]
        pA_u = [r["logitlens_pA_unseen_nonbio"] for r in rows]

        ax1.plot(xs, auroc_u, marker=mk, lw=2.3, ms=6, color=col, label=lbl)
        ax2.plot(xs, pA_u, marker=mk, lw=2.3, ms=6, color=col, label=lbl)

    ax1.axvspan(0.54, 0.88, color="#d8f3dc", alpha=0.35, label="Pre-Erasure Peak Window")
    ax1.axvspan(0.88, 1.02, color="#ffe5ec", alpha=0.45, label="Late-Layer Erasure Zone")
    ax1.axhline(0.5, color="gray", ls=":", lw=1.2)
    ax1.set_xlim(-0.02, 1.03)
    ax1.set_ylim(0.42, 1.02)
    ax1.set_xlabel("Normalized Layer Depth (l / L)")
    ax1.set_ylabel("Linear Probe AUROC on UNSEEN Non-Bio Cues")
    ax1.set_title("(a) Mid-Layer Non-Bio Representation & Late Erasure")
    ax1.legend(loc="lower right", fontsize=8.8)

    ax2.axvspan(0.76, 1.02, color="#ffe5ec", alpha=0.45, label="Late Bio-Smile Overwrite")
    ax2.axhline(0.5, color="gray", ls=":", lw=1.2)
    ax2.set_xlim(-0.02, 1.03)
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_xlabel("Normalized Layer Depth (l / L)")
    ax2.set_ylabel("Logit-Lens P(A: Cheerful) on UNSEEN Non-Bio Cues")
    ax2.set_title("(b) Final-20% Biological-Smile Residual Overwrite")
    ax2.legend(loc="upper left", fontsize=8.8)

    fig.tight_layout()
    out = fig_dir / "fig2_layerwise_mechanistic.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(f"Saved {out}")


def plot_fig3_dissociation(out_root: Path, fig_dir: Path):
    files = sorted(glob.glob(str(out_root / "exp16_1pass_surgery" / "*.json")))
    if not files:
        print("[fig3] No exp16 JSONs found")
        return

    order = [
        ("InternVL3-8B-hf", "InternVL3-8B"),
        ("InternVL3-14B-hf", "InternVL3-14B"),
        ("llava-onevision-qwen2-7b-ov-hf", "LLaVA-OV-7B"),
        ("Qwen2.5-VL-7B-Instruct", "Qwen2.5-VL-7B"),
        ("Qwen3-VL-32B-Instruct", "Qwen3-VL-32B"),
    ]
    data = {}
    for f in files:
        d = json.load(open(f))
        data[d["model"]] = d

    names, d_attn, d_mlp, base_nat, surg_nat = [], [], [], [], []
    for key, nice in order:
        if key not in data:
            continue
        t6 = data[key]["test6_7var"]
        p0 = t6["baseline_none"]["mean_pA_conflicts"]
        pa = t6["uncond_ATTN_only_a1.0"]["mean_pA_conflicts"]
        pm = t6["uncond_MLP_only_a1.0"]["mean_pA_conflicts"]
        names.append(nice)
        d_attn.append((p0 - pa) * 100.0)
        d_mlp.append((p0 - pm) * 100.0)
        nat = data[key]["native_32imgs"]
        base_nat.append(nat["baseline_none"]["macro"] * 100.0)
        surg_nat.append(nat["gated_1PASS_BOTH_a1.5"]["macro"] * 100.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.8), dpi=220)
    x = np.arange(len(names))
    w = 0.35

    b1 = ax1.bar(x - w / 2, d_attn, w, label="Ablate Late Self-Attn ONLY", color="#264653")
    b2 = ax1.bar(x + w / 2, d_mlp, w, label="Ablate Late MLP ONLY", color="#e76f51")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15, ha="right")
    ax1.set_ylabel("Causal Drop in False-Cheerful P(A) (% pts)")
    ax1.set_title("(a) Architectural Double-Dissociation: Attn vs. MLP")
    ax1.legend(loc="upper left", fontsize=9.5)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.1f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    b3 = ax2.bar(x - w / 2, base_nat, w, label="Direct Baseline (1.0× Pass)", color="#8d99ae")
    b4 = ax2.bar(x + w / 2, surg_nat, w, label="1-Pass Gated Surgery (1.0× Pass)", color="#2a9d8f")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=15, ha="right")
    ax2.set_ylim(0, 95)
    ax2.set_ylabel("Macro Accuracy on 32 Native Anime Images (%)")
    ax2.set_title("(b) Zero-Shot Synth→Native Transfer (1.0× Pass Cost)")
    ax2.legend(loc="upper right", fontsize=9.5)
    for bar in list(b3) + list(b4):
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 1.2, f"{h:.1f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    fig.tight_layout()
    out = fig_dir / "fig3_causal_double_dissociation.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(f"Saved {out}")


def plot_fig4_mouth_copy(out_root: Path, fig_dir: Path):
    files = sorted(glob.glob(str(out_root / "exp21b_mouth_copy" / "*.json")))
    if not files:
        print("[fig4] No exp21b JSONs found")
        return

    styles = {
        "InternVL3-8B-hf": ("InternVL3-8B", "#1d3557", "o"),
        "Qwen2.5-VL-7B-Instruct": ("Qwen2.5-VL-7B", "#e76f51", "^"),
        "llava-onevision-qwen2-7b-ov-hf": ("LLaVA-OV-7B", "#2a9d8f", "s"),
        "Qwen3-VL-32B-Instruct": ("Qwen3-VL-32B", "#7209b7", "D"),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.9), dpi=220)

    # Panel (a): L0 Spatial Controls (Delta Conf5 P(A) relative to baseline!)
    m_names, d_bg, d_eye, d_mouth = [], [], [], []
    for f in files:
        d = json.load(open(f))
        m = d["model"]
        lbl, col, mk = styles.get(m, (m, "#333", "o"))
        s = d["summary"]
        p0 = s["baseline"]["conf5_pA"]
        m_names.append(lbl)
        d_bg.append((s["ablate_BG_K_L0"]["conf5_pA"] - p0) * 100.0)
        d_eye.append((s["ablate_EYES_L0"]["conf5_pA"] - p0) * 100.0)
        d_mouth.append((s["ablate_MOUTH_L0"]["conf5_pA"] - p0) * 100.0)

        # Panel (b): Layer-by-layer mouth copy curve (normalized depth vs Delta Conf5 P(A))
        nL = d["n_layers"]
        xs, ys = [], []
        for li in d["sweep_layers"]:
            key = f"ablate_MOUTH_L{li}"
            if key in s:
                xs.append(li / float(nL - 1))
                ys.append((s[key]["conf5_pA"] - p0) * 100.0)
        ax2.plot(xs, ys, marker=mk, lw=2.4, ms=6.2, color=col, label=lbl)

    x = np.arange(len(m_names))
    w = 0.25
    b1 = ax1.bar(x - w, d_bg, w, label="Ablate Equal-K BG (L0 Control)", color="#8d99ae")
    b2 = ax1.bar(x, d_eye, w, label="Ablate Eye Tokens (L0 Control)", color="#457b9d")
    b3 = ax1.bar(x + w, d_mouth, w, label="Ablate K Mouth Tokens (L0)", color="#e63946")
    ax1.axhline(0, color="black", lw=1.0)
    ax1.set_xticks(x)
    ax1.set_xticklabels(m_names, rotation=12, ha="right")
    ax1.set_ylim(-35, 10)
    ax1.set_ylabel("Δ False-Cheerful P(A) on Conflicts (% pts)")
    ax1.set_title("(a) L0 Spatial Control: Mouth vs. Eyes vs. Equal-K BG", fontsize=11.5)
    ax1.legend(loc="lower right", fontsize=8.4)
    for bar in list(b1) + list(b2) + list(b3):
        h = bar.get_height()
        va = "bottom" if h >= 0 else "top"
        off = 0.6 if h >= 0 else -0.8
        ax1.text(bar.get_x() + bar.get_width() / 2, h + off, f"{h:+.1f}", ha="center", va=va, fontsize=8.0, fontweight="bold")

    ax2.axvspan(0.35, 0.62, color="#fff3cd", alpha=0.55, label="Mouth-Copy Sigmoid Window")
    ax2.axhline(0, color="black", lw=1.0, ls=":")
    ax2.set_xlim(-0.02, 0.88)
    ax2.set_xlabel("Ablation Layer Depth (l / L)")
    ax2.set_ylabel("Δ False-Cheerful P(A) from Mouth Ablation (% pts)")
    ax2.set_title("(b) Exact Layer Window of Smile Copy-Out", fontsize=11.5)
    ax2.legend(loc="lower right", fontsize=8.8)

    fig.tight_layout()
    out = fig_dir / "fig4_mouth_copy_spatial_causal.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(f"Saved {out}")


def plot_fig5_scale48_and_trilingual(out_root: Path, fig_dir: Path):
    f17c = sorted(glob.glob(str(out_root / "exp17c_scale48_eval" / "*.json")))
    f20b = sorted(glob.glob(str(out_root / "exp20b_trilingual" / "*.json")))
    if not f17c or not f20b:
        print(f"[fig5] Missing inputs: 17c={len(f17c)} 20b={len(f20b)}")
        return

    order6 = [
        ("InternVL3-8B-hf", "InternVL3-8B"),
        ("InternVL3-14B-hf", "InternVL3-14B"),
        ("llava-onevision-qwen2-7b-ov-hf", "LLaVA-OV-7B"),
        ("Qwen2.5-VL-7B-Instruct", "Qwen2.5-7B"),
        ("Qwen2.5-VL-32B-Instruct", "Qwen2.5-32B"),
        ("Qwen3-VL-32B-Instruct", "Qwen3-32B"),
    ]
    d17 = {json.load(open(f))["model"]: json.load(open(f)) for f in f17c}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.9), dpi=220)
    names, dir_m, dir_s, p1_m, p1_s, p2_m, p2_s = [], [], [], [], [], [], []
    for k, nice in order6:
        if k not in d17:
            continue
        cv = d17[k]["cv5_scale48"]
        names.append(nice)
        dir_m.append(cv["direct"]["macro9_mean"] * 100); dir_s.append(cv["direct"]["macro9_std"] * 100)
        p1_m.append(cv["ours_1pass_unified"]["macro9_mean"] * 100); p1_s.append(cv["ours_1pass_unified"]["macro9_std"] * 100)
        p2_m.append(cv["ours_2pass_debiased"]["macro9_mean"] * 100); p2_s.append(cv["ours_2pass_debiased"]["macro9_std"] * 100)

    x = np.arange(len(names))
    w = 0.26
    ax1.bar(x - w, dir_m, w, yerr=dir_s, capsize=3, label="Direct Baseline (1.0×)", color="#8d99ae")
    ax1.bar(x, p1_m, w, yerr=p1_s, capsize=3, label="OURS 1-Pass Unified (1.0×)", color="#2a9d8f")
    ax1.bar(x + w, p2_m, w, yerr=p2_s, capsize=3, label="OURS 2-Pass Debiased", color="#264653")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=16, ha="right")
    ax1.set_ylim(0, 74)
    ax1.set_ylabel("5-Fold CV Macro-9 Accuracy (%) [n=48 Chars, 432 Imgs]")
    ax1.set_title("(a) Scale-48 5-Fold Leave-4-Symbols-Out CV (mean ± std)")
    ax1.legend(loc="upper left", fontsize=8.8)

    # Panel (b): Trilingual EN / JA / ZH Direct vs Zero-Shot EN-Gate 1-Pass Surgery (avg over 3 7B/8B + LLaVA highlight)
    d20 = {json.load(open(f))["model"]: json.load(open(f)) for f in f20b}
    langs = ["EN", "JA", "ZH"]
    lang_lbls = ["English (EN)", "Japanese (JA)", "Chinese (ZH)"]
    m_sub = [("llava-onevision-qwen2-7b-ov-hf", "LLaVA-OV-7B", "#2a9d8f"),
             ("InternVL3-8B-hf", "InternVL3-8B", "#1d3557"),
             ("Qwen2.5-VL-7B-Instruct", "Qwen2.5-7B", "#e76f51")]
    lx = np.arange(len(langs))
    bw = 0.13
    for mi, (mk, mlbl, mcol) in enumerate(m_sub):
        if mk not in d20:
            continue
        off = (mi - 1) * 0.28
        vd = [d20[mk]["results"][L]["direct"]["macro9"] * 100 for L in langs]
        v1 = [d20[mk]["results"][L]["ours_1pass_xling_from_EN"]["macro9"] * 100 for L in langs]
        ax2.bar(lx + off - bw / 2, vd, bw, color=mcol, alpha=0.38, edgecolor=mcol, hatch="//", label=f"{mlbl} Direct")
        ax2.bar(lx + off + bw / 2, v1, bw, color=mcol, alpha=0.95, label=f"{mlbl} EN→Lang 1P")

    ax2.set_xticks(lx)
    ax2.set_xticklabels(lang_lbls)
    ax2.set_ylim(0, 62)
    ax2.set_ylabel("Macro-9 Accuracy on 216 Unseen Images (%)")
    ax2.set_title("(b) Trilingual Override & Zero-Shot EN→JA/ZH Gate Transfer")
    ax2.legend(loc="upper right", ncol=2, fontsize=7.8)

    fig.tight_layout()
    out = fig_dir / "fig5_scale48_and_trilingual.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(f"Saved {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-root", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs"))
    ap.add_argument("--fig-dir", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/plots"))
    args = ap.parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    _style()
    plot_fig1_scaling(args.outputs_root, args.fig_dir)
    plot_fig2_mechanistic(args.outputs_root, args.fig_dir)
    plot_fig3_dissociation(args.outputs_root, args.fig_dir)
    plot_fig4_mouth_copy(args.outputs_root, args.fig_dir)
    plot_fig5_scale48_and_trilingual(args.outputs_root, args.fig_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
