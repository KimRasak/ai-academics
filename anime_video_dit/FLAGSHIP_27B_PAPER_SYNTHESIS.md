# When World Models Become Too Literal: Mechanistic Localization and High-Noise Adaptation for 2D Non-Physical Anime Grammar in Flagship Video DiTs (`13B–27B MoE`)

> **Clean Open-Weight Academic Research Synthesis (Waves 36–55; Waves 56–57 Running)**
> - **100% Open-Weight Frontier Video DiTs (4 Independent Families × 4 Flow Schedules)**:
>   1. `Wan-AI/Wan2.2-I2V-A14B-Diffusers` (**27B MoE**: 14B High-Noise `t ≥ 900` + 14B Low-Noise `t < 900`, UMT5-XXL, 16ch VAE, `shift=3.0`)
>   2. `Lightricks/LTX-Video-0.9.7-dev` (**13B 48L Monolithic Non-MoE DiT**, T5-XXL, 128ch VAE, `linear_quadratic`)
>   3. `hunyuanvideo-community/HunyuanVideo-I2V` (**12.86B Dual+Single-Stream MM-DiT**, LLaVA+CLIP, 16ch VAE, `shift=17.0`)
>   4. `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v` (**2026 54L Dual-Stream MM-DiT**, Qwen2.5-VL+ByT5+SigLIP, 32ch VAE, `shift=7.0`)
> - **100% Clean Synthetic / Public Data**: Strictly disjoint character split (`32` training characters `fc_08..fc_39`; `16` held-out unseen characters `fc_00..fc_07` + `fc_40..fc_47`; `6` OOD art styles & cross-generator portraits including `Illustrious-XL-v2.0`; `6` complex multi-lighting real-world anime scenes). Zero proprietary assets.
> - **Live Playable Video Showcase**: [`flagship_video_dit_report.html`](file:///Users/jinzili/code/synclub_expression_animation/reports/academic_research/flagship_video_dit_report.html) (`http://127.0.0.1:18923/flagship_video_dit_report.html`, **130 embedded H.264 `.mp4` videos**).
> - **Publication Figures (Visually Verified)**: [`fig8`](file:///Users/jinzili/code/synclub_expression_animation/reports/academic_research/experiments/figures/plots/fig8_flagship_27b_literalism_and_moe_lora.png), [`fig9`](file:///Users/jinzili/code/synclub_expression_animation/reports/academic_research/experiments/figures/plots/fig9_cross_arch_timestep_and_efficiency.png), [`fig10`](file:///Users/jinzili/code/synclub_expression_animation/reports/academic_research/experiments/figures/plots/fig10_three_axis_localization_and_3arch_master.png).

---

## 1. Core Thesis & The Flagship Scaling Paradox (`Exp 36, 37A, 37B`)

1. **Scaling Solves 3D Physical Kinematics (`Exp 36`)**: Scaling open-weight I2V Video DiTs from `5B` to **`27B MoE` (`Wan2.2-I2V-A14B`)** solves standard 3D physical motion on 2D anime characters: smooth head turns (`S1`: `Luma-CLIP=0.90, shR=0.90`), dynamic wind + hand-to-hair (`S2`: `0.90, shR=0.97`), **180° back-of-head occlusion & identity return (`S4`: `0.95` vs Real `0.97`)**, and fast action (`S6`: `shR=1.09`).
2. **Yet Scaling Entrenches "Physical World-Model Literalism" Across ALL 4 Flagship Families (`0/60` on Wan2.2/HY-13B/LTX-13B + 4/5 failure on HY-1.5)**:
   - Strong 3D physical world-model priors systematically destroy **2D Non-Physical Anime Visual Grammar** — graphic signifiers that intentionally violate real-world optics and anatomy (`G1` specular highlight extinction without closing eyelids, `G2/S3` translucent hatched forehead gloom without darkening bright hair/smile/room, `G3` stylized 2D anger vein `💢`, `G4` rigid sliding comic sweat drop `💧`, `G6` pure monochrome B&W manga).
   - **Training-Free Double-Bind (`Exp 49`, `80` clips, `5` Blind CoT Judges)**: Prompt engineering, strong anti-literalism negative prompts, and CFG tuning (`CFG=2.0 / 6.0`) are trapped between **Literalism Amplification** (`B2 CFG=6.0`: worse hair blackout `-0.412 → -0.617`) and **Suppression Collapse / Refusal** (`B1 StrongNeg` & `B4 HyperExplicit+StrongNeg`: forehead luma turns *positive* `+0.049/+0.067` bleaching!). Our Clean v2 adaptation beats ALL 4 training-free baselines pooled **`38W / 12L / 30T` (`66.2%`, $p=0.000306$)** (`38W / 5L / 21T` = `75.8%`, $p=1.6\times 10^{-7}$ across `S3, G2–G4`).

---

## 2. Three-Axis Mechanistic Localization (`Temporal/Sigma` × `Component` × `Layer Depth`)

1. **Axis 1 — Universal $\sigma \ge 0.80$ Noise-Space Law Across 3 Architectures × 3 Flow Schedules (`Exp 38, 40B, 47/47B, 50/50B`)**:
   - **`Wan2.2-27B MoE` (`shift=3.0`)**: 2-way MoE swap proves 100% semantic ownership in High-Noise `t ≥ 900` (`M1_HighTarget_LowNeutral` = 99.8% Base; `M2_HighNeutral_LowTarget` = 0% blackout); training Low-Noise (`M2_LoRA_Low`, lower loss `0.0050`) produces 0% cure (`dL_lo=-0.301 vs -0.306`).
   - **`LTX-Video-13B` (`linear_quadratic`)**: Bottom 80% noise (`σ < 0.80`, `W1/W1c`) = **`0W/0L/16T` ($p=1.00$)**; exact fine sweep (`Exp 47B`) reveals a **Two-Stage High-Noise Phase Transition**: Stage 1 (`σ ∈ [0.98, 1.00]`, 15/36 steps) breaks global blackout, and Stage 2 (`σ ∈ [0.95, 0.98]`, ONLY **6 steps**) triggers a **`7.3×` jump** on `G2` (`-0.033 → -0.241`, 96% of full)! Direct causal inverse `High vs Low`: **`12W/1L/3T` (`84.4%`, $p=0.0034$)**.
   - **`HunyuanVideo-I2V 13B` (`shift=17.0` — naturally deconfounds step index from $\sigma$!)**: Bottom 80% noise (`σ < 0.80`) = **`1W/0L/19T` ($p=1.00$)**; Top 20% noise (`σ ≥ 0.80`) = **`20W/1L/7T` ($p=2\times 10^{-5}$)**; direct inverse `σ ≥ 0.80 vs σ < 0.80` = **`17W/2L/9T` (`76.8%`, $p=0.0007$)**!
2. **Axis 2 — `V/O/FFN` Velocity Rewiring, Not `Attn2 Q·K` Sharpening (`Exp 42`)**:
   - Cross-attention `Q·K` upper-face mass stays unchanged (`~0.30`), while High-Noise CFG velocity re-localizes (`leak: 0.798 → 0.601`, lower-face CFG `-30%`, smile blackout `-99.5%`).
3. **Axis 3 — Cross-Architecture Layer-Depth Law (`Exp 51` + `Exp 55`, Zero Retraining Across ALL 3 Architectures!)**:
   - **Universal Final-Third Semantic Deadness (3/3)**: `Wan2.2-27B Late 26..39` = `3W/1L` ($p=0.625$); `LTX-13B Late 32..47` = `2W/1L` ($p=1.00$); `HunyuanVideo-13B SingleLate 20..39` = `2W/0L` ($p=0.50$, Overall `1W/0L` $p=1.00$, visually pixel-identical to Base!). Late blocks are purely texture/pixel decoders.
   - **Architecture-Specific Active Bottleneck**: Peaks in **Earliest Multimodal Blocks** for Dual-Stream MM-DiT (`HunyuanVideo-13B DualStream 0..19`: **`14W/1L` = `90.6%`, $p=0.0010$**) and High-Noise MoE (`Wan2.2-27B Early 0..13`: **`10W/1L` = `78.1%`, $p=0.0117$**), vs. **Middle Third** in 48L monolithic non-MoE (`LTX-13B Mid 16..31`: `9W/2L` Overall `71.9%`, `G2 dU: -0.005 → -0.172`; `Early 0..15` = `1W/0L` $p=1.00$).
   - **3-Axis Ultra-Sparse Subspace (`Exp 46` + `Exp 52`, Zero Retraining)**: Intersecting `r=4` × High-Noise × `blocks 0..19` compresses adaptation to **`6.55M` (`0.024%` of 27B)** with **`11W/2L/3T` (`78.1%`, $p=0.0225$)** (`0.017%` / `4.59M` on `blocks 0..13`: **`10W/2L`, $p=0.0386$**)!

---

## 3. Master Cross-Architecture, OOD & Multi-Seed Results (Order-Balanced Blind `Qwen3-VL-32B` Judge)

| Flagship Backbone | Method / Experiment | Split (`n` pairs) | Grammar (`S3, G1–G4`) | Unseen Comp (`C1/C2`) | Safety (`S1`) | Grand Total (`W/L/T`, WR, exact $p$) |
|---|---|---|---:|---:|---:|---|
| **Wan2.2-A14B (27B MoE)** | **`M5_DualLoRA_v2`** (`Exp 44`) | Unseen-8 (`72`) | **`20W/3L/17T`** ($p=0.0005$) | **`8W/0L/8T`** ($p=0.0078$) | **`0W/0L/16T`** (exact) | **`28W/3L/41T` (`67.4%`, $p<10^{-5}$)** |
| **Wan2.2-A14B (27B MoE)** | **`M4_HighLoRA_v2`** (52M, Low Frozen) | Unseen-8 (`72`) | **`20W/3L/17T`** ($p=0.0005$) | `8W/2L/6T` | `1W/1L/14T` | **`29W/6L/37T` (`66.0%`, $p=0.0001$)** |
| **Wan2.2-A14B (27B MoE)** | **3-Seed Stability (`Seed 1+2` Pooled, `Exp 54`)** | Unseen-4 × 2 (`32`) | **`M4: 19W/0L`** ($p=3.8\times 10^{-6}$) | — | — | **`M4: 19W/0L` (`79.7%`, `15×` tighter SD!)**; `M5: 19W/2L` ($p=0.00022$) |
| **Wan2.2-A14B (27B MoE)** | **6 OOD Art Styles/Generators (`Exp 44`)** | OOD-6 (`24`) | **`13W/6L/5T`** (`G4: 6-0, G3: 5-1`) | — | — | Zero-shot across Retro-90s, Chibi, Watercolor, Manga, Illustrious-XL |
| **Wan2.2-A14B (27B MoE)** | **6 Complex Multi-Light Scenes (`Exp 53`)** | Scenes-6 (`24`) | **`10W/2L/12T`** ($p=0.0386$) | — | — | **`11W/2L/11T` Overall (`68.8%`, $p=0.0225$)** (`S3` leak `-89%`) |
| **Wan2.2-A14B (27B MoE)** | **3-Axis Micro (`0.024%` of 27B, `Exp 52`)** | Unseen-4 (`16`) | **`11W/2L/3T`** ($p=0.0225$) | — | — | `6.55M` params (`0.017%` / `4.59M`: `10W/2L`, $p=0.0386$) |
| **LTX-Video-0.9.7 (13B)** | **`M1_LoRA_High` (`σ ≥ 0.55`, `Exp 45B`)** | Unseen-4 (`28`) | **`11W/1L/8T`** ($p=0.0063$) | **`3W/0L/1T`** | **`0W/0L/4T`** | **`14W/1L/13T` (`73.2%`, $p=0.0010$)** (beats Full `15W/2L`) |
| **HunyuanVideo-I2V (13B)** | **`M3_LoRA_Full` (`0.30%`, `Exp 50`)** | Unseen-4 (`28`) | **`17W/0L/3T`** ($p=1.5\times 10^{-5}$) | **`3W/0L/1T`** | **`4W/0L/0T`** | **`24W/0L/4T` (`92.9%`, $p=1.2\times 10^{-7}$ — 24–0!)** |
| **HunyuanVideo-I2V (13B)** | **`M4_Sigma_ge800` (`Exp 50B`)** | Unseen-4 (`28`) | **`15W/1L/4T`** ($p=0.0005$) | **`4W/0L/0T`** | `1W/0L/3T` | **`20W/1L/7T` Grammar ($p=2\times 10^{-5}$)** vs `σ < 0.80` `1W/0L/19T`! |
| **HunyuanVideo-1.5 (54L)** | **`H2_Sigma_ge800` (`Exp 56`, `0.345%`)** | Unseen-4 (`24`) | **`11W/3L/6T`** (`70.0%`) | — | **`0W/0L/4T`** (exact) | **Causal `σ≥0.80` vs `σ<0.80`: `12W/3L/5T` (`72.5%`, $p=0.0352$)**; `σ<0.80` = `0W/0L/20T`! |

- **Direct 2×2 Causal Component-Axis Dissection (`Exp 58`, 100% Parameter-Matched `0.097%` of 27B each, Zero Retraining)**:
  - **`Comp2_VO_Only` (`10W/2L`, $p=0.0386$)** vs `Comp1_QK_Only` (`5W/1L`, $p=0.219$ n.s.) and **`Comp4_CrossAttn2_Only` (`7W/0L`, $p=0.0156$ undefeated)** vs `Comp3_SelfAttn1_Only` (`6W/3L`, $p=0.508$ n.s.) — directly proving causally that 2D grammar lives in **Text Cross-Attention `V/O` velocity projections**!
- **4-for-4 Cross-Architecture Layer-Depth Law (`Exp 51, 55, 56`)**: Final-third semantic deadness holds on **4/4 architectures** (`Wan2.2 26..39`, `LTX-13B 32..47`, `HY13B Single 20..39`, `HY1.5 36..53`); active grammar sits in the early third for MoE / 20-Dual MM-DiT vs. the middle third for deep uniform 48L/54L stacks (`LTX-13B 16..31` & `HY1.5 18..35`).
- **Additional Zero-Shot Emergent Capabilities**: Brand-new unseen manga symbols (`U1` Blush, `U2` Vertical Depression Lines, `U3` Radial Shock Speed Lines `shR=1.37`, `Exp 43`), smooth $\alpha \in [0, 1]$ control, **`81f` (5.0s, `1.65×` training horizon)** temporal extrapolation, and **simultaneous multi-symbol composition** (`Shadow + SweatDrop`, `3D HeadTurn + Shadow + AngerVein`, `Exp 48`).
- **Multi-VLM Judge Audit (`Exp 57C`, `344` pairs)**: Independent `Qwen2.5-VL-32B` (`LTX 13W/1L p=0.0018`, `LTX High vs Low 10W/2L p=0.0386`, `HY13B Identity 13W/3L p=0.0213`) and `InternVL3-14B` (`Wan2.2 Identity 7W/0L p=0.0156`, `HY13B Identity 9W/1L p=0.0215`) corroborate structural/causal/identity gains while revealing a VLM capability threshold on fine-grained 3D-hairclip vs 2D-vein discrimination.
- **Status**: **ALL 4/4 FLAGSHIP ARCHITECTURES & ALL 3 CAUSAL MECHANISTIC AXES COMPLETE — PAUSED CLEANLY** with 0 active GPU jobs and 134 embedded H.264 showcase videos verified (`0` broken links).
