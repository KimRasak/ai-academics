# Avatars That Keep Feeling: Diagnosing and Measuring Emotional Collapse in Audio-Driven Portrait Avatars

**Working paper package** — targets: ACM MM 2027 (full) / ICME 2027 / WACV 2027 workshop
Status: skeleton v1 (Round 98). Experiments: M1/M2 done; M3 redo in flight (V-A2); M4 baseline anchored; human eval designed.

---

## 1. Title candidates
1. **Avatars That Keep Feeling: An SER↔FER Consistency Benchmark Reveals Emotional Collapse in Audio-Driven Portraits**
2. Born Numb: Audio-Driven Avatars Ignore the Emotion in Your Voice
3. Faces Without Feelings: Quantifying and Mitigating Emotional Collapse in Talking-Head Generation

## 2. Abstract (v0)
Audio-driven portrait avatars have reached photorealistic lip-sync, yet we show they are **emotionally numb**: across 158 generated videos spanning 8 emotion categories, 4 seeds, and two census arms, a state-of-the-art open-source avatar (EchoMimic V1) produces neutral-dominant faces (mean expressiveness 0.079, flat across ground-truth emotions 0.073–0.092) even when the driving audio is clearly emotional (SER expressiveness 0.76–0.80). We introduce an **SER↔FER consistency evaluation pipeline** (speech-emotion recognition judged against facial-emotion recognition with shared 4-class space), plus identity (ArcFace) and lip-sync proxies, and use it to (i) quantify the collapse, (ii) show that naive emotion-faithful SFT lifts expressiveness 10.7× but sacrifices identity and lip-sync, and (iii) analyze why conditioning-based fixes fail. We release the census protocol, judges, and a 2-hour simulated-livestream evaluation matrix with probe windows sampled across the timeline. [Results sections finalize after V-A2 eval.]

## 3. Contributions (draft)
1. **Diagnosis**: first quantitative, seed-resolved evidence of emotional collapse in open-source audio-driven avatars (M2 census: Arm1 158 videos × 4 seeds; Arm2 temporal drift over 146s × 4 seeds; 2h session probes).
2. **Benchmark/pipeline**: SER(audio)↔FER(face) agreement metric (JS-based), expressiveness, identity, sync — with positive controls (FER2013 real faces: expr 0.92–0.99; anchors) validating judges.
3. **Evaluation matrix**: 2-hour simulated livestream protocol with speech-active probe windows every 10 min; audio-level SER drift curve (1800 windows) shows the *source* remains expressive and discriminative (0.76→0.80) while the *rendered face* stays flat (0.065±0.010) — the avatar is the bottleneck.
4. **Training study + methodology findings**: emotion-faithful LoRA SFT variants (pure SFT / +emotion embedding / +arousal weighting / light-touch / RAFT best-of-K) with verifiable rewards; we document three reproduction pitfalls (missing fine-tuned denoising weights, missing add_noise, audio-feature sample-rate coupling) that silently invalidate conclusions — a cautionary methodology contribution for reproducible talking-head training.

## 4. Method
### 4.1 Base avatar
EchoMimic V1 (forked UNet3D + reference attention + face locator + whisper audio features). Weights: sd-image-variations + motion module + denoising_unet.pth + reference_unet.pth. Ref image a.png (1342²).

### 4.2 Judges (all public HF)
- FER: trpakov/vit-face-expression (7-class ViT) + dima806/facial_emotions_image_detection → pooled, mapped to 4-class {neutral, happy, angry, sad} (shared with SER label space; calm→neutral)
- SER: superb/wav2vec2-base-superb-er (4-class)
- Identity: facenet-pytorch InceptionResnetV1 (vggface2) cosine vs reference frame
- Lip-sync proxy: correlation between mouth-region frame-difference energy and audio RMS envelope (SyncNet proper left as future work; joonson/syncnet_python env cost documented)

### 4.3 Metrics
- expressiveness = 1 − P(neutral) (FER pooled)
- agreement = exp(−JS)(SER(audio) ‖ FER(face)) over shared 4-class space
- GT-match = argmax FER == GT emotion (where GT exists)
- identity = ArcFace cosine; sync_proxy as above
- collapse curve: per-window expressiveness across session time

### 4.4 Positive controls
- FER2013 real test faces (n=700): expressive classes 0.92–0.99, neutral 0.27; 4-class acc 48.3% → judges valid
- Anchor control (smiling reference b.png + angry/happy chunks): expr 0.142/0.183 → static reference anchoring does NOT lift the floor (rules out "prompt/ref effect" explanation)
- SER on the 2h session: expr_by_gt angry 0.94 / neutral 0.45 etc. → SER tracks GT

## 5. Experiments & Results (fill as they finalize)
### 5.1 M2 census (DONE)
- Arm1: 40 RAVDESS chunks × 4 seeds: mean expr 0.079; flat across 8 GT emotions (0.073–0.092); seed std 0.012
- Arm2: 4 seeds × 146s continuous: mean 0.073–0.078; drift slope ±1e-4/s ≈ 0 → static poverty, not oscillation
- Verdict B (static emotional poverty); A (transient) disproven; C (judge bias) excluded by positive controls

### 5.2 M3 training study (REDO in flight — bug-fixed pipeline)
Historical (invalidated by pipeline bugs, kept as methodology appendix): naive SFT expr 0.84/identity 0.25. Corrected pipeline results: V-A2 (r=32 α=32 lr 5e-5, 8k steps, fp32, proper add_noise objective + denoising weights) — [PENDING]

### 5.3 M4 2h livestream matrix (baseline DONE; variants pending)
- Audio-level: 1800 × 4s windows, expr thirds 0.763/0.780/0.797 (no collapse in source)
- Video probes (12 × 8s speech-active windows @10-min spacing): baseline expr 0.065±0.010, arc 0.902±0.004, sync 0.366
- [V-A2 probes pending]

### 5.4 Human eval (designed, M4_HUMAN_EVAL_DESIGN.md)
24 clips × 3 versions × 5 Likert items, n=10, H1–H4 hypotheses.

## 6. Reproducibility appendix (the three pitfalls)
1. **Silent weight omission**: from_pretrained_2d(base, motion_module) must be followed by load_state_dict(denoising_unet.pth, strict=False); omitting it produces a plausible-looking but identity-broken generator (arc 0.25 vs 0.90) — easy to miss because outputs still look face-like.
2. **Objective bug**: predicting noise from CLEAN latents (missing scheduler.add_noise) yields constant MSE≈1.0; training appears to work (loss ~1.0 → stable) but learns output-attenuation, not denoising.
3. **Audio feature path coupling**: whisper feature extraction shells out to ffmpeg; missing PATH silently fails per-sample → dataset silently empties (loss 0.0000, steps fly).
Plus ops notes: pkill -f self-match kills your own ssh; unquoted heredocs expand inner variables; GPU co-tenancy causes transient CUBLAS failures — distinguish from divergence by loss trajectory.

## 7. Related work buckets (to fill from refs/)
- Audio-driven talking heads: SadTalker, Wav2Lip, DiffPoseTalk, EMO, Hallo/Hallo2, EchoMimic, AniPortrait, VASA-1
- Emotion in talking heads: EmoTalk, FACE, EAT, EmoGen, MEAD dataset, RAVDESS, CREMA-D
- SER: wav2vec2 SUPERB-ER, emotion2vec
- FER: ViT face expression, FER2013, AffectNet, MAE-FER
- Evaluation: SyncNet, ArcFace/FaceNet, FID/LMD for talking heads; FR metrics
- RLVR/verifiable rewards: RLHF, RAFT, reward hacking lit
- Livestream/digital human systems: commercial agents, TOQA, stream avatars

## 8. Bibliography seed (aim ≥30; extract from refs/merged.jsonl + canonical)
[TO EXPORT: use this project's `paper/refs_seed.bib` and manually verified canonical references:
EchoMimic (2024), Hallo (2024), Hallo2 (2024), EMO (2024), SadTalker (CVPR23), Wav2Lip (CVPR20),
DiffPoseTalk (SIGGRAPH24), AniPortrait (2024), VASA-1 (2024), MakeItTalk (SIGGRAPH20),
FaceFormer (CVPR22), CodeTalker (CVPR23), EmoTalk (ICCV23), FACE (SIGGRAPH Asia23),
EAT (2023), MEAD (CVPR20), RAVDESS (PLOS ONE 2018), CREMA-D (IEEE TAC 2014),
FER2013 (2013), AffectNet (TAC 2019), vit-face-expression (HF), superb-er (HF/SLT21),
wav2vec 2.0 (NeurIPS20), emotion2vec (2024), ArcFace (CVPR19), FaceNet (CVPR15),
SyncNet (ECCV16/amaz), RAFT (2023), DPO (NeurIPS23), RLVR/t tecnologia (2024-26 surveys),
LLM-judge (NeurIPS23), diffusion models survey, LoRA (ICLR22), PEFT, DDIM (ICLR21),
SD image variations, whisper (ICML23), MER2023, MELD (ACL19] → target 35+]

## 9. Venue mapping
- ACM MM 2027 (full, 8pp+ref): benchmark + training study + matrix — main target. Deadline ~Mar 2027.
- ICME 2027 (6pp): census + matrix + human eval. Deadline ~Dec 2026/Jan 2027.
- WACV 2027 workshop (reproducibility/multimedia): the methodology-pitfalls appendix fits a reproducibility track.

## 10. Open TODOs to camera-ready
- [ ] V-A2 eval (12 probes + 30 audios) via merged-weights original pipeline
- [ ] Optional V-B2 (emotion embedding) on fixed base for the 3-variant ablation
- [ ] 2h matrix variant rows + comparison table
- [ ] Human eval execution (n=10)
- [ ] BibTeX export ≥30
- [ ] Figures: census curve, drift curves, 2h thirds bar, probe timeline, matrix heatmap
