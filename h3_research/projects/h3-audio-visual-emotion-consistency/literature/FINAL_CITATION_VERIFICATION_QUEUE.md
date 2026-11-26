# Final-citation verification queue

**Updated:** 2026-09-19  
**Source of the fields below:** local `../../research-work-reproduction/refs/pool_v1.bib`.

## Access-boundary update (2026-09-20)

Scholar Crawler is CAPTCHA-blocked and must not be bypassed. A direct request
to the recorded arXiv/GitHub primary URLs was also blocked by this execution
environment's outbound-host policy. Neither condition establishes that a
candidate source is absent, inaccessible to humans, canonical, or suitable for
the proposal. It only means this automated session cannot promote any pending
row to `approved` from a live fetch.

The queue therefore remains a **verification plan**, not a bibliography count:
the 31 candidates do not satisfy the final ≥30-citation requirement. Promotion
requires separately retained primary-source evidence (for example a locally
saved paper/release checkout or human-verified canonical source), plus the
claim, relevance, and artifact-state fields specified below. Existing local
primary-source audits remain usable only for the claims explicitly bounded in
`CORE_PRIMARY_SOURCE_LEDGER.md`.

This queue upgrades the earlier candidate-key check into a concrete per-paper
review checklist. “Metadata present” means only that a local BibTeX entry
contains a title, year and URL. It does **not** mean that the URL was fetched in
this review, that the venue/version is canonical, or that the work has public
code, weights, data or an executable evaluator. No entry is approved for the
final bibliography yet.

## Required review fields for every final citation

Before marking an entry `approved`, record the canonical author list and
version/venue, inspect its primary source, identify the exact claim it supports,
assess relevance to the selected hypothesis, and record the public
code/weights/data/evaluator state. Papers that only cover talking heads or a
different modality must be removed unless the final hypothesis demonstrably
needs them.

| key | local title / year | local primary URL | intended role | metadata | primary source + release/relevance review |
|---|---|---|---|---|---|
| `fan2023dpok` | DPOK: RL fine-tuning text-to-image diffusion / 2023 | NeurIPS | image diffusion reward-training background | present | pending |
| `clark2024directly` | Directly fine-tuning diffusion models on differentiable rewards / 2024 | ICLR | differentiable-reward training background | present | pending |
| `uehara2024understanding` | Understanding RL-based fine-tuning of diffusion models / 2024 | arXiv:2407.13734 | review of assumptions/risks | present | pending |
| `zhang2024large` | Large-scale reinforcement learning for diffusion models / 2024 | ECCV/Springer | scale/compute context | present | pending |
| `uehara2024fine` | Fine-tuning continuous-time diffusion as entropy-regularized control / 2024 | arXiv:2402.15194 | control formulation | present | pending |
| `deng2024prdp` | PRDP / 2024 | IEEE Xplore 10656824 | reward-difference baseline | present | pending |
| `xiao2024diffppo` | DiffPPO / 2024 | IEEE Xplore 10987802 | PPO-style baseline | present | pending |
| `yang2024using` | Using human feedback without a reward model / 2024 | IEEE Xplore 10656897 | feedback alternative | present | pending |
| `meng2025identity` | Identity-GRPO / 2025 | arXiv:2510.14256 | video RL candidate; likely out-of-scope identity task | present | pending |
| `duan2026mitigating` | Mitigating reward hacking in RLHF / 2026 | arXiv:2602.10623 | reward-hacking risk | present | pending |
| `kurmi2021collaborative` | Collaborative learning to generate audio-video jointly / 2021 | IEEE Xplore 9413802 | early joint AV baseline | present | pending |
| `liu2024syncflow` | SyncFlow / 2024 | arXiv:2412.15220 | text-conditioned joint AV alignment | present | pending |
| `anon2025mmaudio` | MMAudio / 2025 | IEEE Xplore 11092848 | video-to-audio comparator, not automatically joint generation | present | pending |
| `liu2026javisdit` | JavisDiT / 2026 | ICLR | joint AV DiT baseline | present | pending |
| `liu2026javisdita` | JavisDiT++ / 2026 | ICLR | joint AV modeling/optimization | present | pending |
| `team2026mova` | MOVA / 2026 | arXiv:2602.08794 | scalable synchronized AV candidate | present | pending |
| `huang2025jova` | JOVA / 2025 | arXiv:2512.13677 | unified AV learning candidate | present | pending |
| `qiang2026mm` | MM-Sonate / 2026 | arXiv:2601.01568 | controllable AV; voice-cloning scope check required | present | pending |
| `song2026syncphony` | Syncphony / 2026 | ICLR | audio-to-video synchronization | present | pending |
| `weng2026audio` | Audio-sync video generation / 2026 | NeurIPS proceedings URL | multi-stream temporal control | present | pending |
| `wang2026audiogen` | AudioGen-Omni / 2026 | IEEE Xplore 11461581 | video-synchronized audio comparator | present | pending |
| `xing2024survey` | A survey on video diffusion models / 2024 | ACM DOI 10.1145/3696415 | taxonomy/vocabulary | present | pending |
| `ma2025controllable` | Controllable video generation: a survey / 2025 | arXiv:2507.16869 | control framing | present | pending |
| `ma2025video` | Video diffusion generation: review and open problems / 2025 | Springer DOI 10.1007/s10462-025-11331-6 | gaps/evaluation framing | present | pending |
| `xi2025ctrlvdiff` | CtrlVDiff / 2025 | arXiv:2511.21129 | unified multimodal control candidate | present | pending |
| `hua2026vabench` | VABench / 2026 | CVPR 2026 | AV benchmark; local evaluator path separately blocked | present | primary paper checked; release completeness still blocked |
| `wang2020mead` | MEAD / 2020 | ECCV/Springer | public AV-emotion data; talking-face scope warning | present | pending |
| `li2024aves` | AVES / 2024 | IEEE Xplore 10632787 | temporal AV-emotion data candidate | present | pending |
| `goncalves2024versatile` | Versatile AV learning for emotion recognition / 2024 | IEEE Xplore 10609546 | recognition/evaluation, not generator benchmark | present | pending |
| `zhang2024audio` | Audio-synchronized visual animation / 2024 | ECCV/Springer | AV animation comparator | present | pending |
| `mo2024text` | Text-to-audio generation synchronized with videos / 2024 | arXiv:2403.07938 | synchronized-audio comparator | present | pending |

## Non-candidate core citations that require the same review

The final bibliography must separately review and include, where relevant,
MiniMax-H3, Sci-VBench, MM-Diffusion, VBench and every baseline actually used
in the final protocol. Their primary entry points and local evidence are in
`CORE_PRIMARY_SOURCE_LEDGER.md`; that ledger confirms neither a final citation
choice nor a training claim.

## Current conclusion

There are enough *candidate records* to design a bibliography review, but not
yet 30 approved, hypothesis-relevant citations. The final proposal may not use
this queue to claim literature coverage until each selected entry has the
required review fields above.
