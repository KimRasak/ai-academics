# H3/video-generation reference candidate audit

**Status:** preparation ledger, not the final proposal bibliography  
**Prepared:** 2026-09-18  
**Source pool:** `../../research-work-reproduction/refs/pool_v1.bib`

**Mechanical integrity check (2026-09-19):** all 31 candidate keys in this
ledger resolve to a unique entry in the shared 107-entry BibTeX pool. This
checks only citation-key integrity. It does **not** verify a paper's primary
source, artifact release, reported result, citation count, or relevance to a
future hypothesis.

## Purpose and rule

The final H3 proposal must cite at least 30 relevant works, but a raw count of
BibTeX entries is not evidence of coverage. This ledger selects 31 candidate
entries already present in the shared source pool and records why each could be
relevant to a future, training-based H3 or controllable-video question.

Every candidate is **unverified for final inclusion** until its primary paper,
artifact availability, exact claimed result, and relevance to the selected
hypothesis are checked. A candidate in this file is not a reproduced baseline,
does not establish a literature gap, and does not authorize a training run.

## Candidate coverage map

| cluster | count | role in a future proposal |
|---|---:|---|
| diffusion/post-training and preference optimization | 10 | training mechanism, reward and reward-hacking controls |
| joint audio-video generation | 11 | nearest multimodal methods and synchronization objectives |
| control and video-generation framing | 4 | controlled-generation problem definition and exclusions |
| evaluation and public data | 6 | protocol, evaluator-risk and public-data grounding |
| **total unique candidates** | **31** | preparation threshold only |

## 1. Diffusion/post-training and preference optimization

| BibTeX key | year | role to verify before final citation |
|---|---:|---|
| `fan2023dpok` | 2023 | early RL fine-tuning framing for diffusion models |
| `clark2024directly` | 2024 | differentiable-reward fine-tuning baseline |
| `uehara2024understanding` | 2024 | review of RL-based diffusion fine-tuning and assumptions |
| `zhang2024large` | 2024 | scale/compute considerations for RL diffusion training |
| `uehara2024fine` | 2024 | control-theoretic continuous-time fine-tuning formulation |
| `deng2024prdp` | 2024 | reward-difference prediction alternative |
| `xiao2024diffppo` | 2024 | PPO-style diffusion post-training baseline |
| `yang2024using` | 2024 | human-feedback route without a separate reward model |
| `meng2025identity` | 2025 | video-specific RL/post-training candidate; verify task transferability |
| `duan2026mitigating` | 2026 | reward-hacking risk/control candidate |

## 2. Joint audio-video generation

| BibTeX key | year | role to verify before final citation |
|---|---:|---|
| `kurmi2021collaborative` | 2021 | earlier joint audio-video generation baseline |
| `liu2024syncflow` | 2024 | text-conditioned temporal audio-video alignment candidate |
| `anon2025mmaudio` | 2025 | video-to-audio training comparison, not automatically joint generation |
| `liu2026javisdit` | 2026 | joint audio-video diffusion transformer candidate |
| `liu2026javisdita` | 2026 | follow-up joint modeling/optimization candidate |
| `team2026mova` | 2026 | scalable synchronized audio-video generation candidate |
| `huang2025jova` | 2025 | unified multimodal-learning candidate |
| `qiang2026mm` | 2026 | controllable AV generation candidate; inspect voice-cloning scope |
| `song2026syncphony` | 2026 | synchronized audio-to-video generation candidate |
| `weng2026audio` | 2026 | multi-stream temporal-control candidate |
| `wang2026audiogen` | 2026 | unified synchronized audio/speech/song generation candidate |

## 3. Controlled video-generation framing

| BibTeX key | year | role to verify before final citation |
|---|---:|---|
| `xing2024survey` | 2024 | video-diffusion taxonomy and benchmark vocabulary |
| `ma2025controllable` | 2025 | control taxonomy, conditions and open issues |
| `ma2025video` | 2025 | broader video-diffusion gaps and protocol survey |
| `xi2025ctrlvdiff` | 2025 | nearest unified multimodal control candidate |

## 4. Evaluation and public-data grounding

| BibTeX key | year | role to verify before final citation |
|---|---:|---|
| `hua2026vabench` | 2026 | AV benchmark candidate; local protocol is currently blocked by missing join/implementation artifacts |
| `wang2020mead` | 2020 | public AV dataset reference only; talking-face scope is not the preferred main task |
| `li2024aves` | 2024 | public temporal AV-emotion dataset candidate; assess general-video suitability |
| `goncalves2024versatile` | 2024 | AV emotion-recognition evaluation candidate; not a generator benchmark by itself |
| `zhang2024audio` | 2024 | audio-synchronized visual-animation task/evaluation candidate |
| `mo2024text` | 2024 | video-synchronized audio generation comparison candidate |

## Decision gates before conversion to final references

For every selected final citation, record: canonical title/authors/year,
primary URL or DOI, task, public code/weights/data/evaluator state, exact
claim to be used, and which proposal paragraph it supports. Remove candidates
that are talking-head-only, lack relevance to the eventual H3 hypothesis, or
cannot be checked from a primary source.

The final bibliography must add MiniMax-H3, MM-Diffusion, Sci-VBench and any
actual reproduced baseline from their primary sources even when their entries
are not in this current pool. It must not claim that 31 candidates equal 31
verified final citations.

The already checked primary-source entry points for MiniMax-H3, Sci-VBench,
MM-Diffusion, T2V-Turbo, VADER, VideoDPO, MotionCtrl, VBench and VABench are
kept separately in `CORE_PRIMARY_SOURCE_LEDGER.md`. That ledger is a source and
artifact-availability audit, not a substitute for final bibliography review.

The local BibTeX metadata and per-entry approval state are now separately
tracked in `FINAL_CITATION_VERIFICATION_QUEUE.md`. All 31 remain pending
primary-source/release/relevance review unless that file explicitly says
otherwise.
