# MiniMax H3 Research Proposal

**Status:** evidence-building draft, not a submission-ready proposal  
**Updated:** 2026-09-20  
**Active protocol:** reproduction first; public datasets only for exploratory
training

General video-generation evidence, benchmark lineages and claim-level
credibility states are maintained in
`../research-work-reproduction/planning/REPRODUCTION_BLUEPRINT.md`. This project
consumes that evidence but does not own the generic reproduction corpus.
The current candidate-reference audit is maintained in
`literature/REFERENCE_CANDIDATE_AUDIT.md`; it is a traceable preparation list,
not a completed final bibliography.
The verified entry points for the nearest H3, evaluator and reproducible-video
baselines are separated in `literature/CORE_PRIMARY_SOURCE_LEDGER.md`; source
availability there must not be confused with claim-level reproduction.
The H3 release boundary (local Base checkpoints versus unavailable hosted
Context-IR/2K components) is recorded in `H3_RELEASE_BOUNDARY.md` and must
constrain every later research claim.
The requirement-by-requirement decision record is
`PROPOSAL_EVIDENCE_GATE.md`; it is authoritative when this draft and an older
operational note differ.

## Current position

The project does not yet have enough public, reproducible evidence to lock a
new method. Previous self-designed training experiments and their artifacts
have been removed. They do not determine the final proposal.

The active task is to establish three evidence layers:

1. reproduce the official MiniMax H3 768p T2VA, FL2VA and Ref2VA examples;
2. distinguish reproduced public evaluator results on released H3 media from
   newly generated H3 benchmark results;
3. reproduce open video-generation training methods on public datasets,
   including the authors' metrics and training.

The preferred research problem is general video generation rather than digital
humans. Motion, temporal consistency, camera/object control, prompt following,
efficient generation and post-training are in scope. Talking heads, lip sync
and speaker identity are deferred unless their methods transfer directly to
general video generation.

The open release does not expose the complete official system: H3-Context-IR is
hosted and the full 2K workflow depends on official APIs. A local 768p Base
deployment is therefore a checkpoint/deployment reproduction, not a complete
H3-system reproduction. Any future training method must state whether it acts
on frozen local H3-Base inputs/outputs or introduces an independently designed
context processor; the latter is a new method requiring its own public-data
baseline and evaluation evidence.

Only after these layers are complete will this file be rewritten as the final
research proposal.

## Reproduction targets

### R0 — Official H3 deployment reproduction

- Source: `MiniMaxAI/MiniMax-H3`.
- Reproduce the official T2VA, FL2VA and Ref2VA 768p examples.
- Preserve exact official inputs, prompts, software revisions, output media
  and runtime statistics.
- This verifies deployment fidelity; it is not a scientific contribution.
- The authorized four-H100 FL2VA attempt reached `MiniMaxH3Pipeline` and loaded
  its text encoder, but twice OOMed while materializing the 13-shard DiT FSDP
  transformer before opening port 30010, submitting an official request, or
  writing output. The installed H3 `memory` mode was then audited: its tested
  FSDP configuration offloads auxiliary text/VAE components but leaves the DiT
  resident at this failure stage. It is not a supported remedy for this peak.
  The exact command must not be retried; no local H3 media or H3 benchmark
  result exists. See the shared four-H100 failure and memory-mode audit.

### R1 — Public benchmark evidence for H3

- **Completed evaluator lane:** Sci-VBench evaluator commit
  `e8cb3988631d0c22d31c02fc5b1cc3693bfb7031` re-scored all 150 released H3
  `testmini` videos. VT median absolute discrepancy is `6.15e-06`; Dynamic
  Degree matches 149/150. This validates the public evaluator/media path, not
  local H3 generation.
- **Pending generation lane:** VABench remains a candidate only after its code,
  public test-data manifest and license are independently verified. Start with
  an official subset to validate the adapter; do not call a subset score a full
  benchmark result.
- Any future H3 benchmark must report repeated seeds because the existing
  Sci-VBench/Wan study finds within-prompt VT variability of 0.0509 for H3 and
  0.0411 for Wan under a 12-seed public protocol.

### R2 — Open model and metric reproduction

- Primary target: MM-Diffusion, CVPR 2023.
- Official public resources: code, Landscape/AIST++-crop download recipes,
  pretrained checkpoints and FVD/KVD/FAD evaluation scripts.
- First reproduce checkpoint inference and the reported metric protocol.
- The Landscape DDPM-1000/DDIM-25 checkpoint lane has two terminal C3 runs.
  Their direct 64×64 FVD/KVD/FAD×10k values are
  `249.0004/15.4862/10.7107` and `231.8951/14.3733/10.9198`, versus the paper's
  six-run `117.20/5.78/10.72`. This is neither a six-run reproduction nor a
  basis for training; its full provenance and interpretation are in the shared
  reproduction blueprint.
- A separately preregistered run 3/6 is currently collecting the same 2,048
  public-checkpoint samples on GPUs 4--7. Its independent closeout watcher
  evaluates only after exact original/SR/WAV counts and a full decode audit;
  it remains checkpoint evidence only regardless of its value.
- Attempt training reproduction only after dataset license, checksums,
  preprocessing and evaluation have been verified.
- AV-DiT remains a literature comparison until an official code and checkpoint
  release is verified.

### R3 — Optional modern inference comparison

- OmniForcing has a public release.
- OmniForcing training requires at least 8×H200, so only checkpoint inference
  is currently admissible.
- Talking-head-specific systems are not part of the main comparison matrix.

## Evidence gate before proposing a method

A training-based H3 idea may be proposed only after:

- one H3 official reproduction is complete and archived;
- one public H3 benchmark result is complete;
- one open baseline's official metrics are reproduced or the discrepancy is
  explained with evidence;
- the candidate problem remains visible on public data;
- the nearest published method does not already solve it;
- a representative public training dataset and fixed evaluation protocol are
  identified.

Private SynClub assets may be used for visualization, debugging and smoke
tests only. They must not be used for exploratory training or headline
results.

## Required final proposal structure

The final revision must contain:

1. research question and falsifiable hypothesis;
2. literature gap supported by at least 30 relevant references;
3. reproduced baselines and public datasets;
4. training method and why H3 specifically benefits;
5. fixed comparison matrix and evaluation metrics;
6. expected cost, one-month schedule and stopping conditions;
7. risks involving data licensing, evaluator validity and reproducibility.

Until the evidence gate is met, this document intentionally does not claim a
new method.
