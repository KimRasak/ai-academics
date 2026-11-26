# MiniMax H3 release-boundary audit

**Date:** 2026-09-18  
**Primary evidence:** retained official `MiniMaxAI/MiniMax-H3/README.md`,
SHA-256 `819db2fa4bef63feb077b2d425974e640deb3c453b6a12844467374d7ca39ae9`

## What is locally reproducible

The official release describes two task-specific, BF16 H3-Base checkpoints:

| local checkpoint family | official tasks | input conditions | output |
|---|---|---|---|
| `FL2VA/` | T2VA and FL2VA | text, optionally first/last frame | video plus audio |
| `Ref2VA/` | Ref2VA | text plus reference images, video and/or audio | video plus audio |

The retained model index identifies `MiniMaxH3Pipeline` and the two partitions
`fl2va: FL2VA` and `ref2va: Ref2VA`. The README supplies four-GPU SGLang
commands for the Base variants, so a local 768p Base deployment is a legitimate
checkpoint-level reproduction target once four ownership-cleared GPUs exist.

## What is not locally reproducible from this release

The same official README says H3-Context-IR is a multi-stage, hosted
preprocessing/orchestration system and is **not included** in the open-source
release. It interprets multimodal inputs and serializes a structured Context
Intermediate Representation for H3-Base. H3-Regenerate-2K is also described
as an API-assisted full workflow component.

Consequences:

1. Local Base outputs cannot be labelled a reproduction of the complete H3
   system, H3-Context-IR, or the official full 2K workflow.
2. Any comparison must say whether it uses (a) frozen human-written/context-IR
   prompts with local H3-Base, (b) an official hosted Context-IR/API call, or
   (c) a new independently designed context preprocessor.
3. A new context processor is a new method, not an H3 full-system
   reproduction. It needs public data, a fixed baseline, and an evaluation
   protocol before training.
4. The three official 768p cases are suitable only for Base deployment
   verification. They do not establish paper metrics or training-process
   reproduction.

## Implication for a future training proposal

The viable H3-related training question must be scoped to an observable,
publicly evaluable boundary: e.g., a training method operating on a frozen
local H3-Base interface or an independently reproducible multimodal context
representation. It cannot claim to modify, train, or reproduce the unavailable
hosted H3-Context-IR. Before choosing such a question, the project still needs:

- a successful local Base R0 deployment with retained media and runtime logs;
- a public benchmark whose media-to-annotation mapping and evaluator are
  executable (VABench currently does not meet this condition);
- a checkpoint-level open baseline result with its exact evaluator protocol;
- a public training dataset, pre-registered metrics and a falsifiable
  hypothesis.

## Current evidence state

The local checkpoint identity and official deployment commands are C1 artifact
evidence. A read-only gate later found GPUs 4--7 and ports 30010/30011 free,
but it is not launch authorization: explicit human ownership confirmation is
still required. No local H3 Base output has been generated under the reproduction
card, so local generation remains pending. The separately retained three
`sora-2` service outputs are explicitly excluded from all H3 claims.
