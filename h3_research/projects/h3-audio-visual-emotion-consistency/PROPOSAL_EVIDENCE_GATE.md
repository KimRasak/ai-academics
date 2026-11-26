# H3 proposal evidence gate

**Updated:** 2026-09-20  
**Purpose:** prevent a training proposal from being selected merely because a
model checkpoint, a GPU window, or a candidate bibliography exists. This is a
decision record for the final proposal, not a claim that any row is complete.

## Decision rule

A candidate training idea is eligible for a final proposal only when every
required row below has a named, inspectable evidence artifact. A green
deployment gate is not a training gate. A C2 evaluator replay is not a local
generation result; a C3 checkpoint evaluation is not a full training
reproduction; only C4 closes a published training-process claim.

| Required decision input | Current evidence | State | What is still required before a final training idea |
|---|---|---|---|
| Local H3-Base boundary | Official release audit identifies local FL2VA/Ref2VA Base only; Context-IR and 2K workflow are hosted/API-bound | bounded | Do not describe any local result as full H3; any new context processor is a separate method |
| H3 request identity | T2VA, FL2VA and Ref2VA official scripts, inputs, reference media, routing and local wrappers hash-verified | C1 | Preserve launch/server/request/output provenance for at least one local Base case only after the deployment capacity gate closes |
| H3 deployment execution | Authorized GPUs 4--7 ran official FL2VA. It entered `MiniMaxH3Pipeline` and loaded its text encoder, then twice OOMed during 13-shard DiT FSDP materialization before listener/request/output | C1 failure evidence | Stop identical 4×80GB retries. The inspected H3-tested `memory` mode offloads text/VAE but not the failing DiT stage, so it is not a remedy. Require evidence for a supported DiT-peak-reducing configuration or a different hardware/sharding scope; never preempt another user's process |
| H3 benchmark on locally generated media | Sci-VBench C2 replay covers 150 author-released videos, not locally generated media; four-80GB H3 service never reached generation | incomplete | First close the H3 deployment capacity gate. Then run a locally generated H3 case with an executable public evaluator and explicitly matched protocol; use repeated seeds for a quantitative conclusion |
| Formal audio-video benchmark | VABench code/materials audited, but image--annotation join and DNSMOS implementation are absent | blocked | Author-supplied pairing manifest and official DNSMOS wrapper; never infer pairing/order or substitute a metric |
| Reproducible open AV baseline | MM-Diffusion has two complete C3 released-checkpoint 2,048-sample runs on official retained media | partial | Treat individual values only; the public upstream source/permission/preprocess chain remains insufficient for C4 training reproduction |
| Candidate training data | SynClub assets are explicitly not eligible; no public H3-aligned training dataset has passed data-card/license/split/provenance review | blocked | Select a public, inspectable dataset with raw media, labels, license, fixed splits and cache provenance; preregister its role before training |
| Strong comparison | T2V-Turbo/VADER/MotionCtrl/VideoDPO each have separate resource or protocol gaps; no H3-specific trained baseline is yet valid | incomplete | Fix a candidate task and identify an exact public baseline that can be rerun under the same dataset/evaluator protocol |
| Literature gap | 31 candidate keys resolve mechanically in a 107-entry pool; operational core sources are separately mapped; primary-source relevance/resource facts are still candidate-level. Scholar CAPTCHA and outbound-host policy currently prevent live automated promotion | incomplete | Verify ≥30 final citations individually from retained or human-verified primary sources: canonical source, claim used, relevance, release state and proposal paragraph |
| Falsifiable training plan | No hypothesis, objective, data, controls or stopping rule has been selected | intentionally unset | Specify these only after the preceding data, baseline and metric rows close |

## Current permitted work

- Read-only source/resource auditing, literature verification and provenance
  maintenance.
- H3 frozen deployment only after a supported configuration has evidence that
  it fits the assigned hardware or an explicitly different hardware/sharding
  scope is available; it is checkpoint/deployment evidence, not training.
- Repeated checkpoint evaluations only when separately preregistered and when
  they answer a defined protocol question.

## Explicitly prohibited shortcuts

- Treating the cleared GPU gate as permission to launch, or stopping another
  user's process.
- Training with SynClub/private assets, toy subsets, caches without original
  media, or reconstructed substitutes for unavailable paper data.
- Turning an author-media evaluator replay, a demo, a smoke test, or a released
  checkpoint run into a training reproduction claim.
- Selecting an idea solely to satisfy a deadline or citation count.

## Earliest credible proposal shape

If the gates close, the proposal may concern a *publicly evaluable training
method operating at the local H3-Base boundary*, rather than a claim to train
or reproduce hosted H3-Context-IR. It must name: public data and license;
frozen Base interface; training objective and controls; an exact baseline;
sample-level provenance; fixed evaluator revision; repeated-seed protocol;
cost; negative-result and stopping criteria. Until then, no method is selected.
