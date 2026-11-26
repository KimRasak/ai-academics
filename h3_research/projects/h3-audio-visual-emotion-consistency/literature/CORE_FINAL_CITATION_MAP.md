# Core-source citation map for the H3 proposal

**Updated:** 2026-09-19  
**Evidence basis:** `CORE_PRIMARY_SOURCE_LEDGER.md` plus the named reproduction
cards and source manifests. This map covers works and evaluators that the
proposal already relies on operationally. It is intentionally separate from
the 31-item candidate literature queue.

## Rule

An operational source can be cited for only what the retained artifact proves.
For example, an official README supports a release-boundary statement, not a
claim about H3 training; an author-media evaluator replay supports evaluator
execution, not local generation. Before final submission, canonical author,
venue/version and primary URL must be rechecked for every paper-like entry.

| Core object | Current primary-source evidence | Proposal paragraph it can support now | Permitted claim | Prohibited upgrade / remaining citation review |
|---|---|---|---|---|
| MiniMax-H3 | Official `MiniMaxAI/MiniMax-H3` README SHA-256 `819db2…39ae9`; local FL2VA/Ref2VA index presence; frozen official request scripts | release boundary; R0 deployment scope | Local 768p Base has defined T2VA/FL2VA/Ref2VA deployment interfaces | Do not cite as public H3 training, full Context-IR or 2K reproduction; verify authors/version/release paper if one is cited |
| Sci-VBench | Official code commit `e8cb398…b7031`; public video data-card endpoint; 150 released H3 videos replayed | H3 benchmark/evaluator boundary | Official evaluator ran on author-released testmini media (C2) | Do not call it local H3 generation or a cross-model quality result; verify paper authors/venue and score protocol |
| MM-Diffusion | Paper arXiv:2212.09478 PDF hash `39c9…a9b3c`; official code `7250222`; official Landscape archive/checkpoints/evaluator hashes | reproducible AV baseline and C3 evidence | Two released-checkpoint runs on retained official media yield inspectable FVD/KVD/FAD observations | Do not say six-run paper claim or training process was reproduced; review canonical paper metadata and public upstream-data limitation |
| VBench | Official code revision `03e7b94`; prompts/aggregation and evaluator weights retained for existing lanes | shared video-generation evaluation protocol | A pinned evaluator can support checkpoint comparisons only under its exact prompt/seed/frame/version key | Do not compare unaligned paper tables; verify paper citation and every enabled dimension/weight |
| T2V-Turbo | Paper arXiv:2405.18750v2; official inference `10b1c53`; pinned VBench revision | public checkpoint post-training baseline | Released VC2 LoRA has a fixed-seed all-16-dimension VBench checkpoint evaluation | Do not call Table 1 exact or WebVid training reproduced; verify final paper authors/venue and checkpoint identity |
| VADER | Paper arXiv:2407.08737v1; official release tag `d2da343`; public LoRA evaluation artifacts | reward/post-training trade-off baseline | Released LoRA produces a measured reward/dynamics trade-off on the declared partial protocol | Do not equate released LoRA to Table 1 or claim training reproduction; verify final paper metadata and missing protocol fields |
| VideoDPO | Paper arXiv:2412.14167; official code `1febdb4`; 20,000 official MP4 archive decode audit | preference-data provenance limitation | Public pair media/archive can be audited | Do not cite headline reweighted method as rerun; pair weights and evaluator/config identity remain missing |
| MotionCtrl | Paper arXiv:2312.03641; official code `6242d55`; checkpoint hash `efe2…fe46c` | controllable-video comparison and release limitation | A released example checkpoint can be deployed/smoke-tested | Do not cite paper metrics or training as reproduced; test manifests, metrics and matching training code absent |
| VABench | CVPR 2026 paper URL; official code `12b9617`; I2AV image archive/hash audit | AV benchmark limitation and future-gate rationale | Its release can be audited and its missing pairing/DNSMOS dependencies precisely identified | Do not report formal I2AV score or use it as H3 benchmark until author pairing and wrapper are available |

## Minimum final bibliography composition

The final bibliography must not be constructed from the 31 candidate queue
alone. It needs, at minimum, the selected items from this core map plus a
separately approved subset of the 31 candidates that covers: nearest joint AV
generation methods; chosen training family; chosen controlled-generation
formulation; public data; and evaluation. The total must be at least 30
*approved, hypothesis-relevant* citations, not simply 30 keys.

## Current status

This map makes the already-used operational sources traceable to proposal
claims. It does not approve any final paper citation or select a training idea.
