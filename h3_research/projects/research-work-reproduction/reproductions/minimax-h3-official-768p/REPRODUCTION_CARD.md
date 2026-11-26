# MiniMax H3 official 768p reproduction card

**Status:** official four-80GB FL2VA deployment is blocked before serving;
no locally generated H3 media exists  
**Prepared:** 2026-09-16  
**Authority:** official `MiniMaxAI/MiniMax-H3` model repository  
**Scope:** T2VA, FL2VA and Ref2VA official reproducible cases

## Claim being reproduced

The official model repository provides three 768p requests and corresponding
reference outputs:

- T2VA → `assets/t2va.mp4`;
- FL2VA → `assets/fl2va.mp4`;
- Ref2VA → `assets/ref2va.mp4`.

The requests use seed 0 and call an OpenAI-compatible local H3 service.

## Local authoritative assets

Remote model root:

`/gluster_hdd_cv/user/jinzili/weights/MiniMax-H3`

Original checkpoints:

- `FL2VA/`: 135 GiB;
- `Ref2VA/`: 135 GiB.

Official request scripts:

- `scripts/readme/reproducible-768p-t2va-request.sh`;
- `scripts/readme/reproducible-768p-fl2va-request.sh`;
- `scripts/readme/reproducible-768p-ref2va-request.sh`.

Their SHA-256 values are recorded in `source_manifest.jsonl`.

Official reference-media stream metadata and hashes are recorded separately in
`official_reference_media_audit.json`. All three are 1344×768 H.264 at 24 fps
with 32-kHz stereo AAC; their observed container durations are 10.157 s
(T2VA), 8.032 s (FL2VA), and 5.207 s (Ref2VA). Later stochastic local outputs
must be compared by declared conditions, streams, media evidence and content
metrics—not byte identity or exact container duration.

## Environment

- Python 3.11.16
- PyTorch 2.11.0+cu128
- SGLang 0.5.17.dev20260803+gdd6ddc053b
- sglang-kernel 0.4.5+cu129
- diffusers 0.40.0.dev0
- transformers 5.16.1
- accelerate 1.14.0
- DeepSpeed 0.19.6
- PEFT 0.20.0
- PyAV 18.1.0

The official server requires four GPUs. FL2VA (used by both T2VA and FL2VA)
serves port 30010; Ref2VA serves port 30011. The former `30020` value in an
older deployment note is not an official request endpoint and must not be used
for this card.

The frozen request-to-service-to-checkpoint mapping is machine-readable in
`request_routing_preflight.json`: T2VA and FL2VA must use the `FL2VA/`
partition on port 30010, while Ref2VA must use `Ref2VA/` on port 30011. The
two four-GPU services are sequential, never concurrent on the same GPUs.
`R0_FREEZE_MANIFEST.json` combines the release, script, reference-media and
local-tool identities that must be rechecked immediately before submission.
Run `tools/verify_r0_freeze.py --case CASE` at that point; it performs no
network or GPU operation and rejects any hash or routing mismatch.

Before either service is started, run the read-only ownership gate
`tools/check_h3_launch_preflight.sh GPU0 GPU1 GPU2 GPU3`. It rejects an
occupied selected GPU or listener on either official port; a passing result is
still subject to a human ownership confirmation and is never authorization to
stop another user's process.

When that gate passes, use `commands/launch_h3_base_service.sh VARIANT GPU0
GPU1 GPU2 GPU3`, not an unlogged shell fragment. It reruns the gate, fixes the
four visible physical GPUs, stores launch metadata and redirects server stdout/
stderr to `server_logs/`; it starts one service in the foreground and never
preempts, kills, or launches the second variant concurrently.

After the operator stops that foreground service, run
`tools/audit_h3_service_turnover.sh VARIANT GPU0 GPU1 GPU2 GPU3`. This
read-only check records whether the service port and selected GPUs are actually
free; it never kills a residual process. Do not start the next variant or claim
the GPUs released unless its terminal-free decision is recorded.

After a human starts the correctly routed Base service, run exactly one case
with `commands/run_frozen_official_case.sh CASE`. The wrapper uses the archived
durable polling runner: it preserves request JSON, submit response, every
status response, final status, output SHA-256 and ffprobe metadata. It refuses
to overwrite outputs and does not alter the frozen official request payload.
The runner is pinned inside this reproduction card (`tools/`), rather than
depending on a local workspace path absent from the remote container.
Its post-run audit verifies frozen-payload identity and media contract, but
does not call a perceptual similarity result a deployment-reproduction metric.

## Input provenance

The two official request scripts reference three CDN inputs. They were acquired
without transformation and preserved with their source URLs and SHA-256 values
at:

`/gluster_hdd_cv/user/jinzili/data/research-work-reproduction/minimax-h3-official-768p/inputs/`

- FL2VA keyframe PNG: 1,263,485 bytes,
  `352e959e08b886406d3766cee25ebe7bbf679e6d53eeffbd1c607b8dc93e2068`;
- Ref2VA reference MP4: 4,396,484 bytes,
  `d6edc20ac82aa980c9cbd0b05b371f60d12dc8d018492e571b42de0086819c41`;
- Ref2VA reference MP3: 204,432 bytes,
  `2a48d447b343b38a2c504d5281a0c97c688db0ba87931ed9282c598bd6752454`.

The original request scripts still use their public HTTPS URIs. Local copies
are retained for human inspection and CDN-failure diagnosis; they are not
silently substituted into an “unmodified request” claim.

## Reproduction protocol

### Launch-wrapper compatibility correction (2026-09-20)

The first authorized FL2VA launch attempt passed the ownership gate but exited
before invoking SGLang or allocating any GPU. The pre-exec metadata command
used one `nvidia-smi` invocation with four `-i` selectors, which this remote
`nvidia-smi` rejects. The launcher now queries the same four GPUs one at a
time. This changes metadata collection only: it does not change checkpoint,
service flags, routing, request payload, output path, or evaluation contract.
The corrected launcher and its manifest hash must be revalidated before the
next launch.

The corrected launcher reached official transformer loading on 2026-09-20, but
rank 2 failed before a listener or generated video existed: it needed 294 MiB
with only 246 MiB free while PyTorch reported 3.35 GiB reserved but unallocated.
The service automatically exited and released GPUs 4--7. One allocator-only
compatibility retry using `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
also failed; it changed no model, request, seed, routing, sampling, or output
evaluation parameter.

### Authorized FL2VA launch outcome (2026-09-20)

The first launcher invocation passed the four-GPU gate but terminated before
SGLang exec because its metadata-only `nvidia-smi` form was incompatible with
this runtime. After the one-GPU-at-a-time metadata correction, the official
FL2VA service entered `MiniMaxH3Pipeline`, loaded the text encoder, and began
the 13-shard FSDP transformer load on GPUs 4--7. It failed before opening port
30010, submitting a request, or writing output media:

- First load: rank 2 needed 294 MiB with 246 MiB free; PyTorch reported 3.35
  GiB reserved but unallocated.
- Compatibility retry with `expandable_segments:True`: the same 294 MiB
  allocation failed with about 156 MiB free and 1.05 GiB reserved but
  unallocated per rank.

The allocator adjustment reduced fragmentation but did not close the capacity
gap, so the available evidence supports an 80GB-per-rank FSDP load-peak limit,
not a mere allocator defect. SGLang automatically exited; a post-failure check
found no H3 process on GPUs 4--7 and no listener on either official port. This
is a retained deployment-failure result (C1 failure evidence), not an H3
generation, quality, benchmark, paper-metric, or training result. Do not retry
the identical four-GPU load. The subsequent installed-runtime audit is retained
in `H3_FOUR_H100_MEMORY_MODE_AUDIT_20260920.json`: H3 `memory` mode supports
auxiliary text/VAE layerwise offload but retains FSDP and explicitly keeps DiT
CPU/layerwise offload disabled. Since the observed failure is during DiT FSDP
materialization, that supported compatibility mode does not address the load
peak. A next attempt requires a different, evidence-backed sharding/weight
layout or a differently scoped hardware configuration.

1. verify GPUs are free and do not overlap another user's process;
2. serve the local model path with official SGLang semantics;
3. run the unmodified official request script;
4. retain request, server log, generated MP4 and media metadata;
5. compare against the official MP4 using duration, stream metadata and
   perceptual/audio metrics;
6. label stochastic differences explicitly rather than expecting file hashes
   to match.

## Current evidence

- On 2026-09-19, the remote read-only `verify_r0_freeze.py` check passed for
  all three frozen cases (`t2va`, `fl2va`, `ref2va`). It rehashed the official
  H3 README, each selected official request script, routing and reference-media
  audit, and every local runner/gate/launch/audit tool against
  `R0_FREEZE_MANIFEST.json`; each case returned `decision=FREEZE_VERIFIED`.
- A separate read-only request-payload preflight parsed the three official JSON
  heredocs and retained canonical payload hashes, targets and condition
  structures in `request_payload_preflight_20260919.json`: T2VA has no
  condition, FL2VA has one image keyframe, and Ref2VA has video plus audio
  reference conditions. All payloads have seed 0 and match their frozen script
  identities. This neither contacts a service nor establishes generation.
  This verifies protocol identity only and neither starts a service nor creates
  a local H3 output.
- The same read-only observation found ports 30010/30011 free and no H3 Base
  service, but GPU 0--3 had external compute allocations and GPU 4--7 were
  occupied by the active MM-Diffusion checkpoint evaluation. This is not a
  cleared four-GPU window. Even after a future passing gate, a human ownership
  confirmation remains mandatory before calling the four-GPU service launcher.
  The machine-readable evidence is `freeze_verification_20260919.json`.
- model weights and official reference outputs exist locally;
- official scripts and references have recorded hashes;
- SGLang CLI exposes `serve`, `generate` and `version`;
- the checkpoint's pinned `model_index.json` identifies
  `_class_name: MiniMaxH3Pipeline` and partitions `fl2va: FL2VA` and
  `ref2va: Ref2VA`, matching the official four-GPU deployment commands and
  distinguishing this local path from the excluded `sora-2` service outputs;
- a 2026-09-18 index-presence audit found all declared text-encoder shards
  (14; 66,714,912,872 bytes) and transformer shards (13; 66,280,485,936
  bytes) in each task family. This is index-to-file completeness only, not a
  per-shard content-hash claim;
- on 2026-09-18, a no-GPU CLI preflight found that `sglang version` did not
  return within 60 seconds. It was terminated without launching an H3 server,
  listener, or GPU process. Retest this responsiveness only in the actual
  cleared-GPU launch window; it is an environment risk, not checkpoint evidence;
- the read-only ownership gate rechecked GPUs 0, 1, 2 and 4 at
  `2026-09-18T13:55:56Z`: official ports were free, but all four GPUs had
  active compute processes (the current MM-Diffusion checkpoint metric run on
  PIDs 2816799--2816802). It correctly returned `DO_NOT_START_H3_SERVICE`; no
  process was preempted;
- **Historical 2026-09-18 statement:** generation had not yet been executed at
  that time. It is superseded by the 2026-09-22 8×H100 SGLang Base evidence:
  three frozen official cases completed, and the settings-matched Sci-VBench
  lane reached `128/150` before the service disappeared during item 129.
  Those 128 outputs remain incomplete benchmark evidence: no local aggregate
  VT has been computed, and full media/provenance audit remains a hard gate;
- the similarly named `shared/h3-baseline/official_requests/` API artifacts
  were separately audited on 2026-09-18.  Their request/output chains are
  internally consistent, but every final response reports `model: sora-2`.
  They are therefore explicitly excluded from this H3 card and cannot replace
  a local run of the official H3 server;
- on 2026-09-18, no H3 service was listening on 30010/30011 and only GPU 0 was
  free. GPUs 1–7 were occupied, so starting the four-GPU server was not safe.
- A fresh ownership audit at `2026-09-18T06:12:02Z` on
  `osa00-h100-006.osa00.bdjp.io` reached the same conclusion with process-level
  evidence: GPU 0 held 2,838 MiB (PID 1424854); GPU 1–7 each held a large
  62,958–68,396 MiB allocation (PIDs 1419183–1419188, with GPU 3 additionally
  holding 1,878 MiB in PID 3433145). No service was listening on 30010 or
  30011. Four GPUs are therefore not ownership-cleared for this run.
- After the MM-Diffusion sampler terminated, the read-only gate was rerun at
  `2026-09-18T17:54:16Z` for GPUs 0, 1, 2 and 4. Ports 30010/30011 were free,
  but GPUs 0–2 still each had a 2,838 MiB compute allocation (PIDs
  1424854–1424856); only GPU 4 was free. It returned
  `DO_NOT_START_H3_SERVICE`. No process was stopped and no H3 service was
  launched. A later passing gate still requires explicit human ownership
  confirmation before a four-GPU launch.
- A later read-only gate at `2026-09-19T00:43:34Z` passed for physical GPUs
  4--7 and found ports 30010/30011 free, with no compute processes on those
  four cards. Its exact output is retained in
  `h3_ownership_gate_20260919T004334Z.json`. This is readiness evidence only:
  the gate expressly requires an authorized human to confirm ownership before
  the launcher may be called. No service or request was started from this gate.
- The gate was rechecked at `2026-09-19T01:31:15Z` with the same result for
  GPUs 4--7 and ports 30010/30011. The later record is
  `h3_ownership_gate_20260919T013115Z.json`; it does not supersede the human
  confirmation requirement or authorize service startup.

## Non-claims

This reproduction does not reproduce H3 training or a paper benchmark. It
only verifies official checkpoint deployment and official examples.
