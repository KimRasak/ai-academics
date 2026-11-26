# VABench reproduction card

**Status:** resource/protocol audit与 I2AV 输入获取均已完成：原始 ZIP、521 张 JPEG
及逐图 hash 已闭环。正式 I2AV 评分被发布缺口阻断（图像—标注 pairing manifest 与
官方 DNSMOS wrapper 缺失）；未产生 benchmark score。  
**Authority:** VABench paper, official `tanABCC/VABench` repository and the
two repositories linked by its README.  
**Scope:** evaluate a released audio-video generator (initially MiniMax H3)
under the public VABench task protocol. This is an evaluator/weight
reproduction, not a VABench training reproduction.

## What can and cannot be reproduced

VABench (arXiv:2512.09299) evaluates synchronous audio-video generation. Its
official README identifies T2AV, I2AV and stereo audio-video tasks and lists
15 dimensions. It does not release a generator training recipe or claim a
trainable H3 baseline. A completed run can therefore support a claim about a
pinned evaluator on a pinned set of locally generated outputs; it cannot
reproduce H3 pretraining or turn benchmark prompts into a training set.

The candidate official code snapshot is the sole commit currently reachable
from the repository:

- `tanABCC/VABench` commit `12b9617a900a9eca1691da6b7115ce3d58f53a16`
  (2026-07-08);
- the citation in its README names Daili Hua, Xizhi Wang, Bohan Zeng, Xinyi
  Huang, Hao Liang, Junbo Niu, Xinlong Chen, Quanqing Xu and Wentao Zhang;
- the repository itself has no top-level license file. Do not redistribute its
  code or imply a license that was not supplied.

## Public benchmark resources

| resource | pinned revision | contents verified without downloading LFS objects | license/provenance |
|---|---|---|---|
| I2AV test inputs | `tanABBCC/VABench_I2AV@d4a4035643b12e4f20726777db30ffd8cbe8639e` | `images.zip`, LFS SHA-256 `b3364883293842b93e1d4936b1b06a7be2149fe9e20eb6a2029dd1cbed60bd09`, 148,242,034 B; downloaded and archive-tested on 2026-09-18; 521 JPEGs | its dataset card declares Apache-2.0; retain the original ZIP and its source manifest |
| evaluator cache | `tanABBCC/VABENCH_CACHE_DIR@aa0c3da973d86f2994e9c8860f28785ce0cc6152` | 40 LFS objects, 38,381,139,130 B by pointer sizes; includes Qwen2.5-Omni-7B, ViCLIP, ImageBind, Synchformer, CLAP and metric weights | cache repository itself has no dataset card/license; Qwen subdirectory carries Apache-2.0. Audit component licenses before redistribution |
| prompt/QA annotations | code commit above | 14 image JSON files / 521 entries; 24 text JSON files / 848 entries | included in unlicensed code repository; retain commit and do not treat annotations as an independently licensed training corpus |

The I2AV ZIP is original human-inspectable media input. It must be retained
alongside an unpacked inventory, rather than converted into hidden caches.

## Protocol blockers and required repairs

The current official snapshot is not executable verbatim as a reliable
benchmark protocol:

1. `VABench.load_prompt_mapping()` reads `./mapping_text/final_idx_to_prompt.csv`,
   but the checked-in file is `mapping/final_idx_to_prompt.csv`. The original
   invocation therefore fails to map filenames to prompts unless the current
   working directory happens to contain an undocumented extra directory.
2. `evaluate.py` requires every evaluated batch to have matching
   `video/<subset>/`, `audio/<subset>/` and `json/<subset>.json`. The public
   image archive, annotation split names, expected file stems, and the
   repository's text/image JSON layout must be reconciled and frozen before
   generating H3 outputs.
   This reconciliation is currently impossible from the published artifacts:
   the 521 JPEG filenames have no shared identifier with either the 521 I2AV
   annotation entries or the 1,299-row prompt-to-output-name table. Counts are
   equal in every one of the 14 category directories, but neither lexical
   filename order nor ZIP entry order matches JSON order semantically (for
   example, the first physical-interaction JPEG is a basketball while the first
   JSON entry describes two industrial fans). A guessed semantic or order-based
   join would change benchmark identity and is prohibited.
3. The default config requests all 15 dimensions, but it does not pin the
   component weight revisions. Several download URLs use Hugging Face `blob`
   URLs, which are not reliable raw artifact URLs. Use the official cache
   revision above, record every actual SHA-256, and do not allow opportunistic
   downloads during a scored run.
4. The README recommends at least two 80 GB GPUs. The config uses an
   auto-sharded Qwen2.5-Omni 7B with `{0: 80GB, 1: 80GB}`. This evaluator
   requirement is separate from the four-GPU H3 generation service.
5. `first_dnsmos.py` dynamically imports
   `third_party/DNS-Challenge/DNSMOS/dnsmos_local.py`, but that file and its
   parent directory are absent from the pinned official code snapshot and
   there is no Git submodule declaration. The official evaluator cache does
   contain the two DNSMOS ONNX weights, which were independently acquired and
   hash-verified; those weights cannot execute through the released VABench
   code without the missing implementation. Do not substitute another DNSMOS
   implementation and call its result an unmodified VABench score.

The only permitted local patch for an initial plumbing run is the one-line,
recorded mapping-path fix in `patches/mapping-path.patch`. It was verified that
all 521 image annotation prompts exactly match entries in the checked-in
1,299-row mapping table. A patched run is **not** an unmodified-official
evaluation reproduction. It becomes score-bearing only after a small official
subset and the full protocol gate pass.

## H3 evaluation route

1. Preserve the original I2AV ZIP, its hash, source URL, extraction inventory
   and every selected image hash. This is complete: both archive and extracted
   media live under
   `/gluster_hdd_cv/user/jinzili/data/research-work-reproduction/vabench/`;
   `images_inventory.txt` and `images_sha256.txt` contain 521 rows each.
2. Obtain an author-supplied image-to-annotation pairing manifest before
   creating any VABench I2AV request subset. H3's documented local service
   accepts an image condition as a fetchable `uri`, not an undocumented local
   filesystem path. Once the join exists, a controlled temporary HTTP service
   can expose the preserved `images/` root; retain its command and do not
   expose it outside the experiment host.
3. When a fresh four-GPU ownership audit clears a window, run the unmodified
   official H3 I2VA/FL2VA request semantics for that subset. Retain source
   images, H3 requests, generated MP4/WAV, logs and `ffprobe` metadata.
4. Run only the required VABench dimensions against a fully pinned evaluator
   cache; compare the code's discovered coverage with the subset manifest.
5. Scale to the released split only if media coverage is 100%, every metric
   returns a valid per-item result, and all protocol deviations are recorded.

No paper score is imported as a local result, and no VABench result is used to
choose or train a new method until this chain is closed.

## Component-smoke record

On 2026-09-18, the two official cache artifacts needed by `first_dnsmos` were
downloaded from the pinned cache revision to:

`/gluster_hdd_cv/user/jinzili/data/research-work-reproduction/vabench/evaluator-smoke/cache/dns_mos/`

- `sig_bak_ovr.onnx`: 1,157,965 B,
  `269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd`;
- `model_v8.onnx`: 224,860 B,
  `9246480c58567bc6affd4200938e77eef49468c8bc7ed3776d109c07456f6e91`.

Both hashes match the official cache LFS pointers. The released H3
`assets/fl2va.mp4` is inspectable 8-second H.264 + 32-kHz stereo AAC media,
so it is a suitable *media-format* test input. The component score was not run
because the required DNSMOS implementation is absent. This is a C1 release
audit outcome, not an H3 quality result.

## I2AV pairing audit

The full preserved 521-image audit copy was rehashed against the authoritative
remote image inventory: all 521 SHA-256 values matched. This establishes media
identity, but not benchmark pairing. The category image counts equal their JSON
entry counts, yet the release provides no image ID field in JSON, no manifest,
and no mapping from semantic image filename to annotation. The archive's
lexical order demonstrably disagrees with annotation order. Therefore I2AV may
be used only as a human-inspectable public image collection or an explicitly
new protocol until the authors supply a pairing manifest; it cannot support a
VABench I2AV score in this reproduction.
