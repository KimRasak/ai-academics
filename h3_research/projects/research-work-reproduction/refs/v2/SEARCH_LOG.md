# Literature search log v2

**Date:** 2026-09-18

## Intent

Fill named-video-work gaps in the existing generic diffusion/reward pool:

1. controllable video generation benchmarks and evaluation;
2. video diffusion preference alignment and reward fine-tuning.

## Scholar Crawler outcome

- `scholar-crawler --doctor` passed: Playwright, Chrome and the persistent
  `.scholar-profile` are available. The only warning is a non-functional
  installed-version mismatch (pip reports 0.2.0 while the checkout is 0.3.0).
- Query `controllable video generation benchmark evaluation`, 2023 onward,
  one page: Google Scholar returned a CAPTCHA before any result was collected.
  The crawler correctly handed it to a human and was interrupted after 32.2 s;
  the event is retained in `out/challenges.jsonl`.
- A simultaneous second query was not started after Chrome's context closed;
  it wrote zero records and is not evidence of an empty literature result.

## Rule for continuation

Do not retry, rotate proxies or automate a CAPTCHA solution. Once a human has
cleared the Chrome challenge, resume exactly one query at a time with the same
profile and store results in this directory. Until then, use official paper,
repository and benchmark metadata already attached to each reproduction card.

## 2026-09-23 follow-up: controllable-video candidate search

- The controlled Scholar Crawler entry point was verified first: `/opt/miniconda3/bin/scholar-crawler`
  uses Python 3.13.5, Playwright 1.60.0 and local Chrome. The checkout/version mismatch
  (`pip` records 0.2.0 while the source checkout is 0.3.0) is a non-functional warning; it
  does not affect the crawler's documented challenge handling.
- A one-page dry-run for `open source controllable video generation benchmark`, recent three
  calendar years (2024 onward), estimated one request and at most ten records. No request was
  made by the dry-run.
- The corresponding headless one-page run made exactly one request and stopped with
  `challenge_unattended`: Scholar returned `rate_limit` / “unusual traffic” before any result
  was parsed. It produced **zero records**; this is not evidence that the literature is empty.
  The event and resumable state are retained as:
  - `scholar_controllable_video_benchmark_20260923.challenges.jsonl`;
  - `scholar_controllable_video_benchmark_20260923.state.json`;
  - `scholar_controllable_video_benchmark_20260923.jsonl` (empty result log).
- Do not retry headlessly, rotate proxies or automate challenge solving. The only permitted
  continuation is the tool-provided command without `--headless`, after a human can clear the
  visible challenge once, using the same `scholar_profile_20260923` and `--resume`.

## 2026-09-23 primary-metadata access note

For the already retained MM-Diffusion, T2V-Turbo and VADER arXiv PDFs, direct
`export.arxiv.org/api/query` metadata requests were blocked by the outbound-host policy
(`host 'export.arxiv.org' is outside note.com`). This is separate from the Scholar CAPTCHA and
does not establish that any paper metadata or primary source is unavailable. Do not retry the
blocked host automatically; retain these citations as `pending` until a local PDF extraction
tool or human-verified primary page supplies canonical author/venue metadata.

## Current gap

`refs/merged_v1.jsonl` has 148 records but does not include named entries for
MiniMax-H3, VideoDPO, VADER, T2V-Turbo, MotionCtrl, VBench, Sci-VBench or the
named control benchmarks. The generic reward-alignment pool alone is therefore
not sufficient to support the final 30-reference video-generation proposal.
