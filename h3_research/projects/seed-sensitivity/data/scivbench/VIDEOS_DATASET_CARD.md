---
license: cc-by-4.0
language:
- en
task_categories:
- text-to-video
tags:
- video-generation
- scientific-reasoning
- benchmark
- evaluation
- model-outputs
size_categories:
- 10K<n<100K
pretty_name: Sci-VBench Videos
configs:
- config_name: index
  data_files: index.jsonl
- config_name: judge_scores
  data_files: judge_scores.jsonl
- config_name: vt_scores
  data_files: vt_scores.jsonl
- config_name: human_scores
  data_files: human_scores.jsonl
---

# Sci-VBench Videos

**Sci-VBench Videos** is the complete set of model outputs behind the Sci-VBench paper: **11,216 videos** from **16 text-to-video models**, together with the automatic and human scores computed on them. Every video was generated from the **verbatim** benchmark prompt under the model's default configuration — no rewriting, no prompt expansion — so the released prompts and the released videos correspond exactly.

Prompts and evaluation specifications live in the companion repo, [Sci-VBench/Sci-VBench](https://huggingface.co/datasets/Sci-VBench/Sci-VBench). A video joins to its prompt and rubrics on `video_id` alone.

## Coverage

Open-source models were run on the full benchmark; proprietary models on the 150-example `testmini` split.

| Model | Access | Videos | | Model | Access | Videos |
|---|---|---:|---|---|---|---:|
| CogVideoX1.5-5B | open | 1,253 | | Gemini-Omni-Flash | API | 144 |
| Cosmos3-Nano | open | 1,253 | | HappyHorse-1.1 | API | 150 |
| HunyuanVideo-1.5 | open | 1,253 | | Kling-2.6 | API | 150 |
| LTX-2 | open | 1,253 | | Seedance-2.0 | API | 148 |
| LTX-2.3 | open | 1,253 | | Sora-2 | API | 150 |
| LongCat-Video | open | 1,253 | | Veo-3.1 | API | 150 |
| MiniMax-H3 | open | 1,253 | | Veo-3.1-Fast | API | 150 |
| Wan2.2-5B | open | 1,253 | | Wan-2.6 | API | 150 |
| | | | | **Total** | | **11,216** |

Gemini-Omni-Flash is 144/150 and Seedance-2.0 is 148/150: the missing prompts were refused by the provider's input filter, identically on every retry. Six of the eight are wet-chemistry procedures involving a strong acid, a heavy-metal reagent, or heating; the other two — an immunofluorescence staining protocol and a golf putt — have no evident trigger. Because prompts are never rephrased in this benchmark, those examples have no path through those models, and the affected model averages are computed over the videos that were produced.

## Layout

```
videos/<video_id>/<Model>.mp4        11,216 videos, 32.5 GB
settings/<video_id>/<Model>.json      5,310 generator sidecars
index.jsonl                           one row per video
judge_scores.jsonl                    MLLM-as-Judge, 3 rounds per video
vt_scores.jsonl                       VBench Video Quality metrics
human_scores.jsonl                    expert ratings, testmini only
```

## Score files

All three key on `(video_id, model)` and carry nothing but numbers — no reasoning traces, no free-text justifications. The four rubric dimensions are `pg` (Prompt Grounding), `scc` (Scientific and Causal Correctness), `sc` (Spatiotemporal Consistency), and `lpf` (Low-level Perceptual Fidelity), each a 1–5 integer.

### `judge_scores.jsonl` — 33,648 rows

Rubric-conditioned MLLM-as-Judge (Qwen3.5-397B-A17B), scoring one dimension at a time against that dimension's anchored rubric. Every video is scored in three independent rounds; take the mean over rounds.

| Field | Type | Description |
|---|---|---|
| `video_id` | string | Joins to `id` in the dataset repo |
| `model` | string | Generator that produced the video |
| `round` | int | Judge run, 1–3 |
| `pg` `scc` `sc` `lpf` | int | 1–5 score for each dimension, or `null` |

Two of the 134,592 score cells are `null`, where the judge's response could not be parsed on that round: video `c248988dbb77aff1e98f4bc0cbef7a62465f7f28` for HunyuanVideo-1.5 (`scc`, round 2) and for Wan2.2-5B (`pg`, round 2). Neither is in `testmini`, and no video loses a dimension entirely, so averaging over the remaining rounds absorbs both — but guard for `None` when you aggregate.

### `vt_scores.jsonl` — 11,216 rows

The six VBench *Video Quality* metrics and their unweighted mean, `VT`, which the paper uses as the automatic proxy for LPF.

| Field | Type | Description |
|---|---|---|
| `video_id` `model` | string | Join key |
| `subject_consistency` `background_consistency` `motion_smoothness` `dynamic_degree` `aesthetic_quality` `imaging_quality` | float | Individual VBench metrics |
| `VT` | float | Unweighted mean of the six |

### `human_scores.jsonl` — 2,392 rows

Expert ratings on `testmini`, one per video, by a domain annotator who did not author the example and who worked from the prompt, the video, and the released evaluation specification. These are the reference labels behind the paper's human evaluation.

| Field | Type | Description |
|---|---|---|
| `video_id` `model` | string | Join key |
| `pg` `scc` `sc` `lpf` | int | 1–5 expert score for each dimension |

### `index.jsonl` — 11,216 rows

One row per video: `video_id`, `model`, `path`, `settings` (path, or `null`), `testmini`, and `bytes`.

## Generation settings

`settings/<video_id>/<Model>.json` records what actually ran for that video — resolution, frame count, fps, sampler steps, guidance scale, seed, and for locally run models the GPU type and wall time.

Sidecars exist for **six** of the sixteen models: CogVideoX1.5-5B, Cosmos3-Nano, LTX-2.3 and MiniMax-H3 (1,253 each), plus HappyHorse-1.1 and Seedance-2.0, the two API providers that return their parameters. The other four models we ran ourselves — HunyuanVideo-1.5, LTX-2, LongCat-Video, Wan2.2-5B — predate the sidecar format and have none, so `index.settings` is `null` for 5,906 of the 11,216 rows. Settings for every model are tabulated in the paper. Job identifiers have been removed from the sidecars; hardware and timing fields are kept.

## Quick start

Download one model rather than all 32.5 GB:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    "Sci-VBench/Sci-VBench-Videos", repo_type="dataset",
    allow_patterns=["videos/*/Sora-2.mp4", "*.jsonl"],
)
```

Reproduce a model's automatic scores by averaging the three judge rounds. Restrict to `testmini` for the cross-model comparison; drop the filter to get full-benchmark numbers for an open-source model.

```python
import json, collections, statistics

mini = {json.loads(l)["video_id"] for l in open("index.jsonl") if json.loads(l)["testmini"]}

per_video = collections.defaultdict(lambda: collections.defaultdict(list))
for l in open("judge_scores.jsonl"):
    r = json.loads(l)
    for d in ("pg", "scc", "sc", "lpf"):
        if r[d] is not None:            # two rounds failed to parse; see below
            per_video[(r["model"], r["video_id"])][d].append(r[d])

def score(model, dim, testmini_only=True):
    vals = [statistics.mean(v[dim]) for (m, vid), v in per_video.items()
            if m == model and (not testmini_only or vid in mini)]
    return round(statistics.mean(vals), 2)

for d in ("sc", "pg", "scc"):
    print(d, score("Sora-2", d), score("MiniMax-H3", d, testmini_only=False))
```

The official evaluation code expects a model-major tree (`<dir>/<model>/<id>.mp4`). To reshape this repo's id-major layout without copying:

```python
import os, pathlib

src, dst = pathlib.Path("videos"), pathlib.Path("all-videos")
for mp4 in src.glob("*/*.mp4"):
    out = dst / mp4.stem / f"{mp4.parent.name}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        os.link(mp4, out)
```

## License

Released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Outputs of proprietary systems remain subject to their originating provider's terms of service; consult those before redistributing or using them commercially.
