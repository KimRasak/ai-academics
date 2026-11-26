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
- rubric
size_categories:
- 1K<n<10K
pretty_name: Sci-VBench
configs:
- config_name: default
  data_files:
  - split: test
    path: test.jsonl
  - split: testmini
    path: testmini.jsonl
---

# Sci-VBench

**Sci-VBench** is an expert-annotated benchmark for *knowledge- and reasoning-intensive* text-to-video generation. Each example targets a scientific concept whose correctness is verifiable from video evidence alone: a mechanism that unfolds over time, so no single well-chosen frame can satisfy the prompt. Prompts are deliberately minimal — they state the observable initial setup and the intervention and stop there, withholding the expected mechanistic trajectory, so a model must infer the mechanism rather than replay an outcome the prompt already described.

Every example ships with a full **evaluation specification**: a high-level reference guide that fixes how the task should be interpreted, plus 1–5 anchored rubrics. This externalizes the expert knowledge that evaluation requires, so non-expert raters and MLLM judges can score generated videos under the same standard the annotators used — without recruiting domain experts of your own.

The three reasoning-centric rubrics (`pg`, `scc`, `sc`) are written per example — all 1,253 are distinct. `lpf_specification` is a single shared rubric, identical across every example: low-level perceptual fidelity is judged by generic video-quality criteria that do not depend on the scientific content.

Generated videos from 16 models, together with automatic and human scores, are released separately in [Sci-VBench/Sci-VBench-Videos](https://huggingface.co/datasets/Sci-VBench/Sci-VBench-Videos).

## Disciplines and statistics

Sci-VBench covers **1,253 examples** across **60 subjects** in **4 disciplines**. Lengths are mean word counts; the rubric rows count all five anchors of that dimension together.

| Discipline | Examples | Prompt | Ref. guide | LPF rubric | PG rubric | SCC rubric | SC rubric |
|---|---:|---:|---:|---:|---:|---:|---:|
| Natural Science | 567 | 33.3 | 160.1 | 245.0 | 280.2 | 283.3 | 313.1 |
| Healthcare | 300 | 40.5 | 161.3 | 245.0 | 300.0 | 291.9 | 344.6 |
| Engineering | 283 | 42.6 | 163.3 | 245.0 | 288.3 | 294.4 | 327.3 |
| Humanities and Social Sciences | 103 | 35.8 | 164.5 | 245.0 | 295.2 | 315.1 | 321.4 |
| **Total** | **1,253** | | | | | | |

## Splits

| Split | Examples | Description |
|---|---:|---|
| `test` | 1,253 | The full benchmark |
| `testmini` | 150 | A fixed subset on which every evaluated model is run, for cross-model comparison at low cost |

`testmini` is contained in `test` and is also marked there by the `testmini` flag, so either file can be used as the single source.

## Evaluation dimensions

| Key | Dimension | What it measures |
|---|---|---|
| `lpf` | Low-level Perceptual Fidelity | Perceptual quality of the synthesized video: temporal coherence, motion dynamics, and per-frame visual quality |
| `pg` | Prompt Grounding | Whether the explicit conditions stated in the prompt are instantiated — presence and identity of key entities, their initial states, spatial arrangement, and any required instruments |
| `scc` | Scientific and Causal Correctness | Whether the generated dynamics are consistent with the domain knowledge and causal mechanism the prompt targets |
| `sc` | Spatiotemporal Consistency | Whether temporal evolution and spatial relationships stay coherent enough for the intended mechanism to be interpretable |

## Fields

| Field | Type | Description |
|---|---|---|
| `id` | string | Opaque hex example identifier (40–42 chars); also the video directory name in the videos repo |
| `domain` | string | One of the four disciplines |
| `subject` | string | One of the 60 subjects |
| `selected_point` | string | The target scientific concept the example is built around |
| `posted_prompt` | string | **The generation prompt, verbatim.** The exact string given to every evaluated model — no rewriting, no expansion |
| `testmini` | bool | Whether this example belongs to the 150-example `testmini` subset |
| `high_level_evaluation_guide` | string | Target concept, the minimal mechanistic assumptions needed for the scenario to be well-defined, and the expected phenomena as a phase-based storyline |
| `manual` | string | Expert-written procedure and expected observations for the scenario |
| `pg_specification` | dict | 1–5 anchored rubric for Prompt Grounding, keyed `level_1` … `level_5` |
| `scc_specification` | dict | Same, for Scientific and Causal Correctness |
| `sc_specification` | dict | Same, for Spatiotemporal Consistency |
| `lpf_specification` | dict | Same, for Low-level Perceptual Fidelity |

## Quick start

```python
from datasets import load_dataset

full = load_dataset("Sci-VBench/Sci-VBench", split="test")       # 1,253
mini = load_dataset("Sci-VBench/Sci-VBench", split="testmini")   #   150

ex = mini[0]
print(ex["posted_prompt"])                  # feed this to your model, unmodified
print(ex["scc_specification"]["level_5"])   # what a perfect score looks like
```

To assemble the judging context for one video and one dimension:

```python
DIMS = {"pg": "pg_specification", "scc": "scc_specification",
        "sc": "sc_specification", "lpf": "lpf_specification"}

def spec_for(ex, dim):
    rubric = ex[DIMS[dim]]
    anchors = "\n".join(f"{k}: {rubric[k]}" for k in sorted(rubric))
    return f"{ex['posted_prompt']}\n\n{ex['high_level_evaluation_guide']}\n\n{anchors}"
```

Condition the judge on **one** dimension at a time. Keeping the dimensions separate is deliberate: it stops evidence for one dimension from bleeding into another, which is what makes the reasoning-centric scores track expert ratings.

## Notes on this release

- The four `*_specification` dicts use one consistent set of keys, `level_1` … `level_5`. The internal annotation files carried three different key spellings (`1score`, `1 point`, and a stray `5 point`); these were remapped by leading digit.
- Higher anchors routinely build on lower ones ("All conditions for score 3 are met, and additionally…"). Those in-text references originally used the old key spellings, so 2,293 anchors were rewritten to say `score N`. Nothing else in the rubric text changed, and no generation prompt was touched.
- An always-empty `video_type` field was dropped.
- `posted_prompt` is the exact string that produced the videos in the companion repo, so prompts and videos are guaranteed to correspond.

## License

Released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
