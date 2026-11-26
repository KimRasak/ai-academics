# h3_research 目录索引

2026-09-14 起，本工作区从“按文件类型平铺”改为“按研究项目组织”。

## 当前结构

```text
h3_research/
├── projects/
│   ├── seed-sensitivity/
│   ├── emoh3-evaluation/
│   ├── avatar-emotion/
│   ├── research-work-reproduction/
│   └── h3-audio-visual-emotion-consistency/
├── incubator/
├── shared/
├── archive/
└── .cache/
```

顶层只保留上述 5 个目录和导航文件。新增内容按以下顺序落位：

1. 属于已有研究问题：放入对应的 `projects/<project>/`。
2. 尚在调研、未立项：放入 `incubator/<topic>/`。
3. 已被至少两个项目复用：放入 `shared/<component>/`。
4. 仅供追溯且不再维护：放入 `archive/<source>/`。
5. 可重新生成的中间产物：放入 `.cache/<producer>/`。

## 项目边界

- `projects/seed-sensitivity/`：VT seed-sensitivity 论文，包括 Sci-VBench/Wan/LTX 数据、审计、分析代码、论文和 ICASSP 投稿包。
- `projects/emoh3-evaluation/`：EmoH3 judge/evaluation 论文，包括 LoRA 生成实验、VLM 判官、人评、论文工作副本和自包含 release。
- `projects/avatar-emotion/`：EchoMimic 音频驱动数字人的情绪塌缩诊断和情绪忠实化训练。
- `projects/research-work-reproduction/`：科研论文的公开 benchmark、指标演进、逐主张证据账本、权重验证和不超过 8×H100-class 的训练复现。
- `projects/h3-audio-visual-emotion-consistency/`：H3 Audio-Visual Emotion Consistency 课题定义、历史清退记录与 research proposal；不承载复现基础设施。

## 共享实现

- `shared/h3-baseline/`：官方请求、基线结果与分析。
- `shared/vendor/`：第三方代码快照。
- `shared/tools/`：工作区工具。

## 旧路径速查

| 旧路径 | 新路径 |
|---|---|
| `README.md`（旧项目内容） | `projects/seed-sensitivity/README.md` |
| `data/` | `projects/seed-sensitivity/data/` |
| `audits/` | `projects/seed-sensitivity/audits/` |
| `paper/` | `projects/seed-sensitivity/paper/main/` |
| `paper_assets/` | `projects/seed-sensitivity/paper/assets/` |
| `paper_icassp/` | `projects/seed-sensitivity/paper/icassp-build/` |
| `submission/` | `projects/seed-sensitivity/submission/` |
| `avatalk/` | `projects/avatar-emotion/` |
| `h3v2/` | `projects/h3-audio-visual-emotion-consistency/` |
| `projects/h3-avec/` | `projects/h3-audio-visual-emotion-consistency/` |
| `projects/video-generation-reproduction/` | `projects/research-work-reproduction/` |
| `paper_emoh3/` | `projects/emoh3-evaluation/paper/working/` |
| `release/` | `projects/emoh3-evaluation/release/` |
| `judge_logs/` | `projects/emoh3-evaluation/results/judge-logs/` |
| `judge15_logs/` | `projects/emoh3-evaluation/results/judge15-logs/` |
| `human_label/` | `projects/emoh3-evaluation/results/human-label/` |
| `official_requests/` | `shared/h3-baseline/official_requests/` |
| `tools/` | `shared/tools/` |
| `third_party/Sci-VBench/` | `shared/vendor/Sci-VBench/` |
| `refs/survey_*` | `incubator/topic-survey-2026/references/survey_*/` |

`WORKLOG.md` 是历史日志，其中旧路径保持原样；当前操作应以本索引为准。
发布快照、远端运行证据和 `ffprobe` 元数据中的旧绝对路径同样不重写。
