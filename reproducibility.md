# AI/ML 论文复现：活跃会议、出版渠道与社区

> 调研时间：2026-07-21
> 范围：AI 主流领域（视觉、NLP、通用 ML）的论文复现，不含网络测量等话题

## 结论摘要

聚焦 AI/ML 论文复现且 2026 年仍活跃的核心实体：

- **MLRC 2026**：门槛最高但认可度最强，首次成为 NeurIPS 官方 track
- **AAAI RAI Workshop**：纯 AI 复现主题的年度 workshop，自 2019 年持续举办
- **ICML 2026 Agent Reproducibility Challenge**：社区驱动的 AI agent 复现挑战赛，正在进行中
- **ReScience C**：常年开放投稿的复现论文期刊，无截止日期压力
- **TMLR Reproducibility Certification**：滚动审稿，复现论文的正式发表渠道

## 会议与挑战赛

| 项目 | 类型 | 2026 状态 | 产出形式 | 参与门槛 |
|---|---|---|---|---|
| [MLRC 2026](https://reproml.org/) | NeurIPS 2026 官方 track | 仍可投稿，硬截止 2026-09-30 AOE | TMLR 论文 + NeurIPS 展示 | 高 |
| [ACM REP 2026](https://acm-rep.github.io/2026/) | 独立跨学科会议（第 4 届） | 2026-07-20–22 荷兰 Delft，投稿已关闭 | 会议论文 | 中 |
| [AAAI RAI Workshop](https://reproducibleai.github.io/) | AAAI 年度 workshop | 2026-01-26 新加坡已举办 | Workshop 论文（≤8 页） | 低–中 |
| [ICML 2026 Agent Repro Challenge](https://huggingface.co/spaces/ICML-2026-agent-repro/challenge) | 社区挑战赛（HuggingFace） | 正在进行：2026-07-15 → 08-02 | Trackio logbook + 数据集 | 低 |

### MLRC 2026 详情

MLRC 已连续举办八届，2026 年首次被纳入 NeurIPS 官方 track。接收论文将在 NeurIPS 2026（悉尼，12 月 6–13 日）展示。

**参与流程**：
1. 将复现论文投稿至 TMLR（openreview.net/group?id=TMLR）
2. 获得 TMLR Reproducibility Certification
3. 经 MLRC 委员会 compatibility review 后进入 NeurIPS track

**时间线**：
- 意向截止：2026-06-04 23:59 AOE（已过，但仍可自荐）
- 硬截止：2026-09-30 AOE（TMLR 决定须在此前入系统）
- 录用通知：2026-10-07

**接收范围**：
- 复现 2025 年至今顶会/顶刊 ML 论文的研究
- 正面确认、部分复现、复现失败均可
- Meta-analyses、复现方法与工具、AI 辅助复现、AI agent 可复现性研究

### AAAI RAI Workshop 详情

自 2019 年起作为 AAAI workshop 持续举办（RAI-2019 → 2020 → … → 2026），专注 AI 研究特有的复现挑战：非确定性、benchmark 方差、实验文档标准。

**2026 届亮点**：
- Session 覆盖 agent 复现、代码复现、benchmark 复现、图像复现
- 协作撰写 position paper 的工作坊环节
- 组织者：Odd Erik Gundersen, Edward Raff, Thijs Snelleman 等

**注意**：尚未看到 RAI-2027 公告，但该 workshop 已连续举办多年，预计将延续。

### ICML 2026 Agent Reproducibility Challenge 详情

社区主导的开放挑战赛，使用 AI coding agent 逐 claim 复现 ICML 2026 论文。

- 108 篇论文开放复现
- 自动化 Logbook Judge 验证 claim，结果上排行榜
- $4,000 HuggingFace GPU credit 奖金（$2,000 / $1,000 / $500×2）
- 所有完成至少一个 verified logbook 的参与者获参与证书
- 750 个 GPU-credit 名额已分配完，但挑战赛对所有人开放

## 出版渠道

| 渠道 | 特点 | 2026 活跃度 | 适合场景 |
|---|---|---|---|
| [TMLR Reproducibility Certification](https://jmlr.org/tmlr/author-guide.html) | 滚动审稿；复现论文须提供新分析或额外实验 | 持续有论文获得认证 | 严格 ML 复现 + 消融/泛化/新洞见 |
| [ReScience C](https://rescience.github.io/) | 开放获取期刊，GitHub 公开评审 | 2026 Volume 11 已出版，35 个 open issues | 随时投稿，无截止压力 |
| [JMLR MLOSS](https://www.jmlr.org/mloss/) | 开源 ML 软件论文 | 2025–2026 持续出版 | 将复现工作沉淀为开源软件包 |
| [JOSS](https://joss.theoj.org/) | 研究软件期刊 | 2026 年非常活跃 | 支撑可复现实验的研究软件 |

### ReScience C 补充

- 2026 年已发表的 ML 论文包括 "Learning with Noisy Labels [~Re]visited"
- 投稿、评审、代码测试均在 GitHub 公开进行
- 覆盖领域：ML、计算神经科学、计算机图形学等
- 投稿入口：https://github.com/ReScience/submissions

### TMLR Reproducibility Certification 补充

TMLR 是 MLRC 2026 的官方出版渠道。即使不参加 MLRC，单独获得 TMLR Reproducibility Certification 也是复现论文发表的正式途径。要求论文以复现既有 ML 工作为主要目标，同时提供新的分析或实验贡献。

## 活跃组织与平台

| 组织 | 定位 | 推荐用途 |
|---|---|---|
| [OpenML](https://docs.openml.org/) | 开放 ML 实验与基准平台 | 建立标准化复现实验、共享数据集和固定划分 |
| [MLCommons](https://mlcommons.org/benchmarks/) | MLPerf / AlgoPerf 等标准化基准 | 可重复的 ML 性能评测和跨平台比较 |
| [Center for Open Science / OSF](https://www.cos.io/) | 开放科学基础设施 | 管理复现项目、预注册、公开数据/代码 |
| [Papers with Code](https://paperswithcode.com/) | 论文-代码-benchmark 链接平台（Meta 维护） | 查找论文官方实现、验证 SOTA 声明 |
| [Research Software Alliance (ReSA)](https://www.researchsoft.org/) | 国际研究软件组织 | 研究软件质量、FAIR、软件引用 |

### 相关会议补充

- **SC Reproducibility Initiative**：超算会议 SC 的制度化 artifact evaluation（AD/AE 流程 + 结果复现徽章），偏高性能计算和大规模 ML。SC26 继续运行。
- **IRSC 2026 / RSECon26**：研究软件会议，2026 年 9 月英国 Sheffield 举办。关注研究软件质量和计算可复现性。
- **ACM Artifact Review and Badging**：ACM 多个会议采用的 artifact 认证体系（Artifacts Available / Evaluated / Results Reproduced），非独立会议。

## 按目的选择

| 目的 | 推荐路径 |
|---|---|
| 发表 ML 复现论文（顶会级） | MLRC 2026 → TMLR → NeurIPS 展示 |
| 发表 ML 复现论文（无截止） | ReScience C 或单独投 TMLR |
| 参加专门讨论 AI 可复现性的会议 | ACM REP、AAAI RAI Workshop |
| 低门槛参与复现实践 | ICML 2026 Agent Repro Challenge |
| 建立标准化 benchmark 和复现基础设施 | OpenML、MLCommons、OSF |
| 发表支撑复现的研究软件 | JOSS、JMLR MLOSS |
| 做代码与实验材料评审 | SC Reproducibility Initiative、ACM AE Committee |

## 排名限制

目前没有证据支持按"参与人数"建立精确排行榜。会议可能只公布届次而不公布参会人数；ReScience C 的 GitHub issue 数不等于参与人数；平台用户数不能与挑战赛提交数直接比较。本文排序应理解为基于公开证据的综合活跃度与机制相似度排序。

## 主要来源

- [MLRC 2026](https://reproml.org/)
- [MLRC Call for Papers](https://reproml.org/call_for_papers/)
- [NeurIPS 2026 Call for Reproducibility](https://neurips.cc/Conferences/2026/CallForReproducibility)
- [NeurIPS Blog: MLRC 成为官方 Track](https://blog.neurips.cc/2026/05/04/mlrc-2026-reproducibility-as-an-official-track-at-neurips/)
- [ACM REP 2026](https://acm-rep.github.io/2026/)
- [AAAI RAI 2026 Workshop](https://reproducibleai.github.io/)
- [RAI 2026 OpenReview](https://openreview.net/group?id=AAAI.org%2F2026%2FWorkshop%2FRAI)
- [RAI 2020 (历史)](https://folk.idi.ntnu.no/odderik/RAI-2020/)
- [ICML 2026 Agent Repro Challenge](https://huggingface.co/spaces/ICML-2026-agent-repro/challenge)
- [ICML 2026 Agent Repro (HowAIWorks 报道)](https://howaiworks.ai/blog/icml-2026-agent-reproduction-challenge)
- [ReScience C](https://rescience.github.io/)
- [ReScience C 2026 Publications](https://rescience.github.io/read/)
- [ReScience Submissions](https://github.com/ReScience/submissions)
- [TMLR Author Guide](https://jmlr.org/tmlr/author-guide.html)
- [SC26 Reproducibility Initiative](https://sc26.supercomputing.org/program/papers/reproducibility-initiative/)
- [IRSC 2026](https://www.researchsoft.org/irsc/)
- [OpenML](https://docs.openml.org/)
- [MLCommons](https://mlcommons.org/benchmarks/)
- [Center for Open Science](https://www.cos.io/)
- [Papers with Code](https://paperswithcode.com/)
- [JMLR MLOSS](https://www.jmlr.org/mloss/)
