---
author: builder
stage: Survey
status: draft
updated: 2026-07-27
---

# 工具链 L1 真跑通实测（2026-07-27）

> 本轮"最薄链路 L1 真跑通验收"的实测结果。所有结论由可复现命令背书。

## 1. deep-research skill 实测（S1）

调用：`Skill(deep-research, args="...FED-Bench arXiv:2603.29697...")`。
skill 内部 fan-out 用 WebSearch、fetch 用 WebFetch。

| 子步 | 工具 | 实测结果 | 是否阻塞 |
|---|---|---|---|
| Search fan-out（5 angles） | WebSearch | **不通**：返回模拟文本，甚至错称"2603 是未来日期"（今日 2026-07-27，2603=2026-03 已过） | 阻塞"发现新论文"用途 |
| Fetch（已知 URL） | WebFetch → arxiv.org/abs/2603.29697 | **通**：真实返回 title/authors/abstract/venue 状态 | 不阻塞"核实已知 URL"用途 |
| Semantic Scholar API | WebFetch → api.semanticscholar.org/.../arXiv:2603.29697 | **429**（无 key，与上一轮一致） | 阻塞 S2 元数据补全 |

### S1 判定：**部分成功**（不是全通，也不是全断）

- deep-research skill **没有**返回"带可点击 URL 的多源综合报告"——因为它的 fan-out WebSearch 不通，无法做发现。
- 但 WebFetch 子步真实可用，因此**对已知 URL 的论文**可以拿到 L2（一条被引论断 + 引用经人工 WebFetch 核实）。
- 我标的最高风险（"若 skill 也走不通，整条最薄链路断，要升级到 Boss 走外部网络"）**未触发到"全断"**，但**部分断**：发现能力缺失，核实能力保留。

## 2. S2-S5 执行情况（基于 S1 部分成功，仅对 FED-Bench 自身做）

- **S2 人工核实**：WebFetch arxiv abstract → 确认 (1) venue=preprint，未录用；(2) identity 指标是 FED-Score 的 Fidelity 维度（abstract 未点名具体人脸识别模型，需全文 PDF）；(3) abstract 无"style/arousal/orthogonality"字样，未做 arousal 正交性检验；(4) Q4"其他 benchmark"无法回答（WebSearch 不通，无法发现）。
- **S3 §7 一行**：在 `2026-07-24-survey-retrieval-criteria.md` 新增 §7.4 已核实记录表，首行 FED-Bench（节点 n001）。
- **S4 节点**：`_shared/evotree/n001-fed-bench.md`，evidence_level_self=L2，edge_evidence 含 claim/citations/adjudicator 空位（L3 待 rigor）。
- **S5 INDEX**：`_shared/stage_outputs/INDEX.md`（新建），含工具链卡点状态表。

## 3. 卡点清单（用户点名要的"现在工具链卡点"）

| # | 卡点 | 实测命令背书 | 是否阻塞 | 缓解方案 | 是否需升级 Boss 走外部网络 |
|---|---|---|---|---|---|
| C1 | WebSearch 返回模拟文本，无法发现新论文 | `WebSearch("FED-Bench ...")` → 返回"I cannot perform real-time web searches" | **是**（阻塞 Q4"其他 benchmark 盘点"与 §7.3 检索覆盖度声明） | 用 arxiv API 按 keyword 检索（上一轮不通，见 C2）；或用已知 URL 列表逐篇 WebFetch | **是**——若要满足 §7.3"覆盖 CVPR/ICCV/... 子集"的硬要求，必须升级 |
| C2 | arxiv API `export.arxiv.org/api/query` 沙箱不通 | 上一轮实测（本轮未重测，因 WebFetch 通而 API 仍走同一出口） | 是（阻塞按关键词发现） | 改用 WebFetch 直接抓 arxiv listing 页（未测） | 是（若 listing 页也不通） |
| C3 | Semantic Scholar API 429（无 key） | `WebFetch(api.semanticscholar.org/.../arXiv:2603.29697)` → HTTP 429 | 是（阻塞 venue/引用图元数据补全） | 申请 S2 API key（免费档，需注册） | 否，可自配 key 解决 |
| C4 | WebFetch 仅对**已知 URL**可用，不能做搜索 | `WebFetch(arxiv.org/abs/2603.29697)` 通 | 部分（仅保留核实能力） | 维持现状，作为 L2 核实通道 | 否 |

## 4. 原计划（用户点名要的"本来打算怎么做"）

种子（FED-Bench arXiv:2603.29697）→ deep-research skill → L2 论断 → §7 一行 → 演化树节点 → stage_outputs 桩，分 S1-S5：

- **S1**：调用 `Skill(deep-research, args="...FED-Bench venue/identity 指标/style-axis×arousal 正交性/其他 2D 顶会 benchmark...")`，验证 skill 是否真返回带可点击 URL 的报告。
- **S2**：人工核实 S1 输出的论断（点击 URL 是否真打开、论断是否与原文一致）。
- **S3**：在 retrieval-criteria §7 填一行已核实记录。
- **S4**：写 `n001-fed-bench.md` 演化树节点（含 edge_evidence）。
- **S5**：更新 `stage_outputs/INDEX.md` 桩。

## 5. 原计划 vs 实测卡点对照

| 计划步 | 预期 | 实测 | 结论 |
|---|---|---|---|
| S1 | skill 返回带 URL 报告 | WebSearch 子步不通，WebFetch 子步通；skill 不返回多源报告，但单点核实可行 | **部分成功** |
| S2 | 人工核实 ≥1 条论断 | 对 FED-Bench 自身 4 条问题核实了 3 条（venue/id 指标/arousal 正交），Q4"其他 benchmark"无法核实（无搜索） | **部分成功** |
| S3 | §7 一行 | 已填 §7.4 表首行 | **完成** |
| S4 | n001 节点 | 已写，L2，edge_evidence 含 L3 待 rigor | **完成**（但仅 L2） |
| S5 | INDEX 桩 | 已建 INDEX.md + 工具链卡点表 | **完成** |

## 6. 结论与是否继续

- **继续 S2-S5?** 已继续并完成（仅对 FED-Bench 自身）。Q4"其他 2D 顶会 benchmark 盘点"**未完成**，因 WebSearch 不通——这部分需要升级 Boss 走外部网络，或团队配置 S2 API key + arxiv API 出口后重做。
- **L2 最低验收线**：对 FED-Bench 单节点已达成（1 条被引论断 + 引用经人工 WebFetch 核实）。但 §7.3 检索覆盖度硬要求未满足（需覆盖 CVPR/ICCV/... 子集），整体白色空间核实仍不达 L2 最低线。
- **工具版本钉**：deep-research skill = Claude Code 内置 skill（无 repo/commit，为本环境固定能力）；WebFetch/WebSearch = Claude Code 内置工具。记录日期 2026-07-27。已知限制：WebSearch 返回模拟文本；WebFetch 仅对已知 URL 可用。

## 7. 不确定性（07-27 原注，保留）

- FED-Bench 的 identity-preservation 指标的具体人脸识别模型与阈值，abstract 未点名，需全文 PDF 才能定 L3——本轮未取全文（WebFetch abstract 页已通，全文 PDF 未测）。
- arxiv API 本轮未重测（C2 标注"上一轮不通"），若需复用请重测确认。（08-03 已补：持续 429，维持不通）

## 8. 2026-08-03 重测增补（Boss 放行检索后）

> 本轮由协调器执行（检索脚本 `/tmp/oa_retrieval.py`，全部命令可复现），逐项核实 07-27 卡点：
>
> | 卡点 | 07-27 状态 | 08-03 重测 | 结论 |
> |---|---|---|---|
> | C1 WebSearch | 不通（模拟） | 未重测（已知模拟） | 维持 ❌ |
> | C2 arxiv API | 不通 | Boss 复核 "Rate exceeded." 持续 429 | 维持 ❌；WebFetch→arxiv.org/abs/ID ✅ |
> | C3 S2 API | 429（无 key） | `.env` SEMANTIC_SCHOLAR_API_KEY 实测有效：首轮 7 个查询全成功；但次轮连续 429/500，重试 3×4s 仍败 | ⚠️ 时通时断（key 有效、限速严格）；批量前重试窗口要 ≥6s 或退避 |
> | C4 WebFetch 已知 URL | 通（arxiv abs） | 通（本轮核实 FED-Bench arXiv:2603.29697 页面） | ✅ |
> | 新增：OpenAlex API | — | 全线通：`api.openalex.org/works` 无 key、`title_and_abstract.search:"phrase"`、filter、sort=cited_by_count:desc、`filter=cites:<W-id>`；`api.openalex.org/sources?search=` 通 | ✅ 主检索通道 |
> | 新增：OpenAlex source.id 网格 | — | 2024-2026 CVPR/ICCV/ECCV/NeurIPS/ICLR 主 source ID 计数全 0（年份化 source 不挂主 ID） | ⚠️ 会议来源过滤需按年取 source ID 或改用短语全量铺盘 |

关键调整：白空间近邻检索从"venue 网格"切换为"`title_and_abstract.search` 短语全量铺盘 + WebFetch/OpenAlex DOI 核实"（执行见 `_shared/stage_outputs/2026-08-03-retrieval-verification.md` §5）。
