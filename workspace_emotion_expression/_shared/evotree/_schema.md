---
author: builder
stage: Survey
status: draft
updated: 2026-07-27
---

# 演化树节点 Schema

> builder 落地。演化树用于追踪课题相关方法的演化，节点=一个工作（论文/方法/benchmark），边=方法间的承袭/对比/对抗关系。
> **核心约束（来自 builder agent 定义）**：进入决策依据前，边证据必须达到 L3（人工核实+对抗探针过）；L1/L2 仅作线索，不关闭白色空间。

## 1. 节点字段

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `node_id` | string | Y | 形如 `n001-<short-slug>`，全局唯一 |
| `title` | string | Y | 论文/方法/benchmark 标题 |
| `venue` | string | Y | 正式录用会议/期刊（未录用写 `preprint`，不得留空） |
| `year` | int | Y | 正式录用年份；preprint 用首发年份 |
| `arxiv_id` | string | N | arXiv ID（若有） |
| `repo` | string | N | 官方实现 repo URL（含 commit hash，钉版本） |
| `method_category` | enum | Y | `generator` / `editor` / `benchmark` / `representation` / `metric` |
| `modality` | enum | Y | `2D-image` / `3D` / `video` / `audio`（D-003 后只纳入 `2D-image`；3D 仅作背景引用） |
| `claims_verified` | list[enum] | Y | 该工作实际验证了 P1/P2/P3/P4 中的哪几条（见 rigor §1） |
| `evidence_level_self` | enum | Y | 该节点自身证据等级 L1/L2/L3（见 §3） |
| `one_liner` | string | Y | 一句话主旨（≤120 字） |
| `citations` | list[string] | N | 支持 `one_liner` 的可点击 URL |
| `notes` | string | N | builder 装载的复现细节（tricks/坑/许可） |

## 2. 边字段（edge — 承袭/对比/对抗关系）

> **重要**：`edge_evidence` 是**边属性**，不是节点属性。一条边描述"从节点 A 到节点 B 的方法演化关系"，证据等级是这条边本身被核实的程度。

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `from` | node_id | Y | 源节点 |
| `to` | node_id | Y | 目标节点 |
| `edge_type` | enum | Y | `extends` / `compares` / `contradicts` / `evaluated-by` / `reuses-data-from` |
| `edge_claim` | string | Y | 这条边声称的关系（一句话，例如"B 继承 A 的 identity head 并改 AU 监督"） |
| `edge_evidence` | object | Y | **见 §2.1** |
| `adjudicated` | bool | Y | 是否已过对抗探针（rigor 四道闸） |

### 2.1 `edge_evidence` 子字段

| 子字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `claim` | string | Y | 边的具体论断（可被证伪的陈述） |
| `evidence_level` | enum | Y | `L1`（仅预印本/二手引用）/ `L2`（≥1 条被引论断且引用经人工核实）/ `L3`（L2 + 对抗探针过） |
| `citations` | list[string] | Y | 支持该论断的 URL 列表（必须可点击，非模拟） |
| `adjudicator` | string | N | L3 时填：执行对抗探针的角色（如 `rigor`）与日期 |

## 3. 证据等级定义（与 builder agent 定义一致）

- **L1**：仅 arxiv 预印本或二手引用；venue 未核实；**不得关闭白色空间**。
- **L2**：≥1 条被引论断 + 引用经人工核实（点击 URL 真的打开目标，非 404、非模拟）。preprint 仍可作 L2，但论断必须经原文核实。
- **L3**：L2 + 对抗探针过（四道闸：来源可信 / 方法可重复 / 数据无循环 / 反例搜尽）。**进入决策依据（决策依据=Boss/协调器据此决定方向/任务）前的最低线**。

## 4. 文件命名

- 每节点一个 md 文件：`n<n>-<short-slug>.md`，例如 `n001-fed-bench.md`。
- 文件 frontmatter 复制 §1 字段；边以 `## edges` 章节列出（每条边一个 H3）。
