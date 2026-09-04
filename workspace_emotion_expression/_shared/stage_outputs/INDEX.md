---
author: builder
stage: Survey
status: draft
updated: 2026-07-27
---

# stage_outputs 索引

> Survey 阶段产出的目录与一句话主旨。按时间倒序。

| 文件 | 主旨 | 状态 |
|---|---|---|
| 2026-08-03-retrieval-verification.md | 检索验证：证据升级（L1→L2）+ 补漏；全查询可复现（OpenAlex/S2/WebFetch）；方向1白空间在检索范围内确认、方向2单源、方向3动漫域内、171出处两库无果、venue网格法替代 | draft（检索记录） |
| 2026-08-03-paper-directions-synthesis.md | 方向评估组会合成：P0=风格×强度正交检验（W1，不依赖 Boss）、判别式基准+配对双切法、动漫支线带前置、agent 评测划出；默认 3 条 provisional 口径 + 白空间陈述纪律模板 | draft |
| 2026-07-24-survey-synthesis.md | 首轮五视角合成：白色空间真实、头号威胁=arousal 混淆、主张拆 P1-P4、数据合规瓶颈、实验暂缓 | draft（3D 行已被 D-003 推翻） |
| 2026-07-24-survey-retrieval-criteria.md | rigor 检索标准 + §7 白色空间再核实协议 + §8 φ 在 2D 下的边界；2026-07-27 builder 增 §7.4 已核实记录表（首行 FED-Bench） | draft（D-003 后已增补 2D 收窄） |
| _evotree_nodes/n001-fed-bench.md | （路径：`_shared/evotree/n001-fed-bench.md`）FED-Bench 节点：preprint，未触及 arousal 正交性，不占据白色空间；L2 | draft |

## 工具链卡点状态（builder 2026-07-27 实测）

| 工具 | 状态 | 备注 |
|---|---|---|
| OpenAlex API (`api.openalex.org/works`) | ✅ 通（无 key） | 2026-08-03 实测全轮可用；支持 title_and_abstract.search 短语/词 AND、filter、sort=cited_by_count:desc、cites:ID；source.id 对 2024-26 会议年份化（见 verification §0/§5） |
| Semantic Scholar API（`.env` key） | ⚠️ 时通时断 | 08-03 首轮 7 查询成功（含 FED-Bench 定位 arXiv ids），随后连续 429/500，重试 3 次仍败；1 req/s+ 重试仍不稳 |
| WebFetch → arxiv.org/abs/ID | ✅ 通 | 08-03 成功核实 FED-Bench 页面（标题/作者/摘要/三元组细节） |
| arXiv export API (`export.arxiv.org/api/query`) | ❌ 不通 | 延续 07-27；Boss 复核 "Rate exceeded." egress IP 受限；WebFetch 兜底 |
| WebSearch | ❌ 不通 | 返回模拟文本；禁止用于发现 |
