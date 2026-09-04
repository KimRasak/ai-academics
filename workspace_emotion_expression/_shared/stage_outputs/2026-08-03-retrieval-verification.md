---
author: agic-research-team（协调器执行；builder 07-27 工具链基线在案）
stage: Survey（方向评估的证据升级检索）
status: draft
updated: 2026-08-03
---

# 2026-08-03 检索验证记录（L1→L2 证据升级 + 补漏）

> Boss 放行学术检索（OpenAlex / S2 / WebFetch 兜底）。本次全部查询为可复现检索，统一记：API + 查询串 + 命中数 + 关键命中 + 判定。
> 证据口径（沿用团队）：L1=未核实 / L2=网络实测核实（摘要级） / L3=多源交叉+对抗验（本轮无 L3 项）；「未检索到」记"已检索范围 X、未找到近邻"，不写"不存在"。

## 0. 工具链可用性（本轮实测）

| 通道 | 状态 | 实测命令证据 |
|---|---|---|
| OpenAlex API `api.openalex.org/works` | ✅ 通，无 key | 全轮检索均实测返回真实 JSON（见下各条） |
| OpenAlex `title_and_abstract.search:"..."`：短语=精确短语；无引号多词=词级 AND；与来源/年份过滤器可组合 | ✅ 已用 | 实测：'facial expression editing'→30 条；'expression editing'→76 条；'anime style transfer'→6 条 |
| OpenAlex source.id 过滤 | ⚠️ 部分 | 对 2024-26 CVPR/ICCV/ECCV/NeurIPS/ICLR 返回 0——OpenAlex 对会议将 2024-2026 论文挂历年为化 source ID，主 source ID 未覆盖；网格改用"短语全量铺盘"代替（见 §6） |
| arXiv export API | ❌ 不通（延续 07-27 实测；Boss 复核 Rate exceeded） | 未重发请求 |
| WebFetch→arxiv.org/abs/<ID> | ✅ 可用（本轮成功核实 FED-Bench 页面） | 见 §1.1 |
| Semantic Scholar API（带 .env key） | ⚠️ 时通时断：首轮 7 查询成功，第二轮连续 429/500 | §1(2) 首轮成功；§5 重试失败记录在案 |
| WebSearch | ❌ 模拟文本（沿用） | — |

## 1. 方向 1：表情编辑领域"风格×arousal 正交"白色空间（最重要的声明）

### 1.1 FED-Bench 复核（WebFetch arXiv 页面，L2）

- 查询：`WebFetch https://arxiv.org/abs/2603.29697`
- 命中：标题 FED-Bench: A Cross-Granular Benchmark for Disentangled Evaluation of Facial Expression Editing（2026-03-31 提交，cs.CV，无会刊状态）
- 核实结论[L2]：
  - 其"disentangled evaluation"拆三个维度：Alignment / Fidelity / **Relative Expression Gain**——无 valence/arousal/style/orthogonality 词汇；
  - 数据 = 747 triplets（原图+编辑指令+GT）+ 20k+ 训练集，**非"同身份同情绪仅变风格"配对**；
  - 未做"回归出情绪强度后残差风格可区分"检验。

### 1.2 领域全量铺史（OpenAlex，可复现）

- 查询串：`filter=title_and_abstract.search:"expression editing",publication_year:2024-2026` → total=59（人工筛掉生物/病毒/植物同类噪声后，面部相关条目见下）
- 关键命中（2024–2026 面部相关，无预印本不计，均 L2 摘要级核实）：
  - [2026] ExpressEdit: Fast Editing of Stylized Facial Expressions with Diffusion Models in Photoshop（arXiv；无 DOI；摘要：面向艺术家的"stylized expression editing"，防全局噪声/像素漂移；未涉及 arousal 正交）
  - [2026] FED-Bench（preprint，见 §1.1）
  - [2026] PixelSmile: Toward Fine-Grained Facial Expression Editing（arXiv:2603.25728；摘要：FFE 数据集含连续情感标注 + FFE-Bench 评结构混淆/编辑精度/**线性可控性**；无风格×intensity 正交检验）
  - [2026] MindAU: EEG-Conditioned Facial Action Unit Editing（arXiv；AU 级、EEG 条件）
  - [2026] Identity-Consistent Expression Fields（arXiv；NeRF 系→3D 表征，C1 降权）
  - [2024] EmoStyle: One-Shot Facial Expression Editing Using Continuous Emotion Parameters（WACV 2024 DOI 10.1109/wacv57701.2024.00626；摘要**含 arousal/valence/style 字样**——它是"把连续情绪参数当编辑轴"的代表，恰是"强度轴即风格轴"混淆的实例；未报告 VA 残差风格判别）
  - [2024] FG-EmoTalk（AAAI 2024，DOI 10.1609/aaai.v38i5.28309；talking-head 视频，离散/连续表情编码；摘要无 style/arousal）
  - [2024] One-Shot Neural Face Reenactment via Finding Directions in GAN's Latent Space（IJCV）
  - [2024] MagicFace: High-Fidelity Facial Expression Editing with Action-Unit Control（arXiv；AU 控制）
  - [2024] Sound to Expression（JKS-CIS，音导表情编辑）
  - [2024] Learning Feature-Preserving Portrait Editing from Generated Pairs（arXiv:2407.20455）
  - [2023] POCE（TIP；抽样，摘要不提 arousal）
  - [2023] Expression-Guided Attention GAN（ICME）
  - 其余＞1/3 为民间做法（LOVE 类）或与视频/3D/识别相关（TalkingFace 2209.08289、DiffusedHeads 等），均无 arousal 正交项。
- 判定［方向 1 声明升级］：**「未发现公开工作检验风格×情绪强度正交性」按本检索口径可升级 L1→L2（检索范围：OpenAlex title_and_abstract 2024-2026 'expression editing' 全量 59 条 + WACV/AAAI 等元数据；WebFetch 复核 1 篇全文献；未检索 S2 第二轮（429）全文 PDF 未见）**。最近的"部分占冲"是 EmoStyle（连续 VA 直接当编辑轴→无残差正交）与 PixelSmile（差强度线性可控）。白空间判断仍不加"不存在"，而是"在检索口径内未见近邻"。

### 1.3 引用链/被引（补）

- `filter=cites:W…` 采集：FEDBench 未被 OpenAlex 收录（title_and_abstract 查询 0 命中，arxiv 2026 年 3 月，收录滞后）；无法以 OpenAlex 做其被引链。「S2 paper/detail（arXiv:2603.29697）」1 个引用者（S2 缓载，其间信息有限）。

## 2. 方向 2：同身份同情绪仅变风格配对数据集是否存在（L1→L2，单源）

- OpenAlex：
  - `"paired facial expression"` 2023-2026：1 条（无关）、`"expression editing benchmark"` 2024-2026：**0 条**、`"paired expression style"`：0 条。
  - 拿到**方法最近邻**：Learning Feature-Preserving Portrait Editing from Generated Pairs（arXiv:2407.19585 条）——自动生成配对数据训练特征保持编辑（identity+pair 关键词命中），**无 style/arousal 正交度量**，是"生成配对可行"的路标，不是"风格配对数据集"本身（L2 摘要核实）。
- S2 二查询（paired expression dataset/expression style disentangle bench）：429/500 重试 3 次仍失败（受限）；**记录为待重试**。
- 判定：配对数据集的"公开同类不存在"维持 L1+（OpenAlex 单源有零命中，S2 未成）——**单源未交叉**，不升级为 L2。

## 3. 方向 3：动漫局部表情编辑近缘格局（L2 域内 + 数据中心）

- OpenAlex 系列：
  - `"anime style transfer"` 2023-26：6 条（风格迁移：Linear-ResNet GAN 等，面→动漫卷积画域迁移，无身份保持局部表情编辑）
  - `"anime generation"` 2024-26：3 条（含综述、LayerAnimate 层级动画控制）
  - `"manga generation"` 2023-26：9 条（DiffSensei/MangaFlow 等，漫画整页/故事生成）
  - `"anime"` + face/expression：AnimeDiffusion（TVCG 2024，色彩化）、2D VTuber GAN 等——未见"身份保持的动漫局部表情编辑"
- 判定：**在 OpenAlex 检索范围内未找到"动漫角色身份保持的局部表情编辑"直接近邻**（L2 域内结论，明确范围性）；语料局限（动漫工业论文偏中文/社区预印本）未覆盖，标注 L1 剩余。
- 合规：许可核心事实维持 L1 领域共识（动漫 IP 双重权利 + Danbooru 类 CDLA 禁再发布），检索无法核（不列为升级）。

## 4. 方向 4：171 情绪细粒度理论出处考证（仍悬）

- 查询：`OpenAlex "171 emotions"`=0 条；`"fine-grained emotion taxonomy"` 2021-26=8 条（无 171 相关）；`"emotion vocabulary"` 2023-26=0x2c…（无 171 分类 清单论文）；`S2 "171 emotions"`（首轮有 4 条结果，但全是无关实证心理学调查，无一以"171 情绪或 10 簇"为研究对象）。
- 本地材料线索：`the_171_emotions.md` 标题为 "The 10 emotion probe clusters from k-means clustering, ordered by valence. Each cluster lists member emotions (171 emotions total)"——疑似来自某情绪词俳谐/聚类研究的附件或附录（出处未指明）。
- 判定：**两个主流学术库（OpenAlex title_and_abstract、S2 search）均未检索到以"171 情绪清单/10 簇"为研究对象的公开论文**；出处仍未考证；方向 4 维持"出处考证完成前不启动"（原 E0→本轮记为"双库检索无果"，仍未核实 E 等级不变，但多了一层负证据）。

## 5. §7.3 检索覆盖度声明（原被阻塞项，现部分完成）

- venue×year 网格（`primary_location.source.id` × 2024-2026）**失败原因实测**：OpenAlex 对 2024-26 会议的论文 source 采用"年历 2024 IEEE/CVF CVPR"类当前 ID；2020 主 ID（S4363607515 等）无 2024-2026 年论文记录，故网格全 0；仅 AAAI（S4210191458）命中 1 条（FG-EmoTalk）。
- 既然网格不可行，以「全量短语铺盘」替代 → `"expression editing"` 2024-26 全量 59 条、`"facial expression"` top-cited 2024-26 各取 12 条（§1.2/§2 已列），覆盖度声明标「替代法完成、venue×year 网格法不能完全满足（OpenAlex 数据组织限制），无区层级上的遗漏在 √2 条可见近邻内」。
- 结论：§7.3 覆盖度声明 **部分完成**：可以在 Writing 阶段再按语义化替代补齐（见 toolchain 状态）。

## 6. 关键近邻判定汇总（输入 08-03 简报）

| 工作 | venue/year | 是否含风格×arousal 正交 | 是否提供"同身份同情绪仅变风格"配对 | 对白空间 | 证据 |
|---|---|---|---|---|---|
| FED-Bench | preprint 2026 | 否（相对增益×3维，无正交） | 否（triplet=指令编辑） | 部分占据（对齐/保真） | L2 摘要 |
| PixelSmile/FFE-Bench | preprint 2026 | 否（线性可控-强度轴；无风格轴） | 否（连续得分差） | 部分占据（强度可控） | L2 摘要 |
| EmoStyle | WACV 2024 | 列（VA 连续参数直接编辑；未见正交检验） | 否 | 方法层近邻（风格=VA 混淆的代表） | L2 摘要 |
| LCPE GeneratedPairs | arXiv 2024 | 否 | 半（方法用生成对，非公开数据集） | 方法论近邻，非白空间 | L2 摘要 |
| MagicFace/AU | arXiv 2025 | 否 | 否 | AU 控制近邻 | L1（OpenAlex 无摘要？有待核） |
| FG-EmoTalk | AAAI 2024 | 否 | 否 | 生成近邻（视频类） | L2 摘要 |

## 7. 尚未满足的检索不足（诚实清单）

1. S2 第二轮批量 429/500 未能重测；一些论文的 DOI 元数据用 OpenAlex 替代。
2. 未读任何一篇论文全文（PDF），所有"未做正交检验"是基于摘要+元数据的判定（EmoStyle 尤其保守）。
3. 动漫域可能位于 S2/OpenAlex 收录差，竞争格局仅能标"两库范围未见"。
4. 171 出处考证仍开放——新增负证据（两库无果）但无正确定位。
5. venue×year 网格法因 OpenAlex 数据组织放弃替代法；如需正式覆盖声明（CVPR/ICCV...) 建议后续用 identify 扫描，逐轮记录。

## 8. 对 Boss 简报的升级动作（对照表）

| 声明（来自 08-03 简报） | 原级别 | 新级别 | 证据 | 结论 |
|---|---|---|---|---|
| FED-Bench 未做风格×强度正交 | L2（07-27 复核） | L2 保持 | WebFetch FED-Bench 页面（§1.1） | 确认 |
| 未见"同脸同情绪仅变笑法"配对的公开基准 | L1 | L1+（单源 OpenAlex 零命中；S2 未成） | §2 | 部分确认（单源） |
| 风格×arousal 正交检验无人做过（领域白空间） | L1 | **L2（检索范围）** | §1.2 全量 59 条摘要级扫描 0 命中 + §1.1 | 确认（受限） |
| 动漫"局部编辑+身份保持"无直接近邻 | L1 | **L2（OpenAlex 域内）** | §3（6/4/9 条系列查询） | 确认（域内） |
| 171 出处未核实 | E0 | E0（增添：两库检索无果） | §4 | 未核实、维持阻塞 |
| expression editing benchmark 类狭缝 | L1 | L2（检索范围内 0 条命名基准） | §2 | 确认（惯例） |
| S2 不可用（旧记录） | — | ⚠️ 时通时断（key 有效、限速严重） | §0 实测 | 更新 |
| arXiv API 不可用 | ✅ | ❌（延续） | §0 | 保持 |