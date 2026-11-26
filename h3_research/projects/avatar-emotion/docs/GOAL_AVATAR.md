# Goal：Avatars That Keep Feeling（Round 106 立项，用户确认选题）

**Goal ID**: goal-a6c040a7-490c-4539-8f9e-5aed034d98f3 (rev 17, max 120 rounds)

## 完成定义（五项，全部达成才算完）

| # | 里程碑 | 验收标准 |
|---|---|---|
| M1 | 底座+评测管线 | 开源人像模型（Hallo3/SkyReels-Audio/EchoMimic 择优）在 GPU 上跑通；SER↔FER 一致性管线端到端出数 |
| M2 | **第一结果**：情绪塌缩曲线 | ≥2 底座 × 12 seeds × 长音频的塌缩曲线图 + 塌缩起始点/速率/seed 方差表（独立可发表的普查证据） |
| M3 | 训练消融 | 情绪忠实 LoRA（SER↔FER 奖励 + 方差正则）+ 表情自锚定，3 变体（vanilla / +忠实 / +忠实+锚定）训练完成且 held-out 音频验证 |
| M4 | 评测矩阵 | 2h 模拟直播：一致性/身份(ArcFace)/唇同步(SyncNet)/塌缩曲线四维对照 + 小规模人评 |
| M5 | 论文包 | ≥30 引用、数字齐备可溯源（沿用 verify 脚本纪律）、图表齐全、投稿系统材料（目标 ICME 2027 / ACM MM 2027 / WACV 2027 WS） |

## 分相执行（视算力自适应）

- **Phase A（现有单卡即可）**：M1 全部 + M2 的管线与首底座普查 → 第一结果
- **Phase B（需 ≥4×H100）**：M3 训练消融
- **Phase C（需 8×H100）**：M4 全矩阵 + M5 写作冲线
- 设备不足时向用户申请并陈述理由（遵守约定）

## 硬约束（沿用用户规则）

先验证后写作（M2 不过关不启动 M3 写作）；idea 含训练；引用 ≥30；WORKLOG 全程留痕；GPU 不空转；所有数字可溯源。

## 风险登记

底座许可（Hallo3 协议确认，备选 EchoMimic/SkyReels-Audio）；SER/FER 噪声（多检测器集成+置信过滤）；长会话语料成本（人像域省 3-5×）；AsymTalker/SoulX-FlashHead 同期工作（引用区分：他们主攻身份+实时，我们主攻情绪保真）。
