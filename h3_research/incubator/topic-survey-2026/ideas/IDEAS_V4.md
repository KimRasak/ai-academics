# Idea V4（Round 98，广域自查：455 篇 / 31,444 被引；用户点名区之外的自找方向）

累计文献：v1 评估域 115 + v2 前沿 220 + v3 广域 235 → **455 篇去重**。`refs/survey_v2/merged_v2.jsonl`。

## 广域 12 方向地图（含窗口状态，诚实标注）

| 方向 | 锚点（被引） | 窗口状态 |
|---|---|---|
| 多镜头叙事生成 | ShotAdapter(38)/HoloCine(37)/VideoMemory(17)/OneStory(22) | 新热，评估基准也在快速出现（EntityBench/PersonaShot/MUSS 全 2026）——**跟随者窗口** |
| 视频 AIGC 水印 | WMAdapter(68)/SleeperMark(54,T2I)/VideoMark(18)/LVMark(6)/VidStamp(7) | 活跃但**蒸馏×水印存活性无人做**（见 S6） |
| 稀疏注意力 | SparseVideoGen(184)/trainable sparse(81)/VMoBA(39) | 方法竞争激烈 |
| 机器人世界模型 | RoboDreamer(219)/Gen2Act(186)/Ctrl-World(126) | 热但数据工程重 |
| 音视频联合 | MM-Diffusion(408)/Diff-Foley(237) | 需音频模型，重 |
| Tokenizer/VAE | OmniTokenizer(139) | 预训练级算力，重 |
| Flow matching 新范式 | Shortcut(385)/SplitMeanFlow(25)/pi-flow(20) | 方法论文竞争 |
| 数据策展 | ShareGPT4Video(504)/MiraData(179) | 管线成熟，增量空间小 |
| 实时游戏引擎 | GameNGen(316) | 算力重 |
| 个性化定制 | DreamVideo(255)/MotionDirector(241) | 成熟拥挤 |
| 量化压缩 | Q-Diffusion(480)/ViDiT-Q(130) | 成熟拥挤 |
| 4D 生成 | Cat4D(189)/DimensionX(136) | 算力重 |

## 两个自查出的新候选（用户点名区之外）

### S6 ★★强推：少步蒸馏会抹掉水印吗？——加速时代的出处危机
- **观察**：全行业在蒸馏加速视频生成（Self-Forcing 系），水印工作全在 50 步教师上训练（VideoMark/LVMark/VidStamp）。蒸馏彻底改变采样轨迹——**水印在学生模型里还活着吗？没人测过。**
- **查重**：SleeperMark(54) 是 T2I 微调鲁棒；视频域+蒸馏学生角度**空白**（SPDMark/RINGet 均为扰动鲁棒，非蒸馏）。
- **做法**：①用开源水印（VideoMark/LVMark）+ 开源少步学生（Self-Forcing/CausVid Wan1.3B）系统测水印存活率（预期：大幅衰减）②**训练水印保持蒸馏**：蒸馏损失 + 水印恢复正则，让学生既快又可溯源 ③鲁棒性评测（压缩/再生/检测器）。
- **贡献**：指出加速趋势的系统性副作用（所有实时视频生成将不可溯源）+ 训练修复 = 生成+安全两社区。社会相关性强。
- **训练**：实在（蒸馏 + 水印正则，LoRA 级，1 GPU 可行）。

### S1' 多镜头一致性记忆适配器（跟随者位置）
- EntityBench/PersonaShot 证明基准需求，VideoMemory/OneStory 在做记忆模块。可行的差异化：用我们 RAFT 运动管线做跨镜头运动连续性记忆。但整体是跟随打法。

## 最终建议（供用户拍板）

1. **S6（水印×蒸馏）**——最新发现，空白最干净，社会贡献清晰，训练实在，且与已有 C1 共享全部基础设施（少步学生、测量协议、GPU 管线）
2. **C1（计算最优推理 scaling，V3 推荐）**——方法论贡献，资产复用度最高
3. S1'（多镜头跟随者）——若偏好生成侧主战场

三者基础设施高度重叠：Self-Forcing/CausVid 少步 Wan1.3B 检查点 + 我们的 seed-resolved 评估管线 + LoRA 训练能力。选定后第一周即可出第一个测量结果。
