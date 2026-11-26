# HF Papers 前沿收集（Round 103：2026-06 至 2026-09，用户指定 hf-papers 工具）

数据：HF Daily Papers 月度聚合 4 个月，去重 **174 篇**，其中视频/生成相关 **85 篇**（精选 32 篇细读）。
文件：`refs/hf/month_*.json`（原始）+ `refs/hf/video_frontier_2026H2.json`（视频类按 upvote 排序）。
注：HF 面板反映社区热度（upvote），与 Scholar 被引互补——捕捉尚未被引的最新 arXiv。

## 视频域五大新动向（HF 视角）

1. **视频世界模型的记忆问题自成一线**（本批最强信号）：
   - MemLearner(29)：**训练**一个"查询记忆"模块给视频世界模型
   - Addressable Memory(15)：KV cache 作为生长中的视觉记忆，可寻址持久性
   - RECAP-Forcing(6)：长视频生成的外观保持
   - Matrix-Game 3.5(19)：patch memory
   - → "怎么记、记什么、如何查询"是 2026 下半年视频世界模型的核心战场，且全在**训练**记忆模块
2. **流式实时编辑**：SANA-Streaming(42) 实时 V2V 编辑（混合 DiT）；Goku(12) 百万级指令编辑数据集
3. **原生音视频联合**：DreamX-Creator(100！) 2K 原生音视频生成；AVTok(10) 1D 统一 tokenization
4. **Forcing 家族持续迭代**：One-Forcing(8) 稳定一步 AR 视频、Representation Forcing(64) 统一多模态——少步 AR 蒸馏赛道仍活跃（对 C2-Full 有利：赛道热但我们的测量+损失角度无人做）
5. **大世界基础模型**：Orca(508 upvote，本榜第一)——Next-State-Prediction 统一世界潜空间；SimWAM(108) 端到端驾驶世界动作模型

## 与我们选题（C2-Full）的交叉验证

- HF 4 个月面板中**无**"控制保真退化测量"或"控制保持蒸馏损失"工作 → 空白再次确认
- 少步 AR 蒸馏（One-Forcing 等）持续活跃但均不测控制依从性 → 差异化成立
- BlockPilot(77, instance-adaptive speculative decoding for diffusion) 印证自适应算力趋势（影响 C1 备选的窗口判断）

## 结论

C2-Full 选型在 HF 社区热度面板下维持成立：赛道热（蒸馏 AR 持续出论文）、空白净（测量+损失无人做）、记忆/流式等新热点不与我们的能力匹配（需大模型预训练）。
