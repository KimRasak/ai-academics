# 二次元人物 AI 学术前沿与未解真问题调研（2026.11）

> **研究初心**：不堆砌复杂公式、不拼凑三段式工程补丁、不碰任何业务私有数据。回归最纯粹的科学问题：**在 2026 年底的当下，顶会学术界（CVPR / ICCV / SIGGRAPH / NeurIPS / ICLR）在「二次元人物（2D Anime & Stylized Characters）」上到底真正关心什么？哪些赛道已经卷死？哪些本质问题经得起审稿人二号毒打且至今完全没解决？**

---

## 目录结构（整洁、零冗余、按阅读顺序编号）

```text
reports/academic_research/
├── README.md                                   # 本入口导航：盲审后「黄金二次元三强」与 5 小时试金石
├── 01_anime_landscape_2026.md                  # 2025–2026.11 学术界二次元全景图（6 条已卷死赛道 + 底层科学分歧）
├── 02_open_problems_deep_dive.md               # 全部幸存二次元问题总览与横向决选表
├── 03_redteam_and_graveyard.md                 # 审稿人二号排雷记录 & 已排除的 43 个方向墓志铭
├── 04_quick_verification_playbook.md           # 适配 osa00-h100-007 的零训练「眼见为实」试金石清单
├── 05_clean_data_and_open_tools.md             # 合规数据（#CC0 VRM + Blender MToon 1.0）与已核实 HF 开源模型清单
├── 06_clean_storylines_and_decision_matrix.md  # 完整四幕科研故事线与单兵决选矩阵
├── 07_shared_counterfactual_engine_spec.md     # 共享 #CC0 VRM + Blender 引擎规范（盲审修订版：解锁 P6/P3/P2）
├── 08_adversarial_peer_review_and_final_verdict.md # 🌟 4 位独立模拟 AC 盲审毒打报告 & 升级后录用率总表！
├── problems/                                   # 🌟 精选初心问题独立深度档案（已合入盲审升级！）
│   ├── P6_flat_shading_tjunction_depth.md          # 🥇 #1A (CVPR/NeurIPS): 固定平涂画风 T 形交点最小对翻转
│   ├── P3_beyond_facs_symbolic_affect.md           # 🥇 #1B (CVPR/ICCV/NeurIPS): 生物先验霸权（4-Gate + 3-Tier）
│   ├── P2_cel_shading_relighting.md                # 🥈 #3  (CVPR/SIGGRAPH): 真·2D 赛璐璐重打光（去原光 + 滑块 α）
│   ├── P1_cross_modal_subtext_acting.md            # 🥉 音视专精 (NeurIPS/ICCV): 原生 AV-DiT 跨模态言行不一
│   └── P4_abstract_identity_vs_shortcut.md         # ⚠️ 盲审降级归档（VRM 2.2× 大头娃娃自相矛盾）
├── experiments/                                # 🧪 极简零训练 GPU 试金石脚本（直接跑真实验！）
│   ├── README.md
│   ├── exp0_probe_node.py                          # 节点 GPU / 环境 / 已缓存开源权重探针
│   ├── exp1_p6_tjunction_pilot.py                  # P6 零外部资产 15 分钟 T 形交点翻转首发实验
│   └── exp2_p3_nonbio_affect_pilot.py              # P3 4-Gate VLM 阶梯 + 客观像素泄漏评测脚本
└── archive/
    └── general_video_survey_2026_11.md         # 早期通用视频 RLHF (C1*) / AV-DiT (S5/C3*) 调研归档
```

---

## 🧪 真实 H100 实验实证总览（12 轮 GPU 推理 + 轻量化训练，见 [`experiments/EXPERIMENT_LOG.md`](./experiments/EXPERIMENT_LOG.md)）

> **实证第一原则**：用 8×H100 真实模型推理与轻量化训练直接验证假设——**被证伪的果断杀掉，被实证的追根溯源！**

| 实证结论 | 方向 | 关键真机实验证据（100% 开源权重 + 合规数据） | 最终定位 |
| :---: | :--- | :--- | :--- |
| **🏆 #1 实证王牌** | [`P3_beyond_facs_symbolic_affect.md`](./problems/P3_beyond_facs_symbolic_affect.md)<br>*(Beyond-FACS 非生物动漫情绪符号)* | • **4 个开源 SOTA VLM × 3 种视觉编码器**（`Qwen2.5-VL-7B`, `Qwen3-VL-32B`, `InternVL3-8B`, `LLaVA-OneVision-7B`）全部在 **Gate 2 文本知识拿 100%**、Gate 1 可感知符号，但在 **144 张全彩 TV 动漫最小对** 上被微笑嘴巴强烈覆写（`InternVL3` 纯非生物仅 **6.2%** vs 生物脸红 **100%**；`LLaVA-OV` **2.5%** vs **94%**；`Qwen3-VL-32B` 等像素对照 `BIO_brows` **100%** vs `NONBIO_vein 💢` **50%**，且干净笑脸 **50% 妄想**）<br>• **优化实证（7 种方法对比）**：直接字母 LoRA 零迁移（`8.3%`）；**冻结 VLM 隐藏层异常门控 + 解耦绑定（Ours）** 无需改权重，在 **4/4 模型全部提分**（`7B: 35.7→71.4%`，未见符号 `25→66.7%`；`32B` 未见 `💢` 零样本检出 **83.3%**），且 **跨分布零样本迁移至 100% 端到端原生动漫图**（`InternVL3: 75→87.5%`，`Qwen2.5-VL-7B: 62.5→75.0%`）<br>• **生成对偶（`Qwen-Image-Edit-2511`）**：Tier-1「空洞眼」擦毁虹膜、Tier-2「额头黑影」造成 **`-75.5` 全局变暗 + `123.1` 嘴部泄漏**；区域锚定局部化优化消除 **99.7% 泄漏 & `0.00` 嘴部漂移**，同时保持 `32B` 情绪翻转 **100%** | **CVPR / ICCV / NeurIPS / ICLR 首选主攻！** |
| **🥈 #2 实证有效** | [`P2_cel_shading_relighting.md`](./problems/P2_cel_shading_relighting.md)<br>*(真·2D 平涂赛璐璐重打光)* | • **`Qwen-Image-Edit-2511`** 即使被明令禁止 3D 渲染仍彻底塌缩为 3D 径向喷枪渐变（**摧毁 48% 黑墨线**，`ink_retention = 0.518`）<br>• **Depth-Guided 法线展平 ($\alpha=0.82$) + 屏幕空间发丝投射阴影** 在平光立绘上达到 **99.2% 墨线保真 + 零喷枪污染**，肉眼质感逼近番剧赛璐璐<br>• **诚实边界（Exp 3D & Exp 12）**：强预打光立绘的去原光无法靠启发式滤波或 12 角色从零小 U-Net 解决（`+0.09 dB`），必须依托预训练扩散主干 + 大规模 VRM/Blender `(Pre-Lit, Albedo)` 配对数据 | **CVPR / SIGGRAPH 强二选（需中大规模去原光数据）** |
| **❌ 已实证证伪** | [`P6_flat_shading_tjunction_depth.md`](./problems/P6_flat_shading_tjunction_depth.md)<br>*(平涂 T 形交点深度翻转)* | • 真实 H100 压测 **全面证伪**：`Depth-Anything-V2-Small/Large/Metric` 在 400 张最小对（含无黑线平涂、细线同色、大小先验冲突、高低先验冲突、弯曲发丝编织）上全部取得 **93.8%–100%** 配对翻转一致率；`Qwen3-VL-32B` 同样取得 **95%–100%**，零平涂退化！<br>• 15 分钟真实实验避免数月无效投入 | **果断 KILLED（已归档实验证据）** |

---

## 早期理论盲审档案与备选方向

经历 **22 轮文献查新（95+ 次检索、48+ 方向排雷）** 与 **5 位独立模拟顶会审稿人二号严酷盲审**（详评见 [`08_adversarial_peer_review_and_final_verdict.md`](./08_adversarial_peer_review_and_final_verdict.md)）后，已按真实 H100 实验结果完成终极筛选：

- **实验总日志与全部可复现脚本**：[`experiments/EXPERIMENT_LOG.md`](./experiments/EXPERIMENT_LOG.md) · [`experiments/README.md`](./experiments/README.md)
- **原生音视专精备选**：[`P1_cross_modal_subtext_acting.md`](./problems/P1_cross_modal_subtext_acting.md)（傲娇/黑化跨模态模态覆写，待音视模型实测）。
- **已诚实降级/证伪**：`P6`（真实 GPU 证伪：DA2 与 Qwen3-VL-32B 均达 94–100%）、`P4`（VRM `2.2×` 放大头冒充 Q 版自相矛盾）、`P5`（须避让 `Adv-GRPO` RLHF 均质化）、`P7`（真人动捕套二次元“皮套人恐怖谷”）。
