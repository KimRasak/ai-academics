# 审稿人二号红队排雷记录 & 已排除方向墓志铭（防重复踩坑）

> **原则**：宁可现在自己把不够硬的点子亲手毙掉，也绝不在投出半年后被 Reviewer #2 用一篇 2026 年的 arXiv 一刀秒杀。本文件汇总截至 **2026 年 11 月** 已系统审计并**排除或降级**的 **44 个方向**。

---

## 一、 二次元专项已排除 / 降级方向（25 个）

| 方向 | 处置 | 审稿人二号致命理由（已核实 2025–2026 文献） |
| :--- | :---: | :--- |
| **1. 单图拆层 / 自动 Live2D 绑定** | ❌ 排除 | 已被 **`See-through`**（SIGGRAPH 2026, `2602.03749`）与 **`Bunraku`**（`2607.27348`, 9,000 真实 Live2D 模型 + `Live2D-Bench`）正面占领。 |
| **2. 线稿上色 / 动画关键帧补间** | ❌ 排除 | 已被 `MangaNinja`（CVPR'25）、`MangaDiT`、`Cobra`（SIGGRAPH'25）、`ToonComposer`（ICLR）、`OmniColor`、`AniDepth`（SIGGRAPH'25）、`ToonCrafter`、`LayerInbetween`（SIGGRAPH'26）挤满。 |
| **3. 泛二次元音频说话头 / 动作驱动** | ❌ 排除 | 已被 `Animate-X`、`OmniHuman-1.5`、`GenFaceTalk`（ICLR'26）、`AnyTalk`（`2608.16143`）、`CapTalk`（`2605.29316`）、`TokTalk`、`LongCat-Video-Avatar 1.5`、`From Rigging to Waving`（SIGGRAPH Asia'25 TOG）挤爆。 |
| **4. 多角色二次元属性串色/身份泄漏** | ❌ 排除 | 已被 `AnyCrowd`、`FantasyPortrait`、`SDO`（`2608.13820`）、`DreamingComics`、`MultiAnimate`、`ReXMo`、`SCAIL-2` 饱和覆盖。 |
| **5. 泛二次元“3D 写实重力” (`P3`)** | ❌ 排除 | 被腾讯混元 **`AniMatrix`**（`2605.03652`, *"An Anime Video Generation Model that Thinks in Art, Not Physics"*）与 `AniSora` 抢占主叙事。 |
| **6. 泛动漫视频 12 法则 / 艺术形变评分 (`AN-4`)** | ❌ 排除 | **`AnimationBench`**（`2604.15299`, 2026.04）已系统把迪士尼 12 法则（含挤压拉伸、夸张、预备、跟随）+ IP 一致性 + VLM 缺陷定位做全。 |
| **7. 通用视频 RM 对二次元的写实偏置 (`A4`)** | ❌ 排除 | 易被批“水是湿的”OOD 常识；动机被 `AnimeReward`（`2504.10044`）占先；且 `RealGen`（`2512.00473`）证实偏好 RM 实际偏爱油腻 2.5D/CGI 而非纯粹写实。 |
| **8. 宣称“视觉侧缺动漫角色编码器” (`A3`)** | ❌ 排除 | **事实纠偏**：开源社区早有 **`deepghs/ccip`**（`KyaraBench 2610.06013` 甚至直接用它做视觉聚类），且原始 `Danbooru` 打包有版权红旗。*(注：若做 `Anime-P4` 的“互换服装/Q版反事实捷径审计”，则是拿 `CCIP` 当被测基线，完全不冲突！)* |
| **9. 复杂三拼版“一拍二 / 律表时间控制” (`A2`)** | ❌ 排除 | ① 绝不能怪“流匹配连续”（流匹配连续的是去噪时间 $s$ 而非播放时间 $t$）；② 撞 **`Making Time Editable in Video DiTs`**（`2606.10183`）+ Alvdansen 12fps LoRA；③ 硬改三拼标题过于繁杂，违背“回归初心、一眼听懂”原则。 |
| **10. 二次元非欧转头几何 (`S1`)** | ⚠️ 降级 | 撞 **`Toon3D`**（ECCV'24, Efros/Kanazawa）非欧几何叙事，且难以反驳审稿人“平滑 3D 三渲二（如《原神》）只是审美流派选择而非客观 Bug”的质疑。 |
| **11. 动漫线稿在视频中变糊/崩坏 (`S2`)** | ❌ 排除 | 易被批评为“VAE 压缩糊细高频边缘是已知常识”，且 `ToonComposer`/`SketchVideo` 紧邻。 |
| **12. 角色性格 × 情绪 (`S3`)** | ⚠️ 降级 | 撞 EMNLP ***"Trait Collapse in Text-to-Image Generation"***（`ViStoryBench-Personality` + 视觉线索词典 + 免训练 `TBV` 框架涨 +22.1%）。仅当实测加了 `TBV` 显式动作词后视频时间动力学依然坍缩时才可重启。 |
| **13. 动漫首帧“运动吸引子” (`C6`)** | ❌ 排除 | 把 12fps 动漫先验强套动力系统术语，本质是 OOD 分布先验。 |
| **14. 多镜头角色状态持久化（受伤/淋湿/换装跨切）** | ❌ 排除 | Wave 13 查新证实 2026 年已密集拥挤：`SSRS`（2026）、`EntityBench & EntityMem`（ByteDance/Rice, 2026.05）、`StoryMem`、`Unityshots`（2026.06）、`PhysTrace`（2026.10）、`Matrix-Game 4.0`（2026.10）。 |
| **15. 60fps AI 补帧 vs 一拍二的 VBench Smoothness 失效（独立做）** | ❌ 排除 | `Index-Anisora` 与 `ToonComposer` 已明确指出 VBench Smoothness 与动漫一拍二人类偏好相悖，单独拎出来新意不足。 |
| **16. 泛动漫手部/道具交互 & 文本生成精灵表/Live2D 角色** | ❌ 排除 | 已有 `SViMo`（NeurIPS）、`SyncMV4D`、`Hand2World` 与 `Textoon`（`2501.10020`）、`Spiritus`（`2503.09127`）、`Sprite Sheet Diffusion`（`2412.03685`）。 |
| **17. RLHF 标量奖励诱发高光均质化（单独做）** | ⚠️ 避让 | `Adv-GRPO`（CVF'25/26）与 `RubricRM` / `UnifiedReward-Think` 已经指出 `PickScore`/`HPS` 导致高光过饱和均质化；因此 `Anime-P5` 必须专打**纯推理期编辑/I2V 的跨流派向心收缩**。 |
| **18. 动漫多属性/异色瞳组合绑定评测** | ❌ 排除 | Wave 15 核查已有 `TRACE-Bench`、`Danbooru2025` 标签基准、`MangaGen-MetaBench` 与 `GenEval`/`DPG-Bench` 覆盖。 |
| **19. 泛动漫多视图透视/视角规范化** | ❌ 排除 | 已有 `CharacterGen`（SIGGRAPH TOG）、`CoNR`、`NOVA-3D`、`AnimeAdapter` 覆盖常规姿态/视角规范化；纯双焦距非线性透视过于垂类小众。 |
| **20. 2D 二次元长发/衣摆次级物理模拟** | ❌ 排除 | Wave 18 核查已被 **`PhysAnimator`**（`arXiv:2501.16550`, CVPR, Netflix/UCLA — SAM+Delaunay 弹性体网格+风力笔触+草图引导视频扩散）强力覆盖。 |
| **21. 动漫/吉卜力画风记忆归因与遗忘（Style Unlearning）** | ❌ 排除 | Wave 18 核查已高度拥挤：`SAeUron`、`CURE`（NeurIPS）、`AVCE`、`UnlearnCanvas`、`IntroStyle`（ICCV'25）、`CADT`（2026.10）。 |
| **22. 多镜头 180° 轴线法则与视线匹配（Eyeline Match）** | ❌ 排除 | Wave 18 核查已被 **`PersonaShot`**（`arXiv:2608.16717`, 2026.08，显式定义了 `180R` 与 `EM` 指标）、`CineMaster`（SIGGRAPH'25）与 `ShotPlan`（2026.07）正面占领。 |
| **23. 原生透明通道 RGBA 视频生成（二次元发丝抠像）** | ❌ 排除 | Wave 18 核查已密集拥挤：`Wan-Alpha`、`TransPixeler`（CVPR'25）、`UniRGBA`（2026 末）、`LayerFlow`、`TransVDM`、`TransAnimate`。 |
| **24. 漫画分镜格间空白（Gutter）转动画视频** | ❌ 排除 | Wave 21 核查已有 **`Manga2Animate`**（2026.10，多 Agent 编剧/分镜审核闭环）与 `DreamingComics` / `Stage` 正面覆盖。 |
| **25. VRM 版互换服装/Q版抽象身份 (`P4`) & 真人动捕套二次元恐怖谷 (`P7`)** | ⚠️ 降级移出首发 | ① `P4` 经模拟 AC 盲审发现 3D VRM 放大头 `2.2×` 只是大头娃娃 Bug、刚性 3D 头骨与“无骨相”论点自相矛盾（估分 3.5–4.5，已移出首发三强）；② `P7`（“皮套人感”运动-画风恐怖谷）核心论点依赖大规模人类心理物理实验（$N \ge 40\text{–}60$），单兵周期风险高。 |

---

## 二、 通用视频 / 多模态已排除方向（19 个，简要归档）

> 详细反证与论文 ID 见 [`archive/general_video_survey_2026_11.md`](./archive/general_video_survey_2026_11.md)。

- **微表情双峰死区 (`P2` 原版)**：撞 `EmoReward`（`2611.06518` subtlety 门控）与 `Stage-wise Dynamics of CFG`（ICLR'26）。
- **免训练连续情绪强度轨迹控制 (`W4`)**：被 `EmoWorld`（`2608.06231`）1:1 抢发。
- **自然待机呼吸循环 (`S4`)**：被 `DiVA`（`2609.13830`）、`Mobius`（`2502.20307`）、`DreamLoop`（`2601.02646`）占满。
- **首帧情绪状态惯性 (`P4`)** / **流式注意力汇 (`X1*`)** / **改首帧逃逸 (`N2`)**：被 `DyMoS`（`2605.19398`）、`Disen-Forcing`（`2611.02822`）、`DySink`、`GAL` 夹击。
- **少步蒸馏类 (`C4`/`C5`/`W1`/`C7`/`C8`)**：被 `VSF`、`PP-CFG`（`2611.27266`）、`ReDiverse`（`2611.18550`）、`Glance`、`Video-T1` 等围堵。
- **多 Rubric GRPO 方差支配 (`W5`)**：被 `GDPO`、`MO-GRPO`、`Flow-Factory` 挤爆。
- **多镜头跨切 (`W2`)** / **AV-DiT 音色-面容 (`W3`)** / **统一 MLLM 理解-生成割裂 (`Y`)** / **风格-运动纠缠 (`N4`)**：全已高度拥挤。
- **神经运动学定时定律 (`N1*`)** / **时间步分叉 (`N3*`)** / **身份归一化 (`C2*`)**：单兵承载门槛过高或收益不占优。

---

## 三、 合规数据红线（永久备忘）

- **坚决不打包 / 不作为商业友好主释放集**：`Sakuga-42M`（`2405.07425`，arXiv 因版权撤稿）、原始 `Danbooru` / `Safebooru` 打包、`FFHQ`（NC-SA）、`MEAD`（PI 门禁非商用）、`VoxCeleb2`（YouTube）、Pexels/Unsplash 原始打包；**注意 `AnimeRun` 实为 `CC-BY-NC 4.0`（仅限非商用学术评测，勿作商用协议宣称）**。
- **首选洁净学术与合成反事实资源**：**CC0/CC-BY `VRoid` (`VRM`) + Blender NPR/Toon 程序化渲染**（已被 CVPR `PAniC-3D` 与 `LinkTo-Anime` 确立为顶会主流合规范式！）、`LinkTo-Anime`（`2506.02733`）、`AnimationBench`（`2604.15299`）、`Live2D-Bench`（`2607.27348`）、`VideoGen-RewardBench`（Apache-2.0）、`OpenImages V7`（CC-BY）、`LibriSpeech`（CC-BY-4.0）。
