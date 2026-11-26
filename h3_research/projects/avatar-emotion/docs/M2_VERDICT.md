# M2 普查判决报告（Round 90）

## 实验设置
- 底座：EchoMimic V1（antgroup），参照图 a.png（中性人像），512×512，24fps，DDIM 30 步，seed 420-423
- 数据：RAVDESS 语音（8 情绪，段级 GT），前 40 段（≈146s 会话）
- Arm 1：40 chunks × 4 seeds = 160 条独立生成（n=158 有效）
- 裁判：2×ViT-FER 集成（7→4 类）+ wav2vec2-SER（音频侧）；FER 正向对照 = FER2013 真实人脸 700 张

## 关键数字
| 量 | 值 |
|---|---|
| 生成脸表现力 mean_expr | **0.079**（=1−P(neutral)，即 92% 中性概率质量）|
| 真实表情人脸（FER2013 happy/angry/sad/fear/surprise/disgust）| 0.92–0.99 |
| 真实中性人脸（FER2013 neutral）| 0.27 |
| 跨输入情绪平坦度 | angry 0.092 … fearful 0.073（极差 0.019，噪声级）|
| 跨 seed | 0.075–0.083（std 0.012，可忽略）|
| 参照表情锚定对照（b.png 微笑参照）| angry 0.142 / happy 0.183——仅微升，仍塌 |
| FER 裁判 4 类准确率（真实脸）| 48.3%（正向对照通过：裁判能看见表情）|
| SER(audio) 侧 | chunk_0000 SER=ang 0.795（音频情绪信号充分）|

## 判决（三结局表）
- **A（渐进塌缩）**：待 Arm 2 长会话漂移曲线（145.5s 连续生成 × 4 seeds，生成中）——存在"地板效应"风险
- **B（静态情绪贫瘠 / "出生即面瘫"）**：✅ **Arm 1 强证据成立**。生成脸对语音情绪完全不响应，比真实中性脸更中性；参照表情、seed 都救不回来
- **C（证伪）**：排除——裁判正向对照通过，音频侧情绪信号充分，生成侧塌缩是模型真实缺陷

## 含义（对训练设计）
1. 主现象应框架为 **"born numb" 静态情绪贫瘠**，而非单纯"越播越面瘫"；Arm 2 将补充时间轴（是否有进一步漂移）
2. **情绪忠实 LoRA（SER↔FER 一致性 + 表现力方差正则）正是对症药**：可验证奖励直接优化"脸响应语音情绪"
3. 表情自锚定不能只靠参照图静态锚定（对照证明无效）——需要生成过程中的动态锚定模块
4. Arm 1 数据集 = 训练验证集与 baseline 曲线（census_arm1_full.json）

## 产物
- `avatalk/results/census_arm1_full.json`（158 行逐条原始数据）
- `avatalk/results/census_arm1_summary.png`（汇总图）
- `avatalk/results/anchor_eval.json`（锚定对照）
- `avatalk/results/ser_probe40.json`（SER 校验）

## Arm 2 最终结果（Round 92，4 seeds 全量）
| seed | mean expr | 斜率 (/s) |
|---|---|---|
| 420 | 0.076 | -0.00013 |
| 421 | 0.073 | +0.00012 |
| 422 | 0.078 | -0.00002 |
| 423 | 0.075 | -0.00000 |

四条曲线全程平坦（均值 0.073-0.078，斜率量级 1e-4/s ≈ 零），seed 间无差异。图：arm2_drift_all_seeds.png

## 补充：Arm 2 漂移曲线（Round 91，seed420 完成）
- 146s 连续生成，40 窗 FER：mean 0.076，线性斜率 -0.00013/s ≈ 0（平坦贴地）
- 头/中/尾：0.073 → 0.073 → 0.070（无进一步塌缩，无回升）
- 结论：EchoMimic V1 上**地板效应确认**——"出生即面瘫 + 全程不变"。A（动态塌缩）在本底座不成立，但恒定低表现力（B）是更强、更根本的问题
- 论文框架调整：主现象 = static emotional poverty（可加测第二底座查时间轴泛化性，列 limitation/future work）
