# 术语表 / 符号约定

> rigor 维护。所有角色写作时遵循此处的术语与符号定义，保证全文一致。新术语出现时由 rigor 追加。
>
> 状态标记：`draft`（草案，可被推翻）/ `provisional`（已有部分证据）/ `stable`（已验证，本轮 Survey 不出现）。未标注模型的阈值/数据集，待 builder 核实。

## 符号约定（本轮）

- $I$：身份（character identity）
- $E$：目标情绪类别（target emotion，离散或 VA/VAD）
- $P$：姿态（pose）
- $C$：服饰/外观（clothing/appearance）
- $B$：背景（background）
- $S$：表达风格（expression style），本轮研究拟控制的轴
- $\xi$：生成采样噪声
- $\theta=(I,E,P,C,B)$：固定条件向量
- $G(\theta,s,\xi)$：生成器在风格 $s$ 下的输出图像
- $\phi(\cdot)$：表情特征空间（**D-003 后限定到 2D 表征**：AU 向量 / 限定到表情通道的 2D 嵌入；3DMM/3D 表情系数路径已剔除）
- $\Delta_{s_1,s_2}=\phi(G(\theta,s_1,\xi))-\phi(G(\theta,s_2,\xi))$：表达差分

## 待定义（本轮草案）

### expression difference / 表达差分 — `draft`

固定 $\theta=(I,E,P,C,B)$ 后，仅改变表达风格 $s$ 所产生的、在表情特征空间 $\phi$ 中的差分 $\Delta_{s_1,s_2}$。
**不**指像素级原始差分；**不**指身份/姿态/背景带来的差分。若差分中可归因于 $I,E,P,C,B$ 的分量不为零，则该差分被污染，不可作为"表达差分"证据。

### consistency / 一致性（表达差分一致性）— `draft`

一个**声明族**而非单一性质。固定 $\theta$、仅变 $S$ 时，至少包含四条**分别可证伪**的子命题，不得合并陈述：
- P1 可重复性（repeatability）：$\mathrm{Var}_\xi[\phi(G(\theta,s,\xi))]\le\sigma^2_{repeat}$。
- P2 可区分性（distinguishability）：$s_1\ne s_2$ 时，留置分类器/盲评者从 $\phi(x)$ 恢复 $s$ 的准确率高于 chance+δ（含 CI）。
- P3 可控性（controllability）：沿预期方向调整 $S$，$\phi(x)$ 单调/可解释地移动。
- P4 解耦性（disentanglement）：变 $S$ 不改变身份与情绪（见下两条）。
课题名"一致性"最直接对应 P1+P4；"稳定且可区分"=P1∧P2∧P4。**P3 是方法可用性主张，不计入"一致性"本身。**

### identity preservation / 身份保持 — `draft`

变 $S$ 不改变 $I$。操作化：人脸识别嵌入（如 ArcFace 类）在 $G(\theta,s_1)$ 与 $G(\theta,s_2)$ 间的余弦相似度 ≥ 预注册阈值 $\tau_{id}$。
模型与 $\tau_{id}$：待 builder 核实（候选：ArcFace/FaceNet；阈值需在固定数据上校准）。

### emotion preservation / 情绪保持 — `draft`

变 $S$ 不改变目标情绪 $E$。操作化（双条件）：
1. 独立情绪分类器对两种风格输出同一 $E$，概率差 ≤ $\epsilon$；
2. 预测 Valence–Arousal 漂移 $|\Delta v|,|\Delta a|$ ≤ 阈值（用于排除"风格=强度"混淆）。
分类器/VA 回归器：待 builder 核实（候选：AffectNet 训练的 VA 回归器与表情分类器）。

### expression style / 表达风格 — `draft`，关联 Q-004

受控的、定性的表情变化轴，须同时满足：(a) 不可归约为情绪类别；(b) 不可归约为 arousal/valence；(c) 不可归约为身份；(d) 遮蔽非面部区域后盲评仍可识别。
候选轴：克制/外放、眉眼主导/嘴部主导、回避/直视。是否每条都满足上述四性，需经 Q-004 的小规模预试验。

### arousal confound / 唤醒度混淆 — `draft`

最关键的混淆威胁：候选风格轴在经验上只是 arousal（强度）轴。必须通过"回归出预测 arousal 后，残差风格可区分性仍高于 chance"来排除。未排除前，任何"新风格轴"的贡献均属过度声称。

### masking check / 遮蔽检查 — `draft`

评测协议：遮蔽服饰/背景/头发等非面部区域，验证风格信号由面部承载。Q-004 关闭判据之一。

### style card / 风格卡 — `draft`

预注册的风格轴规格：标签集、定义性面部线索（如 AU 模式）、正/负样例、必须通过的遮蔽检查与情绪/身份保持测试。Q-004 的产出载体。

## 稳定区

（本轮 Survey 不出现 stable 项。）
