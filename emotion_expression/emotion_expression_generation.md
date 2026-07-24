# 情绪向量化与表情生成研究综述

## 1. 研究问题

情绪到表情生成属于情感计算（Affective Computing）与可控人脸生成的交叉方向，通常可建模为：

$$
\text{情绪表示} \rightarrow \text{表情参数} \rightarrow \text{人脸图像、视频或 3D 动画}
$$

该方向已有较成熟的研究基础。当前重点已不只是“能否根据情绪生成表情”，而是如何实现可解释、多样化、个性化和时间连续的情绪—表情映射。

## 2. 情绪向量化方法

### 2.1 离散情绪类别

经典方法将情绪编码为 one-hot 或概率向量：

$$
\mathbf e=[p_{happy},p_{sad},p_{angry},p_{fear},p_{disgust},p_{surprise},p_{neutral}]
$$

优点是简单且标注资源丰富；缺点是难以表达情绪强度、混合情绪及连续变化。相比硬标签，soft label 更适合描述表情的不确定性与复合性。

### 2.2 Valence–Arousal / VAD 连续空间

最常见的连续情绪表示是：

$$
\mathbf e=(v,a)
$$

- **Valence（效价）**：从负面到正面；
- **Arousal（唤醒度）**：从平静到激动。

加入 Dominance（支配感）后可形成 VAD/PAD 三维表示：

$$
\mathbf e=(v,a,d)
$$

VA/VAD 表示能够插值，因此特别适合作为生成模型的连续控制条件：

$$
I_{out}=G(I_{identity},v,a)
$$

AffectNet 是这一方向的重要数据集，同时提供离散表情类别和连续 Valence–Arousal 标注。

### 2.3 FACS 与 Action Units

情绪是心理状态，表情是面部肌肉运动，两者常通过面部动作编码系统 FACS 连接。表情可表示为 Action Unit（AU）强度向量：

$$
\mathbf a=[AU_1,AU_2,\ldots,AU_n]
$$

例如，AU6 表示脸颊抬起，AU12 表示嘴角上扬；典型高兴表情通常包含 AU6 和 AU12。由此可以建立可解释的生成链：

$$
(v,a)\rightarrow \mathbf{AU}\rightarrow \text{face}
$$

AU 也可以替换为 3DMM、FLAME 或数字人的 blendshape 参数。

### 2.4 学习得到的隐空间向量

VAE、GAN、StyleGAN 和扩散模型可以学习高维情绪隐变量：

$$
\mathbf z_{emotion}\in\mathbb R^d
$$

模型通常尝试分离身份、姿态和情绪：

$$
\mathbf z=[\mathbf z_{id},\mathbf z_{pose},\mathbf z_{emotion}]
$$

隐空间表达能力较强，但不一定可解释。较有价值的方向是将 VA/VAD、AU 等可解释表示与高维生成隐空间结合。

## 3. 代表性工作

### 3.1 连续情绪控制

**Facial Expression Editing with Continuous Emotion Labels** 使用二维 Valence–Arousal 标签控制人脸表情编辑，表明模型可以在连续情绪空间中操纵表情，而不局限于少数离散类别：

$$
G(x,v,a)\rightarrow x'
$$

### 3.2 多种情绪表示的统一

CVPR 2024 的 **A Unified and Interpretable Emotion Representation and Expression Generation** 尝试统一基本情绪、复合情绪、Action Units 和 Valence–Arousal，建立如下可解释映射：

$$
\text{Emotion Category}\leftrightarrow\text{VA}\leftrightarrow\text{AU}\leftrightarrow\text{Generated Expression}
$$

### 3.3 3D 表情与数字人

**EMOCA: Emotion Driven Monocular Face Capture and Animation** 将情绪信息用于单目 3D 人脸重建与动画。模型不仅关注几何重建误差，也关注生成表情在人类感知上的情绪一致性。常见输出包括 3DMM/FLAME 表情系数、blendshape 权重、面部关键点及数字人动画参数。

## 4. 核心研究难点

情绪到表情不是一一映射，而是条件概率分布：

$$
P(\text{expression}\mid\text{emotion})
$$

同一种“高兴”可以表现为克制微笑、露齿大笑、礼貌微笑或仅有眼部变化。实际表情还受到身份、文化、场景、强度和人格影响：

$$
\text{Expression}=f(\text{emotion},\text{identity},\text{culture},\text{context},\text{intensity},\text{personality})
$$

因此，更合理的建模方式是：

$$
\mathbf x_{expr}\sim p(\mathbf x_{expr}\mid\mathbf e,\mathbf z_{identity},\mathbf c)
$$

而不是将情绪确定性地映射为唯一表情。

## 5. 潜在研究方案

一个值得探索的课题是：**可解释且多样化的连续情绪到表情生成**。

建议流程：

$$
\text{文本、语音或情绪标签}\rightarrow\mathbf e_{VAD}\rightarrow P(\mathbf{AU}\mid\mathbf e,\text{identity})\rightarrow\text{3D 或视频表情}
$$

主要设计包括：

1. 使用 VAD 表示连续情绪；
2. 使用 AU 或 blendshape 作为可解释中间层；
3. 使用扩散模型描述一种情绪对应多种合理表情的分布；
4. 分离身份、姿态和情绪；
5. 联合评估情绪一致性、身份保持、时间连续性及生成多样性。

可进一步研究：

- 文本情绪描述到 VAD/AU 及表情的生成；
- 针对不同人物的个性化情绪表达；
- 跨文化表情生成；
- “悲喜交加”“紧张但期待”等复合情绪；
- 输出表情分布而非唯一结果的不确定性建模；
- 将情绪表示为动态轨迹 $\mathbf e(t)=[v(t),a(t),d(t)]$，生成时间连续的视频表情。

## 参考资料

- [AffectNet: A Database for Facial Expression, Valence, and Arousal Computing in the Wild](https://ar5iv.labs.arxiv.org/html/1708.03985)
- [Facial Expression Editing with Continuous Emotion Labels](https://ar5iv.labs.arxiv.org/html/2006.12210)
- [A Unified and Interpretable Emotion Representation and Expression Generation](https://arxiv.org/html/2404.01243)
- [Circumplex Affect Guided Expression Inference](https://arxiv.org/html/2404.14975)
- [EMOCA: Emotion Driven Monocular Face Capture and Animation](https://arxiv.org/abs/2204.11312)
