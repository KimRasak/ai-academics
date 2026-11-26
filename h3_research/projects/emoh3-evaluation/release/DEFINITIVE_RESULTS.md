# EmoH3 完整实验结果（definitive，用于论文所有数字）

> 判官 = Qwen2-VL-7B-Instruct（do_sample=False，动漫情绪分类，WORD:SCORE）
> ✓ = 判官命中目标情绪。8s/16s = 推理去噪步数。s42/s1234 = 生成 seed。4fr/8fr = 判官抽帧数。
> 生成 = 768×768×124帧，Ref2VA，reference 为立绘。全部 8×H100 实测。

## A. 基座（base，无 LoRA）
| case | 目标 | 8s s42 (4fr) | 16s s42 (8fr) | 16s s1234 (8fr) |
|---|---|---|---|---|
| x01 neutral→joyful | joyful | neutral:50 ✗ | Happy:90 ✓ | Neutral:50 ✗ |
| x02 neutral→sad | sad | neutral:50 ✗ | — | neutral:100 ✗（决定性对照） |
| x03 angry→happy | happy | Happy:80 ✓ | Happy:90 ✓ | — |
| x04 joyful→worried | worried | neutral:100 ✗ | — | — |
| x05 amused→surprised | surprised | Neutral:50 ✗ | — | — |
| x06 anxious→relieved | relieved | Neutral:100 ✗ | — | — |
| x07 sad→cheerful | cheerful | neutral:50 ✗ | — | neutral:50 ✗ |
| x08 curious→angry | angry | Neutral:50 ✗ | — | — |
| 成功率 | | **1/8** | 2/2(测试) | 0/3(测试) |

## B. LoRA v1（9 条纯保持数据，300 步）
| case | 8s s42 (4fr) |
|---|---|
| x01 joyful | Happy:90 ✓ |
| x03 happy | Happy:90 ✓ |
| x07 cheerful | happy:80 ✓ |
| 其余 | ✗ |
| 成功率 | **3/8** |

## C. LoRA v2（18 条纯保持数据，600 步）
| case | 8s s42 (4fr) | 8s s42 (8fr) | 8s s1234 (4fr) | 16s s42 (8fr) | 16s s1234 (8fr) |
|---|---|---|---|---|---|
| x01 joyful | Happy:90 ✓ | Happy:90 ✓ | Neutral:50 ✗ | — | happy:80 ✓ |
| x03 happy | happy:80 ✓ | — | happy:80 ✓ | — | — |
| x05 surprised | Surprised:80 ✓ | Neutral:100 ✗ | Anxious:70 ✗ | Neutral:100 ✗ | — |
| x06 relieved | Sad:80 ✗(部分) | sad:80 ✗(部分) | — | — | — |
| x07 cheerful | neutral:50 ✗ | neutral:100 ✗ | Neutral:50 ✗ | happy:80 ✓ | neutral:50 ✗ |
| 其余 | ✗ | — | ✗ | — | — |
| 成功率 | **3/8** | 1/4 | **1/8** | 1/2 | 1/2 |

## D. LoRA v3 = **Reference 条件化**（22 条：12 跨情绪对 + 10 保持，256² 训练，900 步）★ 论文方法
> 本节为 12 类判官记录（round 19 的 15 类全量重判见 §K/§L，**成功率以 §K 为准**；本节数字保留作历史对照）。
| case | 目标 | 16s s42 (8fr) | 16s s1234 (8fr) | 16s s999 (8fr) | ≥1 seed |
|---|---|---|---|---|---|
| x01 neutral→joyful | joyful | happy:80 ✓ | happy:80 ✓ | — | 2/2 |
| x02 neutral→sad | sad | neutral:100 ✗ | **sad:80 ✓** | neutral:80 ✗ | **1/3（sad 可表达但 seed 依赖）** |
| x03 angry→happy | happy | happy:80 ✓ | happy:90 ✓ | — | 2/2 |
| x04 joyful→worried | worried | neutral:50 ✗ | neutral:100 ✗ | — | 0/2 |
| x05 amused→surprised | surprised | neutral:100 ✗ | happy:80 ✗ | — | 0/2 |
| x06 anxious→relieved | relieved | neutral:50 ✗ | happy:80 ✗ | — | 0/2 |
| x07 sad→cheerful | cheerful | happy:80 ✓ | happy:80 ✓ | Neutral:50 ✗ | **2/3** |
| x08 curious→angry | angry | neutral:100 ✗ | neutral:100 ✗ | — | 0/2 |
| 成功率 | | **3/8** | **4/8** | 0/2 | 4 个不同 case ≥1 命中 |

**决定性对照**：x02 16s s1234 8fr —— base = neutral:100（判官），LoRA v3 = sad:80。同 seed 同步数同协议，唯一变量是 ref 条件化 LoRA → sad 突破是真实效应（使负向情绪"可表达"，但 1/3 seed，非稳健控制）。

### 身份一致性（Qwen2-VL 单帧"是否同一角色"，v3）
| case | s42 | s1234 | 稳定 |
|---|---|---|---|
| x01 | YES:100 | NO:100 | 否 |
| x02 | NO:100 | YES:100 | 否 |
| x03 | YES:100 | YES:100 | **是** |
| x04 | NO:100 | YES:100 | 否 |
| x05 | YES:100 | YES:100 | **是** |
| x06 | YES:100 | NO:100 | 否 |
| x07 | NO:0 | YES:100 | 否 |
| x08 | YES:100 | NO:100 | 否 |

→ 身份判定本身 seed 易变（仅 2/8 稳定）；情绪改变强的 case（x02 sad/x07 cheerful @ s1234）恰是身份判定 YES 的 seed —— 情绪-身份权衡可见，但单帧单模型信号噪声大，需人脸 embedding 的严谨指标。

### 身份一致性（严谨版：CLIP ViT-L/14 余弦相似度，3 帧平均；base=16s，v3=16s）
| case | base | v3 s42 | v3 s1234 |
|---|---|---|---|
| x01 neutral→joyful | 0.730 | 0.767 | 0.775 |
| x02 neutral→sad | 0.819 | 0.694 | 0.700 |
| x03 angry→happy | 0.826 | 0.808 | 0.771 |
| x04 joyful→worried | — | 0.659 | 0.730 |
| x05 amused→surprised | — | 0.808 | 0.754 |
| x06 anxious→relieved | — | 0.831 | 0.730 |
| x07 sad→cheerful | 0.838 | 0.585 | 0.758 |
| x08 curious→angry | — | 0.755 | 0.579 |

→ base 平均 0.80 vs v3 平均 0.73（共享 4 case）——LoRA 以小幅身份保真换取情绪控制；CLIP 跨 seed 稳定（|Δ| 均值 0.078，**5/8 ≤0.10**，x06=0.101/x07=0.173/x08=0.176 超线），优于单帧 VLM 判定；x07 s42(0.585)/x08 s1234(0.579) 最低，恰是情绪渲染最明显的 case。

## E. LoRA v4 = Reference 条件化 @ 768² 训练（同一份数据，900 步）
| case | 16s s42 (8fr) | 16s s1234 (8fr) |
|---|---|---|
| x01 joyful | Neutral:50 ✗ | Neutral:50 ✗ |
| x02 sad | neutral:50 ✗ | **sad:80 ✓** |
| x03 happy | neutral:100 ✗ | Neutral:50 ✗ |
| x04 worried | neutral:50 ✗ | neutral:100 ✗ |
| x05 surprised | Neutral:100 ✗ | Neutral:50 ✗ |
| x06 relieved | Neutral:50 ✗ | neutral:100 ✗ |
| x07 cheerful | neutral:50 ✗ | **happy:80 ✓** |
| x08 angry | neutral:50 ✗ | neutral:50 ✗ |
| 成功率 | **0/8** | **2/8** |

→ 768² 训练比 v3（256²）更差：正向 x01/x03 退化。**分辨率假设被证伪**；x02 sad 跨 v3/v4 都在 s1234 稳健 → 唯一跨模型新能力。

## F. 稳健性关键证据（评估协议贡献）
1. **8 步采样欠解析（旧协议有混淆，已用 15 类重判核验）**：base x01 8s(早期 4fr 判官)=neutral vs 16s(8fr 判官)=Happy:90。⚠️ 早期协议步数与判官帧数同时变化（非"只加步数"）；round 19 已补 base x01 8s 的 8fr 帧并用 15 类判官重判 8s/8fr vs 16s/8fr（固定帧数）后重核此条。
2. **抽帧敏感**：v2 x05 同 seed 同步数，4fr=Surprised:80、8fr=Neutral:100（15 类重判后重核）。
3. **seed 敏感**：v2 x01 8s s42=Happy:90、s1234=Neutral:50；v3 x02 16s s42=neutral、s1234=sad:80。
4. → 单 seed/单配置评估有误导性；多 seed × 步数 × 帧数扫描是必要协议。

## G. 跨模型 16 步 s1234 汇总（论文主表之一）
| case | 目标 | base | v2 | **v3** | v4 |
|---|---|---|---|---|---|
| x01 | joyful | ✗ | ✓ | ✓ | ✗ |
| x02 | sad | ✗(ctrl) | — | **✓** | ✓ |
| x03 | happy | ✓ | — | ✓ | ✗ |
| x07 | cheerful | ✗ | ✗ | **✓** | ✓ |
| x04/x05/x06/x08 | 负向/细腻 | ✗ | ✗ | ✗ | ✗ |

## H. 判官结果数据文件
base8: `pilot_judge_results.jsonl`; v1: `lora_judge_results.jsonl`; v2(4fr s42/s1234): `lora2_judge_results.jsonl`/`lora2s_judge_results.jsonl`; v2(8fr): `judge8_results.jsonl`; 16步控制: `control_judge_results.jsonl`; x02 base 对照: `judge_x02ctrl.jsonl`; v3: `v3_judge_results.jsonl`; v4: `v4_judge_results.jsonl`

## I. round 16：x02 neutral→sad 多 seed 扫描（16 步 × 8 帧判官，新增 5 seed；eff seed = CLI+index）
| eff seed | base | v3 |
|---|---|---|
| 777 | neutral:100 ✗ | neutral:100 ✗ |
| 778 | neutral:100 ✗ | neutral:50 ✗ |
| 779 | neutral:100 ✗ | neutral:100 ✗ |
| 780 | neutral:80 ✗ | neutral:100 ✗ |
| 781 | neutral:100 ✗ | neutral:100 ✗ |
| 合计 | **0/5**（+已知 s42/s1234=0/2 → **0/7**） | **0/5**（+已知 s42/s999/s1234=1/3 → **1/8**） |

→ **v3 的 sad 是罕见事件**：8 seed 中仅 s1234(→eff1236) 成功（1/8）；base 7 seed 从未 sad（0/7）。受控同 seed 对照（s1234：base neutral:100 vs v3 sad:80）仍成立；"使负向情绪可表达"主张据此降频为"1/8 vs 0/7，seed 特异性"。

## K. round 19：15 类判官全量重判（reviewer 驱动的成功判据修正；8fr 主 + 4fr 敏感性）
**动机**：原 12 类 taxonomy 缺 joyful/cheerful/worried 三个目标词 → "判词==目标"的成功判据对 x01/x04/x07 不可能命中（reviewer 指出，主表用 happy 同义当成功但未声明，与附录加粗逻辑矛盾）。修正 = 15 类 taxonomy（加入 joyful/cheerful/worried）+ 全量重判（所有帧目录 × 8fr/4fr，~100 clip-config）+ 同义词规则显式化（joyful/cheerful→happy）。
**判官保守性（最大发现）**：15 类 prompt 下 ~100 clip-config 判官**只输出 4 个词**（neutral×63 / happy×13 / sad×10 / surprised×2，8fr 88 例），**从不出** joyful/cheerful/worried/angry/relieved/anxious/curious/disgusted/fearful/amused → x04(worried)/x06(relieved)/x08(angry) 为**不可测任务**（无法区分模型失败与判官失明）；x05(surprised) 在生成片段中从不出现（但 refswap 里判官能认出 surprised → 模型确实没渲染）。
**15 类主表（8fr，成功=判词==目标，joyful/cheerful→happy 同义）**：
| task | base | v1 | v2 | v3 | v4 |
|---|---|---|---|---|---|
| x01 joyful | 0/3 | 1/1 | 1/1 | **2/2** | 0/2 |
| x02 sad | 0/7 | 0/1 | — | **1/8** | 1/2 |
| x03 happy | 1/1 | 1/1 | — | 1/2 | 0/2 |
| x04 worried | 0/1† | 0/1† | — | 0/2† | 0/2† |
| x05 surprised | — | 0/1 | 0/1 | 0/2 | 0/2 |
| x06 relieved | — | 0/1† | 0/1† | 0/2† | 0/2† |
| x07 cheerful | 0/2 | 1/1 | 0/1 | **2/3** | 0/2 |
| x08 angry | — | 0/1† | — | 0/2† | 0/2† |
† 判官从不出该词（不可测）。→ v3 在可测任务上 4 选 3 改善（x01 2/2 vs 0/2、x02 1/8 vs 0/7、x07 2/3 vs 0/2），x03 退化（1/2 vs 1/1）。

## L. round 19：跨 taxonomy/帧数/seed 敏感性（诚实边界）
1. **seed 翻转（双 taxonomy 复现）**：x02 v3 s42=neutral ↔ s1234=sad:80（12 类与 15 类都如此）；base x02 s1234=neutral:100（双 taxonomy）。
2. **taxonomy 翻转（12 vs 15，8fr）**：x03 v3 s42 happy→neutral；x07 v3 s42/s1234 neutral→happy；base x01 16s s42 happy:90→neutral。→ 具体格子随 taxonomy 变，但定性主张（v3 在 x01/x02/x07 ≥ base）稳定。
3. **帧数翻转（4fr vs 8fr，15 类）**：4/88 例翻转（base x03、v2 x01、refswap 2 例）；原 v2 x05 例子在 15 类下不复现（两档皆 neutral）→ 翻转稀疏且随 clip 变，恰证单配置脆弱。
4. **steps 混淆被证伪**：固定 8fr 后 base x01 8s 与 16s 皆 neutral:50（原"8 vs 16 翻转"来自帧数+taxonomy 同步变化，非步数）→ 早期 F.1 主张作废。
5. **统计诚实**：x02 v3 1/8 vs base 0/7，Fisher 双尾 p=1.00，95% CI [0.3%,52.7%] vs [0%,41%] 完全重叠 → **不显著**；决定性 s1234 对照每模型仅 n=1。定性为"受控演示 + 机制"，非显著效应。

## J. round 16：reference 互换机制测试（prompt 固定 "She looks sad."，同角色 1534125952 四张情绪立绘，16 步 × 8 帧；下表为 12 类判官结果）
> 15 类重判（round 19）复核：base 跟随 reference 在判官可靠词上成立——sad 2/2、surprised 2/2、joyful 1/2（s1234=happy:80，s42 翻为 neutral:50）；angry ref 从不出 angry 词（判官失明）。v3 仅 ref=sad 时 sad:80（双 seed），余皆 neutral → 解耦成立。

| reference 情绪 | base s42 | base s1234 | v3 s42 | v3 s1234 |
|---|---|---|---|---|
| angry | sad:80 | Neutral:50 | neutral:100 | Neutral:50 |
| joyful | **happy:80** | **Happy:80** | neutral:50 | neutral:100 |
| sad | **Sad:80** | **Sad:80** | sad:80 | Sad:80 |
| surprised | **Surprised:100** | **Surprised:100** | neutral:50 | Anxious:80 |

→ **base 输出情绪跟随 reference（6/8，双 seed 一致）**：固定 "sad" prompt 下，joyful→happy、sad→sad、surprised→surprised → **reference-emotion dominance 被直接证实**（angry ref 是弱格：s42 sad / s1234 neutral）。
→ **v3 抑制 dominance**：输出与 reference 解耦（joyful/surprised 不再泄漏情绪，多为 neutral；仅 ref=sad 时两 seed 都 sad）→ 训练确实改变了 reference 调制行为，但目标渲染仍弱（不强于 prompt 目标，多为 neutral）。机制从"推断"变"被证明"。

## 训练数据真实构成（round 14 核实，来自 traindata3/train.jsonl）
- 22 条 = **12 跨情绪对（单一角色 1534125952，ref/target 均为其 v2 立绘/视频）** + **10 保持片段（10 个不同角色 a01/a02/a03/a07/a09/a11/a16/a18/a21/a24，ref=各角色 a-series 立绘，target=该角色 base 视频）**
- 角色 ID 体系：aNN = actor 索引，时间戳 = 角色 ID（b/c/d 系列证实同一角色可有多表情立绘）
- **评估 8 case 覆盖 7 个不同角色**（a01/a02/a03/a07/a11/a15/a18）；其中 **a15（x01/x02 用）不在训练集**
- ⇒ **未见角色泛化证据**：v3 在 a15 上 x01 2/2、x02 1/3（含 sad 突破）→ 非记忆训练角色
- 论文已修正"单角色"错误说法，改为"cross-emotion 监督单角色 + 保持片段 10 角色"

## M. round 19b：决定性 x02 s1234 判官协议稳健性 + v1 8 帧重判（reviewer 二轮响应）
**判官协议稳健性**（同判官 Qwen2-VL-7B，同 8 帧，决定性两 clip；`judge_robust.jsonl`）：
| 协议 | v3 x02 s1234 | base x02 s1234 |
|---|---|---|
| 8 帧分类（原始判词） | **sad:80** | neutral:100 |
| 单中间帧分类（15 类） | **sad:70** | neutral:50 |
| 4 帧分类（15 类） | **sad:80** | neutral:50 |
| 8 帧拼图单图分类 | neutral:50 | neutral:50 |
| 8 帧二元"是否悲伤" | yes | **yes**（yes 偏置，不可用） |
→ 15 类分类协议下决定性判词**对帧数稳健**（v3=sad、base=neutral）；二元提问有 yes 偏置（两 clip 都 yes），如实披露不使用。

**v1 8 帧重判**（`judge15_lora8.jsonl`，主表从 4 帧改为 8 帧）：
x01=Happy:90(1/1)、x02=neutral:100(0/1)、x03=Happy:90(1/1)、x04=neutral:100(0/1)、x05=Neutral:100(0/1)、x06=sad:80(0/1，sad≠relieved)、**x07=neutral:50(0/1，4 帧时为 happy → 又一个帧数翻转实证)**、x08=neutral:100(0/1)。
→ 主表现在全 8 帧一致（此前 v1 列为 4 帧判词，reviewer 指出后重判修正）。

## N. round 19c：第二独立判官（Qwen2.5-VL-7B）——决定性 x02 判词为判官特定

**判官**：Qwen2.5-VL-7B-Instruct（与 Qwen2-VL 完全不同的 VLM 家族），greedy 解码，同一 15 类协议、同一批帧。
**数据**：`release/judge_logs/judge25_results.jsonl`（2 决定性 clip + 35 张自检立绘）。
**决定性 x02 s1234**：v3 = **neutral:85**（Qwen2-VL 判 sad:80）；base = neutral:95（与 Qwen2-VL 一致）。
**自检准确率**：35 张已知表情立绘 → **0/35 严格正确**（连已知 sad、surprised 立绘都判 neutral:85），7/35 宽松（joyful→happy）。

**诚实结论**：
1. x02 的 sad:80 **不被第二判官佐证** → 判词是 **Qwen2-VL 特定**的，论文如实声明"not judge-invariant"。
2. 但 Qwen2.5-VL 全局更保守（0/35，只会说 neutral/happy），其 neutral 是**低信息量默认输出**；它的"分歧"不是对视觉信号的反证。
3. 两判官判词不一致（3/21 vs 0/35；sad:80 vs neutral:85）是"评估是薄弱环节"论点**最强的直接证据**（判官选择本身就是不稳定性来源）→ 已写入论文摘要/Result 1/Finding A/结论。
4. 论文 x02 定位：受控、非显著、**判官特定**示范（不声称判官不变性）；主线贡献转向"评估不稳定性的可复现记录"。

**与 12 类协议一致性**：决定性 x02 只在 15 类主协议下报告；未用 Qwen2.5-VL 跑 12 类（低信息量判官无需额外协议轴）。

## O. round 20（round-4 reviewer 阻塞项修复——完整披露）

两位 reviewer round-3 汇合：11e9e8de=Borderline（阻塞项：judge_robust.jsonl 的 8 帧拼图 Neutral:50 被省略=选择性报告）；d485f11d=Weak Accept（4 处小不一致）。均已修：
1. **拼图完整披露**：Finding A 现在报告全部稳健性单元——单帧 sad:70、4 帧 sad:80、8 帧 sad:80（均逐帧呈现，对"逐帧数量"稳健）；8 帧拼成单图 → v3 翻转为 neutral:50（base neutral:50）——作为"呈现方式敏感"实例报告，契合论文论点。删除"frame-count-robust"，改为"robust to the number of individually presented frames"。
2. **base 措辞**："neutral:50–100"→ 明确 "neutral:50/50/100 (single/4fr/8fr)"。
3. **限制 (i)**：改为两判官（3/21 与 0/35），不再说 "single unvalidated VL model"。
4. **appendix "9 of 15"→"11 of 15"**；完整版协议段同步。
5. **"competent on exactly sad/surprised"**→诚实版 "labels known sad/surprised correctly at least sometimes (sad 1/3, surprised 1/2)"。
6. **两判官比较不可比注记**：21 curated vs 35（超集），相对保守性指示而非严格 head-to-head。
产物：ICASSP 4pp/0 overfull（release/icassp2027/）；generic 8pp、icme 6+1pp、neurips 9pp 全 0 overfull。
状态：两位 reviewer round-4 均 Weak Accept（无阻塞项）→ round-5 携强判官证据寻求升级。

## P. round 21（strong-accept 轮：四判官全景 + Qwen3.8-27B 最强判官佐证）

用户指令："用 3.8"（最新 Qwen 代判官）。部署 **Qwen3.8-27B**（原生多模态，Qwen3_5ForConditionalGeneration，52G bf16，1×H100），同 15 类协议同帧；另补跑 **Qwen3-VL-32B** 与 **Qwen2-VL-7B 全 35 张立绘**（此前只跑 21 张），四判官严格同口径。

### P1. 判官有效性（同一 17 张词表内立绘，严格=精确匹配；宽松=joyful/excited/cheerful→happy）
| 判官 | 严格(精确) | 宽松(同义词族) | 负情绪正确 | 能否识别清晰哭泣立绘(v2raw_sad) |
|---|---|---|---|---|
| Qwen2-VL-7B | 1/17 | 4/17 | 0/5 | **sad** ✓ |
| Qwen2.5-VL-7B | 0/17 | 5/17 | 0/5 | surprised ✗ |
| Qwen3-VL-32B | 1/17 | 6/17 | 0/5 | surprised ✗ |
| Qwen3.8-27B | 0/17 | 7/17 | 0/5 | fearful ✗ |
宽松=论文同义词族规则（joyful/cheerful/happy 一族；窄规则只认 happy 的旧数字 2/17、3/17 作废）。
**任何规模/世代的开放判官都无法通过基本动漫情感自检，且对负情绪（sad/angry/worried/anxious/disgusted 立绘）全部 0/5**——"评估是薄弱环节"在所有尺度成立。

### P2. 决定性 x02 s1234（同 clip 同协议四判官）
| clip | Qwen2-VL | Qwen2.5-VL | Qwen3-VL-32B | Qwen3.8-27B |
|---|---|---|---|---|
| v3 x02 (neutral→sad) | **sad:80** | neutral:85 | worried:85 | **sad:82**（两次判读 sad:85/sad:82，含显式视觉推理） |
| base x02 (control) | neutral:50 | neutral:95 | neutral:95 | neutral:78 |
**base x02 四判官全 neutral（一致）**；v3 x02 在原始判官与**最强判官**下均为 sad。判成非 sad 的两个判官（Qwen2.5-VL→neutral、Qwen3-VL-32B→worried）恰是**认不出哭泣立绘**的两个（都判 surprised）——分歧被解释为"异议判官的负情绪识别缺口"，而非情感缺失。

### P3. 之前"不可测"任务解锁
- **x04 (joyful→worried)**：7B 判官从不说 worried（不可测）；Qwen3-VL-32B=worried:95、Qwen3.8-27B=worried:80 → **v3 x04 可测且成功**。注意两新判官对 subtle-sad 立绘也有轻微 worried 偏置（a18_sad→worried），故 x04 是带已知警示的成功，非干净验证。
- x08 (curious→angry) 仍 neutral/neutral（新判官也不认 angry）。

### P4. 诚实定位（写进论文）
1. x02 sad 由**原始判官 + 最强可用开放判官**（Qwen3.8-27B，最新 Qwen 代）独立佐证 → 从"judge-specific artifact"升级为"bounded, judge-dependent but independently corroborated"。
2. 但**没有判官通过一般自检**（1/17、0/17、1/17、0/17），x02 仍是单 seed(1/8)、非显著（Fisher p=1.00）的单点判断，不是稳健效应。
3. "评估不稳定"论点从两判官（3/21 vs 0/35）扩展到**四判官三代两尺度**（Qwen2-VL→Qwen3.8），全部失败自检 → 论点显著加强。
4. Qwen3.8-27B 判 v2raw_sad=fearful:85（不是 sad），故其 x02=sad 是 clip-specific 判断，不声称其 sad 识别能力。

### P5. 数据文件（均已在 release/judge_logs/）
judge38_results.jsonl（53 行，Qwen3.8-27B 原始含推理输出）、judge38_one.jsonl（v2raw 补判）、judge3vl_results.jsonl（49 行，Qwen3-VL-32B）、judge_extra.jsonl（v2raw 三判官）、judge15_35.jsonl（Qwen2-VL 全 35 张）、MASTER_4JUDGE.md（四判官主表）。

### P6. 论文更新（round-5）
ICASSP：摘要 (i)(ii)(iii)、Result 1(c)、Finding A（判官但受限、佐证表 Table 3=tab:judges）、Finding C（x04 解锁）、限制 (i)、结论——全部更新为四判官叙事；内容 4 页 + 参考文献第 5 页，0 overfull 0 错误。generic 8pp/icme/neurips 同步（5 处核心编辑 + 四判官表）。md 同步。

## Q. round 22：全 88 配置 × 多判官一致性（把"评估是薄弱环节"变成统计量）★ 决定性负结果

**动机**：两位 reviewer 一致认定论文真正的贡献是评估不稳定性，但此前只有少数 clip 的轶事证据。本轮把**全部 88 个 clip-config**（= 附录主矩阵，`frames_all/`）在**同一 15 类协议、同一批 8 帧、greedy** 下过多个判官，得到可统计检验的一致性数字。脚本：`analyze_agreement88.py`（可复现，从原始日志推导）；判官日志：`judge_all88_q25.jsonl`、`judge_all88_q3vl.jsonl`、`judge_all88_q38.jsonl`（进行中）、`judge15_8fr.jsonl`（Qwen2-VL 原有 88 行）。

### Q1. 判官间一致性（88 配置，三判官已完成）
| 指标 | 原始 15 类 | 同义词折叠(joyful/cheerful→happy) |
|---|---|---|
| Qwen2-VL vs Qwen2.5-VL | κ=+0.493 / 一致 71.6% | κ=+0.493 / 71.6% |
| Qwen2-VL vs Qwen3-VL-32B | κ=+0.240 / 一致 42.0% | κ=+0.348 / 51.1% |
| Qwen2.5-VL vs Qwen3-VL-32B | κ=+0.328 / 一致 47.7% | κ=+0.425 / 56.8% |
| **Fleiss' κ（三判官）** | **+0.318** | **+0.396** |
| 三判官完全一致的配置 | 29/88（33.0%） | 37/88（42.0%） |

**同一协议、同一批帧、同一批 clip，跨代判官之间只有 42–52% 的一致率（Qwen2-VL vs Qwen3-VL-32B），三判官完全一致仅 33%，Fleiss' κ=0.32（原始）/0.40（同义词折叠），均落在 "fair" 区间** —— 这是 benchmark 级的定量结论，远强于之前的个案轶事。（同代 7B 之间较高：κ=0.49/71.6%，说明分歧主要来自**世代与规模**，不是随机噪声。）

### Q2. 判官用词量（88 配置）
- Qwen2-VL-7B：**只用 4/15 类**（neutral 63、happy 13、sad 10、surprised 2）
- Qwen2.5-VL-7B：6/15 类（neutral 47、happy 26、sad 10、angry 2、worried 2、surprised 1）
- Qwen3-VL-32B：9/15 类（neutral 23、happy 19、sad 12、surprised 10、worried 9、joyful 6）
原判官的 neutral 占比 **63/88 = 72%**，是最强的中性偏置来源。

### Q3. ⚠️ 撤回：所谓"符号随判官翻转"是配置池不匹配造成的假象
两位 reviewer 独立指出同一个缺陷，**他们是对的，此前的表述已撤回**。全池聚合的 base(21) 与 v3(31) **任务不匹配**：
- 88 配置池本身异质：16 步迁移任务 54、reference-swap 机制探针 16、x02 seed 扫描 13、非 16 步/4 帧 5。
- base 只在 **4 个任务**（x01/x02/x03/x07）上有 16 步片段，v3 覆盖全部 8 个任务 × 2 seed；base 的分母还混进了按"跟随参考"计成功的 refswap 探针。
- 因此 −8.1pp 是分母组成差异，不是效应反向。（另：三个聚合差值的 Fisher p ≈ 0.49/0.50/0.42，本来就全为 null。）
- 同时修正脚本 bug：`frames_refswap_lora_*` 原被误分到 v1，实为 v3（v3 组因此从 23 → 31 配置）。

**任务匹配后的正确结论**（base 16 步存在的 4 个任务）：
| 打分 | base | v3 s42 | v3 s1234 |
|---|---|---|---|
| Qwen2-VL 原始选项序 | 1/4 | 2/4 | **4/4**（Fisher p=0.143） |
| Qwen2-VL 选项序反转 | 2/4 | 3/4 | 4/4 |
| Qwen3-VL-32B 原始 | 3/4 | 3/4 | 3/4 |
| Qwen3-VL-32B 反转 | 3/4 | 3/4 | **4/4** |

→ **任何判官、任何选项序下都不存在 base > v3**。诚实结论是"**增益在更强判官下不可复现（打平），因为它们本来就认可 base**"，而不是"符号翻转"。且原判官下的 1/4→4/4 亦不显著（Fisher p=0.143）。

### Q4. ★ 新决定性发现：判官对**选项顺序**都不稳定（真正的薄弱环节证据）
同一判官、同一批帧、greedy 解码，**只把 15 个候选词的排列顺序反转**：
| 判官 | 自一致率 | κ | 变更数 |
|---|---|---|---|
| Qwen2-VL-7B | **80.7%** | 0.651 | **17/88** |
| Qwen3-VL-32B | **88.6%** | 0.857 | 10/88 |

**而且变更精确落在论文结论依赖的格子上：**
- `base16__x01`（base 在 x01 上失败，是 Finding B 的支柱之一）**neutral → happy**：换个选项序，base 就成功了。
- `lora3s__x02`（**决定性片段**）在 Qwen3-VL-32B 下 **worried → sad**：它原本的"异议"本身就是选项序的产物。

**对照：test–retest 完全稳定**（两个并发进程各判了同一批 88 配置）：88/88 情绪词完全相同，仅 1 例置信分不同（happy:80 vs happy:90）。→ 不稳定性来自**选项顺序这一装饰性变量**，不是采样噪声。

### Q5. 决定性 x02 格子在 4 种打分下的表现（比原论文报告的更稳）
| 片段 | Qwen2-VL 原 | Qwen2-VL 反转 | Qwen3-VL 原 | Qwen3-VL 反转 |
|---|---|---|---|---|
| v3 s1234 x02 | **sad** | **sad** | worried | **sad** |
| base16 x02 | neutral | neutral | neutral | neutral |
| v3 s42 x02 | neutral | neutral | neutral | neutral |

→ base 对照在 **4/4** 种打分下都是 neutral；v3 s1234 在 **3/4** 下判 sad，唯一的异议在选项序反转后也变成 sad。这让 x02 演示比原先报告的更稳，但同时说明**任何单一打分（包括判 sad 的那些）都不可信**。

### Q6. 一致率被共享的 neutral 默认值抬高
| 判官对 | 全池 | 至少一方非 neutral | 双方都非 neutral |
|---|---|---|---|
| Qwen2-VL vs Qwen2.5-VL | 71.6% | **44.4%** | 95.2% |
| Qwen2-VL vs Qwen3-VL-32B | 42.0% | **33.8%** | 88.0% |
| Qwen2.5-VL vs Qwen3-VL-32B | 47.7% | **45.7%** | 88.9% |

→ 分歧几乎全部是"**是否存在情绪**"（一方判 neutral），而一旦双方都认定有情绪，选词几乎一致（88–95%）。这是对判官失效模式的精确刻画：不是分类混乱，而是**中性偏置阈值不同**。

**对论文的影响（已落地）**：撤回"符号翻转"；改为"增益在更强判官下不可复现"+"判官连选项序都不稳定，且不稳定性落在决定性格子上"；披露 88 配置池的异质组成；补 Fisher p；补非 neutral 一致率。

**待补**：Qwen3.8-27B 全 88 配置（76/88）；base 在 x04–x08 的 16 步生成（使 base/v3 全 8 任务任务匹配，两位 reviewer 共同指定的关键实验）。

---

## R. 判帧时间覆盖（round-7 决定性发现，两判官独立确认）

**R1. 原协议只看了片段的前 19%。** 全项目使用的抽帧脚本 `extract8_frames.py` 以 `-ss t` 取帧，`t = i/7` **秒**（i=0..7），即 8 帧全部落在 **0–1.0 s**；而每个片段是 **5.17 s / 124 帧 / 24 fps**。也就是说所有判官只看到片段开头约 19%，**而开头恰恰是参考立绘表情最可能残留、扩散过程尚未展开的部分**。这一点此前从未被披露，两位 reviewer 也没有发现。

**R2. 改为全片段抽帧后，同一判官在同一片段上改判约三分之一。** `extract8_full.py` 在整个时长上均匀取 8 帧（t = 0 … dur−0.05），其余一切不变（同判官、同 15 类、同提示、greedy）。内部一致性验证：全片段集的 f01 与原集的 f05 是同一帧（字节数相同，t≈0.72 s），证明两套帧来自同一视频。

| 判官 | n | 自一致率 | Cohen κ | 改判数 | neutral→有情绪 | 有情绪→neutral |
|---|---|---|---|---|---|---|
| Qwen2-VL-7B | 81 | **66.7%** | 0.459 | **27/81** | 24 | 2 |
| Qwen3-VL-32B | 81 | **63.0%** | 0.549 | **30/81** | 11 | 18(改判到别的情绪) |

**这是本项目发现的最大单一测量方差来源** —— 大于判官身份（跨代一致率 41.6–47.7%）、大于选项顺序（改判 10–17/88）。

**R3. 而且它是系统性偏差，不是噪声。** 非 neutral 判决数：Qwen2-VL **23/81 → 45/81**，Qwen3-VL-32B **60/81 → 70/81**。命中目标情绪的比率（配对精确 McNemar）：

| 判官 | 臂 | 0–1 s | 全片段 | 配对 p |
|---|---|---|---|---|
| Qwen2-VL | base | 4/18 | 9/18 | 0.125 |
| Qwen2-VL | v3/LoRA | 13/63 | **23/63** | **0.002** |
| Qwen3-VL-32B | base | 8/18 | 10/18 | 0.625 |
| Qwen3-VL-32B | v3/LoRA | 20/63 | **33/63** | **0.001** |

**这是全项目唯一达到显著性的效应，而它是关于评测协议本身的**：情绪在片段开头基本不存在，只在后段展开；只看开头会系统性低估情绪控制能力。

**R4. 它直接推翻本文自己的核心正面前提。** 决定性任务 x02（neutral→sad）的 base 对照：

| 判官 | base x02 判为 sad（0–1 s） | base x02 判为 sad（全片段） |
|---|---|---|
| Qwen2-VL | **0/6** | **2/6**（s777, s780） |
| Qwen3-VL-32B | 1/6 | 2/6（s777, s778） |

即"**base 从不表达 sad，所以 v3 让第一个负面情绪成为可能**"这一论断，是**只看片段前 1 秒造成的假象**。同理，base 在 x01 / x07 上的"失败"在全片段判帧下变成成功（`base16__x01`、`base16b__x01`、`base16b__x07` 均 neutral→happy）。

**完全配对的任务匹配矩阵（同 4 个任务、两种判帧协议、8 个键在两套日志中都存在）**：

| 判帧 | Qwen2-VL base | Qwen2-VL v3 | Qwen3-VL-32B base | Qwen3-VL-32B v3 |
|---|---|---|---|---|
| 前 1 s（主协议） | **1/4** | 2/4 | 3/4 | 3/4 |
| 全片段 | **3/4** | 3/4 | 3/4 | 3/4 |

而决定性任务 x02 上，全片段判帧后两臂命中 sad 的比率**完全相同**：base 2/6 与 v3 4/12（两个判官都是这个数）。**因此本文的任务级正面结论全部撤回。**（注意早期报告的"base 1/4 → 3/3"是 62 片段中间集的产物，分母不匹配，已作废。）

**R5. 唯一在四种打分条件下方向一致的仍然是机制项。** 参考情绪跟随率（refswap，prompt 固定 "She looks sad."）：

| 判官 | 判帧 | base | v3 |
|---|---|---|---|
| Qwen2-VL | 0–1 s | 5/8 | 2/8 |
| Qwen2-VL | 全片段 | 3/8 | 2/8 |
| Qwen3-VL-32B | 0–1 s | 4/8 | 2/8 |
| Qwen3-VL-32B | 全片段 | 4/8 | 2/8 |

四种条件（2 判官 × 2 判帧协议）下方向一致（base ≥ v3），但幅度小、n=8、无一显著。**按 ≥2/3 多数共识标签**：base 4/8 对 v3 2/8，2 个不一致配对全部指向抑制，精确 McNemar **p=0.50**；单看最敏感的判官（Qwen2-VL）是 3 个不一致配对、p=0.25。**这是本文唯一敢主张的模型侧结论，且必须标注为欠功效（只报方向，不报效应）。**

**R5b. 全片段判帧是"更可靠"吗？—— 只有弱支持，不能强主张（这是一次没能支持强结论的检验，如实记录）。** 可靠性不需要真值：如果换成全片段后独立判官彼此**更一致**，说明更长的窗口在更稳定地测量同一个量。在 97 个双协议共同片段上，三个判官对全部三个配对都朝同一方向移动，但**只有一个显著**：

| 判官配对 | 前 1 s（一致率 / κ） | 全片段（一致率 / κ） | Δκ | 配对 bootstrap P(Δκ≤0) |
|---|---|---|---|---|
| Qwen2-VL vs Qwen3-VL-32B | 50.5% / 0.348 | **56.7% / 0.462** | **+0.114** | **0.035** |
| Qwen2-VL vs Qwen2.5-VL | 71.1% / 0.487 | 66.0% / 0.498 | +0.011 | 0.448 |
| Qwen2.5-VL vs Qwen3-VL-32B | 56.7% / 0.422 | 53.6% / 0.424 | +0.001 | 0.486 |

**因此只能说全片段判帧"不更差、可能更好"，不能说"系统性更可靠"。** 论文里已按这个强度改写（`reliability_all_pairs()` in `analyze_basefull.py`）。真正无歧义的只是标签移动的**方向**（neutral→committed）。可靠性也不等于正确性，后者仍需人工标注。

**R6. 论文定位因此彻底收敛为测量学论文。** 三个正交且成本极低的评测选择 —— 判官身份、选项顺序、判帧时间覆盖 —— 分别改变 32–58%、11–19%、33–37% 的判决；其中判帧覆盖是系统性的、显著的，并且**足以翻转本文自己的结论**。可复现脚本：`analyze_framespan.py`、`extract8_full.py`、`judge_frameset.py`。

**待补**：base 全 8 任务 × 2 seed × 16 步生成（`run_basefull4.sh`，4 卡可续跑，节点被他人抢占时自动重试）→ 抽帧（两种协议）→ 四判官重判 → 完全任务匹配 × 两种判帧协议的最终矩阵。

## S. 完全任务匹配矩阵（round-9 决定性对照，8 任务 × 2 种子 × 16 步）

**为什么做这个实验**：两位 reviewer 都指出，之前的 base-vs-v3 比较里 base 只有 4 个任务有 16 步片段，任何汇总都混了任务构成，"成功率差异"可能完全来自分母。于是我们**用与 v3 评测完全相同的推理设置，把 base 在全部 8 个任务 × 2 个种子上重新生成**（`run_basefull4.sh`，16 个 mp4 全部产出），再用同一套 15 类判帧协议（前 1 秒 8 帧，贪心）判官。**这是本项目唯一一个完全配对、无分母歧义的模型侧比较。**

| 判官 | base | v3 | Fisher p | 配对 McNemar p |
|---|---|---|---|---|
| Qwen2-VL-7B | 4/16 | 6/16 | 0.704 | 0.688 |
| Qwen2.5-VL-7B | 6/16 | 6/16 | 1.000 | 1.000 |
| Qwen3-VL-32B | **9/16** | 8/16 | 1.000 | 1.000 |

**结论：三个判官全部零结果，其中最强的判官给 base 的分数还更高。** 参考条件 LoRA 在完全匹配的任务矩阵上**没有任何可测的任务级增益**——这不再是"换一种判帧协议后增益消失"的推论，而是配对实验的直接测量。

**两个附带的、独立摧毁旧对照的观察（都在主协议下，无需全片段判帧）**：
1. **base x02 s42 被 Qwen3-VL-32B 判为 sad** → "base 从不表达 sad"的对照直接失效（此前该结论依赖 base 只有 x02ctrl 一个片段）。
2. **base x08 s1234 被 Qwen2-VL 和 Qwen3-VL-32B 都判为 angry** → "frown 类负面目标不可达"的边界也被 base 自己突破。

因此 Finding A / Finding B 的撤回现在**建立在配对实验之上**，不再依赖 base 侧不显著的变化。日志：`judge15_logs/judge_basefull_{q2,q25,q3vl}.jsonl`（q38 判官仍在跑）；脚本 `analyze_basefull.py`。

## T. 三判官 × 97 片段的判帧协议轴（round-9 更新，取代此前 n=81 两判官的数字）

新生成的 16 个任务匹配 base 片段在两种判帧协议下都判了，因此判帧轴的样本从 81 升到 97，判官从 2 个升到 3 个：

| 判官 | 自一致率 | Cohen κ | 翻转 | 方向（neutral→committed） | 非中性 verdicts |
|---|---|---|---|---|---|
| Qwen2-VL-7B | 67.0% | 0.481 | **32/97** | 28 / 32 | 31/97 → 56/97 |
| Qwen2.5-VL-7B | 67.0% | 0.484 | **32/97** | 20 / 32 | 44/97 → 52/97 |
| Qwen3-VL-32B | 64.9% | 0.569 | **34/97** | 13 / 34（另有 20 个非中性↔非中性） | 73/97 → 85/97 |

**命中率（配对 McNemar）**：base 臂 8/34→16/34（**p=0.021**）、12/34→16/34（p=0.125）、17/34→22/34（p=0.125）；LoRA 臂 13/63→23/63（p=0.002）、17/63→18/63（**p=1.000**，Qwen2.5-VL 完全没动）、20/63→33/63（p=0.001）。

**两点比 round-7 更诚实的补充**：① base 臂现在也有一个显著结果（Qwen2-VL p=0.021），所以"唯一显著的效应都在被撤回的那条臂上"这个批评不再成立；② Qwen2.5-VL 在 LoRA 臂上完全没有移动，说明判帧效应本身也是**判官依赖**的 —— 这与全文主题一致，必须写进论文而不是藏起来。

**base x02 判为 sad**：前 1 s 下 0/8、0/8、2/8；全片段下 3/8、2/8、3/8。三个判官在两种协议下都不能维持"base 从不表达 sad"。

## U. round-9 外部审稿发现的 8 个问题（全部已修）

一个独立 reviewer 重跑了三个分析脚本、逐条核对了原始 JSONL，确认**所有 headline 数字都能复现**
（配对矩阵、选项序、判帧、Fleiss、判官效度表、机制表、test–retest 88/88），但指出下列问题：

1. **选择性报告（最严重）**：正文只报了判帧轴 6 个 arm×judge 格子里显著的 3 个，
   把 Qwen2.5-VL 的 LoRA 臂完全不动（17/63→18/63, p=1.000）藏在了 appendix。
   → 四篇论文正文现在**全部报 6 个格子**，并明说"这个平的格子本身就是论点的一个实例：
   判帧效应本身也是判官依赖的"，另加 Bonferroni 说明（6 个检验里 2 个存活）。
2. "the only significant effects we obtain anywhere" 与本文自己的 Δκ p=0.035 矛盾
   → 改为 "only significant **model-side** effects"。
3. 结论里的 "up to 37%" 是被取代的 n=81 旧数字 → 改为 35%（34/97）。
4. Limitations 里的 "81 clips and two re-scored judges" 同样过期 → 97 clips / 三判官。
5. 正文（3 判官/88 配置：33.8–45.7% 与 88–95%）与 appendix（4 判官/77 配置：36.7–50.0% 与 87.9–100%）
   报了同一个统计量的两个范围且都没说子集 → 两处都写明子集，并互相交叉引用。
6. 摘要 "v3 1/8--2/8 in every condition" 不精确：列出的四个条件里 v3 **都是 2/8**，
   1/8 来自第五个（Qwen2.5-VL）→ 已改写。
7. 同一格子的置信度在两份 log 里不同（x04 worried:95 vs 85；q38 x02 sad:82 vs 80）
   → 正文标注取自哪份 log，并说明"差在置信度不在标签"。
8. "across 100+ clip-configs" 无对应 log → 改为 "across the 88 clip-configs"。

附带两条也已修：四个判官写成 "three generations"（改为 "four releases spanning three generations"）；
选项序的 test–retest 88/88 其实是**反序变体**的重复运行（正文已注明）。

**reviewer 结论：Weak Accept（confidence 0.60），全部问题不需要新实验即可修；
唯一能推到 Strong Accept 的缺口仍是 97 个片段的人工标注**（工具已就绪，标注未做）。

## V. 第四个判官补齐：配对矩阵四判官全零

Qwen3.8-27B 在 16 格配对矩阵上补跑完成（`judge_basefull_q38.jsonl`，逐片段数分钟，只跑 0–1 s 协议）：
**base 8/12 vs v3 8/12，Fisher p=1.000，配对 McNemar p=1.000**（它只能解析 16 格中的 12 格）。

因此完整结论是：**四个判官在完全配对的 8 任务 × 2 种子矩阵上全部给出零结果**，
其中最强的判官（Qwen3-VL-32B）给 base 更高（9/16 vs 8/16）。撤回不再有"某个判官没测"的缺口。

## W. round-10：**准确率**证据（外部标签，无需我们打标）—— 这是本项目第一次能谈"正确"而不只是"一致"

之前所有关于"全片段判帧更好"的论证都只能走**可靠性**（判官间 κ 上升），reviewer 反复指出
这不等于**正确**。本轮找到了一个不需要人工打标的外部锚点：

`traindata3/train.jsonl` 引用的 **14 个源片段**（`sad_1534125952_v1_raw.mp4`、
`pres_scared_1538613676_base.mp4` …）来自与 `expressions_0825` 已知情绪立绘**同一批生产**，
情绪写在文件名里。论文本来就把这批立绘的标签当作 ground truth（17 立绘效度检查），
所以用同一来源的视频标签是自洽的。对同样 14 个片段跑两种协议 × 三判官：

| 判官 | strict 0–1 s → 全片段 | lenient 0–1 s → 全片段 |
|---|---|---|
| Qwen2-VL-7B | 0/14 → 3/14 (p=0.250) | 4/14 → 8/14 (p=0.125) |
| Qwen2.5-VL-7B | 0/14 → 4/14 (p=0.125) | 3/14 → **9/14 (p=0.031)** |
| Qwen3-VL-32B | 5/14 → 9/14 (p=0.125) | 7/14 → 11/14 (p=0.125) |

**六个比较全部改善；25 个"对错发生变化"的片段全部（25/25）站在全片段一侧，零个站在开头一侧。**
（McNemar 的 b=0 是精确的：p=0.25 对应 0 vs 3，p=0.125 对应 0 vs 4，p=0.031 对应 0 vs 6。）

改善集中在 4 个 cross 片段（angry/joyful/sad/surprised）——表情确实在片段后段才展开；
`pres_*` 里 anxious/curious/relaxed/scared 两个协议都失败，因为**没有判官能看见这些情绪**
（与 0/5 negative 的效度结论一致）。这两条限制都写进论文了。

**必须诚实标注的两个限制**：(1) 标签是该生产批次的*意图*表情，不是我们对这些片段的独立标注；
(2) 判官看不见的情绪在两个协议下都失败，所以这只在仪器能力范围内衡量协议。

脚本 `analyze_gt_accuracy.py`，日志 `judge_gt{1s,full}_{q2,q25,q3vl}.jsonl`，驱动 `run_gt.sh`。
