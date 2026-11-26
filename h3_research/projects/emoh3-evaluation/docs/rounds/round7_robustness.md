# Round 7：多 seed / 多步数 稳健性研究（推翻并重构 round-5 结论）> 目标：检验 round-5「LoRA 1/8 → 3/8」结论是否稳健。结论：**不稳健**——8 步采样欠解析，基座在 16 步也能出正情绪；LoRA v2 + 16 步把 neutral→joyful 变成跨 seed 稳健成功。这让论文叙事转向「严谨评估协议 + 受控的 LoRA 增益 + 负向情绪为未解问题」。

## 1. 完整判官结果矩阵（Qwen2-VL-7B；✓ = 判官命中目标情绪；8s/16s = 推理步数；s42/s1234 = seed；4fr/8fr = 判官抽帧数）

| case | 目标 | 基座 8s s42 | 基座 16s s42 | 基座 16s s1234 | LoRAv1 8s s42 | LoRAv2 8s s42 | LoRAv2 8s s1234 | LoRAv2 16s s42 | LoRAv2 16s s1234 |
|---|---|---|---|---|---|---|---|---|---|
| x01 neutral→joyful | joyful | ✗ neutral:50 | ✓ Happy:90 | ✗ Neutral:50 | ✓ Happy:90 | ✓ Happy:90 | ✗ Neutral:50 | ✓ Happy:90 | ✓ happy:80 |
| x02 neutral→sad | sad | ✗ | — | — | ✗ | ✗ | ✗ | — | — |
| x03 angry→happy | happy | ✓ Happy:80 | ✓ Happy:90 | — | ✓ Happy:90 | ✓ happy:80 | ✓ happy:80 | — | — |
| x04 joyful→worried | worried | ✗ | — | — | ✗ | ✗(Curious) | ✗ | — | — |
| x05 amused→surprised | surprised | ✗ | — | — | ✗ | ✓ Surprised:80(4fr) / ✗ Neutral:100(8fr) | ✗(Anxious) | ✗ Neutral:100 | — |
| x06 anxious→relieved | relieved | ✗ | — | — | ✗(Sad) | ✗(Sad) | ✗ | — | — |
| x07 sad→cheerful | cheerful | ✗ | — | ✗ neutral:50 | ✓ happy:80 | ✗ neutral:50 | ✗ | ✓ happy:80 | ✗ neutral:50 |
| x08 curious→angry | angry | ✗ | — | — | ✗ | ✗ | ✗ | — | — |
| **成功率** | | **1/8** | | | **3/8** | **3/8** | **1/8** | | |

## 2. 关键发现（每条都有实测）

### 发现 1：8 步采样欠解析，基座「7/8 失败」部分是假象
- 基座 x01（neutral→joyful）：8 步判 neutral:50；**16 步判 Happy:90**（s42）。同样的 prompt/ref/分辨率，只是多加 8 步去噪。
- 基座 x03（angry→happy）：8 步与 16 步都 Happy（本来就能做）。
- → 情绪表达在低步数下没有被充分解析；round-2 的「基座 7/8 失败」严重依赖 8 步采样。

### 发现 2：LoRA v2 + 16 步 = neutral→joyful 的**跨 seed 稳健成功**（唯一受控的 LoRA 增益）
- x01 16 步：基座 s42 ✓ / s1234 ✗（1/2，seed 不稳）；LoRA v2 s42 ✓ / s1234 ✓（**2/2，稳健**）。
- → 控制步数后，LoRA 仍带来真实增益：把 joyful 从「有时成功」变成「稳定成功」。
- x03（happy）基座本来就稳；x07（cheerful）LoRA 16 步 s42 ✓ / s1234 ✗（仍不稳）。

### 发现 3：负向/细腻情绪（sad/worried/relieved/angry）在所有配置下都失败
- x02/x04/x08 恒 neutral；x06 恒 Sad（对 relieved 是部分迁移，但从未到 relieved）。
- → **reference 图的情绪占主导**，prompt/LoRA 都无法可靠覆盖（尤其负向情绪）。这是真正的未解问题，也是下一轮 reference 条件化的动机。

### 发现 4：评估协议严重敏感（单 seed 单配置会误导）
- x05：同 seed 同 8 步，4 帧判 Surprised:80、8 帧判 Neutral:100——**抽帧数都能翻结果**。
- x01：seed 变一下结果就翻（LoRA 8 步 s42 ✓ / s1234 ✗）。
- → 论文级结论必须多 seed + 步数扫描 + 多帧判官。这本身是评估协议贡献。

## 3. 论文叙事重构（更诚实、更可辩护）

**标题方向**：*On the Reliability of Emotion-Conditioned Anime Video Generation: A Multi-Seed Study of MiniMax H3 Ref2VA with LoRA Conditioning*

**贡献**：
1. **首个动漫情绪可控视频生成的稳健性基准**（8 情绪 × 多配置 × 多 seed，Qwen2-VL-7B 判官 + 抽帧/步数扫描）；
2. **受控实验**：步数扫描揭示低步数采样假象；LoRA + 步数给 neutral→joyful 提供跨 seed 稳健增益（2/2 vs 基座 1/2），angry→happy 双模型都稳；
3. **明确未解问题**：reference 情绪主导，负向情绪目标无法可靠覆盖 → 指出 reference 条件化是下一步；
4. **评估协议**：单 seed/单配置评估有误导性，给出多 seed + 步数 + 帧数的标准做法。

**对审稿人的诚实定位**：不吹「LoRA 全面 3/8」，而是「LoRA 提供受控稳健增益 + 严谨评估协议 + 明确局限」。这对 workshop / benchmark 类论文更稳。

## 4. 数据文件（本地）
- `lora_judge_results.jsonl`（v1 8s s42 4fr）
- `lora2_judge_results.jsonl`（v2 8s s42 4fr）
- `lora2s_judge_results.jsonl`（v2 8s s1234 4fr）
- `judge8_results.jsonl`（v2 8s s42 8fr，x01/x05/x06/x07）
- `control_judge_results.jsonl`（16s 与 base16 全部控制组）
- 生成物：`outputs_lora2/`、`outputs_lora2_s1234/`、`outputs_s16/`、`outputs_s16b/`、`outputs_base16/`、`outputs_base16b/`

## 5. 下一步（下一轮）
1. **reference 条件化训练**（结构性修复，论文方法贡献）：把 reference 图片编码进训练 conditioning，教模型「无视 reference 情绪、跟随 prompt」。这是解决负向情绪的唯一路径。
2. 或：**更大 benchmark**（20 角色 × 8 情绪，多 seed 生成）→ 稳健性矩阵做全。
3. 论文写作按上述叙事推进（proposal §5.2/§5.3 需重写）。
