# Round 9：768² 训练——闭合 train/eval ref 分辨率 gap

> round-8 结论：ref 条件化训练解锁 sad（x02 sad:80 @ s1234，base 对照 neutral:100），但 worried(x04)/angry(x08) 仍失败。
> round-9 假设：train 的 ref 是 256²（64 行），eval 的 ref 是 768²（576 行）——**train/eval 布局不匹配**可能让 LoRA 学不到「按 reference 调制」的正确行为。若在 768² 训练（与 eval 完全同布局），worried/angry 可能被解锁。

## 1. 实验设计（受控变量：只有分辨率）
- **同一份 traindata3/train.jsonl**（22 条：12 跨情绪对 + 10 情绪保持）
- prepare_cache 用 `--height 768 --width 768` 重编码 → traindata4/cache
- 其余全部相同：LoRA r16/α32、900 步、8×H100、deepspeed、16 步 eval × seed42/1234 × 8 帧判官

## 2. 已确认
- traindata4/cache 22/22：video=(4032,96)、**ref_video=(576,96)**（与 eval 完全一致）、audio=real
- 8×H100 3 步验证通过：seq=4698（text 15 + ref 576 + audio ~75 + video 4032），loss 2.45→0.31，无 OOM
- **LoRA v4 全量训练 900 步进行中**（~11 步/min，因 seq 大 8×，预计 ~80 min）

## 3. 结果：**分辨率假设被证伪（负结果，且信息量大）**

**LoRA v4（768² 训练）判官结果（16 步 × 8 帧，同协议）：**
| case | 目标 | v4 s42 | v4 s1234 | 对照 v3 |
|---|---|---|---|---|
| x01 neutral→joyful | joyful | neutral:50 ✗ | Neutral:50 ✗ | **v3 2/2 → v4 0/2（退化）** |
| x02 neutral→sad | sad | neutral:50 ✗ | **sad:80 ✓** | v3 1/2（sad 仍只在 s1234） |
| x03 angry→happy | happy | neutral:100 ✗ | Neutral:50 ✗ | **v3 2/2 → v4 0/2（退化）** |
| x04 joyful→worried | worried | neutral:50 ✗ | neutral:100 ✗ | 仍失败 |
| x05 amused→surprised | surprised | Neutral:100 ✗ | Neutral:50 ✗ | 仍失败 |
| x06 anxious→relieved | relieved | Neutral:50 ✗ | neutral:100 ✗ | 仍失败 |
| x07 sad→cheerful | cheerful | neutral:50 ✗ | **happy:80 ✓** | v3 2/2 → v4 1/2（退化） |
| x08 curious→angry | angry | neutral:50 ✗ | neutral:50 ✗ | 仍失败 |

**s42 = 0/8，s1234 = 2/8**。比 v3（3/8、4/8）明显更差。

### 3.1 诚实结论
1. **分辨率假设被证伪**：768² 训练（与 eval 完全同布局）并没有解锁 worried/angry，反而让之前稳的正向（x01/x03）退化到 neutral。
2. **过拟合迹象**：v4 最终 loss 0.30 比 v3 的 0.45 更低，但行为更差——22 样本 × 41 epoch 在更高分辨率下更易记住噪声。同量 LoRA 参数（r16）摊到 8.6× 更长的序列，表达能力被稀释。
3. **x02 sad 是唯一跨模型稳健的新能力**：v3 和 v4 都在 s1234 出 sad:80（v4 还保持了这个）。这不是采样假象（base 对照 neutral:100）。
4. **worried/angry 在所有模型 × 所有配置下从未成功** → 强证据指向 base 模型的表达容量边界（frown 类强负向情绪在 H3 的情绪空间里很难被 LoRA 调制出来），而非 train/eval 布局不匹配。
5. **方法上 v3（256² 训练）仍是最佳**——分辨率匹配不是银弹。

### 3.2 论文叙事（整合 round 8+9）
- **方法贡献**：reference-conditioned LoRA（256²，跨情绪对数据）解锁首个负向情绪 sad（有 base 对照）+ 正向稳健性提升（cheerful 2/2）；
- **负结果贡献**：768² 训练无效（证伪分辨率假设）+ worried/angry 全配置失败 → 严谨地界定 base 模型情绪表达容量边界；
- 这是一个「方法增益 + 明确边界」的诚实 benchmark 故事，适合 workshop 论文。

## 4. 资产
- 本地：`v4_judge_results.jsonl`（16 条）
- 集群：traindata4/cache（22 .pt @768²）、runs/lora4/checkpoint-900.pt、outputs_lora4{,_s1234}、judge_lora4{,_s}_results.jsonl
- 下一步选项：v5 = 更多跨情绪对数据（seed1234 变体/white 变体/更多角色）冲击 worried/angry；或直接进入论文写作。
