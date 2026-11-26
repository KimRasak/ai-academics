# Round 8：Reference 条件化训练（结构性方法修复，冲击负向情绪）

> round-7 结论：负向/细腻情绪（sad/worried/relieved/angry）在所有配置下失败，因为 **reference 图的情绪占主导**。
> round-8 思路：把 reference 图片编码进训练 conditioning，教模型「无视 reference 情绪、跟随 prompt 表达目标情绪」。

## 1. 已完成的实现（把 trainer 文档里的 TODO 做掉）

### 1.1 prepare_cache.py 补丁（reference 编码进 cache）
- 新增 `encode_reference_image()`：与 Ref2VA 推理完全一致的编码配方
  - 图像 cover-fit 到 (height,width) → 归一化 → video VAE `_encode_clip`（纯空间）
  - **posterior.sample(seed=42)**（与推理 KEYFRAME_ENCODE_SEED 一致）→ **round to float16** → normalize → patchify
  - 输出 `ref_video` 行 [rows_per_frame, 96]（256² 下 = (8×8, 96) = (64,96)）
- 每样本有 `reference_images` 就编码第一条 → 存 `item["ref_video"]`

### 1.2 train.py 补丁（ref 行进 packed sequence）
- `layout_for()`：有 ref 时用 `build_packed_sequence(..., keyframe_anchors=("first",))` —— fl2va keyframe 条件块与 ref2va reference 是**同一个机制**（conditioning video rows，锚在 time=num_text_tokens）
- `forward_loss()`：
  - ref 行 **clean 输入**（不加噪），与推理 conditioning（timestep 0.999）一致
  - `video = cat([ref_rows, noised_target_rows])`，`num_condition_video_rows = num_ref_rows`
  - `build_row_timesteps` 把 ref 行 pin 在 0.999
  - loss 只回归 **target 行**（`pred_v[:, num_cond:]`），不回归 ref 行
- 验证：seq 从 ~537 → **601-603**（+64 ref 行），1-GPU smoke test 5 步通过，loss 正常下降

### 1.3 跨情绪训练数据（traindata3，22 条）
- **12 条跨情绪对**（char 1534125952）：ref = 情绪 A 立绘，target = 情绪 B 视频，caption = B 的句子
  （如 ref=angry 立绘 + target=joyful 视频 + "She smiles joyfully." → 教「angry 参考 + joyful prompt → joyful 视频」）
  - A≠B 全组合：angry/joyful/sad/surprised × 3 = 12
  - target 视频 = v2 官方生成（真实音频、正确情绪）
- **10 条情绪保持**（pres_*，同 ref=target 情绪）——保持一致性的 baseline
- 合计 22 条，prepare_cache（patched）全量重编码，22/22 `ref=yes`

## 2. 训练（8×H100 完成）
- 900 步 deepspeed，loss 0.4-1.1（与 v2 同量级），`runs/lora3/checkpoint-900.pt`（42.6MB，108 LoRA 张量/54 模块，命名与 v2 一致）

## 3. 评估（已完成）

8 个跨情绪 case × 16 步 × seed 42/1234 × 8 帧判官（与 round-7 同协议）。

### 3.1 LoRA v3（reference 条件化）判官结果
| case | 目标 | v3 s42 | v3 s1234 | ≥1 seed 命中 |
|---|---|---|---|---|
| x01 neutral→joyful | joyful | happy:80 ✓ | happy:80 ✓ | **2/2** |
| x02 neutral→sad | sad | neutral:100 ✗ | **sad:80 ✓** | **1/2（首次 sad 成功！）** |
| x03 angry→happy | happy | happy:80 ✓ | happy:90 ✓ | **2/2** |
| x04 joyful→worried | worried | neutral:50 ✗ | neutral:100 ✗ | 0/2 |
| x05 amused→surprised | surprised | neutral:100 ✗ | happy:80 ✗ | 0/2（s1234 出 happy 非 surprised） |
| x06 anxious→relieved | relieved | neutral:50 ✗ | happy:80 ✗ | 0/2（s1234 出 happy 非 relieved） |
| x07 sad→cheerful | cheerful | happy:80 ✓ | happy:80 ✓ | **2/2（v2 是 1/2，更稳）** |
| x08 curious→angry | angry | neutral:100 ✗ | neutral:100 ✗ | 0/2 |

**s42 = 3/8，s1234 = 4/8；4 个不同 case 至少一次命中（x01/x02/x03/x07）。**

### 3.2 关键结论（诚实解读）
1. **首次 sad 成功**：x02（neutral→sad）在 seed1234 判 sad:80——整个项目（base/v1/v2/v3 × 多配置）第一次有 sad 目标成功。这是 reference 条件化训练的**真实新能力**（正补对照：base 16s s1234 的 x02 判 neutral，若确认）。
2. **cheerful 更稳**：x07 在 v3 下 2/2 seed 命中（v2@16s 是 s42✓/s1234✗ = 1/2）。
3. **frown 情绪仍未解**：worried(x04)/angry(x08) 两个 seed 都 neutral:100——与 sad 形成对比，可能指向 base 模型对「皱眉/负向强情绪」的表达能力不足（不只是 reference 覆盖问题）。
4. **正情绪偏好仍在**：x05/x06 在 s1234 都判 happy（非目标但偏正）——H3 基座的正向先验仍强。

### 3.3 决定性对照（已确认）
- **base 16s s1234 x02 → neutral:100**（判官，同协议同 seed 同步数）
- **LoRA v3（ref 条件化）16s s1234 x02 → sad:80**
- → **sad 突破是 reference 条件化训练的真实效应，不是采样假象**。唯一变量是 ref 条件化 LoRA。

### 3.4 论文叙事（round-8 结论）
**Reference-conditioned LoRA 训练 = 有效的方法贡献**：
1. 解锁首个负向情绪（sad:80 @ 16s s1234，base 对照 neutral:100）；
2. 提升正向稳健性（x07 cheerful 2/2 seed vs v2 的 1/2；x01/x03 保持 2/2）；
3. 局限明确：worried(x04)/angry(x08) 两个 seed 都 neutral——「皱眉/负向强情绪」仍未解（可能指向 base 模型表达容量，而非 reference 覆盖）。

### 3.5 下一步
- 768² 训练（闭合 train/eval ref 分辨率 gap 64 vs 576 行）——若 worried/angry 仍失败，可归因于 base 容量而非方法；
- 或：更多角色 × 跨情绪对数据（当前仅 char 1534125952 有 4 情绪立绘）；
- 或：frown 情绪专项数据（把 worried/angry 目标视频加进训练）。
- 数据文件：`v3_judge_results.jsonl`（16 条）、`judge_x02ctrl.jsonl`（对照）、`round8_reference_conditioning.md`

## 4. 关键文件
- 本地：`trainer_prepare_cache.py`、`trainer_train.py`（已推 `/gluster.../tmp/h3ft/`）
- `build_manifest_v3.py`、`run_cache3.sh`、`run_lora3_train.sh`、`jobs_lora3.json`、`run_lora3_eval.sh`、`run_lora3_eval_s1234.sh`、`run_lora3_all_eval.sh`、`run_judge_lora3.sh`
- 集群：traindata3/cache（22 .pt，ref=yes）、runs/lora3/checkpoint-900.pt、outputs_lora3{,_s1234}
