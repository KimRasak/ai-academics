# Round 53–54：E4 跨模型 seed-power replication（Wan2.1-Fun，有效批次 v2）

## 协议

- 模型：Wan2.1-Fun-V1.1-1.3B-InP（官方 VideoX-Fun 入口，`predict_t2v.py`）
- 协议：官方默认 81 frames @16 FPS（5.0625 s，与 H3 pilot 5.175 s 对齐）、50 steps、TeaCache 官方推荐、832×480、model_cpu_offload
- prompts：pilot16 的 verbatim `posted_prompt`（与 H3 E3 完全同一批 16 条）
- seeds：0–5 × 16 prompts = 96 条，96 个唯一 MD5，0 失败
- 评分：同一 VT evaluator（同一权重、同一 GPU 栈），96/96 complete，0 errors，299 s
- GPU：034 GPU 3

## v1 作废说明

v1 批次的 prompt 精确字符串替换未命中（官方默认是中文狗 prompt），96 条实为同一 prompt 的 6 个唯一视频 ×16 重复。v1 全部产物已隔离（`wan_batch_invalid_dogprompt`、`vt_invalid_dogprompt`、`e4_wan_seed_power.INVALID.json`），其中数字一律不得引用。

## E4 有效结果（v2）

| 统计量 | H3（E3） | Wan2.1-Fun（E4） |
|---|---:|---:|
| DD pairwise flip | 73/240 = **30.42%** | 38/240 = **15.83%** |
| prompts any-flip | 10/16 = 62.5% | 7/16 = 43.75% |
| VT within-prompt SD（mean） | 0.051963 | 0.038396 |
| VT within-prompt SD（max） | 0.109205 | 0.081148 |
| DD within-prompt SD | 0.306386 | 0.169217 |
| VT overall mean | 0.745554（e3_new_vt 16×6） | 0.678487 |
| DD overall mean | — | 0.1875 |

### 方差分解（16 prompts × 6 seeds，同一 evaluator）

| 模型 | between-prompt SD | within-prompt(seed) SD | ICC(prompt) |
|---|---:|---:|---:|
| H3 | 0.0691 | 0.0486 | 0.670 |
| Wan2.1-Fun | 0.0476 | 0.0471 | 0.505 |

## 结论

1. **seed 敏感性跨模型复现**：两个架构/协议差异很大的模型上，within-prompt(seed) SD 几乎相同（0.0486 vs 0.0471），且都比 released-model 相邻 VT gap（0.0049）大约一个数量级。VT 的 seed 噪声是 metric-pipeline 级别的性质，不是单一模型的现象。
2. **Wan 上 prompt 与 seed 方差近乎各半**（ICC=0.505）；H3 为 2/3 prompt、1/3 seed（ICC=0.670）。仅按 prompt 均值排序模型时，两模型的 seed 项都足以扰动相邻名次。
3. **DD 的 seed 翻转跨模型存在但幅度不同**（30.42% vs 15.83%）；H3 的 DD 阈值行为更剧烈。跨模型一致的方向性支持"DD 阈值化传播 seed 噪声"的机制解释，同时表明翻转率是模型依赖的，不是 DD 的固定属性。
4. 分数不得跨模型直接比较质量（分辨率/协议不同）；对比仅限 metric 行为（SD、flip rate、ICC）。

## 产物

- 生成视频：`/ssd2/jinzili/wan_batch/`（96 个唯一 mp4）
- VT 结果：`/ssd2/jinzili/wan_vt_batch2/wan_batch2/`；本地副本 `data/wan_cross/vt2/`
- 分析脚本：`analyze_e4_wan.py`（输出 `data/wan_cross/e4_wan_seed_power.json`）
- runner：`wan/run_wan_batch.py`（v2，regex prompt 替换 + 断言）


---

**NOTE (Round 70)**: 本审计的 6-seed 数字（38/240=15.83% 等）已被 12-seed 扩展 supersede；以 `round70_12seed_extension.md` 为准（182/1056=17.23%，SD 0.0411）。
