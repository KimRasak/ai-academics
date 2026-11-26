# REPRODUCE.md — 复现指南（E1–E4）

所有远程操作在 `ssh jinzili_mv_260423` 上执行；SSH 需加
`-o ControlPath=none -o ControlMaster=no -T -o RemoteCommand=none`。

## 0. 环境

```sh
# VT evaluator（已存在）
/ssd2/jinzili/envs/scivbench_vt/bin/python -m scivbench_eval.vt.run_vt --help
export SCIVBENCH_VT_WEIGHTS_DIR=/ssd2/jinzili/scivbench_vt_weights

# Wan 独立环境（已存在，Python 3.9，torch 2.8.0+cu128）
/ssd2/jinzili/envs/wan_fun_clean/bin/python -c "import torch; print(torch.cuda.is_available())"
```

## 1. E1 human join（本地）

```sh
cd h3_research/projects/seed-sensitivity
python3 src/e1_human_join.py
# 输出 data/scivbench/e1_human_join.json（2,392 rows）
```

## 2. E2 released VT 复现（远程）

```sh
# 150 条 released 视频 + settings 已在 /ssd2/jinzili/scivbench_released_input/MiniMax-H3/
CUDA_VISIBLE_DEVICES=3 SCIVBENCH_VT_WEIGHTS_DIR=/ssd2/jinzili/scivbench_vt_weights \
/ssd2/jinzili/envs/scivbench_vt/bin/python -m scivbench_eval.vt.run_vt \
  --video_dir /ssd2/jinzili/scivbench_released_input \
  --output_dir /ssd2/jinzili/scivbench_released_vt \
  --models MiniMax-H3 --metrics subject_consistency,background_consistency,motion_smoothness,dynamic_degree,aesthetic_quality,imaging_quality \
  --device cuda --batch-size 8
# 对比：本地 src/compare_released_vt.py → data/scivbench/released_vt_comparison.json
```

## 3. E3 H3 seed-power（远程生成 + VT，本地分析）

- 生成：034 FL2VA 服务（30011），16 prompts × seeds 0–5，输出 `/ssd2/jinzili/e3_seed/outputs/`
- VT：同 E2 命令，输入 `/ssd2/jinzili/e3_vt_input/h3_base`，输出 `/ssd2/jinzili/e3_vt_new`
- 分析（本地）：

```sh
python3 src/analyze_e3_seed_power.py # canonical join → data/scivbench/e3_seed_power.json
```

## 4. E4 Wan 跨模型 replication（远程）

```sh
cd /ssd2/jinzili/wan_fun_src
WAN_SEEDS=0,1,2,3,4,5 WAN_OUT=/ssd2/jinzili/wan_batch CUDA_VISIBLE_DEVICES=3 \
nohup /ssd2/jinzili/envs/wan_fun_clean/bin/python -u run_wan_batch.py \
  > /ssd2/jinzili/wan_batch2.log 2>&1 &
# 96 条，官方 81 frames/16FPS/50 steps；断点续跑：重跑同一命令即可
# VT：输入 /ssd2/jinzili/wan_vt_input/wan_batch2，输出 /ssd2/jinzili/wan_vt_batch2
```

本地拉取结果后：

```sh
python3 src/analyze_e4_wan.py data/wan_cross/vt2 # → data/wan_cross/e4_wan_seed_power.json
python3 src/analyze_rank_stability.py            # → seed-pair ranking stability
python3 src/make_paper_figs.py                   # → paper/assets/fig*.png
```

## 5. 论文

- 初稿 `paper/main/main.tex`（Overleaf 可编译）；图 `paper/assets/`
- 数字一致性 QA：`main.tex` 引用的所有统计量已与
  `released_vt_comparison.json` / `e3_seed_power.json` / `e4_wan_seed_power.json` /
  `e1_human_join.json` 逐项核对（Round 58）

## 注意事项

- v1 Wan 批次因 prompt 替换未生效已作废（`*invalid_dogprompt*`），勿引用
- 不要污染 `/ssd2/jinzili/envs/scivbench_vt`（勿升级其 transformers）
- H3 服务（30011/30020）常驻，勿重启
