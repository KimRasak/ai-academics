# Round 34：Sci-VBench VT 环境与单视频 smoke test

## 环境

在远程 `/ssd2/jinzili/envs/scivbench_vt` 创建了带 system-site-packages 的隔离 venv，复用 CUDA PyTorch 2.11.0+cu128，并安装：`decord`、`clip-anytorch`、`pyiqa`、`vbench` 与本地固定的 Sci-VBench evaluator。

注意：pip 报告该 venv 继承的 SGLang/vLLM 包存在依赖版本冲突，因此它只用于 VT evaluator，不用于 H3 服务或训练。

## 资产

权重目录：`/ssd2/jinzili/scivbench_vt_weights`，约 1.9 GB。最终 `weights --check` 对以下全部返回 OK：DINO ViT-B/16、CLIP ViT-B/32、CLIP ViT-L/14、LAION aesthetic head、AMT-S、RAFT Things、MUSIQ SPAQ、DINO source repository。

本地保存检查结果：`data/scivbench/vt_setup_check_final.txt`。

## Smoke test

输入：`39691e15c62cb0b9ba072532e82eb91ca1f20d21f_seed0.mp4`。

先以 `CUDA_VISIBLE_DEVICES=3` 同时传 `--device cuda:3`，失败为 `invalid device ordinal`：过滤后进程只看到一张卡，逻辑编号应为 `cuda:0`/`cuda`。修正为 `--device cuda` 后成功：

- metric: subject_consistency
- score: 0.908658900046971
- metric runtime: 12.7 s
- total runner runtime: 18.8 s
- errors: 0

结果位于 `data/scivbench/vt_smoke/`。其它五个指标和 composite VT 保持 null，不能将单指标当成 VT。

## Gate

VT dependency/asset/self-test Gate 通过。下一步可以先在 32 条 pilot 上运行六指标，保存 per-video 原始分数与 aggregate；若某个较重指标失败，保留已完成字段并按 evaluator 的 resumable 语义修复，不重跑已有字段。仍不据此启动训练。
