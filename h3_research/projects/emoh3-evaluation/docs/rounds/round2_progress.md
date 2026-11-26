# Round 2 进展：EmoH3 论文选题在 8×H100 上完成实证

## 本 round 做了什么

### 1. 打通 GPU 集群访问
- 通过 SSH 到 `jinzili_mv_260423` 容器（ProxyJump 链）间接访问 gpu_launcher。
- 确认 `osa00-h100-034.osa00.bdjp.io` 的 launcher 可用，**8×H100 80GB 全空闲**。

### 2. 盘点环境资产（都在 `/gluster_hdd_cv/user/jinzili/`）
- 权重：MiniMax-H3（Ref2VA/FL2VA，含 transformer_ref/video_vae/audio_vae）、MiniMax-H3-Turbo(8-merged-a8/a128)、MiniMax-H3-LoRA、ControlNet、reward_models（Qwen2-VL-7B、HPSv2/v3、PickScore、facenet、CLIP）、Qwen3.8-27B。
- 环境：sglang310（torch2.11+tf5.12）、minimax_h3_vxf（diffusers0.40+peft0.20+deepspeed，训练栈齐备）、vllm_omni。
- 部署脚本：`start_h3_8gpu_resident.sh`（8 卡 TP2×Ulysses4，~49GB/GPU）、`run_official_diffusers.sh`。
- 数据：表情参考图 35 张、171 情绪表、anime_style 150 例测试基建、synclub_expression。

### 3. 选定论文选题：EmoH3
**动漫角色情绪表情可控视频生成**（详见 `paper_proposal_emo_h3.md`）：
- 任务：参考图 + 目标情绪 → 免逐角色训练、身份保持的表情视频。
- 三支柱：AnimeEmo-Bench（新 benchmark）+ 情绪条件化方法（含 8×H100 LoRA 训练）+ 自动评估协议（Qwen2-VL judge + facenet + HPSv3）。
- 文献依据：round 1 的 173 条 Scholar 记录（世界模型/参考一致/评测是 2026 热点且无动漫情绪 benchmark）。

### 4. 8×H100 试点实验（M1，已跑通）
- 在 034 上启动 H3 Ref2VA 8 卡 serving（pid 31，仍在跑）。
- **8/8 跨情绪生成成功**：768×768、5.17s、124 帧、H264 + AAC 原生音频；~40s/条。
- Qwen2-VL-7B 情绪判定（8/8）→ **7/8 跨情绪转换失败**（身份保持但目标情绪没出来，仅 angry→happy 成功）。

### 5. 关键科研结论
> 基座 H3 Ref2VA 用一句 prompt 无法可靠转换动漫角色表情 → **问题存在已被 8×H100 数据证明**，直接支撑 LoRA 方法支柱，也符合用户既定"第一步：先证明问题存在"。

## 产出文件（本地 workspace）
- `../plans/paper_proposal_emo_h3.md` — 完整论文选题 proposal（含试点结果）
- `../../results/` — 试点与后续 VLM 判定结果
- `../../src/generation/pilot_expr.py`、`../../src/judges/judge_emotion.py` 与共享基线校验器 — 可复用脚本

## 下一步（round 3）
1. 构建 AnimeEmo-Bench（~10 角色 × 8 情绪）。
2. **8×H100 LoRA 训练**（社区配方：IAmIronMan42/MiniMax-H3-FineTuning，diffusers+peft+deepspeed，r=16 ~10.7M 参数，latent 预缓存）→ 证明方法能提升情绪保真。
3. 3×3 消融 + 完整评估表。
4. 论文写作 → 投 ICME/workshop。
