# 论文复现总台账

**更新：** 2026-09-20  
**状态来源：** `PAPER_REPRODUCTION_TRACKER.jsonl`、各 `../reproductions/<work>/`
的 source manifest、运行 manifest、媒体审计和 claim ledger。  
**使用规则：** 这是今后研究工作的唯一导师汇报入口。任何新论文在下载、运行、
评测或训练前，必须先建一行；任何状态改变必须同时更新本表和 JSONL。

## 判定口径

- **公开权重验证**只回答：论文匹配或明确标注的公开权重能否在指定公开测试协议上
  得到论文所声称的指标。
- **训练复现验证**只回答：从论文所用公开原始数据、预处理、训练、生成到评测的完整
  流程能否恢复论文指标。
- 不以总分更高推断“复现成功”或“超过论文”。若作者 seed、样本数、checkpoint
  选择或 evaluator 版本未固定，只能报告“固定独立协议结果”。
- `一致` 表示在已固定且论文匹配的协议下达到预先登记的容差；`同量级/不一致`不等于
  论文对错；`未验证`不能写成失败。
- `N/A` 仅适用于纯 benchmark/evaluator，不得借此绕过生成模型的权重与训练审计。

| 论文名 | 公开权重 | 公开权重链接 | 验证数据集与指标 | 公开权重验证结果 | 训练复现验证结果 |
|---|---|---|---|---|---|
| **MiniMax-H3** | 有：FL2VA/Ref2VA Base，但非完整 hosted H3 系统 | [官方发布](https://github.com/MiniMaxAI/MiniMax-H3)（本地 README/hash 已固定） | 官方 768p：T2VA（文本）、FL2VA（文本+首帧）、Ref2VA（文本+参考视频/音频）；Sci-VBench testmini 的 150 条公开 prompt、作者 H3 settings、发布视频及 six-VT | **有权重，正式 testmini VT 验证运行中，尚无本地总分。** 150/150 prompt/settings join 冻结为 T2VA、768/16:9、15s、24fps、50 steps、seed 20260803。服务恢复后既有 runner 已续跑至 `146/150`、`0 error`；`/v1/models` 报告 MiniMax-H3/8 GPU/MiniMaxH3Pipeline。第 136 条曾 API 滞后，最终已 completed 且 MP4 落盘；全量媒体完成/审计通过前禁止部分 VT | **不能训练复现。** 无公开 H3 训练程序、训练数据和论文匹配训练协议；不得把本次部署成功写成训练复现 |
| **MM-Diffusion** | 有：Landscape base/SR checkpoint | [官方代码与下载入口](https://github.com/researchmm/MM-Diffusion)（官方 Drive file IDs、hash 已固定） | Landscape；FVD、KVD、FAD×10k；每 run 2,048 original MP4、2,048 SR MP4、2,048 WAV | **有权重但论文指标未复现。** 论文六次汇总 `117.20 / 5.78 / 10.72`；64×64 original：run 1 `249.00 / 15.49 / 10.71`、run 2 `231.90 / 14.37 / 10.92`、run 3 `222.60 / 13.17 / 10.80`。run 3 的 SR diagnostic 为 `137.65 / 7.02 / 10.94`，且三类媒体各 2,048、审计通过；但该 run 启动后 GPU 6 出现约 37.7GB 外部占用，故仅是**受干扰的 C3 checkpoint 观察**，不作干净重复、六次汇总或训练结论 | **不能训练复现。** 上游 YouTube 来源/许可证/精确重建 manifest 不完整；作者 seeds 与六次样本集合未发布；尚未满足 C4 数据与训练门槛 |
| **T2V-Turbo** | 有：VC2 LoRA，且有 VideoCrafter2 base | [T2V-Turbo-VC2 LoRA](https://huggingface.co/jiachenli-ucsb/T2V-Turbo-VC2) | VBench 16 维；论文 four-step Total `81.01`；本地固定独立 protocol Total `82.2872`，4,720 视频 | **有权重但无法严格复现论文指标。** 固定独立协议结果数值更高但作者 seeds、完整生成协议未公开；不能称一致或超越 | **不能训练复现。** 论文 WebVid-10M 精确样本/分片和媒体缺失；InternVideo2 Stage2 1B f4 reward 权重 gated（403） |
| **VADER** | 有：发布的 VideoCrafter2 PickScore、HPS+Aesthetic LoRA | [PickScore LoRA](https://huggingface.co/zheyangqin/VADER_VideoCrafter_PickScore)；[HPS+Aesthetic LoRA](https://huggingface.co/zheyangqin/VADER_VideoCrafter_HPS_Aesthetic) | 固定公共 temporal prompts：PickScore、VBench Dynamic Degree、Motion Smoothness；论文 Table 1 另列 | **有权重但无法复现论文 Table 1。** 360 对固定提示审计可验证发布 LoRA 的 reward/motion trade-off；发布 LoRA 未被证明等同 Table 1 checkpoint | **不能训练复现。** 缺论文 Table 1 的精确 base、seed、样本数、checkpoint 选择和聚合；训练示例与发布 LoRA/论文表不匹配 |
| **VideoDPO** | 无明确论文最终 checkpoint | [官方代码](https://github.com/CIntellifusion/VideoDPO)；[VC2 base](https://huggingface.co/VideoCrafter/VideoCrafter2) | 论文 VBench、HPS(V)、PickScore；官方 archive 的 20,000 MP4 / 10,000 pairs 完整性 | **无权重。** 不存在论文匹配 final checkpoint，不能做论文指标权重验证 | **不能训练复现。** 每对 OmniScore 权重缺失；公开 `dupbeta: 0` 不执行 headline reweighting；evaluator revisions、seeds、checkpoint selection 缺失 |
| **MotionCtrl** | 有：VideoCrafter1/LVDM `motionctrl.pth` | [官方 checkpoint](https://huggingface.co/TencentARC/MotionCtrl/blob/main/motionctrl.pth) | 论文 CamMC、ObjMC、CLIPSim、FID、FVD；本地仅 `test_camera_L` deployment smoke | **有权重但无法复现论文指标。** checkpoint hash/load/单例生成通过；论文 407/283 case protocol 与指标代码不完整 | **不能训练复现。** 论文 VideoCrafter1/LVDM 训练流程与公开 AnimateDiff training branch 不匹配；数据 manifests、reference 视频、评测缺失 |
| **VideoCrafter2** | 有：base checkpoint | [官方权重](https://huggingface.co/VideoCrafter/VideoCrafter2) | 计划为 pinned VBench shared baseline；尚未登记论文匹配独立运行 | **有权重但尚未验证论文指标。** 当前仅作为下游 T2V-Turbo/VADER 的固定 base | **不能训练复现。** 精确预训练 corpus、过滤、训练算力和完整 pipeline 未闭环 |
| **AnimateDiff** | 有：motion module / MotionLoRA releases | [官方代码与权重入口](https://github.com/guoyww/AnimateDiff)；[官方 Hugging Face 模型页](https://huggingface.co/guoyww/animatediff) | 论文涉及 FVD、CLIP score、人评；目前未取得论文匹配 prompt、reference media、evaluator 或表格行级协议 | **有权重但尚未验证论文指标。** 官方仓库 HEAD `e92bd567` 已只读审计，确认 v2/v3 motion module 与 MotionLoRA 的官方下载入口；尚未下载/hash，且未证明任一 release 对应论文表格 checkpoint | **不能训练复现（当前公开物料不足）。** `train.py` 需要 WebVid 风格 CSV + 原始 MP4；官方 config 是路径占位和 100-step 示例，缺论文媒体清单、许可证、过滤/split、完整训练预算与评测协议。Motion Module 与 MotionLoRA 必须分开 |
| **Stable Video Diffusion（候选，未入队）** | 有：SVD/SVD-XT image-to-video checkpoint | [SVD](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid)；[SVD-XT](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt)；[官方代码](https://github.com/Stability-AI/generative-models) | 发布说明：SVD 14 帧、SVD-XT 25 帧，均由单张 context frame 条件生成；尚未取得论文行级测试图像、reference video 或 evaluator | **有权重但尚未验证论文指标。** 官方 SVD 发布 commit `059d8e9` 已审计，含推理脚本/config；权重未下载/hash，发布快照未见论文 FVD/FID 评测脚本、reference media 或逐样本协议 | **不能训练复现（当前公开物料不足）。** 仅有通用 WebDataset training framework；缺 SVD 训练媒体/许可证/split、完整配置/优化轨迹、论文评测和计算预算。不得以自建 WebDataset 训练代替 |
| **JavisDiT++ / JavisDiT-v1.0** | 有：JavisDiT-v1.0 released checkpoint（JavisDiT++ 官方模型页） | [官方代码](https://github.com/JavisVerse/JavisDiT)；[JavisDiT-v1.0](https://huggingface.co/JavisVerse/JavisDiT-v1.0-jav)；[JavisBench](https://huggingface.co/datasets/JavisVerse/JavisBench) | JavisBench-mini：1,000 条公开 text-to-audio-video prompt + FVD/KVD/FAD reference feature cache；官方 240p、4 秒协议。完整 JavisBench 论文表含 FVD、FAD、CLIP、AV-IB、DeSync 等，但 mini 不能直接对照 | **有权重但尚未验证论文指标。** 权重与依赖 76 文件已全量 hash；有效单条 checkpoint smoke、官方 CSV 前四条 MP4/WAV 映射及官方 evaluator 的四条输入读取均已闭环。四条接口结果为 `FVD 3324.9404 / KVD −11.0270 / FAD 11.0482`，只验证 evaluator 输入路径，**不能**视作 JavisBench-mini、论文指标或质量结论 | **不能训练复现（当前公开物料不足）。** Stage 2 依赖不可再发布的 TAVGBench YouTube 原媒体；JavisBench 也只发布 reference feature cache。AV-DPO generated-only 子集不能补齐 Stage 1/2 或 C4 训练链路 |
| **VABench** | N/A：评测工作；有 evaluator cache，不是生成模型 checkpoint | [官方代码](https://github.com/tanABCC/VABench)；[I2AV inputs](https://huggingface.co/datasets/tanABBCC/VABench_I2AV) | I2AV：DNSMOS、NISQA、ViCLIP、CLAP、DeSync、ImageBind、Qwen-Omni 等 | **N/A，正式评测不可执行。** 521 张图与部分 evaluator weights 已审计；image-to-annotation join、DNSMOS wrapper 缺失 | **N/A。** 它不是训练生成模型；正式 benchmark 复现被 pairing/implementation 缺口阻塞 |

## 最新终态记录

MM-Diffusion run 3 是同一 Landscape checkpoint 协议的预注册重复，不是训练。它已满足
`original=sr_mp4=wav=2048`、全量媒体审计通过、original/SR 两次独立 evaluator 完成的
终态门槛。启动后 GPU 6 获得约 37.7GB 的外部计算占用；因此该结果必须标注为受干扰观察，
不能与启动时全卡隔离的 run 1/2 混同为干净重复，也不能用于估计方差、认定论文六次汇总
不成立或启动训练。完整可核验记录见
[`run 3 manifest`](../reproductions/mm-diffusion/run_manifests/ddpm_ddim25_metric_run03_officialrng_gpu4567_20260920.json)。

## 新论文入表最低字段

1. 论文名与精确表格行；
2. 是否公开论文匹配权重；若有，官方链接、revision、hash 和本地路径；
3. 明确测试数据集、split、指标、论文报告值及 evaluator revision；
4. 公开权重验证结论：`无权重` / `有权重未验证` / `有权重但无法复现指标` /
   `有权重且复现指标与论文一致`，并给出证据路径；
5. 训练复现结论：`未开始` / `可训练待验证` / `不能训练复现` /
   `训练复现指标一致`，以及确切卡点或完整运行证据；
6. 原始媒体、标注、许可证、split、缓存与派生特征的 provenance manifest。
