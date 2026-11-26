# 论文主张—公开资源—复现证据图（已归档）

> 本文件的日常展示职责已由
> [`dashboard/PAPER_REPRODUCTION_TRACKER.md`](../../dashboard/PAPER_REPRODUCTION_TRACKER.md)
> 和 [`dashboard/REPRODUCTION_PROCESS_TRACKER.md`](../../dashboard/REPRODUCTION_PROCESS_TRACKER.md)
> 接管。本副本只保留历史主张级摘要，不能覆盖 dashboard 的当前结论。

**更新：** 2026-09-20  
**用途：** 这是研究蓝图的主张级细节图，不是日常状态入口。论文名、公开权重
链接、验证数据集与指标、公开权重验证和训练复现的当前结论，统一以
`../../dashboard/PAPER_REPRODUCTION_TRACKER.md` 为准。每行是一个可审查的论文对象，不是论文
可信度排名。数值只在 `论文主张` 栏属于作者报告；本地证据必须按 C0--C5 分级。
完整定义见 [REPRODUCTION_BLUEPRINT.md](../../planning/REPRODUCTION_BLUEPRINT.md)，逐项原始记录见
`../../claim_ledger.jsonl`、`../../reproduction_registry.jsonl` 和各 `../../reproductions/*/` 卡片。

| 对象 / 公开任务 | 作者主张或待验证对象 | 可核验公开链 | 当前本地证据 | 不能声称什么 | 下一步与训练准入 |
|---|---|---|---|---|---|
| **MM-Diffusion** / Landscape | CVPR'23 `Ours*` DDPM-1000：FVD 117.20、KVD 5.78、FAD×10k 10.72（六次聚合） | 代码 `7250222`、官方 ZIP 中 900/100 原始 MP4、base/SR 权重、I3D/AudioCLIP 与 FVD/KVD/FAD evaluator 均已固定和哈希 | C3 run 1/6 与 run 2/6：每次 2,048 原始 MP4、2,048 SR MP4、2,048 WAV 审计通过；原始 64×64 分别为 249.0004/15.4862/10.7107、231.8951/14.3733/10.9198 | 不能称论文表格行、六次聚合、稳定方差或训练流程已复现；SR 256×256 分数仅诊断；默认 DPM-Solver 路径有公开代码缺陷；928 个 YouTube 上游 URL/许可与精确重建 manifest 未发布 | 两次仅报告逐次值、描述性均值与范围；如继续仅可另行预注册同协议 run 3，**不训练**。C4 还须上游数据来源/许可、预处理、100k/270k steps→checkpoint→评测完整链路 |
| **T2V-Turbo** / VC2 + VBench | NeurIPS'24 四步 VC2：Total/Quality/Semantic 81.01/82.57/74.76 | 论文时代推理代码、VC2 LoRA、VideoCrafter2 与 VBench evaluator 已固定 | C2：独立 4,720 视频、16 维 VBench，82.29/84.44/73.69；作者种子未知 | 不能称 Table 1 精确重现或训练复现 | 仅当论文使用的 WebVid-10M 媒体/分片 manifest 与 InternVideo2 Stage-2 f4 reward 权重可得，才进入一批次→恢复→8 卡训练链路 |
| **VADER** / VideoCrafter2 LoRA | Table 1 奖励、视频质量与训练效率增益 | 代码、发布 LoRA、部分奖励与 VBench 路径可审计 | C2：360 对固定提示下 PickScore 上升、Dynamic Degree 下降的 trade-off | 不能把发布 LoRA 当成 Table 1 checkpoint，也不能把 450-step 示例当作论文训练复现 | 先补齐 Table 1 的 base/checkpoint、种子、样本量、聚合与完整超参；否则只允许披露为独立流程兼容性实验 |
| **VideoDPO** / VC2 偏好训练 | headline OmniScore-reweighted DPO 的 VBench/HPS(V)/PickScore 改进 | 官方 20,000 MP4、10,000 pairs、代码已保存；20,000/20,000 解码通过 | C1：媒体与 pair provenance 完整 | 不能称 headline 权重、指标或训练已复现；不能把无权重 vanilla DPO 叫作 VideoDPO 论文复现 | 缺每对 OmniScore/probability、匹配配置和 evaluator revision；取得后仍须一批次、反传、恢复 smoke 再训练 |
| **MotionCtrl** / camera & object control | SIGGRAPH'24 CamMC/ObjMC/CLIPSim/FID/FVD 与三阶段训练 | 代码、checkpoint、示例条件存在 | C2 deployment smoke：一个 camera 条件视频可解码、checkpoint 键匹配 | 不能声称论文指标或训练复现 | 缺 407/283 case manifests、1,000 reference 视频、metric 实现与 VideoCrafter1 训练程序；可另定义公开控制 benchmark，但须与论文表隔离 |
| **MiniMax-H3** / 768p Base + Sci-VBench | Base T2VA/FL2VA/Ref2VA 可部署；作者 H3 视频的 Sci-VBench 表现 | 官方 Base 权重、请求脚本、参考媒体与 Sci-VBench evaluator 已冻结 | C2：作者 150 个 testmini 视频 evaluator 回放；授权 4×80GB FL2VA 两次在 DiT FSDP 物化 OOM，未监听/请求/生成 | 不能称完整 H3、Context-IR、2K workflow、训练或本地 checkpoint 生成已复现；`sora-2` 历史服务输出永久排除 | 停止同一 4×80GB 命令。安装版 H3 `memory` mode 不减 DiT 峰值；仅有 H3 支持的降峰值配置，或更换硬件/分片范围后才重启冻结 case；H3 无公开训练链 |
| **VABench** / I2AV | H3 或基线的正式音视频维度分数 | I2AV 输入、部分权重与 evaluator checkout 已固定 | C1：521 图像等物料审计 | 不能猜测图像—标注配对，不能以替代 DNSMOS 产生“官方分数” | 仅等待作者提供 pairing manifest 和官方 DNSMOS wrapper；缺一即停止正式计分 |
| **VideoCrafter2** / shared base | 为后训练工作提供共同公开基座和 VBench 基线 | checkpoint 与 VBench 路径公开 | D：可作版本固定的共同基线 | 不能称其预训练重现 | 不进入基座预训练；只服务已满足训练准入的下游论文 |
| **AnimateDiff** / motion module | motion module 与 MotionLoRA 的不同训练主张 | 代码、权重、部分评测公开 | B/D：尚未完成 compute/训练协议审计 | 不能把低成本 MotionLoRA 当成 full motion-module 训练复现 | 先固定完整训练配置、原始媒体和成本；缺任一项只保留文献候选 |

## 数据与缓存的最低审查条件

任何进入 C3 或 C4 的路径都必须保留原始视频/音频/图像、标注、下载来源、许可证、
切分、哈希和缓存映射；`.pt`、latent 或 feature cache 只能是派生物。若任何媒体被
清理、重编码或筛选，manifest 必须记录其输入哈希、工具版本、命令、输出哈希与理由。
这比“缓存可被模型读取”更强，目的是让研究人员仍能抽查语义、偏差和错误样本。

## 目前的决策

当前唯一正在消耗 GPU 的任务是 MM-Diffusion 的已预注册 checkpoint 重复评测，目的
是判断 run 1 的 FVD/KVD 偏差是否稳定。它不是探索训练，也不构成对任意论文训练
流程的认可。研究 idea 的筛选必须等该可比基线、H3 Base 可辨识部署和至少一个正式
H3 相关公开评测的边界进一步闭合后再进行。
