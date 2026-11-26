# H3 / 视频生成核心主源台账

**更新：** 2026-09-19  
**用途：** 为最终 proposal 的文献与复现蓝图保留已核验的主源入口。它不是
最终参考文献，也不代表其中任何论文主张已经成立；最终引用仍须核对作者、标题、
版本、发表 venue 和与具体假设的关系。

Google Scholar 的自动检索目前被 CAPTCHA 阻断，且不应绕过。因此本表只抽取
当前复现卡 `source_manifest.jsonl`、冻结 evaluator checkout 与已保存官方数据卡中
已经存在的可核验来源。每一条同时说明它能支持什么、不能支持什么。

| 对象 | 已核验主源 | 已固定版本 / 本地证据 | 当前可支持的结论 | 不可据此声称的内容 |
|---|---|---|---|---|
| MiniMax-H3 | [官方模型发布](https://github.com/MiniMaxAI/MiniMax-H3) | `README.md` SHA-256 `819db2…39ae9`；FL2VA/Ref2VA index 所列 shard 均存在 | 768p Base 的 T2VA/FL2VA/Ref2VA 本地部署是可定义的 checkpoint 路径 | 完整 H3、Context-IR、2K workflow 或训练过程均未公开可复现 |
| Sci-VBench | [官方代码](https://github.com/Sci-VBench/Sci-VBench)；[公开视频数据卡](https://huggingface.co/datasets/Sci-VBench/Sci-VBench-Videos) | evaluator `e8cb3988631d0c22d31c02fc5b1cc3693bfb7031`；testmini 的 video ID 可与 rubric 相连 | 对作者发布 H3 testmini 媒体的 evaluator 重放（C2） | 本地 H3 生成、训练或跨模型绝对质量比较 |
| MM-Diffusion | [论文 arXiv:2212.09478](https://arxiv.org/abs/2212.09478)；[官方代码](https://github.com/researchmm/MM-Diffusion) | paper SHA-256 `39c9…a9b3c`；当前 code `7250222`，paper-era evaluator `c895b29` | Landscape checkpoint、公开媒体与 FVD/KVD/FAD 的可验证路径 | 默认 DPM-Solver 行；在 2,048 样本与独立评测终态前的论文数值复现 |
| T2V-Turbo | [论文 arXiv:2405.18750v2](https://arxiv.org/abs/2405.18750)；[官方代码](https://github.com/Ji4chenLi/t2v-turbo) | paper SHA-256 `528b…254`；inference `10b1c53`；VBench `03e7b94` | 已发布 VC2 LoRA 的独立固定种子 VBench checkpoint 验证 | 论文作者 seeds 的精确 Table 1 重现，或 WebVid 训练复现 |
| VADER | [论文 arXiv:2407.08737v1](https://arxiv.org/abs/2407.08737)；[官方代码](https://github.com/mihirp1998/vader) | paper SHA-256 `d08a…2111`；release tag `d2da343` | 已发布 LoRA 在固定公共 protocol 下的奖励—动态性 trade-off 检验 | Table 1 权重或训练流程复现；发布 LoRA 并非已证实的 Table 1 checkpoint |
| VideoDPO | [论文 arXiv:2412.14167](https://arxiv.org/abs/2412.14167)；[官方代码](https://github.com/CIntellifusion/VideoDPO) | code `1febdb4`；官方 archive 的 20,000 MP4 均解码通过 | 官方原始视频/pair archive 的 provenance 与完整性审计 | headline OmniScore reweighted 训练或论文评测；pair 权重与完整 evaluator 仍缺 |
| MotionCtrl | [论文 arXiv:2312.03641](https://arxiv.org/abs/2312.03641)；[官方代码](https://github.com/TencentARC/MotionCtrl) | code `6242d55`；checkpoint SHA-256 `efe2…fe46c` | checkpoint example/deployment 检验 | 407/283 case 的论文指标或三阶段 VideoCrafter1 训练；完整 manifests 与评测缺失 |
| VBench | [官方代码](https://github.com/Vchitect/VBench) | paper-day `03e7b94`；已固定 prompts、aggregation 与若干 evaluator 权重 | T2V-Turbo 等公开 checkpoint 的可重复独立生成评测 | 未匹配 prompts、seed、帧数和 evaluator 版本时的论文表横向比较 |
| VABench | [论文页面](https://openaccess.thecvf.com/content/CVPR2026/html/Hua_VABench_A_Comprehensive_Benchmark_for_Audio-Video_Generation_CVPR_2026_paper.html)；[官方代码](https://github.com/tanABCC/VABench) | code `12b9617`；I2AV archive 521 张图的 hash 已保留 | 发布物料与 evaluator 依赖缺口审计 | 正式 I2AV 分数：图像—标注 join 和官方 DNSMOS wrapper 均缺失 |

## 进入最终 bibliography 前的核对字段

每一篇实际引用必须另行填写：规范作者名单、标题、年份与 venue、主 URL/DOI、
所支持的 proposal 段落、具体可检验主张、公开 code/weights/data/evaluator 状态。
不得因“已列入本表”而把 C0 论文数字升级为实测结果，也不得将作者公开媒体的
C2 evaluator 回放写为 C3 checkpoint 生成复现。

这些已实际进入当前复现/提案边界的对象，及其可支持的 proposal 段落和禁止升级，
另见 `CORE_FINAL_CITATION_MAP.md`。它和候选文献队列共同构成最终 bibliography 的
两层审核入口。

## 对当前选题的直接约束

这批主源确认了两个边界：H3 的公开部分是 Base checkpoint 部署而非可训练的完整
系统；而最接近、能在公开数据上持续推进到训练层的联合音视频路径是 MM-Diffusion。
在 H3 本地 Base 输出和一个可执行的公开 H3 benchmark 未闭环前，本表只能用于
限定假设与排除错误声称，不能据此提出或启动新的 H3 训练方法。
