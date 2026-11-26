# EmoH3 Evaluation

研究主题：MiniMax H3 情绪控制评测对 judge、选项顺序和抽帧范围的敏感性。

- `paper/working/`：工作论文的多 venue 版本。
- `src/analysis/`：统计、身份和人评分析。
- `src/judges/`：VLM 情绪与身份判官。
- `src/generation/`：生成、抽帧和输出准备。
- `src/training/`：manifest、cache 和训练器。
- `src/paper/`：附录、图和盲标工具生成。
- `runners/{generation,training,evaluation,ops}/`：远程运行脚本。
- `results/`：原始判官日志、人评材料和验证结果。
- `docs/{plans,reports,rounds}/`：计划、结论和逐轮实验记录。
- `supplemental/`：补充材料工作副本。
- `release/`：自包含发布快照；应视为不可变，不与工作副本去重。
