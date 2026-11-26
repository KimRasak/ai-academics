# 人类标注锚点（blind）— 使用说明

## 为什么要做这一步
我们已经证明：四个开放 VLM 判官对同一批片段只有 42–72% 一致（Fleiss κ=0.32），而且
**干预效果的符号随判官翻转**（原判官 +7.0pp，更强判官 −8.1pp）。缺的唯一一环是：
**哪个判官（如果有）跟人的判断一致**。这批标注能一次性回答它，无论结果朝哪边都有定论价值。

## 怎么标（约 20 分钟，94 题）
1. 双击打开 `index.html`（不需要服务器、不需要联网）。
2. 每题看帧 → 从 15 个词里**必选一个** → 选把握程度（1 低 / 2 中 / 3 高）。
   - 快捷键：`h`happy `j`joyful `s`sad `a`angry `w`worried `c`cheerful `u`surprised
     `r`relieved `n`anxious `i`curious `e`neutral `d`disgusted `f`fearful `m`amused `o`other
   - `Z` 回退上一题；点图片放大；进度自动存在浏览器，可以中断后继续。
3. 标完点「下载」，把 `labels_<代号>.json` 放回本目录（`results/human-label/`）。

## 重要：盲测纪律
- 题目顺序随机、文件名只有 `i001…i094`，**不含模型/条件信息**（base 还是 LoRA、哪个 seed 都看不出来）。
- 里面混了 55 个视频片段（8 帧，和判官看到的完全相同）+ 39 张已知情绪立绘（用来量你自己的标注可靠性）。
- 不要事先看 `MAP.json`（它是解码表，标完再用）。
- **最好有 2–3 位标注者各自独立标一份**（labels_A.json / labels_B.json / …），这样才能报标注者间一致性；
  只有一份也能用，但结论强度会打折。

## 标完之后
运行 `python3 src/analysis/score_human_labels.py`（在 `projects/emoh3-evaluation/` 下），会直接输出：
标注者间一致性、你在已知立绘上的准确率、**每个判官与人类共识的一致率/κ**、
决定性 x02 对的人类判词、以及**用人类标签重算的 base vs v3 效应方向**。
