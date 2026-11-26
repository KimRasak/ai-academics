# ICASSP 2027 投稿操作手册（deadline 2026-09-16，3 天内）

论文已完全就绪。以下是唯一剩下的、必须由作者完成的步骤。

## 0. 文件清单（本目录）

| 文件 | 用途 |
|---|---|
| `icassp_v1.pdf` | 完整编译稿（3 页，限 4+1 ✓）——**含作者占位符，填名后需重编译** |
| `icassp_v1.tex` | 源码（作者处有 TODO 标记，`grep TODO icassp_v1.tex` 可定位） |
| `icassp_abstract.txt` | 投稿系统纯文本摘要（1,208 字符，可直接粘贴） |
| `supplementary_vt.zip` | 补充材料（可作 supplementary 上传） |

重编译命令（本机已装 tectonic）：

```sh
cd h3_research/projects/seed-sensitivity/paper/icassp-build
# 编辑 main.tex 作者块（搜 TODO）后：
../../../../shared/tools/tectonic main.tex
```

## 1. 关键合规点（已核实）

- **页数**：4 页正文 + 第 5 页仅参考文献（我们 3 页，达标）
- **评审制**：官方 Paper Kit 明确 *"ICASSP does not perform blind reviews, so be sure to include the author list in your submitted paper"* —— **单盲，PDF 内必须写真名**（不能保留 Anonymous！）
- **格式**：IEEEtran conference 双栏（已用）
- **Track 建议**：Image, Video, and Multidimensional Signal Processing（IVM SP）
- Deadline：2026-09-16（AoE，系统以当地时间为准——建议 9/15 前提交）

## 2. 投稿步骤

1. 系统：`https://cmsworkshops.com/ICASSP2027/`（Paper Submission）；需 IEEE account
2. 填 Title / Abstract（粘贴 `icassp_abstract.txt`）/ Authors（= PDF 中的名字，顺序一致）
3. 选 Track：IVM SP；Keywords 建议：video quality assessment, evaluation methodology, reproducibility, text-to-video, benchmark
4. 上传填好作者的 PDF（作者名单与系统录入必须一致）
5. 可选上传 `supplementary_vt.zip`
6. 提交前下载系统生成的检查 PDF 再核对一遍

## 3. 提交前最后检查（10 分钟）

- [ ] PDF 作者块无 TODO 字样
- [ ] 系统作者列表与 PDF 完全一致
- [ ] Title 一致：*Seed Sensitivity of Composite Video-Quality Metrics: Three-Model Evidence on the MiniMax-H3 Leaderboard*
- [ ] 摘要与 PDF 一致
- [ ] PDF 共 3 页（≤4+1）

## 4. 若来不及/不投 ICASSP

- 论文主版（长版）`../main/main.tex` v0.5.2 与补充包持续有效，任何 venue 可用
- 备选：WACV 2027 workshop（见 [WACV 2027 dates](https://wacv.thecvf.com/Conferences/2027/Dates)），或 ICME 2027
