# H3 Research Workspace

本目录按论文与研究课题组织。每个 `projects/*` 目录都是一个独立的研究与复现单元。

| 项目 | 状态 | 入口 |
|---|---|---|
| VT seed sensitivity | 论文与投稿包已形成 | `projects/seed-sensitivity/README.md` |
| EmoH3 evaluation | 论文、判官实验与发布快照 | `projects/emoh3-evaluation/README.md` |
| Avatar emotion | EchoMimic 情绪塌缩与训练研究 | `projects/avatar-emotion/README.md` |
| Research-work reproduction | 公开 benchmark、论文指标、权重与训练流程复现 | `projects/research-work-reproduction/README.md` |
| H3 Audio-Visual Emotion Consistency | H3 音视频情绪一致性课题与 research proposal | `projects/h3-audio-visual-emotion-consistency/README.md` |

其他入口：

- `incubator/`：尚未立项或已暂停的候选方向和文献调查。
- `shared/`：跨项目工具、第三方代码和官方 H3 基线资料。
- `archive/`：历史事故恢复或不再活跃的材料。
- `.cache/`：可再生缓存、编译产物和临时生成证据；不作为研究入口。
- `WORKLOG.md`：跨项目的原始时间线，保留历史路径，不作为当前目录导航。
- `INDEX.md`：当前结构和旧路径迁移表。

新增材料先归属到一个具体项目；只有被两个以上项目复用的实现才进入
`shared/`，尚未形成项目的问题才进入 `incubator/`。不要在本目录顶层新增
项目文件或按文件类型建立新的平铺目录。
