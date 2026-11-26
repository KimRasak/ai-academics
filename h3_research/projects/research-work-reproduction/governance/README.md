# 工作区规则

- [WORKSPACE_CONTRACT.md](WORKSPACE_CONTRACT.md)：目录职责、候选与已入队工作的
  边界、必需证据文件、一次状态变更应同步更新的位置。

规则优先于便利：不能为了减少文件而删除原始 manifest、媒体 provenance 或失败记录；
可删除的只有可再生缓存和编译产物，它们已由上级 `.gitignore` 排除。
