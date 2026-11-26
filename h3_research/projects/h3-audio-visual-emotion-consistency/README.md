# H3 Audio-Visual Emotion Consistency

本目录对应旧课题全称 **H3 Audio-Visual Emotion Consistency**，即 H3
音视频情绪一致性研究。此前自行设计的私有小数据训练实验及其派生缓存已清退，
当前仅保留课题定义、历史清退记录和待重写的 research proposal。

论文、公开权重、训练流程和 benchmark 的通用复现工作由相邻的
`../research-work-reproduction/` 维护。本目录不保存复现代码、评测账本或
通用文献池；只有达到明确证据等级的结论才引用回 proposal。

## 当前入口

- `PROPOSAL_H3.md`：复现证据门槛；尚未锁定新方法。
- `PROPOSAL_EVIDENCE_GATE.md`：最终 training proposal 的逐项证据准入表；明确
  已核验事实、未闭环证据与禁止捷径。
- `SYNCLUB_ASSET_BOUNDARY_AUDIT.md`：本地 SynClub assets 的使用边界；在来源、
  许可证、标注和 provenance 闭环前，仅允许可视化、诊断与 smoke test。
- `../research-work-reproduction/planning/REPRODUCTION_FIRST_PLAN.md`：科研工作复现执行计划。
- `../research-work-reproduction/dashboard/PAPER_REPRODUCTION_TRACKER.md`：**论文复现总表**；
  后续汇报统一按论文名、公开权重与链接、验证数据集/指标、权重验证结果、训练复现
  结果组织。
- `../research-work-reproduction/planning/REPRODUCTION_BLUEPRINT.md`：benchmark、
  指标演进和逐主张证据蓝图。
- `CLEANUP_20260916.md`：本轮删除范围、保留资产和后续准入规则。
- `../research-work-reproduction/reproductions/minimax-h3-official-768p/`：
  H3 官方部署复现卡。
- `../research-work-reproduction/refs/`：通用视频生成与后训练文献池。

## 依赖资产

- 官方 H3 权重：`/gluster_hdd_cv/user/jinzili/weights/MiniMax-H3`
- 官方公开输入与复现数据：
  `/gluster_hdd_cv/user/jinzili/data/h3_reproduction/`
- 私有 SynClub 资产仅允许用于可视化、诊断和 smoke test，不得用于探索训练。

## 当前判定

- H3 官方权重与三类 768p 可复现请求已存在于远端；
- 权重、请求路由和冻结工具已核验；授权 4×80GB H100 上的 FL2VA 服务已实际启动
  两次，但均在 DiT FSDP 初始化峰值 OOM，未监听、未提交请求、未生成媒体。安装版
  H3 的受支持 `memory` mode 不降低这个峰值；
- 下一步复现任务由 `research-work-reproduction/` 统一调度；
- 在公开复现门槛通过前，不启动新的 H3 自主探索训练。
