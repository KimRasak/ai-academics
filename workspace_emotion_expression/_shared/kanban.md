# 任务看板

> 协调器维护。角色来此领取/更新自己的任务。状态：`todo` / `doing` / `blocked` / `deferred` / `done`。

## 2026-08-03 晚间：检索证据升级（Boss 放行学术检索）

> 检索已放行：OpenAlex ✅（主通道）、S2 ⚠️（key 有效但限速时断）、arXiv API ❌（WebFetch 兜底）、WebSearch ❌。全部查询可复现，记录在 `stage_outputs/2026-08-03-retrieval-verification.md`。以下被阻塞项解封/部分完成：

| 任务 | 负责 | 原状态 | 新状态 | 备注 |
|---|---|---|---|---|
| 检索覆盖度声明（§7.3 venue 网格） | rigor+builder | blocked（检索不通） | **部分完成** | venue×year 网格法受 OpenAlex 会议年份化 source 限制无法直接跑；改用"`title_and_abstract.search`'expression editing' 2024-26 全量 59 条铺盘"替代，写进 verification §5；正式声明可再按年取 source ID 补全 |
| Q-004 其他 2D 顶会 benchmark 盘点 | builder | todo | **in_progress** | 已在检索中补充（§2/§6 表）；宏至 8 条未见"风格配对基准"，还需全文级核对 |
| 白空间系统核实 P1-P4（逐条记录） | rigor+builder | todo | **in_progress** | n001 FED-Bench（L2）、n002 FFE-Bench（新节点，L2）、n003 EmoStyle（新节点，L2）已核实；行为落在"部分占据但 arousal 正交未排除" |
| W5 工具链自测（S2+arxiv RSS） | builder | todo | **已完成（部分）** | S2 key 实测可通但限流严重；arxiv API 仍 429；RSS 未测（OpenAlex 已可替代发现） |
| 方向 1 白空间核心声明升级 | rigor 复核 | L1 | **L2（检索范围）** | 全量 59 条 + WebFetch 复核，未发现"风格×强度正交"检验；保留"（检索口径内"措辞 |
| 171 出处考证（W4） | rigor | todo | **做了负证据检索，无果** | OpenAlex+S2 两库均无"171 情绪清单"学术条目；维持阻塞，方向 4 前置条件不变 |

## 2026-08-03 方向评估组会（「接下来值得做的 AI 论文方向」）

> 五视角产出已回收（合成见 `_shared/stage_outputs/2026-08-03-paper-directions-synthesis.md`）。团队共识：方向 4（风格轴×情绪强度正交性检验）是本轮唯一 P0，可自给自足开工，不依赖 Boss 拍板；默认假设（判别先行/双造法/硬门槛）标注 provisional 推进。

| # | 动作 | 负责 | 状态 | 备注 |
|---|---|---|---|---|
| W1 | Q-005 正交预试验落地（=方向 4 正文）：2-3 风格卡×2-3 情绪的差分配对 + 固定 VA 后的残差盲评/分类器判别 + 预登记 κ 阈值 | rigor + builder | todo（可即刻开工） | 课题存亡只压它；与 Boss 3 项无关 |
| W2 | Q-006 2D 配对双造管线（C1 prompt 合成 / C2 局部编辑+manifest 审计 + 风格卡 YAML） + W1-W4 定量决策包 | builder | todo | 双造须"构造机制异构 + 至少一线不依赖编辑模型"（critic/rigor 双重审过），否则假双造 |
| W3 | 动漫方向（B）前置子命题：动漫面部身份度量小标定（嵌入对动漫形变的稳定性） | rigor + builder | todo | 未过前动漫方向不进推荐主体；数据合规：动漫权重/IP 只锁内部评测 |
| W4 | 171 情绪理论出处考证（"171 情绪+10 簇"材料源头） | rigor | todo | 出处未核实（E0）前，不可称"基于 171 理论" |
| W5 | 工具链 30 分钟自测：S2 API key（`.env`）+ arxiv RSS 流（WebFetch 抓取） | builder | todo | 通了就解除方向 3（FED-Bench 补正交）的竞品全扫卡死 |
| W6 | FED-Bench 是否兑现开源持续观察（是否正式录用 + 代码/数据） | builder | todo（观察项） | 决定方向 3 是以对标 vs 超越姿态进入 |
| W7 | 术语冻结 4 条入 glossary：正交（强口径）/差分（绑定共享 θ 配对）/双造法（异构独立）/表达强度（冻结前不入清单） | rigor | todo | 组会已采纳意见，待 rigor 落盘 |
| W8 | 方向评估进入 stage_outputs + Boss brief | 协调器 | doing | 本行即协调器产出 |

### 已判出（本轮，不再单独立项）

- **E（科研 agent 评测）已占位**：PaperBench（OpenAI, arXiv 2504.01848, L2 核实）已评"从零复现 20 篇 ICML 论文"类任务；剩余空白窄且难立真值——从候选清单划出（visionary 核实、critic 同意）。
- **D8 低算力非方向**：只是约束，不立项。
- **G（identity-expression Pareto）**："表达强度"标尺未定义即与 B1/方向 4 重叠（rigor 判），暂不承认独立方向；方法论元素可并入 C 作评估轴（visionary 注）。
- **D7〔voice 命名〕Pareto 化的"没动词"叙事**不具贡献（voice/critic/rigor 一致），若日后立项须把动词钉死为"使权衡可选取"。

## 2026-08-03 Blocked 更新（Boss 3 项 + 2 个执行口径）

| 阻塞项 | 阻塞什么 | 等谁 | 起算日 | 备注 |
|---|---|---|---|---|
| Boss 对 3 项待确认事项拍板（主线先判别后生成 / 配对数据双造法 / 预试验硬门槛） | 动作 5 种子选择；主线生成断言；叙事上限 | Boss | 2026-07-24 | 团队默认（①判别先行②双造③门槛+兜底）标注 provisional 全速推进，不空转；Boss 任何一项回复后回链落 `D-xxx` 并调整对应 W 项 |
| 实验禁令是否解除（Q-002） | W1 实验中是否含"生成验证"环节 | Boss | 2026-07-24 | 本轮新识别（visionary 建议）：按"判别为主、生成为证"解释执行，正式口径待 Boss |
| 盲评标注预算 | W1 预实验中盲人招募规模与来源 | Boss | 2026-08-03 | 本轮新识别：小样本盲评可自组织，付费扩大规模需预算批四件套 |

## Survey 阶段（D-002 主线 + D-003 收窄为纯 2D；首轮已合成，第二轮 2D 收窄讨论进行中；2026-08-03 方向评估完成、检索已放行）

### 首轮任务（已完成）

| 任务 | 负责 | 状态 | 备注 |
|---|---|---|---|
| 识别与表情差分一致性相关的顶会研究方向 | visionary | done | 产出 5 候选方向 |
| 形式化主张边界 + 检索纳入/排除标准 + Q-004 操作化 | rigor | done | 写入 stage_outputs 与 glossary |
| 盘点 benchmark 数据集与评测指标 + 契合度矩阵 | builder | done | 含 3D 行（已被 D-003 剔除/标"已排除"） |
| pre-mortem + benchmark 许可审计 + 评测捷径 + 替代方案 | critic | done | 风险登记册 |
| 方法演进对比表 + pitch + Boss 小结框架 | voice | done | 对比表与 pitch |
| 将首轮全组产出合成并写 Boss 小结 | 协调器 | done | briefs/2026-07-24-survey-team-synthesis.md |

### 第二轮（D-003 收窄为纯 2D 后）

| 任务 | 负责 | 状态 | 备注 |
|---|---|---|---|
| 评估纯 2D 下"表达差分一致性"白色空间是否仍真实 + 候选 2D 方向战略价值 | visionary | doing | 本轮全组讨论 |
| 重写检索标准：新增"仅 2D""近 2–3 年顶会"约束 + 重核白色空间 + 更新 Q-004/Q-005 在 2D 下判据 | rigor | doing | 本轮全组讨论 |
| 盘点纯 2D 方向×benchmark×指标矩阵（剔除 3D 行）+ 2D 配对数据瓶颈方案 | builder | doing | 本轮全组讨论 |
| 审 2D 下"同一角色/同情绪/仅变风格"配对数据获取瓶颈 + 3D 剔除后风险重评 | critic | doing | 本轮全组讨论 |
| 更新方法演进对比表与 pitch 至纯 2D 设定 | voice | doing | 本轮全组讨论 |
| 合成第二轮讨论并写 Boss 小结 2026-07-24-survey-2d-narrowing.md | 协调器 | doing | needs-confirmation |

### 待办（跨轮）

| 任务 | 负责 | 状态 | 备注 |
|---|---|---|---|
| 按更新后检索标准系统核实"白色空间是否真无人占据"（逐篇记录验证了 P1/P2/P3/P4） | rigor + builder | todo | 2D 收窄后重新核实 |
| 收口 2D benchmark 许可"待核实"项为 manifest（剔除 3D 资产分支） | builder + critic | todo | 服务 Q-003；聚焦 2D 图像数据集与合成管线 |
| 风格轴存在性预试验草案（2D 资产版：遮蔽+arousal 正交性+预登记阈值） | rigor + builder | todo | Q-005/Q-004；待 Boss 确认是否设为 Q-002 前置门槛 |

## 已降级/关闭

| 任务 | 负责 | 状态 | 备注 |
|---|---|---|---|
| 澄清 FaceWare 名称 | builder→Boss | closed | D-003 排除 3D，该事项失效 |

## Deferred（暂缓）

| 任务 | 负责 | 状态 | 备注 |
|---|---|---|---|
| 复现/评估关键 baseline 的可用实现 | builder | deferred | D-002 指示暂缓实验；关联 Q-002 |
| 试点协议（样本/对照/盲评/阈值）落地 | builder + rigor | deferred | 同上，待 Survey 后重启 |

## Blocked

| 阻塞项 | 阻塞什么 | 等谁 | 起算日 | 备注 |
|---|---|---|---|---|
| Boss 对 3 项待确认事项拍板（主线先判别后生成 / 配对数据双造法 / 预试验硬门槛） | 动作 5（A+B+C 试点）的种子选择；动作 1/2/3/6/7 的目标命题；Q-005 预试验草案能否启动；voice 叙事收敛 | Boss | 2026-07-24 | 来源 `2026-07-24-survey-2d-narrowing-boss.md` 待确认 3 项；已等 3 天；催办 brief `2026-07-27-boss-recheck-3-items.md` |

## 2026-07-27 组会后动作（7 项分诊）

> 7 候选动作来自 2026-07-27 组会（visionary/voice/critic 收敛）。P0 = 动作 5 + 动作 1；其余按阶段/依赖 defer 或 block。

| # | 动作 | 负责 | 状态 | 备注 |
|---|---|---|---|---|
| 1 | 最薄切片 L1 验收（工具链真实负载跑通） | builder | in_progress | builder 正在跑 L1；不依赖 Boss 3 项 |
| 2 | 白空间证伪后的抽象/链路动作 | 待分派 | deferred | 属 Ideation/Design 阶段；等主线定 + 白空间证伪后再启 |
| 3 | 白空间证伪后的固化动作 | 待分派 | deferred | 属 Ideation/Design 阶段；等主线定 + 白空间证伪后再启 |
| 4 | graveyard 落盘（被否决方案记入演化树 fork 分支） | critic | done | 已落盘 `_shared/graveyard/README.md` |
| 5 | A+B+C 试点种子选择（白空间定向证伪） | rigor + builder | blocked | 阻塞 = Boss 主线拍板（动作 5 种子随判别/生成路径不同而不同）；Boss 回第 1 项即可解封种子选择 |
| 6 | 现有演化树/workspace/graveyard 体系修补 | 待分派 | deferred | 格式没坏别修；推到 Ideation 阶段 |
| 7 | 演化树骨架先建类动作 | 待分派 | deferred | 属 Ideation 阶段；不应先建骨架 |
