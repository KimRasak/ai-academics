# Survey 风险登记册（精华）

> 一屏摘要。详细论证见 `notes/survey-pre-mortem-and-benchmark-audit.md`。
> 更新：2026-07-24

## Pre-mortem 五大失败机制
1. **F1 表情-身份纠缠**（最严重）：风格扰动牵动身份，"同一角色"在像素层不成立。早期信号：ArcFace 跨风格 cos<同身份阈。
2. **F2 "一致性"不可证伪**：四种定义（分布集中/身份保持/可插值/盲评复现）互不蕴含，易漂移。
3. **F3 无对应 benchmark**：真实人脸 benchmark 非差分成对数据；自建规模小被质疑 cherry-picking。
4. **F4 identity↔expression trade-off**：方法靠"少改脸"刷身份、"夸张化"刷表情，加权任意。
5. **F5 合规使 Q-003 不闭**：主流 benchmark 不可再发布，复现包只能给代码。

## Benchmark 许可结论
可公开审计/可再发布的真实人脸表情 benchmark ≈ 0；可走只剩合成/3D 资产 + 自标注，需逐项核许可证。具体逐数据集表见 notes。

## 评测捷径 top-3
1. identity↑↔expression↓ 隐性 trade-off（画双轴 Pareto 验证）。
2. FER 分类器循环性（换多分类器看排名稳定性）。
3. 风格被强度偷换（固定 VA 验证）。

## 替代路径
- B1 收窄到 AU 级差分一致性（有 FACS+估计器，可证伪）。
- B2 先判别后生成（先证任务可解）。
- 主线批评：含控制/识别/保持一致三任务，太散，建议先收敛到识别/度量。
