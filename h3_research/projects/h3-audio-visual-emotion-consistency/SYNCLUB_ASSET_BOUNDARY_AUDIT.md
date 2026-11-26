# SynClub assets 使用边界审计

**审计日期：** 2026-09-19  
**审计对象：** `/Users/jinzili/code/synclub_expression/assets`  
**结论：** 未获准进入正式训练、正式 benchmark 或论文核心定量证据。

## 后续交叉检查（2026-09-20）

对 `projects/research-work-reproduction/` 与本项目的 Markdown、JSON、JSONL、
shell 和 Python 路径引用复查后，除本边界说明及明确的禁止性文字外，未发现
`synclub_expression/assets` 被接入任何活跃复现、H3 部署、训练、正式 benchmark
或 proposal 数据路径。正在运行的 MM-Diffusion run 3 仍明确使用公开 Landscape
archive 与其已保留的 manifest。该复查不改变上面的禁止结论。

## 观察到的状态

- 路径存在；只读盘点时约有 **14,540** 个文件、约 **6.4 GB**。
- 顶层及浅层内容包含角色图、场景/提示清单、实验性 JSON、网页图像与局部
  README/应用文件，表明它是混合用途资产目录，而非一个已冻结、已发布的数据集。
- 在当前 H3/reproduction 文档树中未发现该目录对应的正式数据卡、许可证、原始
  来源 URL、统一样本清单、标注 schema、去重/过滤记录、固定 train/validation/test
  切分或媒体—派生缓存 provenance manifest。
- 当前活跃 MM-Diffusion 复现使用的是公开 Landscape archive；其已记录的 manifest、
  权重、评测和输出路径均不指向此目录。

## 可用范围与禁止项

在上述缺口补齐前，本目录只可用于本地可视化、接口诊断或非结论性的 smoke test。
不得用作：

- 任意探索训练、偏好数据构造或模型选择；
- 论文的训练集、验证集、测试集或定量比较依据；
- 替代公开 benchmark、公开原始媒体或缺失的论文训练数据；
- 只保留 latent/`.pt` 后删除原始媒体的缓存来源。

## 如未来申请正式使用，必须先完成

1. 明确每一资产或子集的权利人、许可证、原始来源、采集日期和允许用途；
2. 生成不可变样本 manifest（相对路径、SHA-256、媒体元数据、来源键、标注键）；
3. 审计标注定义、质量、覆盖和潜在身份/版权/隐私风险；
4. 预注册公开可取得的对照数据、固定切分、训练目的和独立评测协议；
5. 将所有 cache 映射回原始媒体与处理命令，并经人工复核后才改变本结论。

这是一项使用边界审计，不对目录内每份文件的版权、真实性或适用性作正面保证。
