# 共享反事实实验引擎设计规范（AC 盲审修订版：一份 `#CC0` VRM + Blender 管线，一键解锁「黄金二次元三强」`P6 · P3 · P2`）

> **AC 盲审修订（2026.11.25）**：已根据独立模拟审稿人二号意见**正式剔除 `P4` 的“VRM 头部等比放大 2.2× 冒充 Q 版 / 异构 VRM 强行换衣”错误设计**，并补齐了 **`P6` 零先验/反常规交织翻转**、**`P3` 等像素生物/安慰剂双对照**、以及 **`P2` 预打光输入 + 渲染域随机化**！

---

## 一、 资产准入合规清单（零版权雷区）

1. **3D 合成资产**：仅选用明确标注 **`CC0-1.0`** 或 **`CC-BY-4.0`** 的 `.vrm` 独立创作者角色（记录 `asset_id / author / source_url / license / sha256` 到 `licenses.csv`）。
   - **明确排除**：新版 VRoid Studio 默认未标注 CC0 的素体/模板、*Alicia Solid*（专属角色条款）、商业游戏解包模型。
2. **真实手绘 2D 生态效度资产（审稿人必查！）**：
   - 从 OpenGameArt（CC0 筛选）、Wikimedia Commons（显式 CC0/公有领域）、合规开源视觉小说分层立绘收集 **30–100 张跨 ≥4 种画风的真实 2D 手绘插画**（`P2` 需 80–100 张；`P3` 需含真实分层符号的 2D 立绘；`P6` 另配真实缠绕电线/白纸/手绘动漫验证集）。
3. **对外开源发布模式**：仿照 SIGGRAPH `CharacterGen` & `UniRig`，开源无头渲染与评测脚本 + 自渲染合规评测集，绝不违规再分发三方受限原始网格。

---

## 二、 共享 Blender 4.3+ 无头渲染引擎的 3 个核心开关（精准服务 `P6 · P3 · P2`）

基于 `blender --background --python render_engine.py` + `saturday06/VRM-Addon-for-Blender` + `MToon 1.0`：

### 开关 1：固定平涂画风下的「最小对 T 形交点翻转」（专供 **`P6`**，1 小时可跑！）
- 在锁死的 **`L3_Flat2D` 纯平涂画风**（另存 `L1_PBR` / `L4_LineArt` 作副轴）下，对以下 3 类场景成对交换局部遮挡 Z 序（`A-over-B` $\leftrightarrow$ `B-over-A`），并输出精确 `Z-Buffer` + 远离边界 ≥1–2 ViT Patch 步长的干净内部采样坐标：
  1. **Regime 1（常规动漫）**：刘海压眼、手挡头发、围巾压衣领；
  2. **Regime 2（零语义先验 / 对称交织）**：Blender 程序化生成的同色交织丝带、平涂纸条、同色双臂交叉（语义先验严格 50/50！）；
  3. **Regime 3（反常规先验冲突）**：丝带/长发/衣袖绕至手掌或脸部前方。
- **质量守门**：自动检查 Blender Freestyle/轮廓线在交点处无漏线，并用 50 对人类盲测确认人类天花板 ≥95%。

### 开关 2：非生物情绪符号最小对 + 等像素双对照组（专供 **`P3`**）
- 在同一张清晰正脸（匹配原生分辨率，避免 2px 瞳孔被 ViT 吞掉）上，同时生成 3 条**等像素修改面积（matched $\Delta$ pixels）**的平行分支：
  1. **非生物符号臂（8 类）**：`eye_highlight`（开/关）、`forehead_shadow`、`blush_lines`、`sweat_drop`、`starry/spiral_eyes`、`cat_mouth :3`、`ear_droop`、`ahoge_droop`，成对输出 `A → B` 过渡对（区分同向组 vs 微笑+无高光冲突组）；
  2. **等像素生物 FACS 对照臂**：相同像素预算的微皱眉/嘴角微撇（验证模型感知上限）；
  3. **等像素安慰剂对照臂（Placebo）**：相同像素预算的发卡/耳环开关（校准模型误翻噪底）。
- **务必叠加真实分层 2D/Live2D 开源立绘**，避免 VRM 贴纸生硬感。

### 开关 3：真重打光（Pre-Lit $\to$ Relit）+ 渲染域随机化 + 连续法线展平滑块 $\alpha$（专供 **`P2`**）
- 同一姿态 360° 扫描方向光与色温，并**彻底修复原版两大漏洞**：
  1. **输入必须含预打光（Pre-Lit）**：随机采样已有单方向光阴影的图（而非只从无阴影 `L3_Flat2D` 输入），强迫模型真正学会**去原光（De-lighting）+ 重打光**，消灭真实手绘图上的双层重影；
  2. **Shader 域随机化 + 内在层导出**：在 Blender 里随机扰动线宽、水彩/厚涂/网点底纹，并分离导出 `去光底图`、`硬边阴影掩码`、`逆光掩码`，防止模型把 50 个低多边形 VRoid 质感焊死进真实手绘输出；
  3. **成对连续法线滑块真值**：同步导出 `L2_Cel_RealNormal`（$\alpha=0$，真实 3D 鼻影）与 `L2_Cel_AnimeNormal`（$\alpha=1$，球面/柱面抹平人脸法线 + 保留锐利刘海投影）！

---

## 三、 统一元数据规范

所有样本统一落入 `manifest.jsonl`，每行记录 `sample_id`、`task_tag`（`P6`/`P3`/`P2`）、`character_id`、`regime_or_control`、`render_level`、`alpha_normal_flatten`、`light_azimuth_deg`、`affect_symbol`、`rgb_path`、`depth_exr_path`、`mask_path`、`license_ref`。
