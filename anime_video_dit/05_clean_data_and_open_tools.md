# 二次元人物科研合规数据、开源渲染管线与评测基座速查手册（2026.11）

> **目的**：确保任何选定的二次元科研题目都能 **100% 合法合规（零业务私有数据、零侵权番剧抓取）**，并可直接落地在 `osa00-h100-007` 科研节点上复现与发顶会。

---

## 一、 3D→2D 程序化反事实渲染资产（已获 CVPR / SIGGRAPH 先例背书）

顶会近期论文（CVPR **`PAniC-3D`**、SIGGRAPH **`CharacterGen`**、**`UniRig`**、**`LinkTo-Anime`**）已经确立了用 **VRM 卡通模型 + Blender 无头批量渲染** 构造二次元精准真值基准的主流范式：

- **重要版权核查纠偏（必读！）**：
  - 早期 VRoid Studio Beta 示例模型带 CC0，但**新版 VRoid Studio 默认自带素体与服装模板并非自动 CC0**（受 Pixiv 服务条款约束）；VRM 官方示例角色 *Alicia Solid* 的代码仓库为 MIT，但角色模型本身受专属角色规范约束，同样不可直接当成 CC0 商用再分发。
  - **100% 合规的两种标准做法**：
    1. 在 VRoid Hub / Sketchfab / OpenGameArt 严格筛选明确标注 **`#CC0`** 或 **`CC-BY 4.0`** 的独立创作者 `.vrm` 模型（记录每个资产的 URL、作者与协议清单）；
    2. 仿照 SIGGRAPH `CharacterGen` 与 `UniRig` 的顶会标准：**开源“无头 Blender + `VRM-Addon-for-Blender` + `MToon 1.0` 渲染与反事实生成脚本”**，只公开发布自渲染且协议洁净的评测集或生成脚本，绝不违规打包再分发三方商业 `.vrm` 源文件。
- **技术栈（纯开源、可无头脚本化，已核实仓库）**：
  - 渲染引擎：**Blender 4.3+**（无头模式 `blender --background --python script.py`）+ **[`saturday06/VRM-Addon-for-Blender`](https://github.com/saturday06/VRM-Addon-for-Blender)**；
  - 二次元着色器：**`MToon 1.0`**（VRM 官方标准赛璐璐着色器，可精确控制硬阴影阈值、阴影色相偏移、轮廓线粗细、边缘逆光 Rim Light）+ 自定义 **Stylized Anime Face Normal（球面/柱面抹平人脸法线）**；
  - 顶会参考先例：**`LinkTo-Anime`**（`arXiv:2506.02733`，395 序列 / 24,000+ 帧，基于 3D/VRM 渲染 + Mixamo 骨骼 + 自定义轮廓线与赛璐璐着色）。
  - 可一键生成的 4 类精准反事实真值：
    1. **重打光真值（支撑 `Anime-P2`）**：同角色同姿态 × 360° 光源方向 × 「3D 物理 PBR vs 真实法线卡通 vs 抹平人脸法线硬色块阴影」；
    2. **非生物情绪符号真值（支撑 `Anime-P3`）**：同张脸 1 像素不动，程序化开关瞳孔高光贴图、额头暗影遮罩、脸颊斜线脸红、汗滴/猫嘴；
    3. **四维抽象身份真值（支撑 `Anime-P4`）**：角色 A 穿角色 B 服装（互换服装）、8 头身 $\leftrightarrow$ 2 头身 Q 版缩放、彩色平涂 $\leftrightarrow$ 黑白网点漫画 Shader；
    4. **多流派画风基线（支撑 `Anime-P5`）**：同一批角色切换不同 NPR Shader（平涂/水彩/粗黑描边/黑白网点/复古低饱和）。

---

## 二、 公开学术基准与协议红线核查表

| 资源 | 用途 | 真实协议 / 风险提示 | 使用建议 |
| :--- | :--- | :--- | :--- |
| **`LinkTo-Anime` (`2506.02733`)** | 80 个 VRM 渲染的 29k 帧赛璐璐光流与动态基准 | 学术开源（合成 VRM 渲染） | ✅ 可作 2D 赛璐璐动态参考基准 |
| **`AnimationBench` (`2604.15299`)** | 迪士尼 12 法则 + 角色动画评测集 | arXiv + 官方 GitHub 学术评测 | ✅ 评测对比基线 |
| **`Live2D-Bench` (`2607.27348`)** | 2D 分层角色拆解与网格驱动基准 | 学术评测发布 | ✅ 评测参考 |
| **`MangaUB` / `Re:Verse` (`2508.08508`)** | 漫画视觉语言理解基准 | 学术评测（漫画素材仅限评测） | ✅ 理解侧对照基线 |
| **`AnimeRun`** | Blender 开源电影渲染的 2D 光流 | ⚠️ **`CC-BY-NC 4.0`（非商用！）** | ⚠️ 仅限非商用学术评测，勿标成商用友好 |
| **`Sakuga-42M` (`2405.07425`)** | 番剧视频片段集 | ❌ **arXiv 已因版权撤稿** | ❌ **严禁使用** |
| **原始 `Danbooru` / `Safebooru` 打包** | 爬图二次元插画 | ❌ **版权与画师争议高风险** | ❌ **严禁作为论文主释放集** |

---

## 三、 `osa00-h100-007` 可直接调用的开源评测模型清单

- **图像编辑与重打光基座（已核实开源权重状态）**：
  - `Qwen-Image-Edit`（Apache-2.0，2511/2512 版支持指令编辑）
  - `FLUX.1-Kontext`（非商用科研评测，强上下文风格保持）
  - `IC-Light`（[`lllyasviel/IC-Light`](https://github.com/lllyasviel/IC-Light)，ICLR'25 Oral，全开源权重）
  - `UniRelight`（[`nv-tlabs/UniRelight`](https://github.com/nv-tlabs/UniRelight)，`arXiv:2506.15673`，NVIDIA 官方代码 + HF 权重）
  - *(注：`LightLab` `2505.09608` 截至 2026.11 仍未公开官方权重，仅作相关工作引用，不列作本地 H100 基线！)*
- **3D 深度、法线与空间理解基座（用于 `Anime-P6` 与 `Anime-P2` 法线对照）**：
  - `depth-anything/DA3-LARGE` / `DA3-GIANT`（`Depth Anything 3`, `arXiv:2511.10647`, `ByteDance-Seed/depth-anything-3`）与 `depth-anything/Depth-Anything-V2-Large`（`2406.09414`）
  - `facebook/VGGT-1B`（`cc-by-nc-4.0`，另有 `facebook/VGGT-1B-Commercial` / `facebook/VGGT-Omega`，`2503.11651`）
  - `Ruicheng/moge-2-vitl-normal`（Microsoft `MoGe-2`，同时输出单目深度与表面法线！）
  - `apple/DepthPro`
- **视频与原生音视生成基座**：
  - `Wan2.1-I2V-14B` / `Wan2.2-TI2V-5B`（Apache-2.0，主视频基座）
  - `AMAP-ML/DreamX-Creator`（7B Apache-2.0，基于 Wan2.2 的原生音视 DiT）
  - `Lightricks/LTX-2.3`（22B，非商用学术评测，原生双流音视 DiT）
- **多模态理解与表征编码器基座**：
  - `Qwen/Qwen2.5-VL-7B-Instruct` / `72B-Instruct`、`Qwen3-VL`（已合入 `transformers`）、`OpenGVLab/InternVL3`
  - `DINOv2` / `SigLIP 2` / `CLIP` / **`deepghs/ccip`**（HuggingFace 开源动漫角色对比学习编码器，附 `deepghs/anime_face_detection`）
  - `HPSv3` / `PickScore` / `VideoReward`（审美与视频偏好奖励模型审计）
