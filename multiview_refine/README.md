# Layer 2: Multi-view Refinement
输入：
- initial Gaussian + FLAME params：来自LAM 前向的输出
- 多视角图片：同一人脸
- COLMAP camera：colmap 重建多视角人脸得到的相机参数
- 每张图的 mask / landmarks / FLAME tracking：考虑到输入图片可能不是严格同一表情动作
输出：
- refined Gaussian：多视角迭代优化后的 Gaussian，核心产物
- refined per-view camera delta / pose delta：gaussian 优化的附带产物，一般是联合优化的，避免colmap得到的相机参数不太准确
- refined / fixed FLAME identity betas：FLAME 基础形态参数
- refined base params：如果允许 FLAME 基础参数的小幅优化
- optional densified Gaussians：如果允许gaussian裂变复制

主要实现闭环：**多视角数据 -> 坐标对齐 -> LAM/GS 渲染 -> loss 反传 -> refined Gaussian**

1. **数据加载**：加载 initial canonical Gaussian、FLAME betas、per-view FLAME params、aligned cameras、RGB/mask。
2. **对齐**：渲染所有视角，检查 mask/landmark/五官大位置是否对齐。
3. **全局几何参数校准**：冻结 Gaussian，只优化 global alignment 或 per-view camera/head pose 小 delta。
4. **Gaussian Appearance refinement**：固定 FLAME identity 和 Gaussian xyz/offset，优化 shs/opacity/scaling。
5. **Gaussian Geometry refinement**：固定 FLAME identity 和绑定结构，小幅优化 Gaussian xyz/offset，并用 offset/scale/smooth 正则约束。
6. 可选表情微调：仅当眼睛/嘴巴状态明显不对时，小幅优化 expr/jaw/eyes delta。


## 数据预处理
1. 图片拍摄
  - 同一人
  - 尽量同一表情 + 同一光照
  - 统一尺寸
  - 文件名按帧/视角排序
2. 前景 mask：可以配合抠图使用
3. colmap 基础重建：获得每张图对应的相机外参+内参，提供相机几何，但没有标准坐标系约束
4. FLAME tracking：获得每张图的人脸状态参数和语义约束
5. 坐标系对齐：求 Sim(3) 全局优化，把 COLMAP camera 校准到 LAM/FLAME render 坐标系。FLAME tracking 的单图相机不作为默认对齐目标。

## 启动

```bash
python app_multiview_gaussian_refine.py
```

默认打开 `http://127.0.0.1:7861`。第 0 页签上传分成两块：

1. `Camera Images ZIP`：纯相机拍摄的多视角图片包，只放目标人物的多视角原图。
2. `LAM Canonical Package ZIP`：单图正面 LAM Layer 1 输出包，放同一个人的 canonical 空间 Gaussian 和 FLAME identity/frame 参数。

`Camera Images ZIP` 推荐结构：

```text
camera_views.zip
└── camera_views/
    ├── images/
    │   ├── 000001.png
    │   ├── 000002.png
    │   └── ...
    └── fg_masks/                   # 可选；也接受 masks/
        ├── 000001.png
        ├── 000002.png
        └── ...
```

也可以不包 `images/` 目录，直接把 `.png/.jpg/.jpeg` 放在 ZIP 根目录；程序会把图片复制到 workspace 的 `images/`。这包里不要混 Layer 1 的 Gaussian/FLAME 文件，避免把“目标多视角观测”和“初始化 identity”混在一起。

`fg_masks/` 是可选的多视角前景 mask。如果你已经有离线抠图结果，可以和图片一起放进 `Camera Images ZIP`，或后续在第 1 页签 `Import` 的 `Masks Dir` 导入；导入后程序会统一放到 workspace 的 `fg_masks/`。mask 推荐用 `.png` 单通道灰度图，白色/高值表示参与优化的人脸/头部前景，黑色/低值表示背景；程序会把 mask 转成灰度并除以 255 作为 `[0, 1]` alpha 权重，所以硬二值和带软边的灰度 mask 都可以。mask 尺寸必须和对应图片一致，文件名 stem 也必须一致，例如 `images/000001.png` 对应 `fg_masks/000001.png`。

如果不提供 `fg_masks/` 或 `masks/`，可以在第 2 页签 `Masks / FLAME` 点击 `Run FLAME Tracking + Generate Masks` 自动生成。自动流程是：每张图单独检测头脸 bbox，扩大 bbox 后裁剪并 resize 到 `1024x1024`，用 human matting 模型生成 crop 内前景 mask，再根据 preprocess metadata 把 crop mask resize/paste 回原始图片坐标，最终写入 `fg_masks/<原图stem>.png`。这个 mask 不是 COLMAP 生成的，也没有做多视角一致性优化；如果自动结果不准，可以只替换对应视角的 mask 文件。

`LAM Canonical Package ZIP` 推荐结构：

```text
layer1_lam.zip
└── layer1_lam/
    ├── xxx_canonical.ply                 # 必需；canonical 空间绝对 xyz Gaussian
    ├── xxx_canonical_flame_param.npz     # 必需；同一个 identity 的 shape/betas
    └── flame_param/                      # 可选；单图正面 frame 参数，导入后仅归档
        └── 00000_00.npz
```

程序会把 `xxx_canonical.ply` 复制成 workspace 下的 `init.ply`，把 `xxx_canonical_flame_param.npz` 复制成 `canonical_flame_param.npz`。这里的 Gaussian 必须是 LAM/FLAME canonical 空间的绝对坐标版本；`*_gs_offset.ply` 或 OAC/h5 包里的 `offset.ply` 只保存 offset，不适合作为这个入口的默认 init Gaussian。

这两个 ZIP 都要来自同一个人。`init.ply` 和 `canonical_flame_param.npz` 是强绑定的 Layer 1 初始化；后续每张多视角图的 mask、landmarks、per-view FLAME 参数由 `Masks / FLAME` 页签生成，或者通过第 1 页签 `Import` 导入已经离线准备好的结果。

文件匹配规则：

- `fg_masks/` 或 `masks/` 中的 mask 文件名 stem 必须和 `images/` 对应，例如 `images/000001.png` 对应 `fg_masks/000001.png`。
- `flame_param/` 和 `landmark2d/` 也按同名 stem 查找，例如 `flame_param/000001.npz`。
- 所有用于 refinement 的图片尺寸必须一致，mask 尺寸也必须和对应图片一致。
- `transforms_aligned.json` 或 `transforms.json` 的 `frames` 里需要包含相机内参 `fl_x/fl_y/cx/cy` 和 `transform_matrix`；通常通过 Gradio 的 `COLMAP` + `Sim3 Alignment` 生成。

第 1 页签 `Import` 不上传文件，而是填写本机目录路径，适合数据已经在磁盘上的情况：

- `Images Dir`：必填，指向多视角图片目录。
- `Masks Dir`：可选，导入后会复制为 `fg_masks/`。
- `FLAME Param Dir`：可选，可指向包含 `flame_param/` 和 `canonical_flame_param.npz` 的目录，也可直接指向 `flame_param/`。
- `COLMAP Dir`：可选，导入到 workspace 的 `colmap/`。
- `Initial PLY Path`：可选，导入后复制为 workspace 下的 `init.ply`。

### 坐标系对齐

同时接入三份来自不同坐标系的数据：

1. LAM initial Gaussian / FLAME identity
  - 坐标系：LAM 的 FLAME canonical 空间，也是最终 refined Gaussian 所在的模型坐标系。
  - 作用：提供 canonical bound Gaussian 和 identity betas。
  - 要求：init.ply 应保存 Gaussian 的真实 canonical xyz，不能只保存 offset。

2. COLMAP camera
  - 坐标系：COLMAP 自己重建出的任意 SfM 坐标系，尺度、朝向、原点都不受 LAM 约束。
  - 作用：提供多视角相机外参/内参。
  - 对齐目标：经过 manual Sim3、显式 calibrated target transforms，或后续 calibrate 阶段的 global/per-view camera delta，变成 LAM/FLAME render 坐标系下的 camera。

3. Per-view FLAME tracking
  - 坐标系：单图 FLAME tracking 的裁剪图像坐标和单图相机假设，不是 COLMAP 多视角坐标系。
  - 作用：只作为语义观测，提供 mask、landmarks、expr/jaw/eyes 等每视角状态约束。
  - 处理：mask/landmarks 会从 tracking crop 坐标还原到原始多视角图片像素坐标；global rotation/translation 固定 canonicalize 为 0，避免把相机绕人拍摄的视角重复编码进头部姿态。
  - 不作为 Sim(3) target：single-image FLAME tracking 产生的 transforms 不代表真实多视角 camera rig，不能直接拿来和 COLMAP camera center 做 Sim(3)。
