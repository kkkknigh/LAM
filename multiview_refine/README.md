# Layer 2: Multi-view Refinement

## 入口和数据位置

启动入口：

```bash
python app_multiview_gaussian_refine.py
```

默认服务地址是 `http://127.0.0.1:7861`。可用环境变量覆盖：

- `GRADIO_SERVER_NAME` / `GRADIO_SERVER_PORT`：Gradio 监听地址和端口。
- `MULTIVIEW_OUTPUT_ROOT`：workspace 根目录，默认 `output/multiview_refine`。
- `COLMAP_PATH`：COLMAP 可执行文件，默认 `colmap`。
- `APP_INFER` / `APP_MODEL_NAME`：LAM 推理配置和权重目录。

每次在页面点击 `Create Workspace` 会创建一个独立 workspace：

```text
output/multiview_refine/<job_id>/
├── inputs/                         # 原始上传 ZIP 归档
├── data/
│   ├── images/                     # 多视角原图
│   ├── fg_masks/                   # 多视角前景 mask
│   ├── flame_param/                # 每个视角的 FLAME tracking 参数
│   ├── landmark2d/                 # 还原到原图像素坐标的 2D landmarks
│   ├── init.ply                    # Layer 1 canonical Gaussian 初始化
│   ├── canonical_flame_param.npz   # Layer 1 identity / betas
│   ├── layer1_metadata.json        # 可选，Layer 1 元数据
│   └── layer1_reference_transforms.json  # 可选，Layer 1 参考相机
├── tracking_work/                  # 单图 FLAME tracking 中间结果
├── colmap/
│   ├── database.db
│   ├── sparse/
│   ├── transforms_colmap_sparse.json
│   └── transforms_colmap_raw.json  # COLMAP 相机，尚未对齐到 LAM 坐标系
├── alignment/
│   ├── sim3_colmap_to_lam.json
│   ├── transforms_aligned.json     # refinement 实际读取的相机
│   ├── initial_sim3_report.json
│   └── landmark_calibration_report.json
├── refine/
│   ├── refined_gaussian.ply        # 核心产物
│   ├── camera_delta.pt
│   ├── intrinsics_delta.pt
│   ├── exposure_delta.pt
│   ├── pose_delta.pt
│   ├── gaussian_geometry_delta.pt
│   ├── refine_config.json
│   ├── loss_history.jsonl
│   └── checkpoints/
├── debug/
│   ├── 00_inputs/
│   ├── 01_flame_tracking/ 或 01_masks_imported/
│   ├── 02_colmap/camera_centers.png
│   ├── 03_alignment/alignment_camera_centers.png
│   ├── 04_preview/
│   └── 05_refine/
└── exports/
    ├── package.zip                 # 导出包
    ├── refined_gaussian.ply
    ├── canonical_flame_param.npz
    ├── transforms_aligned.json     # 已 bake refinement camera/intrinsics delta
    ├── sim3_colmap_to_lam.json     # 已合成 refinement global camera delta
    ├── transforms_aligned.initial.json
    ├── sim3_colmap_to_lam.initial.json
    ├── export_alignment_report.json
    ├── final_review.zip
    └── final_review/
        ├── overlays/
        └── final_review.mp4
```

最重要的产物：

- `refine/refined_gaussian.ply`：多视角优化后的 Gaussian。
- `exports/package.zip`：导出包，包含 refined Gaussian、canonical FLAME、对齐相机、delta、loss 等文件。
- `exports/transforms_aligned.json`：导出时会把优化得到的 camera / intrinsics delta bake 到每一帧。
- `debug/04_preview/`、`debug/05_refine/`、`exports/final_review/`：检查对齐和 refinement 质量的可视化结果。

## 输入

页面第 1 步需要上传两个 ZIP，必须来自同一个人：

1. `Camera Images ZIP`：纯相机拍摄的多视角图片包，只放目标人物的多视角原图，可选包含前景 mask（建议包含一下）。
2. `LAM Canonical Package ZIP`：单图正面 LAM Layer 1 输出包，提供 canonical Gaussian 和 FLAME identity。

`Camera Images ZIP` 结构：

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

mask 推荐用 `.png` 单通道灰度图，白色表示参与优化的头部前景，黑色表示背景。mask 文件名 stem 必须和图片一致，例如 `images/000001.png` 对应 `fg_masks/000001.png`；尺寸不一致时，上传 ZIP 内 mask 会被归一化，外部导入 mask 会按原图尺寸做后处理。

`LAM Canonical Package ZIP` 结构：

```text
layer1_lam.zip
└── layer1_lam/
    ├── xxx_canonical.ply                 # 必需；canonical 空间绝对 xyz Gaussian
    ├── xxx_canonical_flame_param.npz     # 必需；同一个 identity 的 shape/betas
    ├── layer1_metadata.json              # 可选
    └── layer1_reference_transforms.json  # 可选；用于更稳的初始视角/距离估计
```
运行 `app_lam` 时会在 `output/` 下会直接导出。

## 处理流程

UI 6 步执行：

1. `Upload Inputs`
   - 上传 `Camera Images ZIP` 和 `LAM Canonical Package ZIP`。

2. `Masks / FLAME`
   - `Run FLAME Tracking (Keep Uploaded Masks)`：每张图单独做 FLAME tracking，如果上传包里已有同名 mask，会保留原 mask，只做尺寸/灰度归一化。

3. `COLMAP`
   - 对 `data/images/` 跑 COLMAP，得到多视角相机。

4. `Camera Alignment / Calibration`
   - `Initialize Alignment`：用 Layer 1 Gaussian 统计、可选 `layer1_reference_transforms.json` 和 COLMAP 相机中心估计初始 Sim(3)，把 COLMAP 相机对齐到 LAM/FLAME render 坐标系。
   - `Preview Alignment`：渲染 `data/init.ply` 并叠到多视角图上，检查脸部位置、尺度和五官是否对齐。
   - `Calibrate Alignment`：使用 landmarks 做全局 Sim(3) 校准，并 bake 回 `alignment/transforms_aligned.json`。

5. `Refinement`
   - `Run Refine`：依次跑 `camera`、`pose`、`appearance` 三个阶段，主要优化 per-view camera delta、表达/下颌/眼睛小 delta、Gaussian appearance。
   - `Run Geometry`：从 `refine/checkpoints/latest.pt` 续跑 `geometry_light`，小幅优化 Gaussian geometry。
   - `Run Small XYZ Geometry (Optional / Advanced)`：继续从 latest checkpoint 续跑 `geometry_xyz`，允许更直接的小幅 xyz 几何调整，正则更强，属于高级选项，会对五官产生较大改变。

6. `Export / Final Review`
   - `Export Package`：生成 `exports/package.zip`。
   - `Render Final Review`：用 `refine/refined_gaussian.ply` 和 bake 后的导出相机渲染所有视角，生成 overlay 图片、MP4 和 ZIP。

## 其他说明

### 坐标系说明

当前闭环是：**多视角数据 -> COLMAP 相机 -> Layer 1/LAM 坐标对齐 -> LAM/GS 渲染 -> loss 反传 -> refined Gaussian**。

同时接入三份来自不同坐标系的数据：

1. LAM initial Gaussian / FLAME identity
   - 坐标系：LAM 的 FLAME canonical 空间，也是最终 refined Gaussian 所在的模型坐标系。
   - 作用：提供 canonical bound Gaussian 和 identity betas。
   - 要求：`data/init.ply` 必须保存 Gaussian 的真实 canonical xyz，不能只保存 offset。

2. COLMAP camera
   - 坐标系：COLMAP 自己重建出的任意 SfM 坐标系，尺度、朝向、原点都不受 LAM 约束。
   - 作用：提供多视角相机外参/内参。
   - 处理：先写入 `colmap/transforms_colmap_raw.json`，再通过初始 Sim(3) 和 landmark calibration 变成 `alignment/transforms_aligned.json`。

3. Per-view FLAME tracking
   - 坐标系：单图 FLAME tracking 的裁剪图像坐标和单图相机假设，不是 COLMAP 多视角坐标系。
   - 作用：只作为语义观测，提供 mask、landmarks、expr/jaw/eyes 等每视角状态约束。
   - 处理：mask/landmarks 会从 tracking crop 坐标还原到原始多视角图片像素坐标；global rotation/translation 固定 canonicalize 为 0，避免把相机绕人拍摄的视角重复编码进头部姿态。
   - 不作为 Sim(3) target：single-image FLAME tracking 产生的 transforms 不代表真实多视角 camera rig，不能直接拿来和 COLMAP camera center 做 Sim(3)。

### 多视角数据要求

- 多视角图片必须是同一人，尽量同一表情、同一光照、统一尺寸。
- 文件名按帧/视角排序，mask、FLAME 参数、landmark 都按同名 stem 匹配。
- 所有用于 refinement 的图片尺寸必须一致，mask 尺寸必须和对应图片一致。
- `alignment/transforms_aligned.json` 的 `frames` 里需要包含相机内参 `fl_x/fl_y/cx/cy` 和 `transform_matrix`；正常流程通过 `COLMAP` + `Initialize Alignment` + `Calibrate Alignment` 生成。
