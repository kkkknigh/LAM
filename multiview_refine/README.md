# LAM 多视角 Gaussian Refinement

本模块用于在 LAM 单图生成的 canonical avatar 基础上，引入同一人物的多视角观测进行细化。流程以单图 LAM 结果作为强初始化，使用 COLMAP 估计多视角相机，将 COLMAP 相机系统对齐到 LAM canonical 空间，并依次优化相机、表情、外观和局部 Gaussian 几何。

## 核心功能

- 使用多视角 RGB 图像细化 LAM canonical Gaussian。
- 支持外部前景 mask，例如 SAM、rembg 或手工 mask。
- 使用 COLMAP 相机作为多视角几何来源。
- 使用 FLAME tracking 的 landmarks 和表情状态作为语义约束。
- 导出 refined Gaussian、camera delta、pose delta、诊断结果和 review 视频。

## 快速开始

在 LAM 根目录运行 Gradio：

```bash
python app_multiview_gaussian_refine.py
```

默认访问地址：

```text
http://127.0.0.1:7861
```

可以在界面中上传两个 ZIP 包，也可以从本地磁盘导入已有目录。

## 输入数据

### Camera Images ZIP

该压缩包包含目标人物的多视角图像，也可以同时包含可选的前景 mask。

推荐结构：

```text
camera_views.zip
└── camera_views/
    ├── images/
    │   ├── 000001.png
    │   ├── 000002.png
    │   └── ...
    └── fg_masks/        # 可选；也接受 masks/
        ├── 000001.png
        ├── 000002.png
        └── ...
```

注意事项：

- 图像应来自同一人物。
- 尽量保持表情、光照和分辨率一致。
- mask 文件名应和对应图像文件名保持同 stem。
- mask 用作 refinement 阶段的前景约束，默认不会传给 COLMAP。
- 如果不提供 mask，界面可以在 FLAME tracking 阶段自动生成。

### LAM Canonical Package ZIP

该压缩包包含同一人物的 Layer 1 LAM 初始化结果。

推荐结构：

```text
layer1_lam.zip
└── layer1_lam/
    ├── xxx_canonical.ply
    ├── xxx_canonical_flame_param.npz
    └── flame_param/        # 可选
        └── 00000_00.npz
```

其中 canonical PLY 应保存 canonical 空间下的 Gaussian 绝对坐标。只保存 offset 的 PLY，例如部分 OAC/h5 导出包中的 offset 文件，不适合作为默认初始化。

## 处理流程

默认流程如下：

1. **准备工作区**  
   导入多视角图像、可选 mask 和 LAM canonical package。

2. **Masks / FLAME tracking**  
   生成或导入前景 mask，运行单图 FLAME tracking，并将 landmarks 还原到原始图像坐标。

3. **COLMAP 重建**  
   从 RGB 图像中估计多视角相机内参和外参。

4. **相机对齐**  
   根据 LAM canonical 空间和 COLMAP 相机系统初始化全局 Sim(3) 变换，并利用 2D landmarks 进行校准。

5. **对齐预览**  
   渲染 overlay，检查 Gaussian 是否在各视角中基本对齐。

6. **Refinement 优化**  
   按以下阶段依次优化：

   | 阶段 | 优化变量 | 作用 |
   | --- | --- | --- |
   | `camera` | 每视角相机旋转/平移残差 | 修正局部相机投影误差 |
   | `pose` | 每视角 `expression`、`jaw`、`eyes` | 匹配不同视角中的人脸状态差异 |
   | `appearance` | 有界 Gaussian color / SH delta | 改善前景颜色一致性 |
   | `geometry_light` | Gaussian scale / rotation / offset | 轻量局部几何细化 |
   | `geometry_xyz` | Gaussian xyz / offset | 最终小幅几何修正 |

7. **导出与检查**  
   导出 refined package，并渲染 final review overlays/videos。

## 坐标系说明

流程中同时涉及三类坐标来源：

- **LAM canonical 空间**  
  初始 Gaussian 和 canonical FLAME identity 所在的模型空间。最终 refined Gaussian 仍保留在该空间。

- **COLMAP SfM 空间**  
  从多视角图像恢复出的相机系统。该空间的尺度、朝向和原点是任意的。

- **FLAME tracking 图像空间**  
  每张图像独立 tracking 得到的 landmarks 和表情状态。单图 tracking 的相机假设不作为真实多视角 camera rig 使用。

相机对齐阶段会通过全局 Sim(3) 变换和 landmark 校准，将 COLMAP 相机映射到 LAM canonical 空间。

## Mask

前景 mask 可以来自 SAM、rembg、手工编辑或内置 tracking/matting 流程。

mask 语义约定：

- 白色 / 高值：参与优化的前景区域；
- 黑色 / 低值：忽略或降权的背景区域；
- 支持带软边的灰度 mask。

外部 mask 默认会经过最大连通域保留、形态学闭运算和轻微羽化等后处理。它们会在 refinement 中用于 RGB 前景区域选择、mask loss、IoU loss 和边界 loss。

## 输出结果

导出包通常包含：

- refined Gaussian PLY；
- canonical FLAME 参数；
- 对齐后的相机 transforms；
- camera、pose 和 geometry delta 文件；
- loss history 和配置文件；
- final review 资源。

建议通过 final review overlays 和视频检查相机对齐、轮廓质量、前景外观和表情响应。
