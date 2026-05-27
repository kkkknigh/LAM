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
