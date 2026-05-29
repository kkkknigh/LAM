```
$env:DISTUTILS_USE_SDK="1"
$env:PYTORCH3D_NO_NINJA="1"
$env:FORCE_CUDA="1"
$env:MAX_JOBS="4"
$env:DISTUTILS_USE_SDK="1"
$env:PYTORCH3D_NO_NINJA="1"
$env:FORCE_CUDA="1"
$env:MAX_JOBS="4"

cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" && set' |
  ForEach-Object {
    if ($_ -match "^(.*?)=(.*)$") {
      Set-Item -Path "Env:$($matches[1])" -Value $matches[2]
    }
  }

$env:HTTP_PROXY="http://127.0.0.1:7897"
$env:HTTPS_PROXY="http://127.0.0.1:7897"


****A. 数据准备器
读取多视角图、mask、COLMAP camera、FLAME params。

B. 相机/坐标对齐
把 COLMAP camera 对齐到 FLAME/LAM 渲染坐标系。

C. 可微渲染优化循环
每轮选几个视角：
  cano_gs + 当前视角 FLAME pose/expr + camera
  -> renderer.forward_animate_gs()
  -> 得到 render_rgb/render_mask
  -> 和真实图算 loss
  -> 反传更新 Gaussian 参数

D. 参数冻结/解冻策略
先只优化 camera delta，再优化 Gaussian appearance，最后小幅 offset。

E. 导出
保存 refined Gaussian，然后继续复用原来的实时表情/视频渲染。

## FLAME 模型
~~~
shape / betas      # 身份脸型，比如脸宽、鼻梁、下巴
expr               # 表情，比如笑、皱眉、鼓嘴
jaw_pose           # 下巴/张嘴
eyes_pose          # 眼睛姿态
neck_pose          # 脖子姿态
rotation           # 头部整体旋转
translation        # 头部整体平移
camera             # 相机参数
~~~


## 模块

### Layer 1: Avatar Initialization
输入一张正面人脸图，进行 LAM 前向，后续用到的主要的输出是：
- canonical bound Gaussian: 一组绑定在FLAME canonical空间上的3D Gaussian，每个Gaussian带有位置 xyz、局部修正 offset、颜色/SH shs、透明度 opacity、尺度 scaling、旋转 rotation 参数。
- FLAME identity paras: shape/betas, FLAME 的身份形状参数，表征人脸长什么样的低维参数，来自FLAME tracking
- neutral/base flame params：输出的 FLAME 头像参数对应的的姿态参数集合
  
### Layer 2: Multi-view Refinement
输入：
- initial Gaussian：来自步骤 1 的输出
- 多视角图片：同一人脸
- COLMAP camera：colmap 重建多视角人脸得到的相机参数
- 每张图的 mask / landmarks / FLAME tracking：考虑到输入图片可能不是严格同一表情动作
输出：
- refined Gaussian：多视角迭代优化后的 Gaussian，核心产物
- refined per-view camera delta / pose delta：gaussian 优化的附带产物，一般是联合优化的，避免colmap得到的相机参数不太准确
- refined / fixed FLAME identity betas：FLAME 基础形态参数
- refined base params：如果允许 FLAME 基础参数的小幅优化
- optional densified Gaussians：如果允许gaussian裂变复制

#### 数据预处理
1. 图片拍摄
   - 同一人
   - 尽量同一表情 + 同一光照
   - 统一尺寸
   - 文件名按帧/视角排序
2. 前景 mask：可以配合抠图使用
3. colmap 基础重建：获得每张图对应的相机外参+内参，提供相机几何，但没有标准坐标系约束
4. FLAME tracking：获得每张图的人脸状态参数和语义约束
5. 坐标系对齐：求Sim(3)全局优化，把 COLMAP/FLAME tracking 校准到 LAM/FLAME render 坐标系

#### 第一版优化
> 原则上，固定 FLAME identity shape 和 FLAME-Gaussian 绑定结构，优化 canonical bound Gaussian 的 appearance 和小幅 geometry correction。

主要实现闭环：多视角数据 -> 坐标对齐 -> LAM/GS 渲染 -> loss 反传 -> refined Gaussian

0. 初始化：加载 initial canonical Gaussian、FLAME betas、per-view FLAME params、aligned cameras、RGB/mask。
1. 对齐验证：渲染所有视角，检查 mask/landmark/五官大位置是否对齐。
2. 几何校准：冻结 Gaussian，只优化 global alignment 或 per-view camera/head pose 小 delta。
3. Appearance refinement：固定 FLAME identity 和 Gaussian xyz/offset，优化 shs/opacity/scaling。
4. Geometry refinement：固定 FLAME identity 和绑定结构，小幅优化 Gaussian xyz/offset，并用 offset/scale/smooth 正则约束。
5. 可选表情微调：仅当眼睛/嘴巴状态明显不对时，小幅优化 expr/jaw/eyes delta。

#### 第二版优化
> 允许在优化过程中复制/裂变部分 Gaussian，仍然挂在原先面部网格点位上，预期能够提高局部精度

### Layer 3: Expression/Motion Control
把文本控制信号转换成一段 FLAME 非身份参数序列，用于驱动 Layer 2 得到的 refined Gaussian 进行面部表情和动作变化。

输入：
- avatar non-identity FLAME params：来自 Layer 1/2 的头像非身份参数部分（控制表情动作，通常是上一时刻/初始状态的，防止变动太大穿模等）
- control signal：文本
- optional motion settings：fps、duration、强度、是否允许头动、是否允许眼动等硬性动作设置
- optional constraints：表情幅度限制、jaw/eyes 平滑约束、是否保持头部固定等出于稳定性的动作限制

输出：
- FLAME motion sequence：
  - expr[t]
  - jaw_pose[t]
  - eyes_pose[t]
  - neck_pose[t]
  - rotation[t] / translation[t] 可选
- metadata：
  - fps
  - num_frames
  - coordinate convention
  - base params reference

### Layer 4: Realtime Rendering

输入：
- refined canonical bound Gaussian：Layer 2 输出的 GaussianModel
- FLAME identity betas：固定的头像身份参数
- 当前帧 non-identity FLAME params：Layer 3 得到的 expr / jaw_pose / eyes_pose / neck_pose / rotation / translation
- camera：当前渲染视角的 c2w + intrinsics（相机参数另一种表示）
- render settings：分辨率、背景色、fps、是否输出 alpha/depth 等配置信息