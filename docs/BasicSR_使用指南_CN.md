# BasicSR 全面使用指南

[English](BasicSR_Usage_Guide.md) **|** [简体中文](BasicSR_使用指南_CN.md)

本指南提供了使用 BasicSR 框架进行图像复原任务的完整流程，包括超分辨率、去噪和修复等任务的详细配置方法。

## 📋 目录

1. [快速开始](#快速开始)
2. [通用工作流程](#通用工作流程)
3. [环境安装](#环境安装)
4. [数据准备](#数据准备)
5. [超分辨率任务](#超分辨率任务)
6. [去噪任务](#去噪任务)
7. [修复任务](#修复任务)
8. [模型架构选择](#模型架构选择)
9. [训练技巧和最佳实践](#训练技巧和最佳实践)
10. [常见问题解决](#常见问题解决)

## 🚀 快速开始

### 基本要求
- Python 3.7+
- PyTorch 1.7+
- NVIDIA GPU (推荐)

### 一键运行示例
```bash
# 克隆仓库
git clone https://github.com/XPixelGroup/BasicSR.git
cd BasicSR

# 安装依赖
pip install -r requirements.txt
pip install -e .

# 运行超分辨率示例
python basicsr/train.py -opt options/train/SRResNet_SRGAN/train_MSRResNet_x4.yml
```

## 🔄 通用工作流程

所有任务都遵循以下标准流程：

### 1. 环境准备
- 安装 BasicSR 和依赖包
- 配置 GPU 环境
- 下载预训练模型（如需要）

### 2. 数据准备
- 组织数据集目录结构
- 预处理数据（裁剪、格式转换等）
- 创建 LMDB 数据库（可选，用于加速训练）

### 3. 配置文件设置
- 选择合适的配置模板
- 修改数据路径
- 调整网络参数
- 设置训练/测试参数

### 4. 训练模型
```bash
# 单GPU训练
python basicsr/train.py -opt path/to/config.yml

# 多GPU分布式训练
python -m torch.distributed.launch --nproc_per_node=4 basicsr/train.py -opt path/to/config.yml --launcher pytorch
```

### 5. 测试评估
```bash
# 测试模型
python basicsr/test.py -opt path/to/test_config.yml

# 计算指标
python scripts/metrics/calculate_psnr_ssim.py --gt path/to/gt --restored path/to/results
```

## 🛠️ 环境安装

详细安装说明请参考 [INSTALL.md](INSTALL.md)

### 基础安装
```bash
# 创建conda环境
conda create -n basicsr python=3.8
conda activate basicsr

# 安装PyTorch
conda install pytorch torchvision cudatoolkit=11.1 -c pytorch -c nvidia

# 安装BasicSR
pip install basicsr
# 或从源码安装
git clone https://github.com/XPixelGroup/BasicSR.git
cd BasicSR
pip install -e .
```

### 验证安装
```python
import basicsr
print(basicsr.__version__)
```

## 📁 数据准备

详细数据准备指南请参考 [DatasetPreparation_CN.md](DatasetPreparation_CN.md)

### 标准数据目录结构
```
datasets/
├── DIV2K/
│   ├── DIV2K_train_HR/          # 高分辨率训练图像
│   ├── DIV2K_train_LR_bicubic/  # 低分辨率训练图像
│   ├── DIV2K_valid_HR/          # 高分辨率验证图像
│   └── DIV2K_valid_LR_bicubic/  # 低分辨率验证图像
├── Set5/                        # 测试数据集
│   ├── GTmod12/                 # Ground Truth
│   └── LRbicx4/                 # 低分辨率输入
└── other_datasets/
```

### 数据预处理脚本
```bash
# 提取子图像（用于训练）
python scripts/data_preparation/extract_subimages.py

# 创建LMDB数据库
python scripts/data_preparation/create_lmdb.py

# 生成退化数据
python scripts/data_preparation/generate_multiscale_DF2K.py
```

## 🔍 超分辨率任务

超分辨率是 BasicSR 的核心功能，支持多种架构和放大倍数。

### 支持的架构
- **SRResNet/MSRResNet**: 经典残差网络，适合初学者
- **EDSR**: 增强深度超分辨率网络
- **RCAN**: 残差通道注意力网络
- **SwinIR**: 基于Swin Transformer的网络
- **ESRGAN**: 生成对抗网络，适合真实图像
- **Real-ESRGAN**: 面向真实场景的增强版本

### 配置示例

#### 1. SRResNet (推荐入门)
```yaml
# 基本配置
name: train_MSRResNet_x4
model_type: SRModel
scale: 4
num_gpu: 1

# 数据集配置
datasets:
  train:
    name: DIV2K
    type: PairedImageDataset
    dataroot_gt: datasets/DIV2K/DIV2K_train_HR_sub
    dataroot_lq: datasets/DIV2K/DIV2K_train_LR_bicubic_X4_sub
    gt_size: 128
    use_hflip: true
    use_rot: true
    batch_size_per_gpu: 16

# 网络结构
network_g:
  type: MSRResNet
  num_in_ch: 3
  num_out_ch: 3
  num_feat: 64
  num_block: 16
  upscale: 4

# 训练设置
train:
  optim_g:
    type: Adam
    lr: !!float 2e-4
  scheduler:
    type: CosineAnnealingRestartLR
    periods: [250000, 250000, 250000, 250000]
    restart_weights: [1, 1, 1, 1]
  total_iter: 1000000
  pixel_opt:
    type: L1Loss
    loss_weight: 1.0
```

#### 2. EDSR (高性能)
```yaml
network_g:
  type: EDSR
  num_in_ch: 3
  num_out_ch: 3
  num_feat: 64
  num_block: 16
  upscale: 4
  res_scale: 1
  img_range: 255.
  rgb_mean: [0.4488, 0.4371, 0.4040]
```

#### 3. SwinIR (最新架构)
```yaml
network_g:
  type: SwinIR
  upscale: 4
  in_chans: 3
  img_size: 48
  window_size: 8
  img_range: 1.
  depths: [6, 6, 6, 6, 6, 6]
  embed_dim: 180
  num_heads: [6, 6, 6, 6, 6, 6]
  mlp_ratio: 2
  upsampler: 'pixelshuffle'
  resi_connection: '1conv'
```

### 训练命令
```bash
# 单GPU训练
CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/SRResNet_SRGAN/train_MSRResNet_x4.yml

# 多GPU训练 (4张卡)
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/train/SRResNet_SRGAN/train_MSRResNet_x4.yml --launcher pytorch

# 自动恢复训练
python basicsr/train.py -opt options/train/SRResNet_SRGAN/train_MSRResNet_x4.yml --auto_resume
```

### 测试和推理
```bash
# 测试模型性能
python basicsr/test.py -opt options/test/SRResNet_SRGAN/test_MSRResNet_x4.yml

# 单张图像推理
python inference/inference_esrgan.py --input path/to/input --output path/to/output --model_path path/to/model.pth
```

## 🔇 去噪任务

去噪任务主要移除图像中的噪声，恢复清晰图像。

### 支持的架构
- **RIDNet**: 真实图像去噪网络
- **SwinIR**: 支持灰度和彩色图像去噪
- **DnCNN**: 经典去噪网络（可自行实现）

### 配置示例

#### 1. RIDNet 配置
```yaml
name: train_RIDNet_noise25
model_type: SRModel  # 复用SR模型框架
scale: 1  # 去噪任务不改变分辨率
num_gpu: 1

datasets:
  train:
    name: NoiseDataset
    type: PairedImageDataset
    dataroot_gt: datasets/denoise/train/GT
    dataroot_lq: datasets/denoise/train/Noisy
    gt_size: 128
    use_hflip: true
    use_rot: true
    batch_size_per_gpu: 16

network_g:
  type: RIDNet
  num_in_ch: 3
  num_feat: 64
  num_out_ch: 3

train:
  optim_g:
    type: Adam
    lr: !!float 1e-4
  pixel_opt:
    type: L1Loss
    loss_weight: 1.0
```

#### 2. SwinIR 去噪配置
```yaml
network_g:
  type: SwinIR
  upscale: 1
  in_chans: 3
  img_size: 128
  window_size: 8
  img_range: 1.
  depths: [6, 6, 6, 6, 6, 6]
  embed_dim: 180
  num_heads: [6, 6, 6, 6, 6, 6]
  mlp_ratio: 2
  upsampler: ''  # 去噪任务不需要上采样
  resi_connection: '1conv'
```

### 噪声数据准备
```python
# 生成合成噪声数据
import numpy as np
import cv2

def add_noise(img, noise_level=25):
    """添加高斯噪声"""
    noise = np.random.normal(0, noise_level/255.0, img.shape)
    noisy_img = img + noise
    return np.clip(noisy_img, 0, 1)

# 批量处理
import glob
clean_imgs = glob.glob('datasets/clean/*.png')
for img_path in clean_imgs:
    img = cv2.imread(img_path).astype(np.float32) / 255.0
    noisy_img = add_noise(img, 25)
    save_path = img_path.replace('clean', 'noisy')
    cv2.imwrite(save_path, (noisy_img * 255).astype(np.uint8))
```

### 去噪推理
```bash
# 使用RIDNet进行去噪
python inference/inference_ridnet.py --test_path datasets/denoise/test --noise_g 25 --model_path experiments/pretrained_models/RIDNet/RIDNet.pth

# 使用SwinIR进行去噪
python inference/inference_swinir.py --task color_dn --noise 25 --input datasets/denoise/test --output results/swinir_denoise --model_path path/to/swinir_denoise_model.pth
```

## 🖼️ 修复任务

修复任务包括图像修复、JPEG压缩伪影去除等。

### 支持的架构
- **SwinIR**: 支持JPEG压缩伪影去除
- **DFDNet**: 人脸修复专用网络
- **自定义架构**: 可基于现有网络修改

### JPEG压缩伪影去除

#### 配置示例
```yaml
name: train_SwinIR_JPEG_CAR
model_type: SwinIRModel
scale: 1
num_gpu: 1

datasets:
  train:
    name: JPEG_Dataset
    type: PairedImageDataset
    dataroot_gt: datasets/jpeg_car/train/GT
    dataroot_lq: datasets/jpeg_car/train/Compressed
    gt_size: 128
    use_hflip: true
    use_rot: true

network_g:
  type: SwinIR
  upscale: 1
  in_chans: 3
  img_size: 64
  window_size: 7  # JPEG任务使用7
  img_range: 1.
  depths: [6, 6, 6, 6, 6, 6]
  embed_dim: 180
  num_heads: [6, 6, 6, 6, 6, 6]
  mlp_ratio: 2
  upsampler: ''
  resi_connection: '1conv'
```

#### 数据准备
```python
# 生成JPEG压缩数据
import cv2
from PIL import Image

def compress_jpeg(img_path, quality=10):
    """生成JPEG压缩图像"""
    img = Image.open(img_path)
    compressed_path = img_path.replace('.png', f'_q{quality}.jpg')
    img.save(compressed_path, 'JPEG', quality=quality)
    return compressed_path

# 批量处理
import glob
clean_imgs = glob.glob('datasets/clean/*.png')
for img_path in clean_imgs:
    compressed_path = compress_jpeg(img_path, quality=10)
    print(f"Compressed: {compressed_path}")
```

### 人脸修复 (DFDNet)

#### 配置要求
```bash
# 安装dlib (人脸检测依赖)
pip install dlib

# 下载预训练模型
python scripts/download_pretrained_models.py DFDNet
```

#### 推理示例
```bash
# 人脸修复推理
python inference/inference_dfdnet.py --upscale_factor=2 --test_path datasets/faces
```

## 🏗️ 模型架构选择

### 按任务类型选择

| 任务类型 | 推荐架构 | 特点 | 适用场景 |
|---------|---------|------|---------|
| 超分辨率 | SRResNet | 简单快速 | 入门学习、快速原型 |
|         | EDSR | 性能优秀 | 追求高质量结果 |
|         | SwinIR | 最新技术 | 研究、最佳性能 |
|         | ESRGAN | 生成对抗 | 真实图像、视觉效果 |
| 去噪 | RIDNet | 专门设计 | 真实图像去噪 |
|      | SwinIR | 通用性强 | 多种噪声类型 |
| 修复 | SwinIR | 多任务支持 | JPEG、修复等 |
|     | DFDNet | 专门人脸 | 人脸修复专用 |

### 按计算资源选择

| GPU显存 | 推荐架构 | 批大小 | 图像尺寸 |
|---------|---------|-------|----------|
| 8GB | SRResNet | 16 | 128x128 |
| 16GB | EDSR/RCAN | 16 | 192x192 |
| 24GB+ | SwinIR | 8 | 256x256 |

### 按数据集规模选择

- **小数据集 (<1K图像)**: 使用预训练模型微调
- **中等数据集 (1K-10K)**: SRResNet或EDSR从头训练
- **大数据集 (>10K)**: 任意架构，推荐SwinIR或ESRGAN

## 💡 训练技巧和最佳实践

### 数据增强
```yaml
# 在配置文件中启用数据增强
datasets:
  train:
    use_hflip: true      # 水平翻转
    use_rot: true        # 旋转
    use_shuffle: true    # 随机洗牌
    color_jitter: true   # 颜色抖动（自定义）
```

### 学习率调度
```yaml
# 余弦退火重启
scheduler:
  type: CosineAnnealingRestartLR
  periods: [250000, 250000, 250000, 250000]
  restart_weights: [1, 1, 1, 1]
  eta_min: !!float 1e-7

# 多步长衰减
scheduler:
  type: MultiStepLR
  milestones: [100000, 200000, 300000, 400000]
  gamma: 0.5
```

### 损失函数选择
```yaml
# L1损失 (常用)
pixel_opt:
  type: L1Loss
  loss_weight: 1.0

# 感知损失 (更好视觉效果)
perceptual_opt:
  type: PerceptualLoss
  layer_weights:
    'conv1_2': 0.1
    'conv2_2': 0.1
    'conv3_4': 1
    'conv4_4': 1
    'conv5_4': 1
  vgg_type: vgg19
  use_input_norm: true

# 组合损失
pixel_opt:
  type: L1Loss
  loss_weight: 1.0
perceptual_opt:
  type: PerceptualLoss
  loss_weight: 0.1
```

### 训练监控
```yaml
# 验证设置
val:
  val_freq: !!float 5e3  # 每5000次迭代验证一次
  save_img: true         # 保存验证图像
  
  metrics:
    psnr:
      type: calculate_psnr
      crop_border: 4
      test_y_channel: false
    ssim:
      type: calculate_ssim
      crop_border: 4
      test_y_channel: false

# 日志设置
logger:
  print_freq: 100
  save_checkpoint_freq: !!float 5e3
  use_tb_logger: true
  wandb:
    project: my_project_name
    resume_id: ~
```

### 内存优化
```yaml
# 减少批大小
datasets:
  train:
    batch_size_per_gpu: 8  # 从16减少到8

# 使用梯度累积
train:
  accumulate_grad_batches: 2  # 累积2个batch的梯度

# 启用混合精度训练
train:
  use_amp: true  # 自动混合精度
```

## ❓ 常见问题解决

### 1. 内存不足错误
```bash
# 错误: CUDA out of memory
```
**解决方案**:
- 减小 `batch_size_per_gpu`
- 减小 `gt_size` (训练patch大小)
- 使用更小的网络架构
- 启用梯度检查点 (gradient checkpointing)

### 2. 训练不收敛
**可能原因和解决方案**:
- 学习率过大: 降低学习率
- 数据预处理问题: 检查数据归一化
- 网络架构不适合: 尝试其他架构
- 数据集质量差: 检查数据标注

### 3. 验证PSNR不提升
**解决方案**:
- 检查验证数据集路径
- 确认损失函数适合任务
- 调整学习率调度策略
- 增加训练迭代次数

### 4. 分布式训练问题
```bash
# 错误: RuntimeError: Address already in use
```
**解决方案**:
```bash
# 更改端口号
python -m torch.distributed.launch --master_port=4322 ...

# 或设置环境变量
export MASTER_PORT=4322
```

### 5. 模型加载错误
```bash
# 错误: Key mismatch when loading state dict
```
**解决方案**:
```yaml
# 在配置文件中设置
path:
  strict_load_g: false  # 允许部分加载
  param_key_g: params_ema  # 或使用EMA参数
```

## 📚 相关文档

- [安装指南](INSTALL.md)
- [数据准备详细说明](DatasetPreparation_CN.md)
- [训练测试命令](TrainTest_CN.md)
- [配置文件说明](Config.md)
- [模型库](ModelZoo_CN.md)
- [评估指标](Metrics_CN.md)
- [HOWTOs指南](HOWTOs_CN.md)

## 🤝 社区支持

- **技术交流QQ群**: 320960100 (答案: 互帮互助共同进步)
- **GitHub Issues**: [提交问题](https://github.com/XPixelGroup/BasicSR/issues)
- **讨论区**: [GitHub Discussions](https://github.com/XPixelGroup/BasicSR/discussions)

## 📄 许可证

本项目使用 Apache 2.0 许可证。详见 [LICENSE](../LICENSE.txt)。

---

如果这个指南对您有帮助，欢迎给项目点个 ⭐ Star！