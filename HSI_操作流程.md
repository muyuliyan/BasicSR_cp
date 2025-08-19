# HSI 去噪和修复操作流程

本文档详细说明如何使用BasicSR框架进行HSI（高光谱图像）去噪和修复任务的操作流程。

## 📋 环境要求

1. Python 3.7+
2. PyTorch
3. NumPy, SciPy
4. HSI数据文件（.mat 或 .npy 格式）

## 🚀 操作流程

### 步骤1: 数据准备

根据任务类型组织数据：

#### 去噪任务数据结构：
```
datasets/your_hsi_dataset/
├── clean/       # 干净的HSI图像 (.mat 或 .npy)
├── noisy/       # 噪声图像 (可选，可运行时生成)
└── val/
    └── clean/   # 验证用干净图像
```

#### 修复任务数据结构：
```
datasets/your_hsi_dataset/
├── complete/    # 完整的HSI图像 (.mat 或 .npy)
├── masks/       # 掩码图像 (可选，可运行时生成)
└── val/
    └── complete/ # 验证用完整图像
```

#### 使用数据准备脚本：

**去噪数据准备：**
```bash
python scripts/hsi_data_preparation.py \
    --input_dir datasets/original_hsi \
    --output_dir datasets/hsi_denoising \
    --task denoising \
    --noise_type gaussian \
    --noise_levels 10 25 50
```

**修复数据准备：**
```bash
python scripts/hsi_data_preparation.py \
    --input_dir datasets/original_hsi \
    --output_dir datasets/hsi_inpainting \
    --task inpainting \
    --mask_type random_rect \
    --mask_ratios 0.1 0.2 0.3
```

### 步骤2: 配置文件设置

#### 去噪任务配置 (`options/train/HSI/train_HSI_Denoising_SRResNet.yml`)：

```yaml
# 设置数据路径
datasets:
  train:
    dataroot_gt: datasets/hsi_denoising/clean
    add_noise_to_gt: true          # 运行时生成噪声
    noise_type: gaussian           # gaussian, poisson, mixed
    noise_range: [5, 50]           # 噪声等级范围

# 设置光谱通道数
network_g:
  num_in_ch: 31    # 替换为你的HSI波段数
  num_out_ch: 31   # 应与 num_in_ch 相同
  upscale: 1       # 去噪任务不需要上采样

# 调整显存使用
datasets:
  train:
    gt_size: 64            # 如果显存不足可减小
    batch_size_per_gpu: 8  # 如果显存不足可减小
```

#### 修复任务配置 (`options/train/HSI/train_HSI_Inpainting_SRResNet.yml`)：

```yaml
# 设置数据路径
datasets:
  train:
    dataroot_gt: datasets/hsi_inpainting/complete
    generate_mask: true         # 运行时生成掩码
    mask_type: random_rect      # random_rect, random_irregular
    mask_ratio: [0.1, 0.3]      # 10-30%的图像区域

# 设置光谱通道数
network_g:
  num_in_ch: 31    # 替换为你的HSI波段数
  num_out_ch: 31   # 应与 num_in_ch 相同
  upscale: 1       # 修复任务不需要上采样
```

### 步骤3: 训练模型

#### 去噪任务训练：
```bash
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python basicsr/train.py \
    -opt options/train/HSI/train_HSI_Denoising_SRResNet.yml \
    --auto_resume
```

#### 修复任务训练：
```bash
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python basicsr/train.py \
    -opt options/train/HSI/train_HSI_Inpainting_SRResNet.yml \
    --auto_resume
```

### 步骤4: 测试模型

#### 去噪任务测试：

1. 更新测试配置 (`options/test/HSI/test_HSI_Denoising_SRResNet.yml`)：
   ```yaml
   # 设置测试数据路径
   datasets:
     test_1:
       dataroot_gt: datasets/hsi_denoising/test/clean
       dataroot_lq: datasets/hsi_denoising/test/noisy  # 或留空使用生成的噪声
   
   # 设置模型路径
   path:
     pretrain_network_g: experiments/你的实验名称/models/net_g_latest.pth
   ```

2. 运行测试：
   ```bash
   PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python basicsr/test.py \
       -opt options/test/HSI/test_HSI_Denoising_SRResNet.yml
   ```

#### 修复任务测试：

1. 更新测试配置 (`options/test/HSI/test_HSI_Inpainting_SRResNet.yml`)：
   ```yaml
   # 设置测试数据路径
   datasets:
     test_1:
       dataroot_gt: datasets/hsi_inpainting/test/complete
       dataroot_mask: datasets/hsi_inpainting/test/masks  # 或留空使用生成的掩码
   
   # 设置模型路径
   path:
     pretrain_network_g: experiments/你的实验名称/models/net_g_latest.pth
   ```

2. 运行测试：
   ```bash
   PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python basicsr/test.py \
       -opt options/test/HSI/test_HSI_Inpainting_SRResNet.yml
   ```

## ⚙️ 参数调整建议

### 去噪任务参数：

1. **噪声类型选择**：
   - `gaussian`: 适合处理高斯噪声
   - `poisson`: 适合处理泊松噪声（光子噪声）
   - `mixed`: 混合噪声，提高泛化能力

2. **噪声等级设置**：
   - 训练时使用范围：`[5, 50]` 增加多样性
   - 测试时使用固定值：`[25, 25]` 便于比较

### 修复任务参数：

1. **掩码类型选择**：
   - `random_rect`: 随机矩形掩码，计算简单
   - `random_irregular`: 不规则掩码，更接近真实场景

2. **掩码比例设置**：
   - 训练时使用范围：`[0.1, 0.3]` 增加多样性
   - 测试时使用固定值：`[0.2, 0.2]` 便于比较

## 📊 评估指标

系统提供5个关键的HSI评估指标：

| 指标 | 描述 | 期望值 |
|------|------|--------|
| **PSNR** | 峰值信噪比 | 越高越好 ↑ |
| **SSIM** | 结构相似性指数 | 越高越好 ↑ |
| **SAM** | 光谱角映射器 | 越低越好 ↓ |
| **ERGAS** | 全局相对误差 | 越低越好 ↓ |
| **RMSE** | 均方根误差 | 越低越好 ↓ |

### 预期结果范围：

#### 去噪任务：
- **PSNR**: 30-45 dB（取决于噪声等级）
- **SSIM**: 0.85-0.98
- **SAM**: 0.05-0.2 弧度
- **RMSE**: 2-10（取决于噪声等级）

#### 修复任务：
- **PSNR**: 20-35 dB（取决于掩码比例）
- **SSIM**: 0.75-0.95
- **SAM**: 0.1-0.4 弧度
- **RMSE**: 5-20（取决于掩码比例）

## 🔧 常见问题解决

### 显存不足：
```yaml
datasets:
  train:
    gt_size: 32            # 减小补丁大小
    batch_size_per_gpu: 4  # 减小批次大小
    num_worker_per_gpu: 2  # 减少工作进程
```

### 训练不稳定：
- 去噪：尝试混合噪声训练提高泛化能力
- 修复：降低掩码比例，使用渐进训练策略

### 效果不佳：
- 检查数据预处理是否正确
- 确认光谱通道数设置正确
- 验证数据归一化到[0,1]范围

## 📝 使用示例

### 完整的去噪流程示例：

```bash
# 1. 准备数据
python scripts/hsi_data_preparation.py \
    --input_dir datasets/CAVE_dataset \
    --output_dir datasets/CAVE_denoising \
    --task denoising \
    --noise_type gaussian \
    --noise_levels 15 25 35

# 2. 修改配置文件中的路径和通道数（31通道CAVE数据集）
# 编辑 options/train/HSI/train_HSI_Denoising_SRResNet.yml

# 3. 开始训练
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python basicsr/train.py \
    -opt options/train/HSI/train_HSI_Denoising_SRResNet.yml \
    --auto_resume

# 4. 训练完成后进行测试
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python basicsr/test.py \
    -opt options/test/HSI/test_HSI_Denoising_SRResNet.yml
```

### 完整的修复流程示例：

```bash
# 1. 准备数据
python scripts/hsi_data_preparation.py \
    --input_dir datasets/CAVE_dataset \
    --output_dir datasets/CAVE_inpainting \
    --task inpainting \
    --mask_type random_irregular \
    --mask_ratios 0.15 0.25 0.35

# 2. 修改配置文件
# 编辑 options/train/HSI/train_HSI_Inpainting_SRResNet.yml

# 3. 开始训练
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python basicsr/train.py \
    -opt options/train/HSI/train_HSI_Inpainting_SRResNet.yml \
    --auto_resume

# 4. 测试模型
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python basicsr/test.py \
    -opt options/test/HSI/test_HSI_Inpainting_SRResNet.yml
```

## 🎯 总结

通过以上流程，你可以成功使用BasicSR框架进行HSI去噪和修复任务。系统支持：

1. **三种主要任务**：超分辨率、去噪、修复
2. **灵活的数据处理**：支持运行时生成噪声和掩码
3. **完整的评估体系**：提供HSI专用评估指标
4. **易于配置**：提供完整的配置模板和文档

祝你的HSI处理实验顺利！🚀