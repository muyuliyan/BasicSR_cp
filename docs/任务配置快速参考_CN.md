# BasicSR 任务配置快速参考

本文档提供 BasicSR 各种任务配置的快速参考模板，方便用户快速上手。

## 🎯 超分辨率任务配置

### 基础 SRResNet 配置模板

```yaml
# 基本设置
name: your_experiment_name
model_type: SRModel
scale: 4  # 放大倍数: 2, 3, 4, 8
num_gpu: 1
manual_seed: 0

# 数据集设置
datasets:
  train:
    name: YourDataset
    type: PairedImageDataset
    dataroot_gt: datasets/your_dataset/HR  # 高分辨率图像路径
    dataroot_lq: datasets/your_dataset/LR  # 低分辨率图像路径
    filename_tmpl: '{}'
    io_backend:
      type: disk
    
    # 训练参数
    gt_size: 128  # 训练patch大小
    use_hflip: true
    use_rot: true
    num_worker_per_gpu: 6
    batch_size_per_gpu: 16
    dataset_enlarge_ratio: 100
    
  val:
    name: Set5
    type: PairedImageDataset
    dataroot_gt: datasets/Set5/GTmod12
    dataroot_lq: datasets/Set5/LRbicx4
    io_backend:
      type: disk

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
  ema_decay: 0.999
  optim_g:
    type: Adam
    lr: !!float 2e-4
    weight_decay: 0
    betas: [0.9, 0.99]
  
  scheduler:
    type: CosineAnnealingRestartLR
    periods: [250000, 250000, 250000, 250000]
    restart_weights: [1, 1, 1, 1]
    eta_min: !!float 1e-7
  
  total_iter: 1000000
  warmup_iter: -1
  
  # 损失函数
  pixel_opt:
    type: L1Loss
    loss_weight: 1.0
    reduction: mean

# 验证设置
val:
  val_freq: !!float 5e3
  save_img: false
  metrics:
    psnr:
      type: calculate_psnr
      crop_border: 4
      test_y_channel: false

# 日志设置
logger:
  print_freq: 100
  save_checkpoint_freq: !!float 5e3
  use_tb_logger: true

# 分布式训练设置
dist_params:
  backend: nccl
  port: 29500
```

### EDSR 高性能配置

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

# 训练参数调整
datasets:
  train:
    gt_size: 192  # EDSR通常使用更大的patch
    batch_size_per_gpu: 8  # 减少batch size
```

### SwinIR 最新架构配置

```yaml
model_type: SwinIRModel  # 使用专门的SwinIR模型

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

# SwinIR特定训练参数
datasets:
  train:
    gt_size: 192
    batch_size_per_gpu: 4  # SwinIR需要更多显存
    
train:
  scheduler:
    type: MultiStepLR
    milestones: [250000, 400000, 450000, 475000]
    gamma: 0.5
  total_iter: 500000
```

## 🔇 去噪任务配置

### RIDNet 去噪配置

```yaml
# 基本设置
name: RIDNet_denoise_sigma25
model_type: SRModel  # 复用SR模型框架
scale: 1  # 去噪不改变分辨率
num_gpu: 1

# 数据集设置
datasets:
  train:
    name: DenoiseDataset
    type: PairedImageDataset
    dataroot_gt: datasets/denoise/train/GT      # 干净图像
    dataroot_lq: datasets/denoise/train/Noisy   # 含噪图像
    filename_tmpl: '{}'
    io_backend:
      type: disk
    
    gt_size: 128
    use_hflip: true
    use_rot: true
    batch_size_per_gpu: 16
    
  val:
    name: Set12
    type: PairedImageDataset
    dataroot_gt: datasets/denoise/Set12/GT
    dataroot_lq: datasets/denoise/Set12/Noisy
    io_backend:
      type: disk

# 网络结构
network_g:
  type: RIDNet
  num_in_ch: 3
  num_feat: 64
  num_out_ch: 3

# 训练设置
train:
  optim_g:
    type: Adam
    lr: !!float 1e-4  # 去噪通常使用较小学习率
    weight_decay: 0
    betas: [0.9, 0.999]
  
  scheduler:
    type: MultiStepLR
    milestones: [50000, 100000, 150000, 200000]
    gamma: 0.5
  
  total_iter: 300000
  
  pixel_opt:
    type: L1Loss
    loss_weight: 1.0
```

### SwinIR 去噪配置

```yaml
model_type: SwinIRModel

network_g:
  type: SwinIR
  upscale: 1  # 去噪不放大
  in_chans: 3
  img_size: 128
  window_size: 8
  img_range: 1.
  depths: [6, 6, 6, 6, 6, 6]
  embed_dim: 180
  num_heads: [6, 6, 6, 6, 6, 6]
  mlp_ratio: 2
  upsampler: ''  # 去噪不需要上采样器
  resi_connection: '1conv'

# 去噪特定参数
datasets:
  train:
    gt_size: 128
    batch_size_per_gpu: 8
```

## 🖼️ 修复任务配置

### JPEG 压缩伪影去除

```yaml
# 基本设置
name: SwinIR_JPEG_CAR_quality10
model_type: SwinIRModel
scale: 1
num_gpu: 1

# 数据集设置
datasets:
  train:
    name: JPEG_CAR_Dataset
    type: PairedImageDataset
    dataroot_gt: datasets/jpeg_car/train/GT         # 原始图像
    dataroot_lq: datasets/jpeg_car/train/Compressed # JPEG压缩图像
    filename_tmpl: '{}'
    io_backend:
      type: disk
    
    gt_size: 96
    use_hflip: true
    use_rot: true
    batch_size_per_gpu: 16

# 网络结构 (JPEG任务专用参数)
network_g:
  type: SwinIR
  upscale: 1
  in_chans: 3
  img_size: 64
  window_size: 7  # JPEG任务使用7而不是8
  img_range: 1.
  depths: [6, 6, 6, 6, 6, 6]
  embed_dim: 180
  num_heads: [6, 6, 6, 6, 6, 6]
  mlp_ratio: 2
  upsampler: ''
  resi_connection: '1conv'

# 训练设置
train:
  optim_g:
    type: Adam
    lr: !!float 2e-4
  
  scheduler:
    type: CosineAnnealingRestartLR
    periods: [92000]
    restart_weights: [1]
    eta_min: !!float 1e-7
  
  total_iter: 92000
  
  pixel_opt:
    type: L1Loss
    loss_weight: 1.0
```

## 📊 测试配置模板

### 通用测试配置

```yaml
# 基本设置
name: test_experiment_name
model_type: SRModel  # 或 SwinIRModel
scale: 4  # 根据任务调整
num_gpu: 1

# 测试数据集
datasets:
  test_1:
    name: Set5
    type: PairedImageDataset
    dataroot_gt: datasets/Set5/GTmod12
    dataroot_lq: datasets/Set5/LRbicx4
    io_backend:
      type: disk

  test_2:
    name: Set14
    type: PairedImageDataset
    dataroot_gt: datasets/Set14/GTmod12
    dataroot_lq: datasets/Set14/LRbicx4
    io_backend:
      type: disk

# 网络结构 (与训练时保持一致)
network_g:
  type: MSRResNet
  num_in_ch: 3
  num_out_ch: 3
  num_feat: 64
  num_block: 16
  upscale: 4

# 模型路径
path:
  pretrain_network_g: experiments/pretrained_models/your_model.pth
  strict_load_g: true

# 验证设置
val:
  save_img: true
  suffix: ~  # 输出文件后缀
  
  metrics:
    psnr:
      type: calculate_psnr
      crop_border: 4
      test_y_channel: false
      better: higher
    ssim:
      type: calculate_ssim
      crop_border: 4
      test_y_channel: false
      better: higher
```

## 🚀 不同显存配置建议

### 8GB 显存配置
```yaml
datasets:
  train:
    gt_size: 64
    batch_size_per_gpu: 8
    num_worker_per_gpu: 4

network_g:
  type: MSRResNet  # 使用较小网络
  num_feat: 64
  num_block: 16
```

### 16GB 显存配置
```yaml
datasets:
  train:
    gt_size: 128
    batch_size_per_gpu: 16
    num_worker_per_gpu: 6

network_g:
  type: EDSR  # 可使用中等大小网络
  num_feat: 64
  num_block: 16
```

### 24GB+ 显存配置
```yaml
datasets:
  train:
    gt_size: 192
    batch_size_per_gpu: 4  # SwinIR batch较小但patch较大
    num_worker_per_gpu: 8

network_g:
  type: SwinIR  # 可使用大型网络
  embed_dim: 180
  depths: [6, 6, 6, 6, 6, 6]
```

## 🎛️ 常用损失函数配置

### L1 损失
```yaml
train:
  pixel_opt:
    type: L1Loss
    loss_weight: 1.0
    reduction: mean
```

### L2 损失
```yaml
train:
  pixel_opt:
    type: MSELoss
    loss_weight: 1.0
    reduction: mean
```

### 感知损失
```yaml
train:
  perceptual_opt:
    type: PerceptualLoss
    layer_weights:
      'conv3_4': 1
      'conv4_4': 1
      'conv5_4': 1
    vgg_type: vgg19
    use_input_norm: true
    range_norm: false
    perceptual_weight: 1.0
    style_weight: 0
    criterion: l1
```

### 组合损失
```yaml
train:
  pixel_opt:
    type: L1Loss
    loss_weight: 1.0
    reduction: mean
  
  perceptual_opt:
    type: PerceptualLoss
    layer_weights:
      'conv3_4': 1
      'conv4_4': 1
      'conv5_4': 1
    vgg_type: vgg19
    use_input_norm: true
    perceptual_weight: 0.1
    style_weight: 0
    criterion: l1
```

## 📈 学习率调度配置

### 余弦退火重启
```yaml
train:
  scheduler:
    type: CosineAnnealingRestartLR
    periods: [250000, 250000, 250000, 250000]
    restart_weights: [1, 1, 1, 1]
    eta_min: !!float 1e-7
```

### 多步长衰减
```yaml
train:
  scheduler:
    type: MultiStepLR
    milestones: [100000, 200000, 300000, 400000]
    gamma: 0.5
```

### 线性预热 + 余弦衰减
```yaml
train:
  scheduler:
    type: CosineAnnealingLR
    T_max: 400000
    eta_min: !!float 1e-7
  warmup_iter: 10000  # 预热迭代次数
```

## 🔧 使用技巧

### 1. 快速验证配置
设置较小的训练参数用于验证配置是否正确：
```yaml
train:
  total_iter: 1000  # 临时设置小值
val:
  val_freq: 500     # 更频繁验证
logger:
  print_freq: 10    # 更频繁打印
```

### 2. 调试模式
在实验名称中包含 "debug" 可启用调试模式：
```yaml
name: debug_test_config  # 会启用调试模式
```

### 3. 恢复训练
```yaml
path:
  resume_state: experiments/train_experiment_name/training_states/latest.state
```

### 4. 预训练模型加载
```yaml
path:
  pretrain_network_g: experiments/pretrained_models/model.pth
  param_key_g: params  # 或 params_ema
  strict_load_g: true
```

这些配置模板可以根据具体需求进行调整，建议先从基础配置开始，逐步调优参数。