# BasicSR 常见任务一键配置指南

本指南提供最常见的三种任务（超分辨率、去噪、修复）的一键配置方法，让初学者能快速上手。

## 📚 文档导航

- **完整使用指南**: [BasicSR_使用指南_CN.md](BasicSR_使用指南_CN.md)
- **配置模板参考**: [任务配置快速参考_CN.md](任务配置快速参考_CN.md)
- **数据准备**: [DatasetPreparation_CN.md](DatasetPreparation_CN.md)
- **训练测试**: [TrainTest_CN.md](TrainTest_CN.md)

---

## 🔍 任务一：超分辨率 (Image Super-Resolution)

### 步骤 1: 准备数据

```bash
# 创建数据目录
mkdir -p datasets/MyDataset/{train,val}/{HR,LR}

# 数据结构示例
datasets/MyDataset/
├── train/
│   ├── HR/          # 高分辨率训练图像
│   └── LR/          # 低分辨率训练图像 (可用脚本生成)
└── val/
    ├── HR/          # 高分辨率验证图像
    └── LR/          # 低分辨率验证图像
```

### 步骤 2: 生成低分辨率图像 (如需要)

```python
# 生成4倍下采样的LR图像
import cv2
import os
from glob import glob

def generate_lr_images(hr_path, lr_path, scale=4):
    os.makedirs(lr_path, exist_ok=True)
    hr_images = glob(os.path.join(hr_path, '*.png'))
    
    for img_path in hr_images:
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        lr_img = cv2.resize(img, (w//scale, h//scale), interpolation=cv2.INTER_CUBIC)
        
        filename = os.path.basename(img_path)
        save_path = os.path.join(lr_path, filename)
        cv2.imwrite(save_path, lr_img)
        print(f"Generated: {save_path}")

# 使用示例
generate_lr_images('datasets/MyDataset/train/HR', 'datasets/MyDataset/train/LR', scale=4)
generate_lr_images('datasets/MyDataset/val/HR', 'datasets/MyDataset/val/LR', scale=4)
```

### 步骤 3: 配置文件 (保存为 `options/train/my_sr_config.yml`)

```yaml
# 基本设置
name: my_sr_x4_experiment
model_type: SRModel
scale: 4
num_gpu: 1
manual_seed: 0

# 数据集设置
datasets:
  train:
    name: MyDataset
    type: PairedImageDataset
    dataroot_gt: datasets/MyDataset/train/HR
    dataroot_lq: datasets/MyDataset/train/LR
    filename_tmpl: '{}'
    io_backend:
      type: disk
    
    gt_size: 128
    use_hflip: true
    use_rot: true
    num_worker_per_gpu: 4
    batch_size_per_gpu: 16
    dataset_enlarge_ratio: 100
    
  val:
    name: MyValidation
    type: PairedImageDataset
    dataroot_gt: datasets/MyDataset/val/HR
    dataroot_lq: datasets/MyDataset/val/LR
    io_backend:
      type: disk

# 网络结构 (推荐初学者使用)
network_g:
  type: MSRResNet
  num_in_ch: 3
  num_out_ch: 3
  num_feat: 64
  num_block: 16
  upscale: 4

# 路径设置
path:
  pretrain_network_g: ~
  strict_load_g: true
  resume_state: ~

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
    periods: [100000, 100000, 100000, 100000]
    restart_weights: [1, 1, 1, 1]
    eta_min: !!float 1e-7
  
  total_iter: 400000
  warmup_iter: -1
  
  pixel_opt:
    type: L1Loss
    loss_weight: 1.0
    reduction: mean

# 验证设置
val:
  val_freq: !!float 5e3
  save_img: true
  
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
  wandb:
    project: ~
    resume_id: ~

# 分布式训练设置
dist_params:
  backend: nccl
  port: 29500
```

### 步骤 4: 开始训练

```bash
# 单GPU训练
python basicsr/train.py -opt options/train/my_sr_config.yml

# 多GPU训练 (如有4张卡)
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/train/my_sr_config.yml --launcher pytorch
```

---

## 🔇 任务二：图像去噪 (Image Denoising)

### 步骤 1: 准备去噪数据

```bash
# 创建去噪数据目录
mkdir -p datasets/Denoise/{train,val}/{GT,Noisy}
```

### 步骤 2: 生成含噪图像

```python
# 添加高斯噪声
import cv2
import numpy as np
import os
from glob import glob

def add_gaussian_noise(clean_path, noisy_path, noise_level=25):
    """给图像添加高斯噪声"""
    os.makedirs(noisy_path, exist_ok=True)
    clean_images = glob(os.path.join(clean_path, '*.png'))
    
    for img_path in clean_images:
        # 读取图像并归一化到[0,1]
        img = cv2.imread(img_path).astype(np.float32) / 255.0
        
        # 添加高斯噪声
        noise = np.random.normal(0, noise_level/255.0, img.shape)
        noisy_img = img + noise
        noisy_img = np.clip(noisy_img, 0, 1)
        
        # 保存含噪图像
        filename = os.path.basename(img_path)
        save_path = os.path.join(noisy_path, filename)
        cv2.imwrite(save_path, (noisy_img * 255).astype(np.uint8))
        print(f"Generated noisy image: {save_path}")

# 生成训练和验证的含噪图像
add_gaussian_noise('datasets/Denoise/train/GT', 'datasets/Denoise/train/Noisy', noise_level=25)
add_gaussian_noise('datasets/Denoise/val/GT', 'datasets/Denoise/val/Noisy', noise_level=25)
```

### 步骤 3: 去噪配置文件 (保存为 `options/train/my_denoise_config.yml`)

```yaml
# 基本设置
name: my_denoise_sigma25_experiment
model_type: SRModel
scale: 1  # 去噪不改变分辨率
num_gpu: 1
manual_seed: 0

# 数据集设置
datasets:
  train:
    name: DenoiseDataset
    type: PairedImageDataset
    dataroot_gt: datasets/Denoise/train/GT
    dataroot_lq: datasets/Denoise/train/Noisy
    filename_tmpl: '{}'
    io_backend:
      type: disk
    
    gt_size: 128
    use_hflip: true
    use_rot: true
    num_worker_per_gpu: 4
    batch_size_per_gpu: 16
    dataset_enlarge_ratio: 100
    
  val:
    name: DenoiseValidation
    type: PairedImageDataset
    dataroot_gt: datasets/Denoise/val/GT
    dataroot_lq: datasets/Denoise/val/Noisy
    io_backend:
      type: disk

# 网络结构 (使用RIDNet进行去噪)
network_g:
  type: RIDNet
  num_in_ch: 3
  num_feat: 64
  num_out_ch: 3

# 训练设置
train:
  ema_decay: 0.999
  optim_g:
    type: Adam
    lr: !!float 1e-4  # 去噪使用较小学习率
    weight_decay: 0
    betas: [0.9, 0.999]
  
  scheduler:
    type: MultiStepLR
    milestones: [50000, 100000, 150000, 200000]
    gamma: 0.5
  
  total_iter: 300000
  warmup_iter: -1
  
  pixel_opt:
    type: L1Loss
    loss_weight: 1.0
    reduction: mean

# 验证设置
val:
  val_freq: !!float 5e3
  save_img: true
  
  metrics:
    psnr:
      type: calculate_psnr
      crop_border: 0  # 去噪通常不裁剪边界
      test_y_channel: false

# 日志设置
logger:
  print_freq: 100
  save_checkpoint_freq: !!float 5e3
  use_tb_logger: true

dist_params:
  backend: nccl
  port: 29500
```

### 步骤 4: 开始去噪训练

```bash
python basicsr/train.py -opt options/train/my_denoise_config.yml
```

---

## 🖼️ 任务三：JPEG压缩伪影去除

### 步骤 1: 准备JPEG压缩数据

```bash
mkdir -p datasets/JPEG_CAR/{train,val}/{GT,Compressed}
```

### 步骤 2: 生成JPEG压缩图像

```python
# 生成JPEG压缩伪影数据
import cv2
from PIL import Image
import os
from glob import glob

def compress_jpeg(clean_path, compressed_path, quality=10):
    """生成JPEG压缩图像"""
    os.makedirs(compressed_path, exist_ok=True)
    clean_images = glob(os.path.join(clean_path, '*.png'))
    
    for img_path in clean_images:
        # 使用PIL进行JPEG压缩
        img = Image.open(img_path)
        filename = os.path.splitext(os.path.basename(img_path))[0]
        
        # 保存为JPEG格式（有损压缩）
        jpeg_path = os.path.join(compressed_path, f"{filename}.jpg")
        img.save(jpeg_path, 'JPEG', quality=quality)
        
        # 再读取并保存为PNG用于训练
        compressed_img = Image.open(jpeg_path)
        png_path = os.path.join(compressed_path, f"{filename}.png")
        compressed_img.save(png_path, 'PNG')
        
        # 删除中间JPEG文件
        os.remove(jpeg_path)
        print(f"Generated compressed image: {png_path}")

# 生成质量因子为10的JPEG压缩图像
compress_jpeg('datasets/JPEG_CAR/train/GT', 'datasets/JPEG_CAR/train/Compressed', quality=10)
compress_jpeg('datasets/JPEG_CAR/val/GT', 'datasets/JPEG_CAR/val/Compressed', quality=10)
```

### 步骤 3: JPEG修复配置文件 (保存为 `options/train/my_jpeg_car_config.yml`)

```yaml
# 基本设置
name: my_jpeg_car_q10_experiment
model_type: SwinIRModel
scale: 1
num_gpu: 1
manual_seed: 0

# 数据集设置
datasets:
  train:
    name: JPEG_CAR_Dataset
    type: PairedImageDataset
    dataroot_gt: datasets/JPEG_CAR/train/GT
    dataroot_lq: datasets/JPEG_CAR/train/Compressed
    filename_tmpl: '{}'
    io_backend:
      type: disk
    
    gt_size: 96
    use_hflip: true
    use_rot: true
    num_worker_per_gpu: 4
    batch_size_per_gpu: 16
    dataset_enlarge_ratio: 100
    
  val:
    name: JPEG_CAR_Validation
    type: PairedImageDataset
    dataroot_gt: datasets/JPEG_CAR/val/GT
    dataroot_lq: datasets/JPEG_CAR/val/Compressed
    io_backend:
      type: disk

# 网络结构 (SwinIR专用于JPEG压缩伪影去除)
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
    periods: [92000]
    restart_weights: [1]
    eta_min: !!float 1e-7
  
  total_iter: 92000
  warmup_iter: -1
  
  pixel_opt:
    type: L1Loss
    loss_weight: 1.0
    reduction: mean

# 验证设置
val:
  val_freq: !!float 2e3
  save_img: true
  
  metrics:
    psnr:
      type: calculate_psnr
      crop_border: 0
      test_y_channel: false

# 日志设置
logger:
  print_freq: 100
  save_checkpoint_freq: !!float 2e3
  use_tb_logger: true

dist_params:
  backend: nccl
  port: 29500
```

### 步骤 4: 开始JPEG修复训练

```bash
python basicsr/train.py -opt options/train/my_jpeg_car_config.yml
```

---

## 📊 测试和评估

### 创建测试配置文件

以超分辨率为例，保存为 `options/test/my_sr_test.yml`:

```yaml
name: test_my_sr_x4
model_type: SRModel
scale: 4
num_gpu: 1

datasets:
  test_1:
    name: MyTestSet
    type: PairedImageDataset
    dataroot_gt: datasets/MyDataset/val/HR
    dataroot_lq: datasets/MyDataset/val/LR
    io_backend:
      type: disk

network_g:
  type: MSRResNet
  num_in_ch: 3
  num_out_ch: 3
  num_feat: 64
  num_block: 16
  upscale: 4

path:
  pretrain_network_g: experiments/my_sr_x4_experiment/models/net_g_latest.pth
  strict_load_g: true

val:
  save_img: true
  suffix: ~
  
  metrics:
    psnr:
      type: calculate_psnr
      crop_border: 4
      test_y_channel: false
    ssim:
      type: calculate_ssim
      crop_border: 4
      test_y_channel: false
```

### 运行测试

```bash
# 测试模型
python basicsr/test.py -opt options/test/my_sr_test.yml

# 计算更多指标
python scripts/metrics/calculate_psnr_ssim.py --gt datasets/MyDataset/val/HR --restored results/test_my_sr_x4/MyTestSet --crop_border 4
```

---

## 🛠️ 实用工具脚本

### 1. 批量图像转换

```python
# convert_images.py - 批量转换图像格式
import cv2
import os
from glob import glob

def convert_images(input_dir, output_dir, target_format='png'):
    """批量转换图像格式"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 支持的输入格式
    patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    
    for pattern in patterns:
        files = glob(os.path.join(input_dir, pattern))
        files.extend(glob(os.path.join(input_dir, pattern.upper())))
        
        for file_path in files:
            img = cv2.imread(file_path)
            if img is not None:
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                output_path = os.path.join(output_dir, f"{base_name}.{target_format}")
                cv2.imwrite(output_path, img)
                print(f"Converted: {file_path} -> {output_path}")

# 使用示例
# convert_images('input_folder', 'output_folder', 'png')
```

### 2. 图像质量检查

```python
# check_image_quality.py - 检查图像质量
import cv2
import numpy as np
from glob import glob

def check_images(folder_path):
    """检查图像基本信息"""
    images = glob(os.path.join(folder_path, '*.png'))
    
    resolutions = {}
    corrupted = []
    
    for img_path in images:
        try:
            img = cv2.imread(img_path)
            if img is None:
                corrupted.append(img_path)
                continue
                
            h, w = img.shape[:2]
            resolution = f"{w}x{h}"
            resolutions[resolution] = resolutions.get(resolution, 0) + 1
            
        except Exception as e:
            corrupted.append(img_path)
            print(f"Error reading {img_path}: {e}")
    
    print(f"Total images: {len(images)}")
    print(f"Corrupted images: {len(corrupted)}")
    print("Resolutions found:")
    for res, count in resolutions.items():
        print(f"  {res}: {count} images")
    
    return corrupted

# 使用示例
# corrupted = check_images('datasets/MyDataset/train/HR')
```

### 3. 训练监控脚本

```python
# monitor_training.py - 监控训练进度
import os
import time
from glob import glob

def monitor_training(experiment_dir):
    """监控训练进度"""
    log_file = os.path.join(experiment_dir, 'train.log')
    
    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        return
    
    print(f"Monitoring: {log_file}")
    print("Press Ctrl+C to stop monitoring")
    
    try:
        with open(log_file, 'r') as f:
            # 移动到文件末尾
            f.seek(0, 2)
            
            while True:
                line = f.readline()
                if line:
                    # 过滤包含关键信息的行
                    if any(keyword in line for keyword in ['iter:', 'psnr:', 'loss:', 'lr:']):
                        print(line.strip())
                else:
                    time.sleep(1)
                    
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

# 使用示例
# monitor_training('experiments/my_sr_x4_experiment')
```

---

## ❓ 常见问题解决

### Q1: 训练过程中显存不足
```bash
# 解决方案：减少batch size和patch size
datasets:
  train:
    gt_size: 64        # 从128减少到64
    batch_size_per_gpu: 8  # 从16减少到8
```

### Q2: 训练速度太慢
```bash
# 解决方案：调整数据加载参数
datasets:
  train:
    num_worker_per_gpu: 8      # 增加数据加载进程
    prefetch_mode: cuda        # 使用CUDA预取
    dataset_enlarge_ratio: 1   # 减少数据重复次数
```

### Q3: 验证PSNR不提升
```yaml
# 检查路径是否正确
datasets:
  val:
    dataroot_gt: datasets/correct/path/to/GT  # 确保路径正确
    dataroot_lq: datasets/correct/path/to/LR
```

### Q4: 模型加载失败
```yaml
# 设置更宽松的加载模式
path:
  strict_load_g: false  # 允许部分参数不匹配
```

---

## 📈 进阶技巧

1. **使用预训练模型加速训练**:
   ```yaml
   path:
     pretrain_network_g: experiments/pretrained_models/MSRResNet.pth
   ```

2. **启用EMA(指数移动平均)**:
   ```yaml
   train:
     ema_decay: 0.999  # 通常能提升模型稳定性
   ```

3. **混合精度训练节省显存**:
   ```yaml
   train:
     use_amp: true  # 启用自动混合精度
   ```

4. **使用Wandb监控训练**:
   ```yaml
   logger:
     wandb:
       project: my_project_name
       entity: your_username
   ```

这份指南涵盖了最常见的使用场景，如需更详细的说明，请参考完整的使用指南。