# BasicSR Comprehensive Usage Guide

[English](BasicSR_Usage_Guide.md) **|** [简体中文](BasicSR_使用指南_CN.md)

This guide provides a complete workflow for using the BasicSR framework for image restoration tasks, including detailed configuration methods for super-resolution, denoising, and inpainting tasks.

## 📋 Contents

1. [Quick Start](#quick-start)
2. [General Workflow](#general-workflow)
3. [Environment Setup](#environment-setup)
4. [Data Preparation](#data-preparation)
5. [Super-Resolution Tasks](#super-resolution-tasks)
6. [Denoising Tasks](#denoising-tasks)
7. [Inpainting Tasks](#inpainting-tasks)
8. [Model Architecture Selection](#model-architecture-selection)
9. [Training Tips and Best Practices](#training-tips-and-best-practices)
10. [Troubleshooting](#troubleshooting)

## 🚀 Quick Start

### Basic Requirements
- Python 3.7+
- PyTorch 1.7+
- NVIDIA GPU (recommended)

### One-click Example
```bash
# Clone repository
git clone https://github.com/XPixelGroup/BasicSR.git
cd BasicSR

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run super-resolution example
python basicsr/train.py -opt options/train/SRResNet_SRGAN/train_MSRResNet_x4.yml
```

## 🔄 General Workflow

All tasks follow this standard workflow:

### 1. Environment Setup
- Install BasicSR and dependencies
- Configure GPU environment
- Download pretrained models (if needed)

### 2. Data Preparation
- Organize dataset directory structure
- Preprocess data (cropping, format conversion, etc.)
- Create LMDB database (optional, for training acceleration)

### 3. Configuration Setup
- Choose appropriate configuration template
- Modify data paths
- Adjust network parameters
- Set training/testing parameters

### 4. Model Training
```bash
# Single GPU training
python basicsr/train.py -opt path/to/config.yml

# Multi-GPU distributed training
python -m torch.distributed.launch --nproc_per_node=4 basicsr/train.py -opt path/to/config.yml --launcher pytorch
```

### 5. Testing and Evaluation
```bash
# Test model
python basicsr/test.py -opt path/to/test_config.yml

# Calculate metrics
python scripts/metrics/calculate_psnr_ssim.py --gt path/to/gt --restored path/to/results
```

## 🛠️ Environment Setup

For detailed installation instructions, please refer to [INSTALL.md](INSTALL.md)

### Basic Installation
```bash
# Create conda environment
conda create -n basicsr python=3.8
conda activate basicsr

# Install PyTorch
conda install pytorch torchvision cudatoolkit=11.1 -c pytorch -c nvidia

# Install BasicSR
pip install basicsr
# Or install from source
git clone https://github.com/XPixelGroup/BasicSR.git
cd BasicSR
pip install -e .
```

### Verify Installation
```python
import basicsr
print(basicsr.__version__)
```

## 📁 Data Preparation

For detailed data preparation guide, please refer to [DatasetPreparation.md](DatasetPreparation.md)

### Standard Data Directory Structure
```
datasets/
├── DIV2K/
│   ├── DIV2K_train_HR/          # High-resolution training images
│   ├── DIV2K_train_LR_bicubic/  # Low-resolution training images
│   ├── DIV2K_valid_HR/          # High-resolution validation images
│   └── DIV2K_valid_LR_bicubic/  # Low-resolution validation images
├── Set5/                        # Test dataset
│   ├── GTmod12/                 # Ground Truth
│   └── LRbicx4/                 # Low-resolution input
└── other_datasets/
```

### Data Preprocessing Scripts
```bash
# Extract subimages (for training)
python scripts/data_preparation/extract_subimages.py

# Create LMDB database
python scripts/data_preparation/create_lmdb.py

# Generate degraded data
python scripts/data_preparation/generate_multiscale_DF2K.py
```

## 🔍 Super-Resolution Tasks

Super-resolution is the core functionality of BasicSR, supporting multiple architectures and scale factors.

### Supported Architectures
- **SRResNet/MSRResNet**: Classic residual network, suitable for beginners
- **EDSR**: Enhanced Deep Super-Resolution network
- **RCAN**: Residual Channel Attention Network
- **SwinIR**: Swin Transformer-based network
- **ESRGAN**: Generative Adversarial Network, suitable for real images
- **Real-ESRGAN**: Enhanced version for real-world scenarios

### Configuration Examples

#### 1. SRResNet (Recommended for beginners)
```yaml
# Basic configuration
name: train_MSRResNet_x4
model_type: SRModel
scale: 4
num_gpu: 1

# Dataset configuration
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

# Network structure
network_g:
  type: MSRResNet
  num_in_ch: 3
  num_out_ch: 3
  num_feat: 64
  num_block: 16
  upscale: 4

# Training settings
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

#### 2. EDSR (High performance)
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

#### 3. SwinIR (Latest architecture)
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

### Training Commands
```bash
# Single GPU training
CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/SRResNet_SRGAN/train_MSRResNet_x4.yml

# Multi-GPU training (4 GPUs)
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/train/SRResNet_SRGAN/train_MSRResNet_x4.yml --launcher pytorch

# Auto-resume training
python basicsr/train.py -opt options/train/SRResNet_SRGAN/train_MSRResNet_x4.yml --auto_resume
```

### Testing and Inference
```bash
# Test model performance
python basicsr/test.py -opt options/test/SRResNet_SRGAN/test_MSRResNet_x4.yml

# Single image inference
python inference/inference_esrgan.py --input path/to/input --output path/to/output --model_path path/to/model.pth
```

## 🔇 Denoising Tasks

Denoising tasks mainly remove noise from images to restore clear images.

### Supported Architectures
- **RIDNet**: Real Image Denoising Network
- **SwinIR**: Supports grayscale and color image denoising
- **DnCNN**: Classic denoising network (can be implemented)

### Configuration Examples

#### 1. RIDNet Configuration
```yaml
name: train_RIDNet_noise25
model_type: SRModel  # Reuse SR model framework
scale: 1  # Denoising doesn't change resolution
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

#### 2. SwinIR Denoising Configuration
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
  upsampler: ''  # No upsampling for denoising
  resi_connection: '1conv'
```

### Noise Data Preparation
```python
# Generate synthetic noise data
import numpy as np
import cv2

def add_noise(img, noise_level=25):
    """Add Gaussian noise"""
    noise = np.random.normal(0, noise_level/255.0, img.shape)
    noisy_img = img + noise
    return np.clip(noisy_img, 0, 1)

# Batch processing
import glob
clean_imgs = glob.glob('datasets/clean/*.png')
for img_path in clean_imgs:
    img = cv2.imread(img_path).astype(np.float32) / 255.0
    noisy_img = add_noise(img, 25)
    save_path = img_path.replace('clean', 'noisy')
    cv2.imwrite(save_path, (noisy_img * 255).astype(np.uint8))
```

### Denoising Inference
```bash
# Denoising with RIDNet
python inference/inference_ridnet.py --test_path datasets/denoise/test --noise_g 25 --model_path experiments/pretrained_models/RIDNet/RIDNet.pth

# Denoising with SwinIR
python inference/inference_swinir.py --task color_dn --noise 25 --input datasets/denoise/test --output results/swinir_denoise --model_path path/to/swinir_denoise_model.pth
```

## 🖼️ Inpainting Tasks

Inpainting tasks include image restoration, JPEG compression artifact removal, etc.

### Supported Architectures
- **SwinIR**: Supports JPEG compression artifact removal
- **DFDNet**: Dedicated face restoration network
- **Custom architectures**: Can be modified based on existing networks

### JPEG Compression Artifact Removal

#### Configuration Example
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
  window_size: 7  # Use 7 for JPEG tasks
  img_range: 1.
  depths: [6, 6, 6, 6, 6, 6]
  embed_dim: 180
  num_heads: [6, 6, 6, 6, 6, 6]
  mlp_ratio: 2
  upsampler: ''
  resi_connection: '1conv'
```

#### Data Preparation
```python
# Generate JPEG compressed data
import cv2
from PIL import Image

def compress_jpeg(img_path, quality=10):
    """Generate JPEG compressed images"""
    img = Image.open(img_path)
    compressed_path = img_path.replace('.png', f'_q{quality}.jpg')
    img.save(compressed_path, 'JPEG', quality=quality)
    return compressed_path

# Batch processing
import glob
clean_imgs = glob.glob('datasets/clean/*.png')
for img_path in clean_imgs:
    compressed_path = compress_jpeg(img_path, quality=10)
    print(f"Compressed: {compressed_path}")
```

### Face Restoration (DFDNet)

#### Configuration Requirements
```bash
# Install dlib (face detection dependency)
pip install dlib

# Download pretrained models
python scripts/download_pretrained_models.py DFDNet
```

#### Inference Example
```bash
# Face restoration inference
python inference/inference_dfdnet.py --upscale_factor=2 --test_path datasets/faces
```

## 🏗️ Model Architecture Selection

### Selection by Task Type

| Task Type | Recommended Architecture | Features | Use Cases |
|-----------|-------------------------|----------|-----------|
| Super-Resolution | SRResNet | Simple and fast | Learning, quick prototyping |
|                  | EDSR | Excellent performance | High-quality results |
|                  | SwinIR | Latest technology | Research, best performance |
|                  | ESRGAN | Generative adversarial | Real images, visual effects |
| Denoising | RIDNet | Specialized design | Real image denoising |
|           | SwinIR | Versatile | Various noise types |
| Restoration | SwinIR | Multi-task support | JPEG, restoration, etc. |
|            | DFDNet | Face-specific | Face restoration only |

### Selection by Computing Resources

| GPU Memory | Recommended Architecture | Batch Size | Image Size |
|------------|-------------------------|------------|------------|
| 8GB | SRResNet | 16 | 128x128 |
| 16GB | EDSR/RCAN | 16 | 192x192 |
| 24GB+ | SwinIR | 8 | 256x256 |

### Selection by Dataset Size

- **Small dataset (<1K images)**: Fine-tune pretrained models
- **Medium dataset (1K-10K)**: Train SRResNet or EDSR from scratch
- **Large dataset (>10K)**: Any architecture, recommend SwinIR or ESRGAN

## 💡 Training Tips and Best Practices

### Data Augmentation
```yaml
# Enable data augmentation in config file
datasets:
  train:
    use_hflip: true      # Horizontal flip
    use_rot: true        # Rotation
    use_shuffle: true    # Random shuffle
    color_jitter: true   # Color jittering (custom)
```

### Learning Rate Scheduling
```yaml
# Cosine annealing restart
scheduler:
  type: CosineAnnealingRestartLR
  periods: [250000, 250000, 250000, 250000]
  restart_weights: [1, 1, 1, 1]
  eta_min: !!float 1e-7

# Multi-step decay
scheduler:
  type: MultiStepLR
  milestones: [100000, 200000, 300000, 400000]
  gamma: 0.5
```

### Loss Function Selection
```yaml
# L1 loss (common)
pixel_opt:
  type: L1Loss
  loss_weight: 1.0

# Perceptual loss (better visual effects)
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

# Combined loss
pixel_opt:
  type: L1Loss
  loss_weight: 1.0
perceptual_opt:
  type: PerceptualLoss
  loss_weight: 0.1
```

### Training Monitoring
```yaml
# Validation settings
val:
  val_freq: !!float 5e3  # Validate every 5000 iterations
  save_img: true         # Save validation images
  
  metrics:
    psnr:
      type: calculate_psnr
      crop_border: 4
      test_y_channel: false
    ssim:
      type: calculate_ssim
      crop_border: 4
      test_y_channel: false

# Logging settings
logger:
  print_freq: 100
  save_checkpoint_freq: !!float 5e3
  use_tb_logger: true
  wandb:
    project: my_project_name
    resume_id: ~
```

### Memory Optimization
```yaml
# Reduce batch size
datasets:
  train:
    batch_size_per_gpu: 8  # Reduce from 16 to 8

# Use gradient accumulation
train:
  accumulate_grad_batches: 2  # Accumulate gradients from 2 batches

# Enable mixed precision training
train:
  use_amp: true  # Automatic mixed precision
```

## ❓ Troubleshooting

### 1. Out of Memory Error
```bash
# Error: CUDA out of memory
```
**Solutions**:
- Reduce `batch_size_per_gpu`
- Reduce `gt_size` (training patch size)
- Use smaller network architecture
- Enable gradient checkpointing

### 2. Training Not Converging
**Possible causes and solutions**:
- Learning rate too high: Reduce learning rate
- Data preprocessing issues: Check data normalization
- Inappropriate network architecture: Try other architectures
- Poor dataset quality: Check data annotations

### 3. Validation PSNR Not Improving
**Solutions**:
- Check validation dataset paths
- Confirm loss function suitable for task
- Adjust learning rate scheduling strategy
- Increase training iterations

### 4. Distributed Training Issues
```bash
# Error: RuntimeError: Address already in use
```
**Solutions**:
```bash
# Change port number
python -m torch.distributed.launch --master_port=4322 ...

# Or set environment variable
export MASTER_PORT=4322
```

### 5. Model Loading Error
```bash
# Error: Key mismatch when loading state dict
```
**Solutions**:
```yaml
# Set in config file
path:
  strict_load_g: false  # Allow partial loading
  param_key_g: params_ema  # Or use EMA parameters
```

## 📚 Related Documentation

- [Installation Guide](INSTALL.md)
- [Data Preparation Details](DatasetPreparation.md)
- [Training and Testing Commands](TrainTest.md)
- [Configuration File Explanation](Config.md)
- [Model Zoo](ModelZoo.md)
- [Evaluation Metrics](Metrics.md)
- [HOWTOs Guide](HOWTOs.md)

## 🤝 Community Support

- **GitHub Issues**: [Submit Issues](https://github.com/XPixelGroup/BasicSR/issues)
- **Discussions**: [GitHub Discussions](https://github.com/XPixelGroup/BasicSR/discussions)
- **QQ Group**: 320960100 (Answer: 互帮互助共同进步)

## 📄 License

This project is licensed under the Apache 2.0 License. See [LICENSE](../LICENSE.txt) for details.

---

If this guide is helpful to you, please give the project a ⭐ Star!