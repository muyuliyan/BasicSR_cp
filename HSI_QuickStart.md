# HSI Processing Quick Start Guide

This guide provides step-by-step instructions to get started with HSI (Hyperspectral Image) processing using the BasicSR framework, including super-resolution, denoising, and inpainting tasks.

## 📋 Prerequisites

1. Python 3.7+
2. PyTorch
3. NumPy, SciPy
4. MATLAB files (.mat) or NumPy files (.npy) with HSI data

## 🚀 Quick Start

### Step 1: Prepare Your Dataset

Choose your task and organize data accordingly:

#### For Super-Resolution:
```
datasets/your_hsi_dataset/
├── HR/          # High-resolution HSI images (.mat or .npy)
├── LR/          # Low-resolution HSI images 
└── val/
    ├── HR/      # Validation HR images
    └── LR/      # Validation LR images
```

#### For Denoising:
```
datasets/your_hsi_dataset/
├── clean/       # Clean HSI images (.mat or .npy)
├── noisy/       # Noisy HSI images (optional, can generate on-the-fly)
└── val/
    └── clean/   # Validation clean images
```

#### For Inpainting:
```
datasets/your_hsi_dataset/
├── complete/    # Complete HSI images (.mat or .npy)
├── masks/       # Mask images (optional, can generate on-the-fly)
└── val/
    └── complete/ # Validation complete images
```

For super-resolution, generate LR data using bicubic downsampling:
```bash
python scripts/hsi_bicubic_preprocessing.py \
    --input_folder datasets/your_hsi_dataset/HR \
    --output_folder datasets/your_hsi_dataset/LR \
    --scale 4 \
    --file_format mat \
    --data_key gt
```

### Step 2: Configure Training

Choose your task configuration:

#### Super-Resolution Configuration:
Update `options/train/HSI/train_HSI_SRResNet_x4.yml`:
```yaml
# Set your dataset paths
datasets:
  train:
    dataroot_gt: datasets/your_hsi_dataset/HR
    dataroot_lq: datasets/your_hsi_dataset/LR

# Set spectral channels (IMPORTANT!)
network_g:
  num_in_ch: 31   # Replace with your HSI band count
  num_out_ch: 31  # Should match num_in_ch
```

#### Denoising Configuration:
Update `options/train/HSI/train_HSI_Denoising_SRResNet.yml`:
```yaml
# Set your dataset paths
datasets:
  train:
    dataroot_gt: datasets/your_hsi_dataset/clean
    add_noise_to_gt: true  # Generate noise on-the-fly
    noise_type: gaussian   # gaussian, poisson, mixed
    noise_range: [5, 50]   # Noise level range

# Set spectral channels
network_g:
  num_in_ch: 31   # Replace with your HSI band count
  num_out_ch: 31  # Should match num_in_ch
  upscale: 1      # No upscaling for denoising
```

#### Inpainting Configuration:
Update `options/train/HSI/train_HSI_Inpainting_SRResNet.yml`:
```yaml
# Set your dataset paths
datasets:
  train:
    dataroot_gt: datasets/your_hsi_dataset/complete
    generate_mask: true        # Generate masks on-the-fly
    mask_type: random_rect     # random_rect, random_irregular
    mask_ratio: [0.1, 0.3]     # 10-30% of image area

# Set spectral channels
network_g:
  num_in_ch: 31   # Replace with your HSI band count
  num_out_ch: 31  # Should match num_in_ch
  upscale: 1      # No upscaling for inpainting
```

Adjust for your GPU memory:
```yaml
datasets:
  train:
    gt_size: 64           # Reduce if out of memory
    batch_size_per_gpu: 8 # Reduce if out of memory
```

### Step 3: Train the Model

Choose the appropriate training command:

#### Super-Resolution:
```bash
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python basicsr/train.py \
    -opt options/train/HSI/train_HSI_SRResNet_x4.yml \
    --auto_resume
```

#### Denoising:
```bash
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python basicsr/train.py \
    -opt options/train/HSI/train_HSI_Denoising_SRResNet.yml \
    --auto_resume
```

#### Inpainting:
```bash
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python basicsr/train.py \
    -opt options/train/HSI/train_HSI_Inpainting_SRResNet.yml \
    --auto_resume
```

### Step 4: Test the Model

Update test configurations and run testing:

#### Super-Resolution:
1. Update `options/test/HSI/test_HSI_SRResNet_x4.yml`:
   ```yaml
   # Set test dataset paths
   datasets:
     test_1:
       dataroot_gt: datasets/your_hsi_dataset/test/HR
       dataroot_lq: datasets/your_hsi_dataset/test/LR
   
   # Set model path
   path:
     pretrain_network_g: experiments/your_experiment/models/net_g_latest.pth
   ```

2. Run testing:
   ```bash
   PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python basicsr/test.py \
       -opt options/test/HSI/test_HSI_SRResNet_x4.yml
   ```

#### Denoising:
1. Update `options/test/HSI/test_HSI_Denoising_SRResNet.yml`:
   ```yaml
   # Set test dataset paths
   datasets:
     test_1:
       dataroot_gt: datasets/your_hsi_dataset/test/clean
       dataroot_lq: datasets/your_hsi_dataset/test/noisy  # or leave empty for generated noise
   
   # Set model path
   path:
     pretrain_network_g: experiments/your_experiment/models/net_g_latest.pth
   ```

2. Run testing:
   ```bash
   PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python basicsr/test.py \
       -opt options/test/HSI/test_HSI_Denoising_SRResNet.yml
   ```

#### Inpainting:
1. Update `options/test/HSI/test_HSI_Inpainting_SRResNet.yml`:
   ```yaml
   # Set test dataset paths
   datasets:
     test_1:
       dataroot_gt: datasets/your_hsi_dataset/test/complete
       dataroot_mask: datasets/your_hsi_dataset/test/masks  # or leave empty for generated masks
   
   # Set model path
   path:
     pretrain_network_g: experiments/your_experiment/models/net_g_latest.pth
   ```

2. Run testing:
   ```bash
   PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python basicsr/test.py \
       -opt options/test/HSI/test_HSI_Inpainting_SRResNet.yml
   ```

## 📊 Available Metrics

The framework provides 5 key HSI evaluation metrics:

| Metric | Description | Better |
|--------|-------------|--------|
| **PSNR** | Peak Signal-to-Noise Ratio | Higher ↑ |
| **SSIM** | Structural Similarity Index | Higher ↑ |
| **SAM** | Spectral Angle Mapper | Lower ↓ |
| **ERGAS** | Global Relative Error | Lower ↓ |
| **RMSE** | Root Mean Square Error | Lower ↓ |

## 🔧 Customization

### Different Spectral Bands

Common HSI datasets and their channel counts:
- **CAVE**: 31 channels
- **Harvard**: 31 channels  
- **ICVL**: 31 channels
- **Pavia University**: 103 channels
- **Pavia Centre**: 102 channels

Update `num_in_ch` and `num_out_ch` accordingly.

### Task-Specific Settings

#### Denoising Parameters:
```yaml
# Noise settings
noise_type: gaussian     # gaussian, poisson, mixed
noise_range: [5, 50]     # Noise level range for training diversity
add_noise_to_gt: true    # Generate noise on-the-fly vs. using pre-computed

# For mixed noise training:
noise_type: mixed        # Randomly choose between gaussian and poisson
```

#### Inpainting Parameters:
```yaml
# Mask settings
mask_type: random_rect      # random_rect, random_irregular
mask_ratio: [0.1, 0.3]      # Range of mask ratio (10-30% of image area)
generate_mask: true         # Generate masks on-the-fly vs. using pre-defined

# For irregular masks:
mask_type: random_irregular # More natural, complex mask shapes
```

### Memory Optimization

If you encounter out-of-memory errors:

```yaml
# Reduce these values
datasets:
  train:
    gt_size: 32            # Smaller patches
    batch_size_per_gpu: 4  # Smaller batches
    num_worker_per_gpu: 2  # Fewer workers
```

### Different Architectures

To use RCAN instead of SRResNet:

```yaml
network_g:
  type: RCAN
  num_in_ch: 31
  num_out_ch: 31
  num_feat: 64
  num_group: 10
  num_block: 20
  squeeze_factor: 16
  upscale: 4
```

## 📁 File Structure Overview

```
├── basicsr/
│   ├── data/
│   │   ├── hsi_dataset.py              # HSI super-resolution dataset
│   │   ├── hsi_denoising_dataset.py    # HSI denoising dataset
│   │   └── hsi_inpainting_dataset.py   # HSI inpainting dataset
│   └── metrics/hsi_metrics.py          # HSI-specific metrics
├── scripts/hsi_bicubic_preprocessing.py  # Preprocessing script for super-resolution
├── options/
│   ├── train/HSI/
│   │   ├── train_HSI_SRResNet_x4.yml           # Super-resolution training
│   │   ├── train_HSI_Denoising_SRResNet.yml    # Denoising training
│   │   └── train_HSI_Inpainting_SRResNet.yml   # Inpainting training
│   └── test/HSI/
│       ├── test_HSI_SRResNet_x4.yml           # Super-resolution testing
│       ├── test_HSI_Denoising_SRResNet.yml    # Denoising testing
│       └── test_HSI_Inpainting_SRResNet.yml   # Inpainting testing
├── docs/HSI_DatasetPreparation.md        # Detailed documentation
└── datasets/example_hsi_dataset/         # Example structure
```

## 🔍 Testing Components

Test HSI metrics independently:
```bash
python hsi_metrics_test.py
```

Test preprocessing:
```bash
python scripts/hsi_bicubic_preprocessing.py --help
```

## 📚 Documentation

- **Detailed Guide**: `docs/HSI_DatasetPreparation.md`
- **Chinese Version**: `docs/HSI_DatasetPreparation_CN.md` 
- **Configuration Help**: `options/HSI_README.md`

## ⚠️ Common Issues

### General Issues:
1. **Shape mismatch**: Ensure all HSI files have the same number of spectral bands
2. **Memory error**: Reduce batch size and patch size
3. **File format**: Verify .mat files contain the correct data keys
4. **Path error**: Check all dataset paths in configuration files

### Task-Specific Issues:

#### Denoising:
5. **Unrealistic noise**: Adjust noise_range to match your real data
6. **Training instability**: Try mixed noise for better generalization
7. **Poor validation**: Use fixed noise level for consistent validation

#### Inpainting:
8. **Mask ratio too high**: Reduce mask_ratio if training becomes unstable
9. **Artifacts at mask boundaries**: Try irregular masks for more natural boundaries
10. **Poor reconstruction**: Ensure mask generation matches your test scenario

## 🎯 Expected Results

### Super-Resolution:
Typical metric ranges for good HSI super-resolution:
- **PSNR**: 25-40 dB (higher is better)
- **SSIM**: 0.8-0.95 (higher is better)  
- **SAM**: 0.1-0.3 radians (lower is better)
- **ERGAS**: 5-20 (lower is better)
- **RMSE**: 5-15 (lower is better)

### Denoising:
Typical metric ranges for good HSI denoising:
- **PSNR**: 30-45 dB (depending on noise level)
- **SSIM**: 0.85-0.98 (higher is better)
- **SAM**: 0.05-0.2 radians (lower is better)
- **RMSE**: 2-10 (depending on noise level)

### Inpainting:
Typical metric ranges for good HSI inpainting:
- **PSNR**: 20-35 dB (depending on mask ratio)
- **SSIM**: 0.75-0.95 (higher is better)
- **SAM**: 0.1-0.4 radians (lower is better)
- **RMSE**: 5-20 (depending on mask ratio)

Good luck with your HSI processing experiments! 🚀