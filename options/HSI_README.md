# HSI (Hyperspectral Image) Super-Resolution Configurations

This directory contains configuration templates specifically designed for hyperspectral image (HSI) super-resolution experiments using BasicSR framework.

## Directory Structure

```
options/
├── train/HSI/
│   └── train_HSI_SRResNet_x4.yml    # Training configuration template
└── test/HSI/
    └── test_HSI_SRResNet_x4.yml     # Testing configuration template
```

## Features

### HSI-Specific Modifications

1. **Dataset Type**: Uses `HSIDataset` class for handling `.mat` and `.npy` files
2. **Network Configuration**: Configurable input/output channels for different spectral bands
3. **Memory Optimization**: Smaller batch sizes and patch sizes for HSI data
4. **Comprehensive Metrics**: Includes PSNR, SSIM, SAM, ERGAS, and RMSE

### Supported Metrics

- **PSNR** (Peak Signal-to-Noise Ratio)
- **SSIM** (Structural Similarity Index)
- **SAM** (Spectral Angle Mapper) - HSI-specific
- **ERGAS** (Erreur Relative Globale Adimensionnelle de Synthèse) - HSI-specific  
- **RMSE** (Root Mean Square Error)

## Quick Start

### 1. Prepare Your Dataset

Follow the [HSI Dataset Preparation Guide](../../docs/HSI_DatasetPreparation.md) to organize your data.

### 2. Configure for Your Dataset

**Training Configuration** (`train_HSI_SRResNet_x4.yml`):

```yaml
# Update dataset paths
datasets:
  train:
    dataroot_gt: datasets/your_hsi_dataset/HR
    dataroot_lq: datasets/your_hsi_dataset/LR

# Set spectral channels (example for 31-band HSI)
network_g:
  num_in_ch: 31
  num_out_ch: 31
```

**Testing Configuration** (`test_HSI_SRResNet_x4.yml`):

```yaml
# Update dataset paths
datasets:
  test_1:
    dataroot_gt: datasets/your_hsi_dataset/test/HR
    dataroot_lq: datasets/your_hsi_dataset/test/LR

# Set model path
path:
  pretrain_network_g: experiments/pretrained_models/your_model.pth
```

### 3. Train the Model

```bash
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python basicsr/train.py \
    -opt options/train/HSI/train_HSI_SRResNet_x4.yml \
    --auto_resume
```

### 4. Test the Model

```bash
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python basicsr/test.py \
    -opt options/test/HSI/test_HSI_SRResNet_x4.yml
```

## Configuration Parameters

### Key Parameters to Modify

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `num_in_ch` / `num_out_ch` | Number of spectral bands | 31, 102, 103, 128, 200 |
| `gt_size` | Patch size for training | 32, 64, 96 |
| `batch_size_per_gpu` | Batch size per GPU | 4, 8, 16 |
| `scale` | Super-resolution scale factor | 2, 3, 4 |
| `dataroot_gt` / `dataroot_lq` | Dataset paths | Your dataset paths |

### Memory Considerations

HSI data requires significant memory due to the high number of spectral channels. Adjust these parameters based on your GPU memory:

- **32GB GPU**: `batch_size_per_gpu: 16`, `gt_size: 96`
- **16GB GPU**: `batch_size_per_gpu: 8`, `gt_size: 64`
- **8GB GPU**: `batch_size_per_gpu: 4`, `gt_size: 32`

## Architecture Support

The current configurations use SRResNet architecture. To use other architectures:

### RCAN
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

### EDSR
```yaml
network_g:
  type: EDSR
  num_in_ch: 31
  num_out_ch: 31
  num_feat: 64
  num_block: 16
  upscale: 4
  res_scale: 1
  img_range: 255.
  rgb_mean: [0.4488, 0.4371, 0.4040]  # Adjust for HSI
```

## Common Issues and Solutions

### 1. Out of Memory Error
- Reduce `batch_size_per_gpu`
- Reduce `gt_size`
- Use gradient checkpointing (add `use_checkpoint: true` in network_g)

### 2. Shape Mismatch
- Ensure all HSI files have the same number of spectral bands
- Verify `num_in_ch` and `num_out_ch` match your data

### 3. File Format Issues
- Check data keys in .mat files
- Ensure .npy files contain 3D arrays with shape [H, W, C]

## Experiment Tracking

The configurations include Weights & Biases (wandb) integration:

```yaml
logger:
  wandb:
    project: HSI_SuperResolution
```

To use:
1. Install wandb: `pip install wandb`
2. Login: `wandb login`
3. Run training with wandb enabled

## References

- [BasicSR Documentation](https://github.com/XPixelGroup/BasicSR)
- [HSI Dataset Preparation Guide](../../docs/HSI_DatasetPreparation.md)
- [Configuration Documentation](../../docs/Config.md)