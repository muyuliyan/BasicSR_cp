# MRI (Medical Resonance Imaging) Processing Configurations

This directory contains configuration templates specifically designed for medical MRI super-resolution experiments using BasicSR framework, optimized for OASIS brain MRI and MM-WHS cardiac MRI datasets.

## Directory Structure

```
options/
├── train/MRI/
│   ├── train_OASIS_SRResNet_x4.yml      # OASIS brain MRI training config
│   └── train_MMWHS_SRResNet_x4.yml      # MM-WHS cardiac MRI training config
└── test/MRI/
    ├── test_OASIS_SRResNet_x4.yml       # OASIS brain MRI testing config
    └── test_MMWHS_SRResNet_x4.yml       # MM-WHS cardiac MRI testing config
```

## Features

### MRI-Specific Modifications

1. **Medical Dataset Types**: Uses `MRIDataset`, `OASISDataset`, and `MMWHSDataset` classes
2. **Single-Channel Processing**: Optimized for grayscale MRI (1 channel vs 31-200 HSI channels)
3. **Medical Format Support**: NIfTI (.nii/.nii.gz), NumPy (.npy), MATLAB (.mat)
4. **Robust Normalization**: Percentile-based clipping + [0,1] scaling
5. **3D to 2D Conversion**: Automatic slice extraction from 3D volumes
6. **Memory Optimization**: Larger batch sizes possible due to single-channel data

### Supported Metrics

- **PSNR** (Peak Signal-to-Noise Ratio) - Standard image quality metric
- **SSIM** (Structural Similarity Index) - Perceptual similarity metric

## Quick Start

### 1. Prepare Your Dataset

#### OASIS Brain MRI Dataset:
```bash
mkdir -p datasets/OASIS/{train,val,test}/{HR,LR}
# Copy your .nii/.nii.gz brain MRI files to HR folders
python scripts/mri_data_preparation.py \
    --input datasets/OASIS/train/HR \
    --output datasets/OASIS/train/LR \
    --scale 4 --extract-2d
```

#### MM-WHS Cardiac MRI Dataset:
```bash
mkdir -p datasets/MM-WHS/{train,val,test}/{HR,LR}  
# Copy your cardiac MRI files to HR folders
python scripts/mri_data_preparation.py \
    --input datasets/MM-WHS/train/HR \
    --output datasets/MM-WHS/train/LR \
    --scale 4 --extract-2d
```

### 2. Configure for Your Dataset

Choose the appropriate configuration file and update the parameters:

#### OASIS Configuration (`train_OASIS_SRResNet_x4.yml`):
```yaml
datasets:
  train:
    dataroot_gt: datasets/OASIS/train/HR      # Update path
    dataroot_lq: datasets/OASIS/train/LR      # Update path
    normalize_to_01: true
    clip_percentiles: [2, 98]                 # Conservative for brain MRI
```

#### MM-WHS Configuration (`train_MMWHS_SRResNet_x4.yml`):
```yaml
datasets:
  train:
    dataroot_gt: datasets/MM-WHS/train/HR     # Update path  
    dataroot_lq: datasets/MM-WHS/train/LR     # Update path
    normalize_to_01: true
    clip_percentiles: [1, 99]                 # Wider range for cardiac MRI
```

### 3. Train the Model

```bash
# Train OASIS model
PYTHONPATH="./:${PYTHONPATH}" python basicsr/train.py \
    -opt options/train/MRI/train_OASIS_SRResNet_x4.yml --auto_resume

# Train MM-WHS model  
PYTHONPATH="./:${PYTHONPATH}" python basicsr/train.py \
    -opt options/train/MRI/train_MMWHS_SRResNet_x4.yml --auto_resume
```

### 4. Test the Model

```bash
# Test OASIS model
PYTHONPATH="./:${PYTHONPATH}" python basicsr/test.py \
    -opt options/test/MRI/test_OASIS_SRResNet_x4.yml

# Test MM-WHS model
PYTHONPATH="./:${PYTHONPATH}" python basicsr/test.py \
    -opt options/test/MRI/test_MMWHS_SRResNet_x4.yml
```

## Configuration Parameters

### Key Parameters to Modify

```yaml
# Dataset Configuration
datasets:
  train:
    type: OASISDataset                    # or MMWHSDataset
    dataroot_gt: path/to/your/HR/data     # High-resolution MRI path
    dataroot_lq: path/to/your/LR/data     # Low-resolution MRI path
    
    # MRI-specific settings
    normalize_to_01: true                 # Normalize to [0,1] range
    clip_percentiles: [2, 98]             # Percentile clipping [min, max]
    slice_axis: 2                         # Axis for 2D slice extraction (0,1,2)
    extract_2d_slices: true               # Convert 3D→2D for training
    
    # Training parameters
    gt_size: 128                          # Training patch size
    batch_size_per_gpu: 16                # Batch size (larger than HSI)
    use_hflip: true                       # Horizontal flip augmentation
    use_rot: true                         # Rotation augmentation

# Network Architecture  
network_g:
  type: MSRResNet                         # Architecture type
  num_in_ch: 1                            # Single-channel MRI input
  num_out_ch: 1                           # Single-channel output
  num_feat: 64                            # Feature dimensions
  num_block: 16                           # Number of residual blocks
  upscale: 4                              # Super-resolution scale

# Training Settings
train:
  optim_g:
    lr: !!float 2e-4                      # Learning rate
  total_iter: 500000                      # Training iterations
  pixel_opt:
    type: L1Loss                          # Loss function
```

### Memory Considerations

For limited GPU memory, adjust these parameters:

```yaml
# Reduce memory usage
batch_size_per_gpu: 8                     # Smaller batch size
gt_size: 96                               # Smaller patch size
num_worker_per_gpu: 2                     # Fewer data loading workers

# For very limited memory
batch_size_per_gpu: 4
gt_size: 64
num_worker_per_gpu: 1
```

## Architecture Support

The current configurations use MSRResNet architecture. To use other architectures:

### RCAN
```yaml
network_g:
  type: RCAN
  num_in_ch: 1
  num_out_ch: 1
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
  num_in_ch: 1
  num_out_ch: 1
  num_feat: 64
  num_block: 16
  upscale: 4
  res_scale: 1
  img_range: 1.0                          # Adjusted for [0,1] normalized MRI
```

## Dataset-Specific Settings

### OASIS Brain MRI Settings
```yaml
# Conservative normalization for brain tissue contrast
clip_percentiles: [2, 98]               # Remove extreme outliers
slice_axis: 2                            # Axial slices typical
extract_2d_slices: true                  # 2D slice training
```

### MM-WHS Cardiac MRI Settings  
```yaml
# Wider range for cardiac tissue contrast
clip_percentiles: [1, 99]               # Preserve more intensity range
slice_axis: 2                            # Axial or short-axis views
extract_2d_slices: true                  # 2D slice training
```

## Common Issues and Solutions

### 1. **File Format Issues**
```bash
# Install medical imaging libraries
pip install nibabel          # For NIfTI support
pip install pydicom         # For DICOM support (optional)
```

### 2. **Memory Problems**
```yaml
# Reduce memory usage
batch_size_per_gpu: 4
gt_size: 64
num_worker_per_gpu: 1
```

### 3. **Data Loading Errors**
```bash
# Test dataset loading
python -c "
from basicsr.data.mri_dataset import OASISDataset
# Test with your config
"
```

### 4. **Poor Normalization Results**
```yaml
# Adjust percentiles based on your data
clip_percentiles: [5, 95]   # Less aggressive clipping
# or
clip_percentiles: [1, 99.5] # More aggressive clipping
```

## Experiment Tracking

The configurations support both TensorBoard and Weights & Biases:

```yaml
logger:
  use_tb_logger: true
  wandb:
    project: MRI_OASIS_SuperResolution    # Customize project name
    resume_id: ~
```

View training progress:
```bash
# TensorBoard
tensorboard --logdir experiments/

# Or check wandb.ai if configured
```

## Expected Performance

| Dataset | Architecture | PSNR (dB) | SSIM | Training Time* |
|---------|--------------|-----------|------|----------------|
| OASIS | MSRResNet | 28-35 | 0.88-0.95 | ~10 hours |
| MM-WHS | MSRResNet | 26-32 | 0.85-0.93 | ~12 hours |

*Training time on RTX 3090 GPU

## References

- [OASIS Dataset](https://www.oasis-brains.org/)
- [MM-WHS Challenge](https://zmiclab.github.io/zxh/0/mmwhs/)
- [BasicSR Documentation](https://github.com/XPixelGroup/BasicSR)
- [MRI Dataset Preparation Guide](../../docs/MRI_DatasetPreparation.md)