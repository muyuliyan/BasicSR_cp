# MRI Dataset Preparation Guide

This guide provides instructions for preparing MRI datasets for super-resolution experiments using BasicSR, specifically for OASIS and MM-WHS datasets.

## Supported MRI Datasets

### OASIS Dataset
- **Type**: Brain MRI
- **Format**: Primarily T1-weighted images
- **Spatial resolution**: Various (typically 1mm³ isotropic)
- **Download**: [OASIS-1](https://www.oasis-brains.org/#data) and [OASIS-3](https://www.oasis-brains.org/files/oasis_longitudinal.pdf)

### MM-WHS Dataset
- **Type**: Cardiac MRI
- **Format**: Multi-modal cardiac images
- **Applications**: Whole heart segmentation
- **Download**: [MM-WHS Challenge](https://zmiclab.github.io/zxh/0/mmwhs/)

## Dataset Structure

Organize your MRI datasets in the following structure:

```
datasets/
├── OASIS/
│   ├── train/
│   │   ├── HR/                 # High-resolution images
│   │   │   ├── subject001.nii  # or .nii.gz, .npy, .mat
│   │   │   ├── subject002.nii
│   │   │   └── ...
│   │   └── LR/                 # Low-resolution images (generated)
│   │       ├── subject001.nii
│   │       ├── subject002.nii
│   │       └── ...
│   ├── val/                    # Validation set
│   │   ├── HR/
│   │   └── LR/
│   └── test/                   # Test set
│       ├── HR/
│       └── LR/
└── MM-WHS/
    ├── train/
    │   ├── HR/
    │   └── LR/
    ├── val/
    │   ├── HR/
    │   └── LR/
    └── test/
        ├── HR/
        └── LR/
```

## Supported File Formats

### NIfTI Files (.nii, .nii.gz)
- Standard medical imaging format
- Requires `nibabel` package: `pip install nibabel`
- Supports both compressed (.nii.gz) and uncompressed (.nii) formats
- Preserves spatial orientation and metadata

### NumPy Files (.npy)
- Direct numpy array storage
- Shape: `[H, W]` for 2D or `[H, W, D]` for 3D
- Example loading: `volume = np.load('image.npy')`

### MATLAB Files (.mat)
- Requires `scipy` package (included in requirements)
- Data should be stored with keys: `'data'`, `'image'`, `'volume'`, etc.
- Example loading in MATLAB: `data = load('image.mat'); mri = data.image;`

## Data Preprocessing

### Step 1: Install Dependencies

```bash
# Basic requirements
pip install -r requirements.txt

# For NIfTI support
pip install nibabel

# For DICOM support (optional)
pip install pydicom
```

### Step 2: Prepare High-Resolution Data

Place your original MRI volumes in the `HR` folders. Ensure consistent formatting:

- **OASIS**: T1-weighted brain MRI volumes
- **MM-WHS**: Cardiac MRI volumes (T2-weighted typically)

### Step 3: Generate Low-Resolution Data

Use the MRI data preparation script:

```bash
# For OASIS dataset
python scripts/mri_data_preparation.py \
    --input datasets/OASIS/train/HR \
    --output datasets/OASIS/train/LR \
    --scale 4 \
    --extract-2d

# For MM-WHS dataset
python scripts/mri_data_preparation.py \
    --input datasets/MM-WHS/train/HR \
    --output datasets/MM-WHS/train/LR \
    --scale 4 \
    --extract-2d
```

### Step 4: Data Normalization

The MRI datasets automatically apply robust normalization:

- **Percentile clipping**: Removes outliers using configurable percentiles
- **[0,1] normalization**: Scales intensity values to 0-1 range
- **OASIS**: Uses [2, 98] percentiles (conservative for brain MRI)
- **MM-WHS**: Uses [1, 99] percentiles (wider range for cardiac MRI)

## Configuration

### Training Configuration

Modify the training configuration files:

#### OASIS Configuration (`train_OASIS_SRResNet_x4.yml`):
```yaml
datasets:
  train:
    type: OASISDataset
    dataroot_gt: datasets/OASIS/train/HR
    dataroot_lq: datasets/OASIS/train/LR
    normalize_to_01: true
    clip_percentiles: [2, 98]  # Conservative for brain MRI
```

#### MM-WHS Configuration (`train_MMWHS_SRResNet_x4.yml`):
```yaml
datasets:
  train:
    type: MMWHSDataset
    dataroot_gt: datasets/MM-WHS/train/HR
    dataroot_lq: datasets/MM-WHS/train/LR
    normalize_to_01: true
    clip_percentiles: [1, 99]  # Wider range for cardiac MRI
```

### Network Architecture

The configurations use single-channel networks optimized for MRI:

```yaml
network_g:
  type: MSRResNet
  num_in_ch: 1   # Single-channel grayscale MRI
  num_out_ch: 1  # Single-channel output
  num_feat: 64
  num_block: 16
  upscale: 4
```

## Training

### OASIS Dataset Training
```bash
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python basicsr/train.py \
    -opt options/train/MRI/train_OASIS_SRResNet_x4.yml --auto_resume
```

### MM-WHS Dataset Training
```bash
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python basicsr/train.py \
    -opt options/train/MRI/train_MMWHS_SRResNet_x4.yml --auto_resume
```

## Testing

### Test OASIS Model
```bash
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python basicsr/test.py \
    -opt options/test/MRI/test_OASIS_SRResNet_x4.yml
```

### Test MM-WHS Model
```bash
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python basicsr/test.py \
    -opt options/test/MRI/test_MMWHS_SRResNet_x4.yml
```

## Evaluation Metrics

The framework provides standard image quality metrics suitable for MRI:

| Metric | Description | Better | Typical Range |
|--------|-------------|--------|---------------|
| **PSNR** | Peak Signal-to-Noise Ratio | Higher ↑ | 25-35 dB |
| **SSIM** | Structural Similarity Index | Higher ↑ | 0.85-0.95 |

## Tips and Best Practices

### 1. **Memory Management**
- MRI volumes can be large. Use `extract_2d_slices=True` for 2D training
- Adjust `batch_size_per_gpu` based on available GPU memory
- Consider using smaller `gt_size` (patch size) for memory efficiency

### 2. **Data Quality**
- Ensure consistent image orientation across the dataset
- Check for artifacts or motion corruption in original volumes
- Consider skull-stripping for brain MRI (OASIS) if needed

### 3. **Data Augmentation**
- Horizontal flipping is generally safe for MRI
- Rotation may not preserve anatomical orientation - use carefully
- Consider anatomy-specific augmentations for cardiac MRI

### 4. **Preprocessing Options**
- `slice_axis`: Choose appropriate axis for 2D extraction (0=sagittal, 1=coronal, 2=axial)
- `clip_percentiles`: Adjust based on image characteristics and contrast
- `normalize_to_01`: Recommended for stable training

### 5. **Validation Strategy**
- Use separate patients/subjects for train/val/test splits
- Monitor validation metrics to prevent overfitting
- Save validation images for visual inspection

### 6. **Architecture Selection**
- Start with MSRResNet for baseline results
- Consider RCAN or EDSR for potentially better performance
- Single-channel networks are sufficient for grayscale MRI

## Troubleshooting

### Common Issues

1. **File Format Errors**
   - Ensure `nibabel` is installed for NIfTI files
   - Check file integrity and format consistency

2. **Memory Issues**
   - Reduce `batch_size_per_gpu`
   - Use smaller `gt_size` 
   - Enable `extract_2d_slices` for 3D volumes

3. **Normalization Problems**
   - Check intensity ranges in your data
   - Adjust `clip_percentiles` if needed
   - Verify data loading with simple tests

4. **Poor Results**
   - Ensure proper train/val/test split
   - Check data quality and preprocessing
   - Consider domain-specific preprocessing

## Example Dataset Preparation Workflow

```bash
# 1. Download OASIS dataset
# Follow instructions from https://www.oasis-brains.org/

# 2. Organize data structure
mkdir -p datasets/OASIS/{train,val,test}/{HR,LR}

# 3. Copy high-resolution volumes to HR folders
# (Manual step - copy your .nii/.nii.gz files)

# 4. Generate low-resolution data
python scripts/mri_data_preparation.py \
    --input datasets/OASIS/train/HR \
    --output datasets/OASIS/train/LR \
    --scale 4 --extract-2d

# 5. Update configuration paths
# Edit options/train/MRI/train_OASIS_SRResNet_x4.yml

# 6. Start training
PYTHONPATH="./:${PYTHONPATH}" python basicsr/train.py \
    -opt options/train/MRI/train_OASIS_SRResNet_x4.yml --auto_resume
```

## References

- [OASIS Dataset](https://www.oasis-brains.org/)
- [MM-WHS Challenge](https://zmiclab.github.io/zxh/0/mmwhs/)
- [BasicSR Documentation](https://github.com/XPixelGroup/BasicSR)
- [NiBabel Documentation](https://nipy.org/nibabel/)