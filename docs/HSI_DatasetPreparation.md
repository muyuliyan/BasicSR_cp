# HSI Dataset Preparation Guide

[中文版本](HSI_DatasetPreparation_CN.md)

This guide provides instructions for preparing hyperspectral image (HSI) datasets for super-resolution experiments using BasicSR.

## Dataset Structure

Organize your HSI datasets in the following structure:

```
datasets/
├── your_hsi_dataset/
│   ├── HR/                 # High-resolution images
│   │   ├── image001.mat    # or .npy
│   │   ├── image002.mat
│   │   └── ...
│   ├── LR/                 # Low-resolution images (generated)
│   │   ├── image001.mat    # or .npy
│   │   ├── image002.mat
│   │   └── ...
│   ├── val/               # Validation set (optional)
│   │   ├── HR/
│   │   └── LR/
│   └── test/              # Test set
│       ├── HR/
│       └── LR/
```

## Supported File Formats

### MATLAB Files (.mat)
- Data should be stored with key `'gt'`, `'data'`, or any custom key
- Shape: `[H, W, C]` where C is the number of spectral bands
- Example loading in MATLAB: `data = load('image.mat'); hsi = data.gt;`

### NumPy Files (.npy)
- Direct numpy array storage
- Shape: `[H, W, C]` where C is the number of spectral bands
- Example loading in Python: `hsi = np.load('image.npy')`

## Data Preprocessing

### Step 1: Prepare High-Resolution (HR) Data

Place your original HSI data in the `HR` folder. Ensure all images have the same number of spectral bands.

### Step 2: Generate Low-Resolution (LR) Data

Use the provided preprocessing script to generate LR data through bicubic downsampling:

```bash
# For MATLAB files
python scripts/hsi_bicubic_preprocessing.py \
    --input_folder datasets/your_hsi_dataset/HR \
    --output_folder datasets/your_hsi_dataset/LR \
    --scale 4 \
    --file_format mat \
    --data_key gt

# For NumPy files
python scripts/hsi_bicubic_preprocessing.py \
    --input_folder datasets/your_hsi_dataset/HR \
    --output_folder datasets/your_hsi_dataset/LR \
    --scale 4 \
    --file_format npy
```

Parameters:
- `--input_folder`: Path to HR images
- `--output_folder`: Path to save LR images
- `--scale`: Downsampling factor (typically 2, 3, or 4)
- `--file_format`: File format (`mat` or `npy`)
- `--data_key`: Key for MATLAB files (default: `data`)

### Step 3: Verify Data

After preprocessing, verify that:
1. HR and LR folders contain the same number of files
2. File names match between HR and LR
3. LR image dimensions are HR dimensions divided by scale factor
4. All images have the same number of spectral bands

## Configuration

### Training Configuration

Modify the training configuration file `options/train/HSI/train_HSI_SRResNet_x4.yml`:

1. **Update dataset paths**:
   ```yaml
   datasets:
     train:
       dataroot_gt: datasets/your_hsi_dataset/HR
       dataroot_lq: datasets/your_hsi_dataset/LR
   ```

2. **Set spectral channels**:
   ```yaml
   network_g:
     num_in_ch: 31   # Replace with your number of spectral bands
     num_out_ch: 31  # Should match num_in_ch
   ```

3. **Adjust batch size and patch size** based on your GPU memory:
   ```yaml
   datasets:
     train:
       gt_size: 64           # Patch size
       batch_size_per_gpu: 8 # Batch size
   ```

### Testing Configuration

Modify the testing configuration file `options/test/HSI/test_HSI_SRResNet_x4.yml`:

1. **Update dataset paths**:
   ```yaml
   datasets:
     test_1:
       dataroot_gt: datasets/your_hsi_dataset/test/HR
       dataroot_lq: datasets/your_hsi_dataset/test/LR
   ```

2. **Set model path**:
   ```yaml
   path:
     pretrain_network_g: experiments/pretrained_models/your_model.pth
   ```

## Common HSI Datasets

### CAVE Dataset
- **Spectral bands**: 31 (400-700nm)
- **Spatial resolution**: Various (mainly 512×512)
- **Download**: [Columbia CAVE dataset](http://www1.cs.columbia.edu/CAVE/databases/multispectral/)

### Harvard Dataset  
- **Spectral bands**: 31 (400-700nm)
- **Spatial resolution**: Various
- **Download**: [Harvard dataset](http://vision.seas.harvard.edu/hyperspec/explore.html)

### ICVL Dataset
- **Spectral bands**: 31 (400-700nm) 
- **Spatial resolution**: Various
- **Download**: [ICVL dataset](https://icvl.cs.bgu.ac.il/)

### Pavia University/Centre
- **Spectral bands**: 103/102
- **Spatial resolution**: 610×340 / 1096×715
- **Download**: [Grupo de Inteligencia Computacional](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes)

## Training

Start training with the following command:

```bash
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python basicsr/train.py \
    -opt options/train/HSI/train_HSI_SRResNet_x4.yml \
    --auto_resume
```

## Testing

Test the trained model:

```bash
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python basicsr/test.py \
    -opt options/test/HSI/test_HSI_SRResNet_x4.yml
```

## Metrics

The HSI configuration includes five key metrics for hyperspectral image evaluation:

1. **PSNR** (Peak Signal-to-Noise Ratio): Overall image quality
2. **SSIM** (Structural Similarity Index): Structural similarity
3. **SAM** (Spectral Angle Mapper): Spectral similarity (HSI-specific)
4. **ERGAS** (Erreur Relative Globale Adimensionnelle de Synthèse): Global relative error (HSI-specific)
5. **RMSE** (Root Mean Square Error): Pixel-wise error

## Tips and Best Practices

1. **Memory Management**: HSI data requires more memory. Reduce batch size and patch size if you encounter out-of-memory errors.

2. **Spectral Normalization**: Consider normalizing spectral values to [0, 1] range for better training stability.

3. **Data Augmentation**: The HSI dataset supports horizontal flipping and rotation. Disable if spectral order matters.

4. **Validation**: Use a separate validation set to monitor training progress and prevent overfitting.

5. **Model Selection**: Start with SRResNet architecture and experiment with other models like RCAN or EDSR for better performance.

## Troubleshooting

### Common Issues

1. **Shape mismatch errors**: Ensure all HSI images have the same number of spectral bands
2. **File format errors**: Verify file format and data keys for .mat files
3. **Memory errors**: Reduce batch size and patch size
4. **Path errors**: Check that all dataset paths in configuration files are correct

### Debug Commands

```bash
# Check HSI data shape and format
python -c "
import scipy.io as sio
import numpy as np
data = sio.loadmat('path/to/your/file.mat')
print('Keys:', list(data.keys()))
print('Shape:', data['your_key'].shape)
print('Data type:', data['your_key'].dtype)
"
```