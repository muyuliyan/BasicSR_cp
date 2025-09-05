#!/usr/bin/env python3
"""
MRI Data Preparation Script for BasicSR

This script prepares MRI data for super-resolution training by creating low-resolution 
versions from high-resolution MRI volumes using bicubic downsampling.

Supports:
- NIfTI format (.nii, .nii.gz)
- NumPy format (.npy) 
- MATLAB format (.mat)

For OASIS and MM-WHS datasets.
"""

import argparse
import os
import numpy as np
from tqdm import tqdm
import cv2

# Optional imports for medical formats
try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False

try:
    import scipy.io as sio
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def bicubic_downsample_2d(img, scale):
    """Bicubic downsampling for 2D images."""
    h, w = img.shape[:2]
    new_h, new_w = h // scale, w // scale
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def bicubic_downsample_3d(volume, scale):
    """Bicubic downsampling for 3D volumes slice by slice."""
    if len(volume.shape) == 2:
        return bicubic_downsample_2d(volume, scale)
    
    h, w, d = volume.shape
    new_h, new_w = h // scale, w // scale
    downsampled = np.zeros((new_h, new_w, d), dtype=volume.dtype)
    
    for i in range(d):
        downsampled[:, :, i] = bicubic_downsample_2d(volume[:, :, i], scale)
    
    return downsampled


def modcrop_2d(img, scale):
    """Crop image to be divisible by scale."""
    h, w = img.shape[:2]
    h = h - h % scale
    w = w - w % scale
    return img[:h, :w]


def modcrop_3d(volume, scale):
    """Crop 3D volume to be divisible by scale."""
    if len(volume.shape) == 2:
        return modcrop_2d(volume, scale)
    
    h, w, d = volume.shape
    h = h - h % scale
    w = w - w % scale
    return volume[:h, :w, :]


def load_mri_data(filepath, data_key='data'):
    """Load MRI data from various formats."""
    if filepath.endswith(('.nii', '.nii.gz')):
        if not NIBABEL_AVAILABLE:
            raise ImportError("nibabel is required for NIfTI format. Install with: pip install nibabel")
        img = nib.load(filepath)
        return img.get_fdata()
    
    elif filepath.endswith('.npy'):
        return np.load(filepath)
    
    elif filepath.endswith('.mat'):
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is required for .mat format")
        mat_data = sio.loadmat(filepath)
        # Try common keys for MRI data
        possible_keys = [data_key, 'data', 'image', 'volume', 'mri']
        for key in possible_keys:
            if key in mat_data:
                return mat_data[key]
        
        # Use the first non-metadata key
        keys = [k for k in mat_data.keys() if not k.startswith('__')]
        if keys:
            print(f"Warning: Using key '{keys[0]}' for {filepath}")
            return mat_data[keys[0]]
        else:
            raise ValueError(f"No valid data key found in {filepath}")
    
    else:
        raise ValueError(f"Unsupported file format: {filepath}")


def save_mri_data(data, filepath, original_format):
    """Save MRI data in the same format as input."""
    if original_format in ['.nii', '.nii.gz']:
        if not NIBABEL_AVAILABLE:
            raise ImportError("nibabel is required for NIfTI format")
        img = nib.Nifti1Image(data, np.eye(4))
        nib.save(img, filepath)
    
    elif original_format == '.npy':
        np.save(filepath, data)
    
    elif original_format == '.mat':
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is required for .mat format")
        sio.savemat(filepath, {'lq': data})


def process_mri_dataset(input_folder, output_folder, scale, data_key='data', extract_2d=False):
    """Process MRI dataset for super-resolution."""
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Get list of supported files
    supported_extensions = ['.nii', '.nii.gz', '.npy', '.mat']
    file_list = [f for f in os.listdir(input_folder) 
                 if any(f.endswith(ext) for ext in supported_extensions)]
    
    if not file_list:
        print(f"No supported MRI files found in {input_folder}")
        return
    
    print(f"Processing {len(file_list)} files...")
    print(f"Scale factor: {scale}")
    print(f"Extract 2D slices: {extract_2d}")
    
    for filename in tqdm(file_list, desc="Processing MRI files"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        try:
            # Load MRI data
            mri_data = load_mri_data(input_path, data_key)
            
            # Convert to float for processing
            mri_data = mri_data.astype(np.float32)
            
            # Extract 2D slices if requested and data is 3D
            if extract_2d and len(mri_data.shape) == 3:
                # Extract middle slice along the last axis
                middle_idx = mri_data.shape[2] // 2
                mri_data = mri_data[:, :, middle_idx]
            
            # Modcrop to ensure divisible by scale
            if len(mri_data.shape) == 3:
                mri_data = modcrop_3d(mri_data, scale)
                lr_data = bicubic_downsample_3d(mri_data, scale)
            else:
                mri_data = modcrop_2d(mri_data, scale)
                lr_data = bicubic_downsample_2d(mri_data, scale)
            
            # Determine original format
            if filename.endswith('.nii.gz'):
                original_format = '.nii.gz'
            elif filename.endswith('.nii'):
                original_format = '.nii'
            elif filename.endswith('.npy'):
                original_format = '.npy'
            elif filename.endswith('.mat'):
                original_format = '.mat'
            
            # Save LR data
            save_mri_data(lr_data, output_path, original_format)
            
            print(f"Processed {filename}: {mri_data.shape} -> {lr_data.shape}")
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue


def main():
    parser = argparse.ArgumentParser(description='MRI Data Preparation for BasicSR')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input folder containing high-resolution MRI data')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output folder for low-resolution MRI data')
    parser.add_argument('--scale', '-s', type=int, default=4,
                        help='Downsampling scale factor (default: 4)')
    parser.add_argument('--data-key', '-k', type=str, default='data',
                        help='Key name for .mat files (default: data)')
    parser.add_argument('--extract-2d', action='store_true',
                        help='Extract 2D slices from 3D volumes')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.input):
        print(f"Error: Input folder '{args.input}' does not exist")
        return
    
    if args.scale <= 1:
        print(f"Error: Scale factor must be > 1, got {args.scale}")
        return
    
    print("MRI Data Preparation for BasicSR")
    print("=" * 40)
    print(f"Input folder: {args.input}")
    print(f"Output folder: {args.output}")
    print(f"Scale factor: {args.scale}")
    print(f"Data key (for .mat): {args.data_key}")
    print(f"Extract 2D slices: {args.extract_2d}")
    print()
    
    # Check dependencies
    if not NIBABEL_AVAILABLE:
        print("Warning: nibabel not available. NIfTI files (.nii/.nii.gz) will not be supported.")
    if not SCIPY_AVAILABLE:
        print("Warning: scipy not available. MATLAB files (.mat) will not be supported.")
    print()
    
    # Process dataset
    process_mri_dataset(
        args.input, 
        args.output, 
        args.scale, 
        args.data_key, 
        args.extract_2d
    )
    
    print("Processing complete!")


if __name__ == '__main__':
    main()