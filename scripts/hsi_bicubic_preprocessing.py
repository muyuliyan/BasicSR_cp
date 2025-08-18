#!/usr/bin/env python3
"""
HSI Bicubic Downsampling Script for BasicSR

This script processes hyperspectral images (HSI) by applying bicubic downsampling
to create LR (Low Resolution) datasets from HR (High Resolution) datasets.

Usage:
    python scripts/hsi_bicubic_preprocessing.py \
        --input_folder datasets/your_hsi_dataset/HR \
        --output_folder datasets/your_hsi_dataset/LR \
        --scale 4 \
        --file_format mat

Supported formats: .mat, .npy
For .mat files, assumes data is stored with key 'data' or 'gt'
For .npy files, assumes data is stored as numpy array with shape [H, W, C]
"""

import argparse
import os
import numpy as np
import scipy.io as sio
from scipy import ndimage
from tqdm import tqdm


def bicubic_downsample_3d(image, scale_factor):
    """
    Bicubic downsampling for 3D hyperspectral image.
    
    Args:
        image (ndarray): Input image with shape [H, W, C]
        scale_factor (int): Downsampling scale factor
        
    Returns:
        ndarray: Downsampled image with shape [H//scale, W//scale, C]
    """
    h, w, c = image.shape
    new_h, new_w = h // scale_factor, w // scale_factor
    
    # Process each spectral band separately
    downsampled = np.zeros((new_h, new_w, c), dtype=image.dtype)
    
    for i in range(c):
        # Use scipy's zoom for bicubic-like interpolation
        downsampled[:, :, i] = ndimage.zoom(
            image[:, :, i], 
            (new_h / h, new_w / w), 
            order=3,  # cubic interpolation
            prefilter=True
        )
    
    return downsampled


def modcrop_3d(image, modulo):
    """
    Crop image to be divisible by modulo.
    
    Args:
        image (ndarray): Input image with shape [H, W, C]
        modulo (int): Modulo value
        
    Returns:
        ndarray: Cropped image
    """
    h, w, c = image.shape
    h_crop = h - (h % modulo)
    w_crop = w - (w % modulo)
    return image[:h_crop, :w_crop, :]


def process_hsi_dataset(input_folder, output_folder, scale, file_format, data_key='data'):
    """
    Process HSI dataset for bicubic downsampling.
    
    Args:
        input_folder (str): Path to input HR folder
        output_folder (str): Path to output LR folder
        scale (int): Downsampling scale factor
        file_format (str): File format ('mat' or 'npy')
        data_key (str): Key for .mat files (default: 'data')
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Get list of files
    if file_format == 'mat':
        file_list = [f for f in os.listdir(input_folder) if f.endswith('.mat')]
    elif file_format == 'npy':
        file_list = [f for f in os.listdir(input_folder) if f.endswith('.npy')]
    else:
        raise ValueError(f"Unsupported file format: {file_format}")
    
    print(f"Found {len(file_list)} {file_format} files to process")
    
    for filename in tqdm(file_list, desc="Processing HSI files"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        try:
            # Load HSI data
            if file_format == 'mat':
                data_dict = sio.loadmat(input_path)
                # Try common keys for HSI data
                if data_key in data_dict:
                    hsi_data = data_dict[data_key]
                elif 'gt' in data_dict:
                    hsi_data = data_dict['gt']
                elif 'data' in data_dict:
                    hsi_data = data_dict['data']
                else:
                    # Use the first non-metadata key
                    keys = [k for k in data_dict.keys() if not k.startswith('__')]
                    if keys:
                        hsi_data = data_dict[keys[0]]
                        print(f"Warning: Using key '{keys[0]}' for {filename}")
                    else:
                        raise ValueError(f"No valid data key found in {filename}")
            else:  # npy
                hsi_data = np.load(input_path)
            
            # Ensure data is in HWC format
            if len(hsi_data.shape) != 3:
                raise ValueError(f"Expected 3D data (H, W, C), got shape {hsi_data.shape}")
            
            # Convert to float for processing
            hsi_data = hsi_data.astype(np.float32)
            
            # Modcrop to ensure divisible by scale
            hsi_data = modcrop_3d(hsi_data, scale)
            
            # Bicubic downsampling
            lr_data = bicubic_downsample_3d(hsi_data, scale)
            
            # Save LR data
            if file_format == 'mat':
                sio.savemat(output_path, {'lq': lr_data})
            else:  # npy
                np.save(output_path, lr_data)
                
            print(f"Processed {filename}: {hsi_data.shape} -> {lr_data.shape}")
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue


def main():
    parser = argparse.ArgumentParser(description='HSI Bicubic Downsampling for BasicSR')
    parser.add_argument('--input_folder', type=str, required=True,
                        help='Path to input HR folder containing HSI files')
    parser.add_argument('--output_folder', type=str, required=True,
                        help='Path to output LR folder')
    parser.add_argument('--scale', type=int, default=4,
                        help='Downsampling scale factor (default: 4)')
    parser.add_argument('--file_format', type=str, choices=['mat', 'npy'], default='mat',
                        help='File format: mat or npy (default: mat)')
    parser.add_argument('--data_key', type=str, default='data',
                        help='Key for .mat files (default: data)')
    
    args = parser.parse_args()
    
    print(f"HSI Bicubic Downsampling Configuration:")
    print(f"  Input folder: {args.input_folder}")
    print(f"  Output folder: {args.output_folder}")
    print(f"  Scale factor: {args.scale}")
    print(f"  File format: {args.file_format}")
    if args.file_format == 'mat':
        print(f"  Data key: {args.data_key}")
    print()
    
    # Validate input folder
    if not os.path.exists(args.input_folder):
        raise ValueError(f"Input folder does not exist: {args.input_folder}")
    
    # Process dataset
    process_hsi_dataset(
        args.input_folder,
        args.output_folder,
        args.scale,
        args.file_format,
        args.data_key
    )
    
    print("HSI bicubic downsampling completed successfully!")


if __name__ == '__main__':
    main()