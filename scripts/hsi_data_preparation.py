#!/usr/bin/env python3
"""
HSI Data Preparation Script for Denoising and Inpainting

This script helps prepare HSI datasets for denoising and inpainting tasks.
"""

import os
import argparse
import numpy as np
import scipy.io as sio
from pathlib import Path
import random


def add_noise_to_hsi(clean_data, noise_type='gaussian', noise_level=25):
    """Add noise to clean HSI data"""
    if noise_type == 'gaussian':
        noise = np.random.normal(0, noise_level/255.0, clean_data.shape)
        noisy_data = clean_data + noise
    elif noise_type == 'poisson':
        # Convert to 0-255 range, apply Poisson noise, then back to 0-1
        clean_255 = clean_data * 255
        noisy_255 = np.random.poisson(clean_255)
        noisy_data = noisy_255 / 255.0
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")
    
    return np.clip(noisy_data, 0, 1)


def create_mask(height, width, mask_type='random_rect', mask_ratio=0.2):
    """Create a mask for inpainting"""
    mask = np.ones((height, width), dtype=np.float32)
    
    if mask_type == 'random_rect':
        mask_h = int(height * np.sqrt(mask_ratio))
        mask_w = int(width * np.sqrt(mask_ratio))
        
        start_h = random.randint(0, height - mask_h)
        start_w = random.randint(0, width - mask_w)
        
        mask[start_h:start_h+mask_h, start_w:start_w+mask_w] = 0
        
    elif mask_type == 'random_irregular':
        # Create irregular mask using random walks
        num_walks = random.randint(5, 15)
        
        for _ in range(num_walks):
            start_h = random.randint(0, height-1)
            start_w = random.randint(0, width-1)
            walk_length = random.randint(10, min(height, width) // 4)
            
            curr_h, curr_w = start_h, start_w
            for _ in range(walk_length):
                dh = random.randint(-1, 1)
                dw = random.randint(-1, 1)
                
                curr_h = max(0, min(height-1, curr_h + dh))
                curr_w = max(0, min(width-1, curr_w + dw))
                
                region_size = random.randint(1, 3)
                for i in range(max(0, curr_h - region_size), min(height, curr_h + region_size + 1)):
                    for j in range(max(0, curr_w - region_size), min(width, curr_w + region_size + 1)):
                        mask[i, j] = 0
    
    return mask


def apply_mask_to_hsi(complete_data, mask):
    """Apply mask to HSI data for inpainting"""
    masked_data = complete_data.copy()
    
    # Apply mask to all spectral bands
    for i in range(complete_data.shape[2]):
        masked_data[:, :, i] = complete_data[:, :, i] * mask
    
    return masked_data


def prepare_denoising_data(input_dir, output_dir, noise_type='gaussian', noise_levels=[10, 25, 50]):
    """Prepare data for HSI denoising"""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    clean_dir = output_path / 'clean'
    noisy_dir = output_path / 'noisy'
    clean_dir.mkdir(parents=True, exist_ok=True)
    noisy_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Preparing denoising data from {input_dir} to {output_dir}")
    print(f"Noise type: {noise_type}, Noise levels: {noise_levels}")
    
    # Process each file
    for file_path in input_path.glob('*.mat'):
        print(f"Processing {file_path.name}...")
        
        # Load clean data
        data = sio.loadmat(str(file_path))
        clean_hsi = data['gt'] if 'gt' in data else data[list(data.keys())[0]]
        
        # Normalize to [0, 1] if needed
        if clean_hsi.max() > 1.1:
            clean_hsi = clean_hsi / 255.0
        
        # Save clean data
        clean_output = clean_dir / file_path.name
        sio.savemat(str(clean_output), {'gt': clean_hsi})
        
        # Generate noisy versions
        for noise_level in noise_levels:
            noisy_hsi = add_noise_to_hsi(clean_hsi, noise_type, noise_level)
            
            # Save noisy data
            noisy_filename = f"{file_path.stem}_noise{noise_level}{file_path.suffix}"
            noisy_output = noisy_dir / noisy_filename
            sio.savemat(str(noisy_output), {'lq': noisy_hsi})
    
    print(f"✓ Denoising data preparation completed!")


def prepare_inpainting_data(input_dir, output_dir, mask_type='random_rect', mask_ratios=[0.1, 0.2, 0.3]):
    """Prepare data for HSI inpainting"""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    complete_dir = output_path / 'complete'
    masked_dir = output_path / 'masked'
    masks_dir = output_path / 'masks'
    complete_dir.mkdir(parents=True, exist_ok=True)
    masked_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Preparing inpainting data from {input_dir} to {output_dir}")
    print(f"Mask type: {mask_type}, Mask ratios: {mask_ratios}")
    
    # Process each file
    for file_path in input_path.glob('*.mat'):
        print(f"Processing {file_path.name}...")
        
        # Load complete data
        data = sio.loadmat(str(file_path))
        complete_hsi = data['gt'] if 'gt' in data else data[list(data.keys())[0]]
        
        # Normalize to [0, 1] if needed
        if complete_hsi.max() > 1.1:
            complete_hsi = complete_hsi / 255.0
        
        # Save complete data
        complete_output = complete_dir / file_path.name
        sio.savemat(str(complete_output), {'gt': complete_hsi})
        
        # Generate masked versions
        height, width, _ = complete_hsi.shape
        
        for mask_ratio in mask_ratios:
            mask = create_mask(height, width, mask_type, mask_ratio)
            masked_hsi = apply_mask_to_hsi(complete_hsi, mask)
            
            # Save masked data and mask
            masked_filename = f"{file_path.stem}_mask{int(mask_ratio*100)}{file_path.suffix}"
            mask_filename = f"{file_path.stem}_mask{int(mask_ratio*100)}_mask{file_path.suffix}"
            
            masked_output = masked_dir / masked_filename
            mask_output = masks_dir / mask_filename
            
            sio.savemat(str(masked_output), {'lq': masked_hsi})
            sio.savemat(str(mask_output), {'mask': mask})
    
    print(f"✓ Inpainting data preparation completed!")


def main():
    parser = argparse.ArgumentParser(description='HSI Data Preparation for Denoising and Inpainting')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory with HSI .mat files')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--task', type=str, choices=['denoising', 'inpainting'], required=True, help='Task type')
    
    # Denoising options
    parser.add_argument('--noise_type', type=str, default='gaussian', choices=['gaussian', 'poisson'], 
                       help='Type of noise for denoising')
    parser.add_argument('--noise_levels', type=int, nargs='+', default=[10, 25, 50], 
                       help='Noise levels for denoising')
    
    # Inpainting options
    parser.add_argument('--mask_type', type=str, default='random_rect', choices=['random_rect', 'random_irregular'],
                       help='Type of mask for inpainting')
    parser.add_argument('--mask_ratios', type=float, nargs='+', default=[0.1, 0.2, 0.3],
                       help='Mask ratios for inpainting')
    
    args = parser.parse_args()
    
    if args.task == 'denoising':
        prepare_denoising_data(args.input_dir, args.output_dir, args.noise_type, args.noise_levels)
    elif args.task == 'inpainting':
        prepare_inpainting_data(args.input_dir, args.output_dir, args.mask_type, args.mask_ratios)


if __name__ == '__main__':
    main()