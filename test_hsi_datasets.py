#!/usr/bin/env python3
"""
Simple test script for HSI dataset classes without full BasicSR dependencies
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import tempfile
import scipy.io as sio

# Copy the dataset classes without registry dependency
import importlib.util

def test_hsi_datasets():
    """Test HSI dataset classes independently"""
    
    print("🧪 Testing HSI Dataset Classes")
    print("=" * 50)
    
    # Create temporary test data
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create some dummy HSI data
        test_data = np.random.rand(32, 32, 31).astype(np.float32)  # 32x32 image with 31 bands
        test_file = os.path.join(temp_dir, "test_hsi.mat")
        sio.savemat(test_file, {'gt': test_data})
        
        print(f"✓ Created test data: {test_data.shape}")
        
        # Test 1: HSI Denoising Dataset
        print("\n1️⃣  Testing HSI Denoising Dataset")
        try:
            # Load the module directly
            spec = importlib.util.spec_from_file_location(
                "hsi_denoising_dataset", 
                "basicsr/data/hsi_denoising_dataset.py"
            )
            hdd_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(hdd_module)
            
            # Test dataset creation
            opt = {
                'dataroot_gt': temp_dir,
                'gt_size': 16,
                'noise_type': 'gaussian',
                'noise_range': [5, 50],
                'add_noise_to_gt': True,
                'use_hflip': False,
                'use_rot': False
            }
            
            dataset = hdd_module.HSIDenoisingDataset(opt)
            print(f"✓ Dataset created with {len(dataset)} samples")
            
            # Test data loading
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"✓ Sample loaded: GT shape {sample['gt'].shape}, LQ shape {sample['lq'].shape}")
                
                # Check that noise was added
                mse = torch.mean((sample['gt'] - sample['lq']) ** 2)
                print(f"✓ MSE between GT and noisy: {mse.item():.6f}")
            
        except Exception as e:
            print(f"✗ HSI Denoising Dataset test failed: {e}")
        
        # Test 2: HSI Inpainting Dataset
        print("\n2️⃣  Testing HSI Inpainting Dataset")
        try:
            # Load the module directly
            spec = importlib.util.spec_from_file_location(
                "hsi_inpainting_dataset", 
                "basicsr/data/hsi_inpainting_dataset.py"
            )
            hid_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(hid_module)
            
            # Test dataset creation
            opt = {
                'dataroot_gt': temp_dir,
                'gt_size': 16,
                'mask_type': 'random_rect',
                'mask_ratio': [0.1, 0.3],
                'generate_mask': True,
                'use_hflip': False,
                'use_rot': False
            }
            
            dataset = hid_module.HSIInpaintingDataset(opt)
            print(f"✓ Dataset created with {len(dataset)} samples")
            
            # Test data loading
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"✓ Sample loaded: GT shape {sample['gt'].shape}, LQ shape {sample['lq'].shape}")
                print(f"✓ Mask shape: {sample['mask'].shape}")
                
                # Check mask properties
                mask_ratio = 1 - sample['mask'].mean().item()
                print(f"✓ Actual mask ratio: {mask_ratio:.3f}")
            
        except Exception as e:
            print(f"✗ HSI Inpainting Dataset test failed: {e}")
        
        print("\n🎉 Testing completed!")

if __name__ == "__main__":
    test_hsi_datasets()