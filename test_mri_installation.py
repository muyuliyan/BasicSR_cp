#!/usr/bin/env python3
"""
Test script to verify MRI BasicSR installation and functionality.

This script performs basic checks to ensure the MRI-focused BasicSR
implementation is working correctly.
"""

import os
import sys
import numpy as np
import tempfile
import shutil


def test_imports():
    """Test if all required modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        import basicsr
        print("  ✅ BasicSR imported successfully")
    except ImportError as e:
        print(f"  ❌ BasicSR import failed: {e}")
        return False
    
    try:
        from basicsr.data.mri_dataset import MRIDataset, OASISDataset, MMWHSDataset
        print("  ✅ MRI dataset classes imported successfully")
    except ImportError as e:
        print(f"  ❌ MRI dataset import failed: {e}")
        return False
    
    try:
        from basicsr.utils.registry import DATASET_REGISTRY
        mri_datasets = [name for name in DATASET_REGISTRY._obj_map.keys() 
                       if any(x in name for x in ['MRI', 'OASIS', 'MMWHS'])]
        print(f"  ✅ MRI datasets registered: {mri_datasets}")
    except ImportError as e:
        print(f"  ❌ Registry import failed: {e}")
        return False
    
    return True


def test_configs():
    """Test if configuration files are valid."""
    print("\n🔍 Testing configuration files...")
    
    try:
        import yaml
        
        configs = [
            'options/train/MRI/train_OASIS_SRResNet_x4.yml',
            'options/train/MRI/train_MMWHS_SRResNet_x4.yml',
            'options/test/MRI/test_OASIS_SRResNet_x4.yml',
            'options/test/MRI/test_MMWHS_SRResNet_x4.yml'
        ]
        
        for config_path in configs:
            if not os.path.exists(config_path):
                print(f"  ❌ Config file missing: {config_path}")
                return False
                
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Verify single-channel configuration
            if config['network_g']['num_in_ch'] != 1:
                print(f"  ❌ {config_path}: Expected 1 input channel, got {config['network_g']['num_in_ch']}")
                return False
                
            if config['network_g']['num_out_ch'] != 1:
                print(f"  ❌ {config_path}: Expected 1 output channel, got {config['network_g']['num_out_ch']}")
                return False
        
        print("  ✅ All configuration files valid")
        return True
        
    except Exception as e:
        print(f"  ❌ Config test failed: {e}")
        return False


def test_dataset_functionality():
    """Test dataset functionality with mock data."""
    print("\n🔍 Testing dataset functionality...")
    
    try:
        from basicsr.data.mri_dataset import OASISDataset
        
        # Create temporary directory with mock data
        with tempfile.TemporaryDirectory() as tmp_dir:
            hr_dir = os.path.join(tmp_dir, 'HR')
            lr_dir = os.path.join(tmp_dir, 'LR')
            os.makedirs(hr_dir)
            os.makedirs(lr_dir)
            
            # Create mock MRI data
            mock_hr = np.random.randn(128, 128).astype(np.float32)
            mock_lr = np.random.randn(64, 64).astype(np.float32)
            
            np.save(os.path.join(hr_dir, 'test.npy'), mock_hr)
            np.save(os.path.join(lr_dir, 'test.npy'), mock_lr)
            
            # Test dataset loading
            config = {
                'dataroot_gt': hr_dir,
                'dataroot_lq': lr_dir,
                'normalize_to_01': True,
                'clip_percentiles': [2, 98],
                'extract_2d_slices': True,
                'gt_size': 64,
                'use_hflip': True,
                'use_rot': True,
                'phase': 'train'
            }
            
            dataset = OASISDataset(config)
            
            if len(dataset) == 0:
                print("  ❌ Dataset is empty")
                return False
            
            # Test sample loading
            sample = dataset[0]
            
            if 'lq' not in sample or 'gt' not in sample:
                print("  ❌ Sample missing required keys")
                return False
            
            # Check tensor shapes
            lq_shape = sample['lq'].shape
            gt_shape = sample['gt'].shape
            
            if len(lq_shape) != 3 or lq_shape[0] != 1:
                print(f"  ❌ LQ tensor shape invalid: {lq_shape}, expected [1, H, W]")
                return False
                
            if len(gt_shape) != 3 or gt_shape[0] != 1:
                print(f"  ❌ GT tensor shape invalid: {gt_shape}, expected [1, H, W]")
                return False
        
        print(f"  ✅ Dataset test passed - LQ: {lq_shape}, GT: {gt_shape}")
        return True
        
    except Exception as e:
        print(f"  ❌ Dataset test failed: {e}")
        return False


def test_preprocessing_script():
    """Test MRI preprocessing script."""
    print("\n🔍 Testing preprocessing script...")
    
    try:
        import subprocess
        
        # Test help command
        result = subprocess.run(['python', 'scripts/mri_data_preparation.py', '--help'], 
                              capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"  ❌ Script failed to run: {result.stderr}")
            return False
            
        if 'MRI Data Preparation' not in result.stdout:
            print("  ❌ Script output doesn't contain expected text")
            return False
            
        print("  ✅ Preprocessing script is working")
        return True
        
    except Exception as e:
        print(f"  ❌ Preprocessing test failed: {e}")
        return False


def check_optional_dependencies():
    """Check optional dependencies and provide warnings."""
    print("\n🔍 Checking optional dependencies...")
    
    warnings = []
    
    try:
        import nibabel
        print("  ✅ nibabel available - NIfTI format supported")
    except ImportError:
        warnings.append("nibabel not available - NIfTI format (.nii/.nii.gz) not supported")
    
    try:
        import pydicom
        print("  ✅ pydicom available - DICOM format supported")
    except ImportError:
        warnings.append("pydicom not available - DICOM format not supported")
    
    try:
        import scipy.io
        print("  ✅ scipy available - MATLAB format supported")
    except ImportError:
        warnings.append("scipy not available - MATLAB format (.mat) not supported")
    
    if warnings:
        print("\n⚠️  Warnings:")
        for warning in warnings:
            print(f"  • {warning}")
        print("\nTo install optional dependencies:")
        print("  pip install nibabel pydicom")
    
    return warnings


def main():
    """Run all tests."""
    print("🏥 MRI BasicSR Installation Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_configs,
        test_dataset_functionality,
        test_preprocessing_script
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ❌ Test {test_func.__name__} crashed: {e}")
            failed += 1
    
    # Check optional dependencies (warnings only)
    warnings = check_optional_dependencies()
    
    print(f"\n📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("\n🎉 All tests passed! MRI BasicSR is ready to use.")
        print("\n📖 Next steps:")
        print("  1. Prepare your MRI dataset (see docs/MRI_DatasetPreparation.md)")
        print("  2. Follow the quick start guide (MRI_QuickStart.md)")
        print("  3. Start training your models!")
        
        if warnings:
            print(f"\n⚠️  Note: {len(warnings)} optional dependencies missing")
            print("     Install them for full format support")
    else:
        print("\n❌ Some tests failed. Please check your installation.")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())