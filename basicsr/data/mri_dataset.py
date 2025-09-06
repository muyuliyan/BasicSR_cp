import numpy as np
import torch
from torch.utils import data
import os
import random
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.utils import img2tensor
from basicsr.data.transforms import augment, paired_random_crop

# Import libraries for medical image formats
try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    print("Warning: nibabel not available. Only .npy and .mat formats will be supported.")

try:
    import scipy.io as sio
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. .mat format will not be supported.")


@DATASET_REGISTRY.register()
class MRIDataset(data.Dataset):
    """MRI Dataset for super-resolution tasks.
    
    Supports multiple MRI file formats:
    - .nii/.nii.gz (NIfTI format using nibabel)
    - .npy (NumPy arrays)
    - .mat (MATLAB format)
    
    Assumes HR/LR data are stored with shape [H, W] for 2D or [H, W, D] for 3D.
    For 3D data, slices are extracted as 2D images.
    """

    def __init__(self, opt):
        super(MRIDataset, self).__init__()
        self.opt = opt
        self.gt_folder = opt['dataroot_gt']
        self.lq_folder = opt['dataroot_lq']
        
        # Get list of files
        self.file_list = sorted([f for f in os.listdir(self.gt_folder) 
                                if f.endswith(('.nii', '.nii.gz', '.npy', '.mat'))])
        
        # MRI-specific parameters
        self.scale = opt.get('scale', 4)
        self.gt_size = opt.get('gt_size', 128)
        self.use_hflip = opt.get('use_hflip', True)
        self.use_rot = opt.get('use_rot', True)
        
        # MRI normalization parameters
        self.normalize_to_01 = opt.get('normalize_to_01', True)
        self.clip_percentiles = opt.get('clip_percentiles', [1, 99])  # Robust normalization
        
        # For 3D MRI volumes, specify how to handle slices
        self.slice_axis = opt.get('slice_axis', 2)  # Default: axial slices
        self.extract_2d_slices = opt.get('extract_2d_slices', True)

    def _load_mri_data(self, filepath):
        """Load MRI data from various formats."""
        if filepath.endswith(('.nii', '.nii.gz')):
            if not NIBABEL_AVAILABLE:
                raise ImportError("nibabel is required for NIfTI format")
            img = nib.load(filepath)
            data = img.get_fdata()
        elif filepath.endswith('.npy'):
            data = np.load(filepath)
        elif filepath.endswith('.mat'):
            if not SCIPY_AVAILABLE:
                raise ImportError("scipy is required for .mat format")
            mat_data = sio.loadmat(filepath)
            # Try common keys for MRI data
            if 'data' in mat_data:
                data = mat_data['data']
            elif 'image' in mat_data:
                data = mat_data['image']
            elif 'volume' in mat_data:
                data = mat_data['volume']
            else:
                # Use the first non-metadata key
                keys = [k for k in mat_data.keys() if not k.startswith('__')]
                if keys:
                    data = mat_data[keys[0]]
                else:
                    raise ValueError(f"No valid data key found in {filepath}")
        else:
            raise ValueError(f"Unsupported file format: {filepath}")
        
        return data.astype(np.float32)

    def _normalize_mri(self, data):
        """Apply MRI-specific normalization."""
        if self.normalize_to_01:
            # Robust normalization using percentiles
            if self.clip_percentiles:
                p_low, p_high = np.percentile(data, self.clip_percentiles)
                data = np.clip(data, p_low, p_high)
            
            # Normalize to [0, 1]
            data_min, data_max = data.min(), data.max()
            if data_max > data_min:
                data = (data - data_min) / (data_max - data_min)
            else:
                data = np.zeros_like(data)
        
        return data

    def _extract_2d_slice(self, data_3d, slice_idx=None):
        """Extract 2D slice from 3D volume."""
        if slice_idx is None:
            # Random slice for training, middle slice for validation
            if hasattr(self, 'phase') and self.phase == 'val':
                slice_idx = data_3d.shape[self.slice_axis] // 2
            else:
                slice_idx = random.randint(0, data_3d.shape[self.slice_axis] - 1)
        
        if self.slice_axis == 0:
            return data_3d[slice_idx, :, :]
        elif self.slice_axis == 1:
            return data_3d[:, slice_idx, :]
        else:  # axis == 2
            return data_3d[:, :, slice_idx]

    def __getitem__(self, index):
        fname = self.file_list[index]
        
        # Load GT data
        path_gt = os.path.join(self.gt_folder, fname)
        gt_data = self._load_mri_data(path_gt)
        gt_data = self._normalize_mri(gt_data)
        
        # Load LQ data
        path_lq = os.path.join(self.lq_folder, fname)
        lq_data = self._load_mri_data(path_lq)
        lq_data = self._normalize_mri(lq_data)
        
        # Handle 3D to 2D conversion if needed
        if len(gt_data.shape) == 3 and self.extract_2d_slices:
            slice_idx = None
            if hasattr(self, 'phase') and self.phase == 'val':
                slice_idx = gt_data.shape[self.slice_axis] // 2
            
            gt_data = self._extract_2d_slice(gt_data, slice_idx)
            lq_data = self._extract_2d_slice(lq_data, slice_idx)
        
        # Ensure 2D data has proper format
        if len(gt_data.shape) == 2:
            gt_data = gt_data[:, :, np.newaxis]  # Add channel dimension
            lq_data = lq_data[:, :, np.newaxis]
        
        # Convert to torch tensors (HWC -> CHW)
        gt = img2tensor(gt_data, bgr2rgb=False, float32=True)
        lq = img2tensor(lq_data, bgr2rgb=False, float32=True)
        
        # Random crop for training
        if self.gt_size is not None and hasattr(self, 'phase') and self.phase == 'train':
            gt, lq = paired_random_crop(gt, lq, self.gt_size, self.scale)
        
        # Data augmentation
        if hasattr(self, 'phase') and self.phase == 'train':
            gt, lq = augment([gt, lq], self.use_hflip, self.use_rot)
        
        return {'lq': lq, 'gt': gt, 'lq_path': path_lq, 'gt_path': path_gt}

    def __len__(self):
        return len(self.file_list)


@DATASET_REGISTRY.register()
class OASISDataset(MRIDataset):
    """OASIS Dataset for MRI super-resolution.
    
    OASIS (Open Access Series of Imaging Studies) dataset specific configurations.
    """
    
    def __init__(self, opt):
        # OASIS-specific defaults
        opt.setdefault('normalize_to_01', True)
        opt.setdefault('clip_percentiles', [2, 98])  # More conservative for OASIS
        opt.setdefault('slice_axis', 2)  # Axial slices
        opt.setdefault('extract_2d_slices', True)
        
        super(OASISDataset, self).__init__(opt)


@DATASET_REGISTRY.register()
class MMWHSDataset(MRIDataset):
    """MM-WHS Dataset for MRI super-resolution.
    
    Multi-Modality Whole Heart Segmentation (MM-WHS) dataset specific configurations.
    """
    
    def __init__(self, opt):
        # MM-WHS-specific defaults
        opt.setdefault('normalize_to_01', True)
        opt.setdefault('clip_percentiles', [1, 99])
        opt.setdefault('slice_axis', 2)  # Axial slices
        opt.setdefault('extract_2d_slices', True)
        
        super(MMWHSDataset, self).__init__(opt)