import numpy as np
import torch
from torch.utils import data
import os
import random
from basicsr.utils.registry import DATASET_REGISTRY

import scipy.io as sio  # 用于读取 .mat


@DATASET_REGISTRY.register()
class HSIDenoisingDataset(data.Dataset):
    """HSI Dataset for denoising tasks.
    
    For denoising, we use clean HSI images as GT and add noise to generate noisy LQ images.
    Assume HR data are stored as .mat or .npy, shape [H, W, C].
    """

    def __init__(self, opt):
        super(HSIDenoisingDataset, self).__init__()
        self.opt = opt
        self.gt_folder = opt['dataroot_gt']
        self.lq_folder = opt.get('dataroot_lq', None)  # Optional for noisy data
        
        self.file_list = sorted(os.listdir(self.gt_folder))
        self.gt_size = opt.get('gt_size', 128)
        self.use_hflip = opt.get('use_hflip', True)
        self.use_rot = opt.get('use_rot', True)
        
        # Noise parameters
        self.noise_type = opt.get('noise_type', 'gaussian')  # gaussian, poisson, mixed
        self.noise_range = opt.get('noise_range', [0, 50])  # For gaussian noise
        self.add_noise_to_gt = opt.get('add_noise_to_gt', True)  # Add noise on-the-fly

    def _add_noise(self, img):
        """Add noise to clean image."""
        if self.noise_type == 'gaussian':
            noise_level = random.uniform(*self.noise_range)
            noise = torch.randn_like(img) * (noise_level / 255.0)
            noisy_img = img + noise
        elif self.noise_type == 'poisson':
            # Convert to 0-255 range for Poisson noise
            img_255 = img * 255
            noisy_img_255 = torch.poisson(img_255)
            noisy_img = noisy_img_255 / 255.0
        elif self.noise_type == 'mixed':
            # Random mix of gaussian and poisson
            if random.random() < 0.5:
                noise_level = random.uniform(*self.noise_range)
                noise = torch.randn_like(img) * (noise_level / 255.0)
                noisy_img = img + noise
            else:
                img_255 = img * 255
                noisy_img_255 = torch.poisson(img_255)
                noisy_img = noisy_img_255 / 255.0
        else:
            raise ValueError(f"Unsupported noise type: {self.noise_type}")
        
        return torch.clamp(noisy_img, 0, 1)

    def __getitem__(self, index):
        fname = self.file_list[index]

        # --- load GT ---
        path_gt = os.path.join(self.gt_folder, fname)
        if fname.endswith('.mat'):
            gt = sio.loadmat(path_gt)['gt']  # 假设 key = 'gt'
        elif fname.endswith('.npy'):
            gt = np.load(path_gt)
        else:
            raise ValueError("Unsupported file format")

        # Normalize to [0, 1] if needed
        if gt.max() > 1.1:  # Likely in [0, 255] range
            gt = gt / 255.0

        # --- load or generate LQ ---
        if self.lq_folder and not self.add_noise_to_gt:
            # Load pre-computed noisy data
            path_lq = os.path.join(self.lq_folder, fname)
            if fname.endswith('.mat'):
                lq = sio.loadmat(path_lq)['lq']
            else:
                lq = np.load(path_lq)
            
            if lq.max() > 1.1:
                lq = lq / 255.0
        else:
            # Generate noisy data on-the-fly
            lq = gt.copy()

        # HWC -> CHW
        gt = torch.from_numpy(gt.astype(np.float32)).permute(2, 0, 1)
        lq = torch.from_numpy(lq.astype(np.float32)).permute(2, 0, 1)

        # Add noise if needed
        if self.add_noise_to_gt:
            lq = self._add_noise(gt)

        # 随机裁剪
        if self.gt_size is not None:
            h, w = gt.shape[1], gt.shape[2]
            if h >= self.gt_size and w >= self.gt_size:
                rnd_h = random.randint(0, h - self.gt_size)
                rnd_w = random.randint(0, w - self.gt_size)
                gt = gt[:, rnd_h:rnd_h+self.gt_size, rnd_w:rnd_w+self.gt_size]
                lq = lq[:, rnd_h:rnd_h+self.gt_size, rnd_w:rnd_w+self.gt_size]

        # 增广
        if self.use_hflip and random.random() < 0.5:
            gt = torch.flip(gt, dims=[2])
            lq = torch.flip(lq, dims=[2])
        if self.use_rot:
            rot = random.randint(0, 3)
            gt = torch.rot90(gt, rot, [1, 2])
            lq = torch.rot90(lq, rot, [1, 2])

        return {'lq': lq, 'gt': gt, 'key': fname}

    def __len__(self):
        return len(self.file_list)