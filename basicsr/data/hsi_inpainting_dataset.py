import numpy as np
import torch
from torch.utils import data
import os
import random
from basicsr.utils.registry import DATASET_REGISTRY

import scipy.io as sio  # 用于读取 .mat


@DATASET_REGISTRY.register()
class HSIInpaintingDataset(data.Dataset):
    """HSI Dataset for inpainting tasks.
    
    For inpainting, we use complete HSI images as GT and create masked versions as LQ.
    Assume HR data are stored as .mat or .npy, shape [H, W, C].
    """

    def __init__(self, opt):
        super(HSIInpaintingDataset, self).__init__()
        self.opt = opt
        self.gt_folder = opt['dataroot_gt']
        self.mask_folder = opt.get('dataroot_mask', None)  # Optional for pre-defined masks
        
        self.file_list = sorted(os.listdir(self.gt_folder))
        self.gt_size = opt.get('gt_size', 128)
        self.use_hflip = opt.get('use_hflip', True)
        self.use_rot = opt.get('use_rot', True)
        
        # Mask parameters
        self.mask_type = opt.get('mask_type', 'random_rect')  # random_rect, random_irregular, fixed
        self.mask_ratio = opt.get('mask_ratio', [0.1, 0.3])  # Range of mask ratio
        self.generate_mask = opt.get('generate_mask', True)  # Generate masks on-the-fly

    def _generate_random_rect_mask(self, h, w):
        """Generate random rectangular mask."""
        mask_ratio = random.uniform(*self.mask_ratio)
        mask_h = int(h * np.sqrt(mask_ratio))
        mask_w = int(w * np.sqrt(mask_ratio))
        
        # Random position
        start_h = random.randint(0, h - mask_h)
        start_w = random.randint(0, w - mask_w)
        
        mask = torch.ones((h, w), dtype=torch.float32)
        mask[start_h:start_h+mask_h, start_w:start_w+mask_w] = 0
        
        return mask

    def _generate_random_irregular_mask(self, h, w):
        """Generate random irregular mask using random walks."""
        mask_ratio = random.uniform(*self.mask_ratio)
        mask = torch.ones((h, w), dtype=torch.float32)
        
        # Number of random walks
        num_walks = random.randint(5, 15)
        
        for _ in range(num_walks):
            # Random starting point
            start_h = random.randint(0, h-1)
            start_w = random.randint(0, w-1)
            
            # Random walk length
            walk_length = random.randint(10, min(h, w) // 4)
            
            curr_h, curr_w = start_h, start_w
            for _ in range(walk_length):
                # Random direction
                dh = random.randint(-1, 1)
                dw = random.randint(-1, 1)
                
                curr_h = max(0, min(h-1, curr_h + dh))
                curr_w = max(0, min(w-1, curr_w + dw))
                
                # Create a small region around current position
                region_size = random.randint(1, 3)
                for i in range(max(0, curr_h - region_size), min(h, curr_h + region_size + 1)):
                    for j in range(max(0, curr_w - region_size), min(w, curr_w + region_size + 1)):
                        mask[i, j] = 0
        
        # Ensure mask ratio is approximately correct
        current_ratio = 1 - mask.mean().item()
        if current_ratio < mask_ratio * 0.5:
            # Add more masked regions
            num_additional = int((mask_ratio - current_ratio) * h * w / 100)
            for _ in range(num_additional):
                rh = random.randint(0, h-1)
                rw = random.randint(0, w-1)
                size = random.randint(1, 5)
                for i in range(max(0, rh - size), min(h, rh + size + 1)):
                    for j in range(max(0, rw - size), min(w, rw + size + 1)):
                        mask[i, j] = 0
        
        return mask

    def _create_masked_image(self, img, mask):
        """Create masked image by setting masked regions to 0 or random values."""
        # Expand mask to all channels
        mask_expanded = mask.unsqueeze(0).expand_as(img)  # [C, H, W]
        
        if random.random() < 0.5:
            # Set masked regions to 0
            masked_img = img * mask_expanded
        else:
            # Set masked regions to random values
            random_vals = torch.rand_like(img)
            masked_img = img * mask_expanded + random_vals * (1 - mask_expanded)
        
        return masked_img

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

        # HWC -> CHW
        gt = torch.from_numpy(gt.astype(np.float32)).permute(2, 0, 1)

        # 随机裁剪
        if self.gt_size is not None:
            h, w = gt.shape[1], gt.shape[2]
            if h >= self.gt_size and w >= self.gt_size:
                rnd_h = random.randint(0, h - self.gt_size)
                rnd_w = random.randint(0, w - self.gt_size)
                gt = gt[:, rnd_h:rnd_h+self.gt_size, rnd_w:rnd_w+self.gt_size]

        # --- Generate or load mask ---
        if self.mask_folder and not self.generate_mask:
            # Load pre-defined mask
            mask_fname = fname.replace('.mat', '_mask.mat').replace('.npy', '_mask.npy')
            path_mask = os.path.join(self.mask_folder, mask_fname)
            if os.path.exists(path_mask):
                if mask_fname.endswith('.mat'):
                    mask = sio.loadmat(path_mask)['mask']
                else:
                    mask = np.load(path_mask)
                mask = torch.from_numpy(mask.astype(np.float32))
            else:
                # Fallback to generated mask
                h, w = gt.shape[1], gt.shape[2]
                mask = self._generate_random_rect_mask(h, w)
        else:
            # Generate mask on-the-fly
            h, w = gt.shape[1], gt.shape[2]
            if self.mask_type == 'random_rect':
                mask = self._generate_random_rect_mask(h, w)
            elif self.mask_type == 'random_irregular':
                mask = self._generate_random_irregular_mask(h, w)
            else:
                raise ValueError(f"Unsupported mask type: {self.mask_type}")

        # Create masked LQ image
        lq = self._create_masked_image(gt, mask)

        # 增广
        if self.use_hflip and random.random() < 0.5:
            gt = torch.flip(gt, dims=[2])
            lq = torch.flip(lq, dims=[2])
            mask = torch.flip(mask, dims=[1])
        if self.use_rot:
            rot = random.randint(0, 3)
            gt = torch.rot90(gt, rot, [1, 2])
            lq = torch.rot90(lq, rot, [1, 2])
            mask = torch.rot90(mask, rot, [0, 1])

        return {'lq': lq, 'gt': gt, 'mask': mask, 'key': fname}

    def __len__(self):
        return len(self.file_list)