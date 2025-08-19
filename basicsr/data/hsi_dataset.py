import numpy as np
import torch
from torch.utils import data
import os
import random
from basicsr.utils.registry import DATASET_REGISTRY

import scipy.io as sio  # 用于读取 .mat

@DATASET_REGISTRY.register()
class HSIDataset(data.Dataset):
    """HSI Dataset for SR tasks.
    Assume HR/LR data are stored as .mat or .npy, shape [H, W, C].
    """

    def __init__(self, opt):
        super(HSIDataset, self).__init__()
        self.opt = opt
        self.gt_folder = opt['dataroot_gt']
        self.lq_folder = opt['dataroot_lq']

        self.file_list = sorted(os.listdir(self.gt_folder))
        self.scale = opt.get('scale', 4)
        self.gt_size = opt.get('gt_size', 128)
        self.use_hflip = opt.get('use_hflip', True)
        self.use_rot = opt.get('use_rot', True)

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

        # --- load LQ ---
        path_lq = os.path.join(self.lq_folder, fname)
        if fname.endswith('.mat'):
            lq = sio.loadmat(path_lq)['lq']
        else:
            lq = np.load(path_lq)

        # HWC -> CHW
        gt = torch.from_numpy(gt.astype(np.float32)).permute(2,0,1)
        lq = torch.from_numpy(lq.astype(np.float32)).permute(2,0,1)

        # 随机裁剪
        if self.gt_size is not None:
            h, w = gt.shape[1], gt.shape[2]
            rnd_h = random.randint(0, h - self.gt_size)
            rnd_w = random.randint(0, w - self.gt_size)
            gt = gt[:, rnd_h:rnd_h+self.gt_size, rnd_w:rnd_w+self.gt_size]
            lq = lq[:, rnd_h//self.scale:(rnd_h+self.gt_size)//self.scale,
                        rnd_w//self.scale:(rnd_w+self.gt_size)//self.scale]

        # 增广
        if self.use_hflip and random.random() < 0.5:
            gt = torch.flip(gt, dims=[2])
            lq = torch.flip(lq, dims=[2])
        if self.use_rot:
            rot = random.randint(0,3)
            gt = torch.rot90(gt, rot, [1,2])
            lq = torch.rot90(lq, rot, [1,2])

        return {'lq': lq, 'gt': gt, 'key': fname}

    def __len__(self):
        return len(self.file_list)
