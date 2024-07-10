
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import (paired_paths_from_folder,
                                    paired_DP_paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_meta_info_file)
from basicsr.data.transforms import augment, paired_random_crop, paired_random_crop_DP, random_augmentation
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding, padding_DP, imfrombytesDP

import random
import numpy as np
import torch
import cv2
from glob import glob
from natsort import natsorted
import os
from PIL import Image


class Dataset_PairedImage(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            geometric_augs (bool): Use geometric augmentations.

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, main_path, cropsize=None, random_crop=False, phase='train', val_percent=1):
        super().__init__()
        self.crop_size = cropsize
        self.random_crop = random_crop

        self.files = generate_paths_data(main_path) # list of paired paths: lr_files, hr_files
        print('Total files in dataset: ', len(self.files))
        every = np.rint(100/val_percent)
        if phase == 'train':
            self.files = [file for i,file in enumerate(self.files) if i%every != 0]
        elif phase == 'val':
            self.files = [file for i,file in enumerate(self.files) if i%every == 0]
        print(f'For {phase} - files in dataset: ', len(self.files))


    def __getitem__(self, index):
        # scale = self.opt['scale']
        # index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        lq_path, gt_path = self.files[index]
        # Read iamges:
        lq = Image.open(lq_path)
        gt = Image.open(gt_path)
        assert lq.size == gt.size

        if self.crop_size>0:
            w, h = lq.size
            if self.random_crop:
                h_loc = np.random.uniform(0, h-self.crop_size)
                w_loc = np.random.uniform(0, w-self.crop_size)
            else:
                h_loc, w_loc = h//2, w//2
            crop = (w_loc, h_loc, w_loc+self.crop_size, h_loc+self.crop_size)
            lq = lq.crop(crop)
            gt = gt.crop(crop)

        lq = np.asarray(lq) / 255
        gt = np.asarray(gt) * (2 / 255) - 1  # change to diffusion [-1,1]

        # flip, rotation augmentations
        # if self.geometric_augs:
        #     img_gt, img_lq = random_augmentation(img_gt, img_lq)

        # BGR to RGB, HWC to CHW, numpy to tensor
        lq, gt = img2tensor([lq, gt],
                                    bgr2rgb=False,
                                    float32=True)
        # normalize - None
        # cond_dict =  {
        #     'lq': lq,
        #     'gt': gt,
        #     'lq_path': lq_path,
        #     'gt_path': gt_path
        # }
        cond_dict = {'lq': lq,}
        return gt, cond_dict

    def __len__(self):
        return len(self.files)

class Dataset_PairedImage_crops_less(Dataset_PairedImage):
    # using less crops then maximum available
    def __init__(self, main_path):
        data.Dataset.__init__(self)
        self.crop_size = 0
        self.random_crop = False

        self.files = generate_paths_data_crops(main_path)  # list of paired paths: lr_files, hr_files
        assert len(self.files)>0, f'main_path: {main_path}'
        print('Total files in dataset: ', len(self.files))


def generate_paths_data(path):
    files = natsorted(glob(os.path.join(path, '*', '*.PNG')))
    files2 = natsorted(glob(os.path.join(path, '*','*', '*.PNG')))
    files.extend(files2)
    lr_files, hr_files = [], []
    for file_ in files:
        filename = os.path.split(file_)[-1]
        if 'GT' in filename:
            hr_files.append(file_)
        if 'NOISY' in filename:
            lr_files.append(file_)

    files = [(i, j) for i, j in zip(lr_files, hr_files)]
    return files

def generate_paths_data_crops(path):
    files = natsorted(glob(os.path.join(path, 'input_crops', '*.png')))
    files2 = natsorted(glob(os.path.join(path, 'target_crops', '*.png')))
    # print(len(files))
    # print(len(glob(os.path.join(path, '*'))))
    # print(len(natsorted(glob(os.path.join(path, '*', '*')))))
    lr_files, hr_files = files, files2 # [], []
    # for file_ in files:
    #     filename = os.path.split(file_)[-1]
    #     if 'GT' in filename:
    #         hr_files.append(file_)
    #     if 'NOISY' in filename:
    #         lr_files.append(file_)
    print(len(lr_files), len(hr_files))

    files = [(i, j) for i, j in zip(lr_files, hr_files)]
    return files

class Dataset_PairedImage_crops_less_clip(Dataset_PairedImage_crops_less):
    def __init__(self, *kargs, **kwargs):
        self.clip_dataset = kwargs.pop('clip_dataset', None)
        super().__init__(*kargs, **kwargs)

        if self.clip_dataset:
            fname_tag = 'val' if 'val' in kwargs['main_path'] else 'train4'
            clip_embd_data_path = f'pretrained_models/clip_embd_{fname_tag}.pt'
            self.clip_cache = torch.load(clip_embd_data_path, map_location='cpu')
            print('total keys in clip_cache: ', len(self.clip_cache.keys()))

    def __getitem__(self, index):
        gt, out_dict = super().__getitem__(index) # out_dict['lq'] = noisy image

        if self.clip_dataset:
            img_clip_embd = self.clip_cache.get(index, None)
            if img_clip_embd is None:
                print(f'Missing clip_embd for index {index}; got None')
            # print('fix clip0', end=' ')
            out_dict['clip_condition'] = img_clip_embd

        return gt, out_dict
