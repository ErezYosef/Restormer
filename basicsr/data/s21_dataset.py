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
import os
import json

class Dataset_s21(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        main_path_dataset = '/storage/erez/datasets/denoising/s21_256/'
        clip_dataset=False
        train_unlabeld=0
        xf_width=512
        mode = 'val' if opt['name']=='ValSet' else 'train'
        trim_len=500 if mode=='train' else 0
        fpath_gt=None
        fpath_noisy=None
        super().__init__()

        self.main_path = main_path_dataset
        self.fpath_gt = fpath_gt or 'iso50_t1-50'
        self.fpath_noisy = fpath_noisy or 'iso3200_t1-12000'
        self.trim_len = trim_len
        self.clip_dataset = False # clip_dataset
        self.train_unlabeld = 0
        self.xf_width = xf_width
        self.mode = mode
        print('unlabeled', train_unlabeld, 'clip size:', xf_width)

        # (12000/50) / log_2(3200/50) # log_2(3200/50)=64=2^6stops for iso
        self.amplitude_factors = {'iso50_t1-50': 1,
                                  'iso50_t1-45': 0.9,
                                  'iso3200_t1-12000': 3.75,
                                  'iso3200_t1-6000': 1.875,
                                  'to_coco_mean': 110,  # 122,
                                  'scale_fix': (2**16-1)/(2**10-1),  # rescaling to int, scale it to 10bit image
                                  }

        self.files = os.listdir(
            os.path.join(main_path_dataset, self.mode, self.fpath_gt))  # list of paired paths: lr_files, hr_files
        assert len(self.files) > 0, f'main_path: {main_path_dataset}, {len(self.files)}'
        print('Total files in dataset: ', len(self.files))

        self.sample_every_img = True
        print('using sample_every_img: ', self.sample_every_img, len(self), mode)

        # self.variance_per = 'image'
        # if self.variance_per == 'image' and mode == 'val':
        #     noise_std_est = 'pretrained_models/data_save_rbgg_std_est.json'
        # elif self.variance_per == 'image' and mode == 'train':
        #     noise_std_est = 'pretrained_models/data_save_rbgg_std_est_train.json'
        # elif self.variance_per == 'channel':
        #     print('Warning - channels mode')
        #     noise_std_est = 'pretrained_models/data_save_channels_std_est.json'
        #
        # with open(noise_std_est, "r") as f:
        #     self.noise_std_est = json.load(f)
        # assert self.fpath_noisy in self.noise_std_est['path'], f'{self.noise_std_est["path"]} not compatible to data'

        self.mean = None
        self.std = None

        imgs = []
        for i in range(len(self)):
            data = self[i]
            imgs.append(data['gt'])
            #print(i)
        imgs = torch.stack(imgs, 0)
        print(imgs.shape)
        self.mean = torch.mean(imgs, (0,2,3), keepdim=False)
        self.std = torch.std(imgs, (0,2,3), keepdim=False)

        print('MEAN STD:', self.mean, self.std)


    def __len__(self):
        if self.trim_len>0:
            return min(self.trim_len, len(self.files))
        if self.sample_every_img:
            return len(self.files)
        else:
            raise NotImplementedError

    def __getitem__(self, index):
        img_index, embd_index = self._get_imgtxt_index(index)

        img_path = os.path.join(self.main_path, self.mode, self.fpath_gt, f'{img_index:06}_raw4c.pt')
        raw_img = torch.load(img_path).float().squeeze()  # [..., ::2, ::2]
        raw_img = raw_img * self.amplitude_factors[self.fpath_gt] * self.amplitude_factors['to_coco_mean']  # scale mean 0.0022 to the coco 0.196

        img_path_lq = os.path.join(self.main_path, self.mode, self.fpath_noisy, f'{img_index:06}_raw4c.pt')
        lq = torch.load(img_path_lq).float().squeeze()  # [..., ::2, ::2]
        lq = lq * self.amplitude_factors[self.fpath_noisy] * self.amplitude_factors['to_coco_mean']  # scale mean 0.0024 to the coco 0.196

        raw_img = torch.clamp(raw_img, 0, 1)
        lq = torch.clamp(lq, 0, 1)

        #print(raw_img.shape)

        gt = raw_img# * 2 - 1  # change to diffusion [-1,1]

        # cond_dict = {'lq': lq}
        # fname = f'{img_index:06}_raw4c.pt'


        # BGR to RGB, HWC to CHW, numpy to tensor
        # gt, lq = img2tensor([gt, lq],
        #                             bgr2rgb=True,
        #                             float32=True)
        # normalize

        # todo check
        # if self.mean is not None or self.std is not None:
        #     normalize(lq, self.mean, self.std, inplace=True)
        #     normalize(gt, self.mean, self.std, inplace=True)

        return {
            'lq': lq,
            'gt': gt,
            'lq_path': img_path_lq,
            'gt_path': img_path
        }

    def _get_imgtxt_index(self, index):
        if self.sample_every_img:
            return index, 0
        img_index = index // 5
        embd_index = index % 5
        return img_index, embd_index

class Dataset_PairedImage2(data.Dataset):
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

    def __init__(self, opt):
        super(Dataset_PairedImage2, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt[
            'meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path))

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path)

            # flip, rotation augmentations
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)
