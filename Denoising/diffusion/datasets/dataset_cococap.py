
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import (paired_paths_from_folder,
                                    paired_DP_paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_meta_info_file)
from basicsr.data.transforms import augment, paired_random_crop, paired_random_crop_DP, random_augmentation
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding, padding_DP, imfrombytesDP

import random
# import numpy as np
import torch
import cv2
from glob import glob
from natsort import natsorted
import os
from PIL import Image


class Dataset_Cococap(data.Dataset):

    def __init__(self, main_path_dataset, clip_dataset=False, trim_len=0, deterministic_seed_noise=None,
                 train_unlabeld=0, xf_width=512, deterministic_noise_lvl=0.1, **kwargs):
        super().__init__()
        self.main_path = main_path_dataset
        self.trim_len = trim_len
        self.clip_dataset = clip_dataset
        self.train_unlabeld = train_unlabeld
        self.xf_width = xf_width
        self.mode = kwargs['mode']
        print('train_unlabeld', train_unlabeld, ', defined clip size:', xf_width)

        self.files = os.listdir(main_path_dataset)  # list of paired paths: lr_files, hr_files
        assert len(self.files) > 0, f'main_path: {main_path_dataset}, {len(self.files)}'
        print('Total files in dataset: ', len(self.files))

        if self.clip_dataset:
            # fname_tag = self.mode  # 'val' if 'val' in main_path_dataset[-4:] else 'train'
            #clip_embd_data_path = f'/home/erez/PycharmProjects/raw_dn_related/CycleISP/pretrained_models/clip_embd_{fname_tag}.pt'
            clip_embd_data_path = 'pretrained_models/cococap/clip_embd_L14_{}.pt'
            clip_embd_data_path = kwargs.get('clip_embd_data_path', None) or clip_embd_data_path
            clip_embd_data_path = clip_embd_data_path.format(self.mode)
            self.clip_cache = torch.load(clip_embd_data_path, map_location='cpu')
            print('total keys in clip_cache: ', len(self.clip_cache.keys()))

        sample_every_image = True
        single_sample_in_data = self.clip_dataset and self.clip_cache[0].ndim == 1
        self.sample_every_img_once = sample_every_image and (self.mode == 'val' or trim_len>0 or single_sample_in_data)
        print('using sample_every_img_once: ', self.sample_every_img_once)
        self.len_data = min(len(self.files), len(self.clip_cache)) if self.clip_dataset else len(self.files)
        self.deterministic_seed_noise = deterministic_seed_noise
        self.deterministic_noise_lvl = deterministic_noise_lvl


    def __getitem__(self, index):
        img_index, embd_index = self._get_imgtxt_index(index)

        # print(img_index, embd_index)
        # target = target[:5] #discarded
        # assert len(target)== 5, f'data index {index} len captions: {len(target)}'
        # return img, target
        #embd = self.clip_embd[index // 5][index % 5]  # 512 vector


        # Read iamges:
        img_path = os.path.join(self.main_path, f'{img_index:06}.pt')
        raw_img = torch.load(img_path).float() #[..., ::2, ::2]
        #print(raw_img.shape)

        gt = raw_img * 2 - 1  # change to diffusion [-1,1]
        shot_noise, read_noise = self.random_noise_levels()
        if self.deterministic_seed_noise is not None:
            #print('DETER')
            shot_noise, read_noise = self.deterministic_noise_levels()
            torch.manual_seed(self.deterministic_seed_noise)
        lq = self.add_noise(raw_img, shot_noise, read_noise)

        cond_dict = {'lq': lq}

        if self.clip_dataset and torch.rand(1).item() > self.train_unlabeld:
            #img_clip_embd = self.clip_cache.get(img_index, None)
            img_clip_embd_data = self._load_embd(img_index, embd_index)
            if img_clip_embd_data is None:
                print(f'Missing clip_embd for index {img_index}; got None')
            # print('fix clip0', end=' ')
            img_clip_embd_data /= torch.linalg.norm(img_clip_embd_data)
            cond_dict['clip_condition'] = img_clip_embd_data
        else:
            cond_dict['clip_condition'] = torch.zeros(self.xf_width, dtype=torch.float32)

        return gt, cond_dict

    def _get_imgtxt_index(self, index):
        if self.sample_every_img_once:
            return index, 0
        img_index = index // 5
        embd_index = index % 5
        return img_index, embd_index

    def _load_embd(self, img_index, embd_index):
        img_clip_embd = self.clip_cache.get(img_index, None)
        if img_clip_embd is None:
            print('LOADING NONE IN DATALOADER')
            return None
        if img_clip_embd.ndim == 2:
            return img_clip_embd[embd_index]
        return img_clip_embd # 512/768 vector

    def __len__(self):
        if self.trim_len > 0:
            return min(self.trim_len, self.len_data)
        elif self.sample_every_img_once:
            return self.len_data

        return self.len_data * 5 # 5 embeds per img for training

    def random_noise_levels(self):
        """ Where read_noise in SIDD is not 0 """
        log_min_shot_noise = torch.log10(torch.Tensor([0.00068674])) # -3.16
        log_max_shot_noise = torch.log10(torch.Tensor([0.02194856])) # -1.65
        log_min_shot_noise = torch.log10(torch.Tensor([0.1])) # -1 > -4
        log_max_shot_noise = torch.log10(torch.Tensor([0.31])) # -0.5
        distribution = torch.distributions.uniform.Uniform(log_min_shot_noise, log_max_shot_noise)

        log_shot_noise = distribution.sample()
        shot_noise = torch.pow(10, log_shot_noise)

        distribution = torch.distributions.normal.Normal(torch.Tensor([0.0]), torch.Tensor([0.20]))
        distribution = torch.distributions.normal.Normal(torch.Tensor([0.0]), torch.Tensor([0.50]))
        read_noise = distribution.sample()
        line = lambda x: 1.85 * x + 0.5  ### Line SIDD test set
        line = lambda x: 1.5 * x + 0.05  ### Line 2
        log_read_noise = line(log_shot_noise) + read_noise
        read_noise = torch.pow(10, log_read_noise)
        return shot_noise, read_noise

    def deterministic_noise_levels(self):
        """ Where read_noise in SIDD is not 0 """
        log_min_shot_noise = torch.log10(torch.Tensor([0.1])) # -1
        log_max_shot_noise = torch.log10(torch.Tensor([0.31])) # -0.5

        log_shot_noise = torch.log10(torch.Tensor([0.05]))
        log_shot_noise = torch.log10(torch.Tensor([self.deterministic_noise_lvl]))
        shot_noise = torch.pow(10, log_shot_noise)

        read_noise = 0
        line = lambda x: 1.5 * x + 0.05  ### Line SIDD test set
        log_read_noise = line(log_shot_noise) + read_noise
        read_noise = torch.pow(10, log_read_noise)
        return shot_noise, read_noise

    def add_noise(self, image, shot_noise=0.01, read_noise=0.0005, use_cuda=False):
        """Adds random shot (proportional to image) and read (independent) noise."""
        variance = image * shot_noise + read_noise
        mean = torch.Tensor([0.0])
        if use_cuda:
            mean = mean.cuda()
        distribution = torch.distributions.normal.Normal(mean, torch.sqrt(variance))
        noise = distribution.sample()
        return image + noise

class Dataset_Cococap_dualclipdata(Dataset_Cococap):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.clip_dataset:
            #clip_embd_data_path = f'/home/erez/PycharmProjects/raw_dn_related/CycleISP/pretrained_models/clip_embd_{fname_tag}.pt'
            clip_embd_data_path = 'pretrained_models/cococap/clip_embd_L14_{}.pt'.format(self.mode)
            # clip_embd_data_path = clip_embd_data_path.format(self.mode)
            self.clip_cache_default = torch.load(clip_embd_data_path, map_location='cpu')
            print('total keys in clip_cache: ', len(self.clip_cache.keys()))
        self.len_data = min(len(self.files), max(len(self.clip_cache), len(self.clip_cache_default)))
        print(f'dual clip data: dataset len: {self.len_data}')

    def _load_embd(self, img_index, embd_index):
        img_clip_embd = self.clip_cache.get(img_index, None)
        if img_clip_embd is None:
            # Changed:
            img_clip_embd = self.clip_cache_default.get(img_index, None)
        if img_clip_embd.ndim == 2:
            return img_clip_embd[embd_index]
        return img_clip_embd # 512/768 vector

class Dataset_Cococap_clipimg(Dataset_Cococap):
    # Each image has 6 captions embeds: 5 from coco and one from CLIP on the image itself
    def __init__(self, main_path_dataset, clip_dataset=False, trim_len=0):
        super().__init__(main_path_dataset, clip_dataset, trim_len)
        assert clip_dataset, 'must be clip dataset for coco captions with clip img embed'

        fname_tag = 'val' if 'val' in main_path_dataset[-4:] else 'train'
        clip_embd_data_path = f'/home/erez/PycharmProjects/raw_dn_related/CycleISP/pretrained_models/clip_embd_imgs_{fname_tag}.pt'
        self.clip_cache_imgs = torch.load(clip_embd_data_path, map_location='cpu')
        print('total keys in clip_cache_imgs! : ', len(self.clip_cache_imgs.keys()))

    def _get_imgtxt_index(self, index):
        img_index = index // 6
        embd_index = index % 6
        if self.sample_every_img_once:
            img_index, embd_index = index, 0
        return img_index, embd_index
    def _load_embd(self, img_index, embd_index):
        if embd_index == 5:
            img_clip_embd = self.clip_cache_imgs.get(img_index, None)
            return img_clip_embd # a 512 tensor in each dict entry

        img_clip_embd = self.clip_cache.get(img_index, None) # # a 5x512 tensor in each dict entry
        if img_clip_embd is None:
            return None
        return img_clip_embd[embd_index]

    def __len__(self):
        if self.trim_len>0:
            return self.trim_len
        elif self.sample_every_img_once:
            return len(self.files)

        return len(self.files)*6 # 6 embeds per img (with image CLIP)
