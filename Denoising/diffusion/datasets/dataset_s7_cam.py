
from torch.utils import data as data
from torchvision.transforms.functional import normalize
import random
# import numpy as np
import torch
import cv2
from glob import glob
from natsort import natsorted
import os
from PIL import Image
import yaml

class Dataset_s7(data.Dataset):

    def __init__(self, main_path_dataset, clip_dataset=False, trim_len=0,
                 train_unlabeld=0, xf_width=512, mode='train', **kwargs):
        super().__init__()
        self.main_path = main_path_dataset
        self.trim_len = trim_len
        self.clip_dataset = clip_dataset
        self.train_unlabeld = train_unlabeld
        self.xf_width = xf_width
        print('unlabeled', train_unlabeld, 'clip size:', xf_width)

        self.files = os.listdir(main_path_dataset)  # list of paired paths: lr_files, hr_files
        assert len(self.files) > 0, f'main_path: {main_path_dataset}, {len(self.files)}'
        print('Total files in dataset: ', len(self.files))
        text_file = os.path.join(main_path_dataset, 'text_dict.json')
        with open(text_file, 'r') as file:
            text_data = yaml.safe_load(file)
        self.num_crops_per_image = [len(d.keys()) for f_name, d in text_data.items()]
        self.samples = []
        for f_name, d in text_data.items():
            for c_num_name, crop_text in d.items():
                self.samples.append((f_name, c_num_name))
        # sort_text_data = sorted(text_data.items(), key=lambda x: x[0])
        self.samples = self.samples[:-7] if mode=='train' else self.samples[-7:-1]
        if self.clip_dataset:
            fname_tag = 'val' if 'val' in main_path_dataset[-4:] else 'train'
            #clip_embd_data_path = f'/home/erez/PycharmProjects/raw_dn_related/CycleISP/pretrained_models/clip_embd_{fname_tag}.pt'
            clip_embd_data_path = f'pretrained_models/clip_embd_s7data_L14_train.pt'
            self.clip_cache = torch.load(clip_embd_data_path, map_location='cpu')
            print('total keys in clip_cache: ', len(self.clip_cache.keys()))

        sample_every_image = True
        self.sample_every_img_for_val = ('val' in main_path_dataset[-4:]) and sample_every_image
        print('using sample_every_img_for_val: ', self.sample_every_img_for_val, len(self), mode)


    def __getitem__(self, index):
        folder_name, crop_num_name = self.samples[index]
        img_path = os.path.join(self.main_path, folder_name)
        imgs_inpath = [a for a in os.listdir(img_path) if a.endswith('pt')]
        # Read iamges:
        gt_img = [a for a in imgs_inpath if ('medium' in a and a.startswith(crop_num_name[4:]+'_'))]
        lq_img = [a for a in imgs_inpath if ('short' in a and a.startswith(crop_num_name[4:]+'_'))]
        assert len(gt_img)==1, gt_img
        assert len(lq_img)==1, lq_img
        med_crop_path = os.path.join(img_path, gt_img[0])
        raw_img = torch.load(med_crop_path).float().squeeze() #[..., ::2, ::2]
        raw_img = raw_img*85 # scale mean 0.0022 to the coco 0.196
        raw_img = torch.clamp(raw_img, 0, 1)

        short_crop_path = os.path.join(img_path, lq_img[0])
        lq = torch.load(short_crop_path).float().squeeze() #[..., ::2, ::2]
        lq = lq*85 # scale mean 0.0024 to the coco 0.196
        lq = torch.clamp(lq, 0, 1)

        #print(raw_img.shape)

        gt = raw_img * 2 - 1  # change to diffusion [-1,1]
        # shot_noise, read_noise = self.random_noise_levels()

        cond_dict = {'lq': lq}
        if self.train_unlabeld>0:
            print('Warning, self.train_unlabeld >0 with s7 dataset')
        if self.clip_dataset and torch.rand(1).item() > self.train_unlabeld:
            #img_clip_embd = self.clip_cache.get(img_index, None)
            img_clip_embd_data = self._load_embd(folder_name, crop_num_name)
            if img_clip_embd_data is None:
                print(f'Missing clip_embd for index {folder_name, crop_num_name}; got None')
            # print('fix clip0', end=' ')
            img_clip_embd_data /= torch.linalg.norm(img_clip_embd_data)
            cond_dict['clip_condition'] = img_clip_embd_data
        else:
            cond_dict['clip_condition'] = torch.zeros(self.xf_width, dtype=torch.float32)

        return gt, cond_dict

    def _load_embd(self, folder_name, crop_num_name):
        img_clip_embd = self.clip_cache.get(folder_name, None)
        if img_clip_embd is None:
            return None
        crop_embd = img_clip_embd.get(crop_num_name, None)
        return crop_embd

    def __len__(self):
        if self.trim_len>0:
            return min(self.trim_len, len(self.samples))
        return len(self.samples)

class Dataset_s21(data.Dataset):

    def __init__(self, main_path_dataset, clip_dataset=False, trim_len=0,
                 train_unlabeld=0, xf_width=512, mode='train',fpath_gt=None, fpath_noisy=None, **kwargs):
        super().__init__()
        self.main_path = main_path_dataset
        self.fpath_gt = fpath_gt or 'iso50_t1-50'
        self.fpath_noisy = fpath_noisy or 'iso3200_t1-12000'
        self.trim_len = trim_len
        self.clip_dataset = clip_dataset
        self.train_unlabeld = train_unlabeld
        self.xf_width = xf_width
        self.mode = mode
        print('unlabeled', train_unlabeld, 'clip size:', xf_width)

        # (12000/50) / 64 # 64=2^6stops iso
        self.amplitude_factors = {'iso50_t1-50' : 1,
                                  'iso50_t1-45' : 0.9,
                                  'iso3200_t1-12000': 3.75,
                                  'iso3200_t1-6000': 1.875,
                                  'to_coco_mean': 110, # 122
        }


        self.files = os.listdir(os.path.join(main_path_dataset, self.mode, self.fpath_gt))  # list of paired paths: lr_files, hr_files
        assert len(self.files) > 0, f'main_path: {main_path_dataset}, {len(self.files)}'
        print('Total files in dataset: ', len(self.files))
        if self.clip_dataset:
            #fname_tag = 'val' if 'val' in main_path_dataset[-4:] else 'train'
            clip_embd_data_path = f'pretrained_models/cococap/clip_embd_L14_{mode}.pt'
            self.clip_cache = torch.load(clip_embd_data_path, map_location='cpu')
            print('total keys in clip_cache: ', len(self.clip_cache.keys()))

        sample_every_image = True
        self.sample_every_img_for_val = mode=='val' and sample_every_image
        print('using sample_every_img_for_val: ', self.sample_every_img_for_val, len(self), mode)
        self.noise_upscale_factor = 1


    def __getitem__(self, index):
        img_index, embd_index = self._get_imgtxt_index(index)

        img_path = os.path.join(self.main_path, self.mode, self.fpath_gt, f'{img_index:06}_raw4c.pt')
        raw_img = torch.load(img_path).float().float().squeeze()  # [..., ::2, ::2]
        raw_img = raw_img * self.amplitude_factors[self.fpath_gt] * self.amplitude_factors['to_coco_mean']  # scale mean 0.0022 to the coco 0.196

        img_path = os.path.join(self.main_path, self.mode, self.fpath_noisy, f'{img_index:06}_raw4c.pt')
        lq = torch.load(img_path).float().float().squeeze()  # [..., ::2, ::2]
        lq = lq * self.amplitude_factors[self.fpath_noisy] * self.amplitude_factors['to_coco_mean']  # scale mean 0.0024 to the coco 0.196

        if self.noise_upscale_factor > 1:
            raw_img, lq = self.noise_upscale(raw_img, lq)
        raw_img = torch.clamp(raw_img, 0, 1)
        lq = torch.clamp(lq, 0, 1)

        #print(raw_img.shape)

        gt = raw_img * 2 - 1  # change to diffusion [-1,1]

        cond_dict = {'lq': lq}
        if self.train_unlabeld>0:
            print('Warning, self.train_unlabeld >0 with erezcam dataset')
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
        img_index = index // 5
        embd_index = index % 5
        if self.sample_every_img_for_val:
            img_index, embd_index = index, 0
        return img_index, embd_index

    def _load_embd(self, img_index, embd_index):
        img_clip_embd = self.clip_cache.get(img_index, None)  # todo change 80-img_index
        print("mixing caps")
        if img_clip_embd is None:
            return None
        return img_clip_embd[embd_index]

    def noise_upscale(self, raw_img, lq):
        noise = lq - raw_img
        noise = noise * self.noise_upscale_factor
        lq = raw_img + noise
        return raw_img, lq

    def __len__(self):
        if self.trim_len > 0:
            return min(self.trim_len, len(self.files))
        elif self.sample_every_img_for_val:
            return len(self.files)
        return len(self.files) * 5  # 5 embeds per img
