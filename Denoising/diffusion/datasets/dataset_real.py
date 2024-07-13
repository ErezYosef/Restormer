import os
import torch
from torch.utils import data as data
'''
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import (paired_paths_from_folder,
                                    paired_DP_paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_meta_info_file)
from basicsr.data.transforms import augment, paired_random_crop, paired_random_crop_DP, random_augmentation
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding, padding_DP, imfrombytesDP

import random
# import numpy as np

import cv2
from glob import glob
from natsort import natsorted

from PIL import Image
'''

class Dataset_Realcam(data.Dataset):

    def __init__(self, main_path_dataset, clip_dataset=False, trim_len=0, xf_width=512, deterministic_seed_noise=None, **kwargs):
        super().__init__()
        self.main_path = main_path = main_path_dataset
        self.trim_len = trim_len
        self.clip_dataset = clip_dataset
        self.xf_width = xf_width

        self.files = os.listdir(main_path)  # list of paired paths: lr_files, hr_files
        self.files = list(filter(lambda x: x.endswith(".pt"), self.files))
        self.files.sort()
        assert len(self.files) > 0, f'main_path: {main_path}, {len(self.files)}'
        print('Total files in dataset: ', len(self.files))

        if self.clip_dataset:
            # import clip
            from guided_diffusion.glide.clip_util import clip_model_wrap
            clip_model = clip_model_wrap(model_name="ViT-L/14", device='cpu')
            caps_clip_embd = clip_model.get_txt_embd(self.clip_dataset)
            self.clip_cache = caps_clip_embd[0] # 512 vector
            print(self.clip_dataset, self.clip_cache.shape, 'should be 768')
            # fname_tag = 'val' if 'val' in main_path[-4:] else 'train'
            # clip_embd_data_path = f'/home/erez/PycharmProjects/raw_dn_related/CycleISP/pretrained_models/clip_embd_{fname_tag}.pt'
            # self.clip_cache = torch.load(clip_embd_data_path, map_location='cpu')

    def __getitem__(self, index):
        img_index, embd_index = self._get_imgtxt_index(index)

        # print(img_index, embd_index)
        # target = target[:5] #discarded
        # assert len(target)== 5, f'data index {index} len captions: {len(target)}'
        # return img, target
        #embd = self.clip_embd[index // 5][index % 5]  # 512 vector


        # Read iamges:
        img_path = os.path.join(self.main_path, img_index)
        raw_img = torch.load(img_path).float()

        gt = raw_img * 2 - 1  # change to diffusion [-1,1]
        # shot_noise, read_noise = self.random_noise_levels()
        # if self.deterministic_seed_noise is not None:
        #     shot_noise, read_noise = self.deterministic_noise_levels()
        #     torch.manual_seed(self.deterministic_seed_noise)
        # lq = self.add_noise(raw_img, shot_noise, read_noise)
        lq = raw_img

        cond_dict = {'lq': lq}

        if self.clip_dataset:
            #img_clip_embd = self.clip_cache.get(img_index, None)
            img_clip_embd_data = self.clip_cache
            if img_clip_embd_data is None:
                print(f'Missing clip_embd for index {img_index}; got None')
            # print('fix clip0', end=' ')
            img_clip_embd_data /= torch.linalg.norm(img_clip_embd_data)
            cond_dict['clip_condition'] = img_clip_embd_data
        else:
            cond_dict['clip_condition'] = torch.zeros(self.xf_width, dtype=torch.float32)

        return gt, cond_dict

    def _get_imgtxt_index(self, index):
        img_index = self.files[index]
        embd_index = None
        return img_index, embd_index

    def __len__(self):
        if self.trim_len>0:
            return min(self.trim_len, len(self.files))
        return len(self.files)

class Dataset_Realcam_multiple(data.ConcatDataset):

    def __init__(self, main_path_dataset, clip_dataset, **kwargs):
        assert isinstance(main_path_dataset, list)
        if not clip_dataset:
            clip_dataset = [False] * len(main_path_dataset)
        datasets = [Dataset_Realcam(path_dataset, clip_dataset=clip_d, **kwargs) for path_dataset, clip_d in zip(main_path_dataset, clip_dataset)]
        #self.concat_datasets = torch.utils.data.ConcatDataset(datasets)
        super().__init__(datasets)
        start=0
        lengths = []
        for data in datasets:
            lengths.append(range(start, start+len(data)))
            start += len(data)

        self.batch_loader = lengths

        self.lengths = lengths
        print('lengths', lengths)

    @property # 'dataset.batch_loader' return a generator without using '()' to call the function
    def batch_loader2(self):
        for length in self.lengths:
            start, end = length
            yield range(start, end)