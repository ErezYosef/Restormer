## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

import numpy as np
import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
# import torch.nn.functional as F
import utils

from basicsr.models.archs.restormer_arch import Restormer
from skimage import img_as_ubyte
# import h5py
import scipy.io as sio
# from pdb import set_trace as stx

import sys
sys.path.append('..')
from diffusion.datasets import get_dataset # create_model_wrap
from guided_diffusion.process_raw2rgb_torch import process

from guided_diffusion.saving_imgs_utils import save_img,tensor2img


parser = argparse.ArgumentParser(description='Real Image Denoising using Restormer')

parser.add_argument('--input_dir', default='./Datasets/test/SIDD/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='/data1/erez/Documents/sidd/restormer_train/', type=str, help='Directory for results')
parser.add_argument('--weights', default='./pretrained_models/real_denoising.pth', type=str, help='Path to weights')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')
# CUDA_VISIBLE_DEVICES=2 python test_real_denoising_sidd.py --save_images --input_dir /data1/erez/datasets/sidd/srgb_test/
args = parser.parse_args()

####### Load yaml #######
yaml_file = 'Options/RealDenoising_Restormer.yml'
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

s = x['network_g'].pop('type')
##########################
device=torch.device('cuda')

# args.result_dir = os.path.join(args.result_dir, 'pretrained_results_real_s21')
args.result_dir = os.path.join(args.result_dir, 'pretrained_results_real_s21_set7')
os.makedirs(args.result_dir, exist_ok=True)

args.save_images = True
if args.save_images:
    result_dir_png = os.path.join(args.result_dir, 'png')
    os.makedirs(result_dir_png, exist_ok=True)

model_restoration = Restormer(**x['network_g'])

checkpoint = torch.load(args.weights)
model_restoration.load_state_dict(checkpoint['params'])
print("===>Testing using weights: ",args.weights)
model_restoration.to(device)
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

# Process data
from guided_diffusion.script_util import parse_yaml

# args.main_data_path_val = '/data2/erez/datasets/coco_captions/raw_imgs/val'
args.config_file = 'diffusion/coco_configs/concat_sample_config.yaml'
# args.config_file = 'diffusion/coco_configs/s_concat_sample_config.yaml'
args.config_file = 'diffusion/coco_configs/concat_real_sample_config.yaml'
if os.path.exists(args.config_file):
    print('ex')

args = parse_yaml(args)
print(args.dataset_type)


Dataset_class = get_dataset(args.dataset_type)  # ConcatModel_wrappret_class(UNetModel) #ConcatModelConv
dict_args = {k: v for k, v in vars(args).items()}
val_ds = Dataset_class(main_path_dataset=args.main_data_path_val, mode='val', **dict_args)

val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False,
                                         num_workers=1,
                                         batch_sampler=getattr(val_ds, 'batch_loader', None))
total_saved_samples=0
for sample_condition_data in val_loader:
    gt_imgs, data_dict = sample_condition_data
    print(gt_imgs.max(), gt_imgs.min())
    lq = data_dict['lq']
    print(lq.max(), lq.min())
    gt_rgb = process(gt_imgs, min_max=(-1, 1))
    lq_rgb = process(lq, min_max=(0, 1))
    print(gt_rgb.shape)
    noisy_patch = lq_rgb.to(device).clip(0,1)
    with torch.no_grad():
        restored_patch = model_restoration(noisy_patch)
    restored_patch = torch.clamp(restored_patch, 0, 1).cpu().detach()

    # gt_rgb = gt_rgb.squeeze()
    # lq_rgb = lq_rgb.squeeze()
    for im_i in range(gt_rgb.shape[0]):
        gt_rgbi = gt_rgb[im_i].clone()
        lq_rgbi = lq_rgb[im_i].clone()
        restored_patchi = restored_patch[im_i]
        gt_path = os.path.join(args.result_dir, f'gt{total_saved_samples:03d}.png')
        low_res_path = os.path.join(args.result_dir, f'sample{total_saved_samples:03d}_low_res.png')
        sample_path = os.path.join(args.result_dir, f'sample{total_saved_samples:03d}.png')

        # img_path = os.path.join(dir_to_save, f'sample{total_saved_samples:03d}.png')
        # save_img(tensor2img(gt_rgb, min_max=(0,1)), os.path.join(result_dir_png, 'gt.png'))

        save_img(tensor2img(gt_rgbi, min_max=(0,1)), gt_path)
        save_img(tensor2img(lq_rgbi, min_max=(0,1)), low_res_path)
        save_img(tensor2img(restored_patchi, min_max=(0,1)), sample_path)
        torch.save(gt_rgbi, gt_path.replace('.png', '.pt'))
        torch.save(lq_rgbi, low_res_path.replace('.png', '.pt'))
        torch.save(restored_patchi, sample_path.replace('.png', '.pt'))
        total_saved_samples += 1


    # print(data_dict.keys())
    # print(data_dict['lq'].shape)
    # print(data_dict['clip_condition'])
    #gt_imgs = gt_imgs.to(dtype=torch.float32, device=device)
exit()

filepath = os.path.join(args.input_dir, 'ValidationNoisyBlocksSrgb.mat')
img = sio.loadmat(filepath)
Inoisy = np.float32(np.array(img['ValidationNoisyBlocksSrgb']))
Inoisy /=255.
restored = np.zeros_like(Inoisy)
with torch.no_grad():
    for i in tqdm(range(40)):
        for k in range(32):
            noisy_patch = torch.from_numpy(Inoisy[i,k,:,:,:]).unsqueeze(0).permute(0,3,1,2).cuda()
            restored_patch = model_restoration(noisy_patch)
            restored_patch = torch.clamp(restored_patch,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0)
            restored[i,k,:,:,:] = restored_patch

            if args.save_images:
                save_file = os.path.join(result_dir_png, '%04d_%02d.png'%(i+1,k+1))
                utils.save_img(save_file, img_as_ubyte(restored_patch))

# save denoised data
sio.savemat(os.path.join(args.result_dir, 'Idenoised.mat'), {"Idenoised": restored,})
