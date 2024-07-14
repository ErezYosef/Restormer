import os
# import requests
# import torch
# import torchvision.transforms
# from PIL import Image
import torch
import yaml
import argparse

# import yaml
import glob
import numpy as np
#
# import requests
# from PIL import Image
# from io import BytesIO
from guided_diffusion.glide.ssim import ssim


import glob

from collections import OrderedDict
import lpips
from DISTS_pytorch import DISTS
import torchvision
from PIL import Image

def evaluator_mse_ssim(pred, gt, metrics_dict, i=None):
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
    if gt.dim() == 3:
        gt = gt.unsqueeze(0)
    elif gt.shape[0] != 1:
        metrics_dict['batch_size'] = gt.shape[0]
    mse_loss = torch.nn.MSELoss()
    mse_prev = metrics_dict.get('mse', 0)
    mse_counter = metrics_dict.get('mse_c', 0)
    mse_item = mse_loss(pred, gt) * pred.shape[0]
    metrics_dict['mse'] = (mse_prev * mse_counter + mse_item) / (mse_counter + pred.shape[0])
    metrics_dict['mse_c'] = mse_counter + pred.shape[0]

    ssim_prev = metrics_dict.get('ssim', 0)
    ssim_counter = metrics_dict.get('ssim_c', 0)
    ssim_item = ssim(pred, gt, val_range=1) * pred.shape[0]
    metrics_dict['ssim'] = (ssim_prev * ssim_counter + ssim_item) / (ssim_counter + pred.shape[0])
    metrics_dict['ssim_c'] = mse_counter + pred.shape[0]
    if i is not None:
        metrics_dict[f'mse_{i}'] = mse_item
        metrics_dict[f'ssim_{i}'] = ssim_item

    return metrics_dict

class evaluator_lpips():
    def __init__(self, device='cuda'):
        self.d = device
        self.loss_fn = lpips.LPIPS(net='alex').to(device)
        self.dists_fn = DISTS().to(device)
    def __call__(self, pred, gt, metrics_dict, i=None):
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
        if gt.dim() == 3:
            gt = gt.unsqueeze(0)
        elif gt.shape[0] != 1:
            metrics_dict['batch_size_lpips'] = gt.shape[0]

        _prev = metrics_dict.get('lpips', 0)
        _counter = metrics_dict.get('lpips_c', 0)
        predd = pred.to(self.d)
        gtd = gt.to(self.d)
        lpips_item = self.loss_fn(predd, gtd, normalize=True).mean() * pred.shape[0]
        metrics_dict['lpips'] = (_prev * _counter + lpips_item) / (_counter + pred.shape[0])
        metrics_dict['lpips_c'] = _counter + pred.shape[0]


        _prev = metrics_dict.get('dists', 0)
        _counter = metrics_dict.get('dists_c', 0)
        # image = totensor(image).unsqueeze(0).to(device)
        # loss = metric_func(image, gt_image)
        lpips_item = self.dists_fn(predd, gtd).mean() * pred.shape[0]
        metrics_dict['dists'] = (_prev * _counter + lpips_item) / (_counter + pred.shape[0])
        metrics_dict['dists_c'] = _counter + pred.shape[0]

        if i is not None:
            metrics_dict[f'lpips_{i}'] = lpips_item
            metrics_dict[f'dists_{i}'] = lpips_item

        return metrics_dict

class evaluator_general():
    def __init__(self, metric_name, metric_fun, device='cuda'):
        self.d = device
        self.name = metric_name
        self.metric = metric_fun

    def __call__(self, pred, gt, metrics_dict, i=None):
        _prev = metrics_dict.get(self.name, 0)
        _counter = metrics_dict.get(self.name+'_c', 0)
        # predd = pred.to(self.d)
        # gtd = gt.to(self.d)
        _item = self.metric(pred, gt)
        metrics_dict[self.name] = (_prev * _counter + _item) / (_counter + 1)
        metrics_dict[self.name+'_c'] = _counter + 1

        if i is not None:
            metrics_dict[f'{self.name}_{i}'] = _item

        return metrics_dict


def eval_nafnet(folder):
    #folder = os.path.join('/data1/erez/Documents/sidd/restormer_train/', folder)
    #folder = os.path.join('/data1/erez/Documents/sidd/NAFnet_sample/', folder)
    NAFNet_folder_name = 'NAFnet_FT100_ma10_sample'
    folder = os.path.join(f'/data1/erez/Documents/sidd/{NAFNet_folder_name}/', folder)
    files = os.listdir(folder)
    res_dict = {}

    names = ['gt{:03}.png', 'sample{:03}_low_res.png', 'sample{:03}.png']
    res_dict = {}
    print('ffolderL:', folder)

    gt_img = glob.glob(os.path.join(folder.replace(NAFNet_folder_name, 'restormer_train'), 'gt*.pt'))
    print('ffolderL:', folder)
    pred_img = glob.glob(os.path.join(folder, 'sample*[0-9].pt'))
    low_res_img = glob.glob(os.path.join(folder, 'sample*_low_res.pt'))
    [g.sort() for g in [gt_img, pred_img, low_res_img]]
    lpips_f = evaluator_lpips()
    metrics_dict = {}
    assert len(gt_img) == len(
        low_res_img), f"Lengths do not match: len(gt_img) = {len(gt_img)}, len(low_res_img) = {len(low_res_img)}"
    assert len(gt_img) == len(
        pred_img), f"Lengths do not match: len(gt_img) = {len(gt_img)}, len(pred_img) = {len(pred_img)}"
    print('ffolderL:', folder, pred_img[0])
    for i, gt_im_path in enumerate(gt_img):
        gt = torch.load(gt_im_path)
        res = torch.load(pred_img[i]) # gt+torch.randn_like(gt)*0.00000000001
        metrics_dict = evaluator_mse_ssim(res, gt, metrics_dict, i)
        metrics_dict = lpips_f(res, gt, metrics_dict, i)

    metrics_dict['psnr'] = -10 * torch.log10(metrics_dict['mse'])

    with open(os.path.join(folder, f'metric_dict.yaml'), 'w') as outfile:
        yaml.dump(dict_representor_yaml(metrics_dict), outfile, indent=4)

    #show_image(image)
    print('DONE!')
    for k,v in metrics_dict.items():
        if '_' not in k:
            print(k,v)

def eval_restormer(folder):
    folder = os.path.join('/data1/erez/Documents/sidd/restormer_train/', folder)
    files = os.listdir(folder)
    res_dict = {}

    names = ['gt{:03}.png', 'sample{:03}_low_res.png', 'sample{:03}.png']
    res_dict = {}
    print('ffolderL:', folder)

    gt_img = glob.glob(os.path.join(folder, 'gt*.pt'))
    print('ffolderL:', folder)
    pred_img = glob.glob(os.path.join(folder, 'sample*[0-9].pt'))
    low_res_img = glob.glob(os.path.join(folder, 'sample*_low_res.pt'))
    [g.sort() for g in [gt_img, pred_img, low_res_img]]
    lpips_f = evaluator_lpips()
    metrics_dict = {}
    assert len(gt_img) == len(low_res_img)
    assert len(gt_img) == len(pred_img)
    print('ffolderL:', folder, pred_img[0])
    for i, gt_im_path in enumerate(gt_img):
        gt = torch.load(gt_im_path)
        res = torch.load(pred_img[i]) # gt+torch.randn_like(gt)*0.00000000001
        metrics_dict = evaluator_mse_ssim(res, gt, metrics_dict, i)
        metrics_dict = lpips_f(res, gt, metrics_dict, i)

    metrics_dict['psnr'] = -10 * torch.log10(metrics_dict['mse'])

    with open(os.path.join(folder, f'metric_dict.yaml'), 'w') as outfile:
        yaml.dump(dict_representor_yaml(metrics_dict), outfile, indent=4)

    #show_image(image)
    print('DONE!')
    for k,v in metrics_dict.items():
        if '_' not in k:
            print(k,v)


def metrics_orig_env_RGB(mode=None):
    import lpips
    from guided_diffusion.glide.ssim import ssim
    from functools import partial


    device = torch.device('cuda')
    mse_loss = torch.nn.MSELoss()
    loss_fn = lpips.LPIPS(net='alex').to(device)
    lpips_f = partial(loss_fn, normalize=True)
    ssim_f = partial(ssim, val_range=1)
    totensor = torchvision.transforms.ToTensor()

    #image, image_bytes = load_image_from_url(image_url)
    def metric_get_dict(folder, metric_func: partial = loss_fn):
        #print('lpips')
        #files = os.listdir(folder)
        #images = [f for f in files if f.endswith('.png')]
        val_names = ['sample{:03}_low_res.png', 'sample{:03}.png']
        val_names = ['sample{:03}.png']
        gt_name = 'gt{:03}.png'
        res_dict = {}
        num_samples = len(glob.glob(os.path.join(folder, 'gt*.png')))
        print(f'WARNING evaluating {num_samples} images')
        for i in range(num_samples):
            gt_image = os.path.join(folder, gt_name.format(i))
            gt_image = Image.open(gt_image)
            gt_image = totensor(gt_image).unsqueeze(0).to(device)


            for im_name in val_names:
                img_path = os.path.join(folder, im_name.format(i))
                image = Image.open(img_path)
                image = totensor(image).unsqueeze(0).to(device)
                loss = metric_func(image, gt_image)
                #print(loss.shape)
                loss = loss.mean()
                res_dict[im_name.format(i)] = loss.item()# numpy()
                res_dict[im_name.format(999)] = res_dict.get(im_name.format(999), 0) + loss.item()

        for im_name in val_names:
            res_dict[im_name.format(999)] /= num_samples
            print(im_name, res_dict[im_name.format(999)])
        return res_dict

    test_fname = '230813_1200_concat_n03_1.2m'

    mode = mode or 'cat_30'
    if mode == 'cond':
        process_folders = ['230813_1135_cond_n02_1.2m', '230813_1142_cond_n03_1.2m', '230813_1308_lora_cond_s21_13m', '230813_1225_basecond_s21']
    elif mode == 'cond_30':
        process_folders = ['231026_1425_cond30_n03_1.2m', '231026_1504_cond30_n02_1.2m']
        process_folders = ['231106_1411_cond30_n04_1.2m', '231106_1410_cond30_n01_1.2m']
        process_folders = ['231113_1219_basecond_s21all']
    elif mode == 'cat':
        process_folders = ['230813_1149_concat_n02_1.2m', '230813_1200_concat_n03_1.2m', '230813_1318_lora_concat_s21_13m', '230813_1242_baseconcat_s21']
    elif mode == 'cat_30':
        process_folders = ['231026_1553_concat30_n03_1.2m', '231026_1642_concat30_n02_1.2mgh']
        # process_folders = ['231106_1311_concat30_n04_1.2m', '231106_1310_concat30_n01_1.2m']
        # process_folders = ['231113_1253_baseconcat_s21all']
    elif mode == 'cat_s21all':  # for s21all:
        process_folders = ['231009_1539_lora_concat_s21all_13m']
    elif mode == 'cond_s21all':
        process_folders = ['231009_1510_lora_cond_s21all_13m']
    elif mode == 'cond_s21all_mix':
        process_folders = ['240418_1232_loracond_s21all13m_mixcap']
    else:
        process_folders = ['']

    #process_folders = process_folders_cat


    for test_fname in process_folders:
        # cat: 230803_1923_basecat_Nlvl_L14n/ # cond: 230803_1653_basecond_Nlvl_L14norm
        if 'cond' in mode:
            folder = f'/data1/erez/Documents/sidd/diffusion_coco_storage/230803_1653_basecond_Nlvl_L14norm/{test_fname}/save/' # todo update path to model before run
        if 'cat' in mode:
            folder = f'/data1/erez/Documents/sidd/diffusion_coco_storage/230803_1923_basecat_Nlvl_L14n/{test_fname}/save/' # todo update path to model before run
        if 'n2v_01' in mode:
            folder = '/home/erez/PycharmProjects/raw_dn_related/n2v/models/n2v_coco_2210/save01b/'
            folder = '/home/erez/PycharmProjects/raw_dn_related/n2v/models/n2v_coco/save01/'
        if 'n2v_02' in mode:
            folder = '/home/erez/PycharmProjects/raw_dn_related/n2v/models/n2v_coco/save0.2/'
        if 'n2v_03' in mode:
            folder = '/home/erez/PycharmProjects/raw_dn_related/n2v/models/n2v_coco/save0.3/'
        if 'n2v_04' in mode:
            folder = '/home/erez/PycharmProjects/raw_dn_related/n2v/models/n2v_coco_2210/save04/'
        if 'n2v_s21' in mode:
            folder = '/home/erez/PycharmProjects/raw_dn_related/n2v/models/n2v_coco_2210cp/resume100_s21'
            folder = '/home/erez/PycharmProjects/raw_dn_related/n2v/models/n2v_coco_2210/s21'

        print('test_fname', test_fname)
        # print('lpips')
        # total_dict = metric_get_dict(folder, metric_func=lpips)
        # #print('lpips dict: ', total_dict)
        # torch.save(total_dict, os.path.join(folder, f'lpips_{test_fname}.pt'))
        #
        # print('ssim')
        # total_dict = metric_get_dict(folder, metric_func=ssim_f)
        # #print('ssim dict: ', total_dict)
        # torch.save(total_dict, os.path.join(folder, f'ssim_{test_fname}.pt'))

        print(f'mode == {mode} :: {folder}')
        dists_f = DISTS().to(device)
        total_dict = {}
        use_metrics = {'ssim': ssim_f, 'lpips': lpips_f, 'dists': dists_f, 'mse': mse_loss}

        for metric_name, metric in use_metrics.items():
            print(metric_name)
            metric_dict = metric_get_dict(folder, metric_func=metric)
            total_dict.update({f'{metric_name}_{k}':v for k,v in metric_dict.items()})

        #total_dict = metric_get_dict(folder, metric_func=dists)

        #print('ssim dict: ', total_dict)
        #torch.save(total_dict, os.path.join(folder, f'ssim_{test_fname}.pt'))
        if 'mse_sample999.png' in total_dict:
            total_dict['psnr'] = -10 * torch.log10(torch.tensor(total_dict['mse_sample999.png'])).item()
            print('psnr ',total_dict['psnr'])
        with open(os.path.join(folder, f'metrics_PSNRRGB_{test_fname}.yaml'), 'w') as outfile:
            yaml.dump(total_dict, outfile, indent=4)

def eval_folder(folder):
    folder = os.path.join('/data1/erez/Documents/sidd/restormer_train/', folder)

    files = os.listdir(folder)
    res_dict = {}

    names = ['gt{:03}.png', 'sample{:03}_low_res.png', 'sample{:03}.png']
    res_dict = {}
    gt_img = glob.glob(os.path.join(folder, 'gt*.pt'))
    pred_img = glob.glob(os.path.join(folder, 'sample*[0-9].pt'))
    low_res_img = glob.glob(os.path.join(folder, 'sample*_low_res.pt'))
    [g.sort() for g in [gt_img, pred_img, low_res_img]]
    lpips_f = evaluator_lpips()
    metrics_dict = {}
    assert len(gt_img) == len(low_res_img)
    assert len(gt_img) == len(pred_img)

    for i, gt_im_path in enumerate(gt_img):
        gt = torch.load(gt_im_path)
        res = torch.load(pred_img[i]) # gt+torch.randn_like(gt)*0.00000000001
        metrics_dict = evaluator_mse_ssim(res, gt, metrics_dict, i)
        metrics_dict = lpips_f(res, gt, metrics_dict, i)

    metrics_dict['psnr'] = -10 * torch.log10(metrics_dict['mse'])

    with open(os.path.join(folder, f'metric_dict.yaml'), 'w') as outfile:
        yaml.dump(dict_representor_yaml(metrics_dict), outfile, indent=4)

    #show_image(image)
    print('DONE!')
    for k,v in metrics_dict.items():
        if '_' not in k:
            print(k,v)


def dict_representor_yaml(dict_in):
    out = {}
    for k,v in dict_in.items():
        data = v.tolist() if isinstance(v, (np.ndarray, torch.Tensor)) else v
        out[k] = data
    return out

from guided_diffusion.glide import clip_util

def clip_score_nafnet(folder):
    #folder = os.path.join('/data1/erez/Documents/sidd/restormer_train/', folder)
    NAFNet_folder_name = 'NAFnet_FT100_ma10_sample'
    folder = os.path.join(f'/data1/erez/Documents/sidd/{NAFNet_folder_name}/', folder)
    files = os.listdir(folder)
    res_dict = {}

    names = ['gt{:03}.png', 'sample{:03}_low_res.png', 'sample{:03}.png']
    res_dict = {}
    print('ffolderL:', folder)

    gt_img = glob.glob(os.path.join(folder.replace(NAFNet_folder_name, 'restormer_train'), 'gt*.png'))
    print('ffolderL:', folder)
    pred_img = glob.glob(os.path.join(folder, 'sample*[0-9].png'))
    low_res_img = glob.glob(os.path.join(folder, 'sample*_low_res.png'))
    [g.sort() for g in [gt_img, pred_img, low_res_img]]
    metrics_dict = {}
    assert len(gt_img) == len(low_res_img)
    assert len(gt_img) == len(pred_img)
    print('ffolderL:', folder, pred_img[0])

    device = torch.device('cuda')
    # clipscore = 1 # ViT-L/14@336px
    clip_model = clip_util.clip_model_wrap2(model_name='ViT-L/14@336px', device=device)
    mode='val'
    clip_embd_data_path = f'pretrained_models/cococap/clip_embd_L14_{mode}.pt'
    clip_cache = torch.load(clip_embd_data_path, map_location='cpu')
    with open(f'pretrained_models/cococap/clip_caps_val_end_at_30.yaml', 'r') as s:
        captions_data = yaml.load(s, yaml.SafeLoader)

    clipscore = evaluator_general('clipscore', clip_model.get_cosine, device=device)

    for i, gt_im_path in enumerate(gt_img):
        img_clip_embd = clip_cache.get(i, None)[0]
        img_clip_embd = captions_data.get(i, None)[0]

        gt_image = Image.open(gt_im_path)

        image = Image.open(pred_img[i])
        metrics_dict = clipscore(image, img_clip_embd, metrics_dict, i)

    with open(os.path.join(folder, f'clipscore336_dict.yaml'), 'w') as outfile:
        yaml.dump(dict_representor_yaml(metrics_dict), outfile, indent=4)

    #show_image(image)
    print('DONE!')
    for k,v in metrics_dict.items():
        if '_' not in k:
            print(k,v)


def clip_score_restormer(folder):
    folder = os.path.join('/data1/erez/Documents/sidd/restormer_train/', folder)
    files = os.listdir(folder)
    res_dict = {}

    names = ['gt{:03}.png', 'sample{:03}_low_res.png', 'sample{:03}.png']
    res_dict = {}
    print('ffolderL:', folder)

    gt_img = glob.glob(os.path.join(folder, 'gt*.png'))
    print('ffolderL:', folder)
    pred_img = glob.glob(os.path.join(folder, 'sample*[0-9].png'))
    low_res_img = glob.glob(os.path.join(folder, 'sample*_low_res.png'))
    [g.sort() for g in [gt_img, pred_img, low_res_img]]
    metrics_dict = {}
    assert len(gt_img) == len(low_res_img)
    assert len(gt_img) == len(pred_img)
    print('ffolderL:', folder, pred_img[0])

    device = torch.device('cuda')
    # clipscore = 1 # ViT-L/14@336px
    clip_model = clip_util.clip_model_wrap2(model_name='ViT-L/14@336px', device=device)
    mode='val'
    clip_embd_data_path = f'pretrained_models/cococap/clip_embd_L14_{mode}.pt'
    clip_cache = torch.load(clip_embd_data_path, map_location='cpu')
    with open(f'pretrained_models/cococap/clip_caps_val_end_at_30.yaml', 'r') as s:
        captions_data = yaml.load(s, yaml.SafeLoader)


    clipscore = evaluator_general('clipscore', clip_model.get_cosine, device=device)

    for i, gt_im_path in enumerate(gt_img):
        img_clip_embd = clip_cache.get(i, None)[0]
        img_clip_embd = captions_data.get(i, None)[0]


        gt_image = Image.open(gt_im_path)

        image = Image.open(pred_img[i])
        metrics_dict = clipscore(image, img_clip_embd, metrics_dict, i)


    with open(os.path.join(folder, f'clipscore_dict.yaml'), 'w') as outfile:
        yaml.dump(dict_representor_yaml(metrics_dict), outfile, indent=4)

    #show_image(image)
    print('DONE!')
    for k,v in metrics_dict.items():
        if '_' not in k:
            print(k,v)



if __name__ == '__main__':
    # diffnoise env:
    # rerun since deleted:
    # eval_restormer('pretrained_results_s21') # evaluate metrics for restormer
    # eval_restormer('finetune/experiments/RealDenoisingS21_Restormer/models/net_g_100/') # evaluate metrics for restormer
    # eval_nafnet('pretrained_results03') # evaluate metrics for restormer
    #eval_nafnet('pretrained_results_s21') # evaluate metrics for restormer
    #metrics_orig_env_RGB('cond_s21all_mix') # evaluate my method for psnr in rgb space

    # rerun to get better save of clipscore_dict.yaml
    # clip_score_restormer('pretrained_results01') # evaluate metrics for restormer
    # clip_score_restormer('pretrained_results03') # evaluate metrics for restormer
    # clip_score_restormer('pretrained_results_s21') # evaluate metrics for restormer
    #clip_score_restormer('finetune/experiments/RealDenoisingS21_Restormer/models/net_g_500/') # evaluate metrics for restormer

    #
    # clip_score_nafnet('pretrained_results01') # evaluate metrics for restormer _s21 01 03
    # clip_score_nafnet('pretrained_results03') # evaluate metrics for restormer _s21 01 03
    clip_score_nafnet('pretrained_results_s21') # evaluate metrics for restormer _s21 01 03

