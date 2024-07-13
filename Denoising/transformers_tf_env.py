import os
import requests
import torch
import torchvision.transforms
from PIL import Image

# USE: transformers38 conda env on runway server!
import yaml
import glob
import numpy as np

from guided_diffusion.glide import clip_util


def captioning_crops_save_dict():
    # use transformer env for BLIP paragraph to images
    from transformers import BlipProcessor, BlipForConditionalGeneration
    main_path = '/data1/erez/datasets/s7_isp_b/'
    folders = [folder for folder in os.listdir(main_path) if os.path.isdir(os.path.join(main_path, folder))]
    folders.sort()
    data_dict = {}
    iters=1

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda")

    # folder = os.path.join(main_path, folders[0])
    # files = os.listdir(folder)
    # files = [f for f in files if 'jpg_large' in f]
    # file = os.path.join(folder, files[0])
    c=0
    print(len(folders))
    for folder in folders:
        folder_path = os.path.join(main_path, folder)
        files = os.listdir(folder_path)
        files = [f for f in files if 'jpg_large' in f]
        files.sort()
        #print(len(files))
        print(folder, ':')
        folder_dict = {}
        for file in files:
            file_path = os.path.join(folder_path, file)
            print(file, end='\t\t > \t\t')

            raw_image = Image.open(file_path).convert('RGB')
            width, height = raw_image.size
            #for iter in range(iters):
            # unconditional image captioning
            #raw_image = raw_image.crop((iter*10, iter*10, width, height))
            inputs = processor(raw_image, return_tensors="pt").to("cuda")
            #print('=== typeof inputes', type(inputs))

            out = model.generate(**inputs)
            text=processor.decode(out[0], skip_special_tokens=True)
            print(text)
            folder_dict[f'text{file[0]}'] = text
            c+=1
            #torch.save(data_dict, os.path.join(main_path, 'text_dict.pt'))
            #exit()
        data_dict[folder] = folder_dict
    torch.save(data_dict, os.path.join(main_path, 'text_dict.pt'))
    print(c)

def musiq_eval_folders():
    # from: https://colab.research.google.com/github/google-research/google-research/blob/master/musiq/Inference_with_MUSIQ.ipynb#scrollTo=1wO1PVJqlQxr
    # use env tf24python38 for MUSIC metric
    import tensorflow as tf
    import tensorflow_hub as hub

    import requests
    from PIL import Image
    from io import BytesIO

    #import matplotlib.pyplot as plt
    import numpy as np

    selected_model = 'spaq'  # @param ['spaq', 'koniq', 'paq2piq', 'ava']

    NAME_TO_HANDLE = {
        # Model trained on SPAQ dataset: https://github.com/h4nwei/SPAQ
        'spaq': 'https://tfhub.dev/google/musiq/spaq/1',

        # Model trained on KonIQ-10K dataset: http://database.mmsp-kn.de/koniq-10k-database.html
        'koniq': 'https://tfhub.dev/google/musiq/koniq-10k/1',

        # Model trained on PaQ2PiQ dataset: https://github.com/baidut/PaQ-2-PiQ
        'paq2piq': 'https://tfhub.dev/google/musiq/paq2piq/1',

        # Model trained on AVA dataset: https://ieeexplore.ieee.org/document/6247954
        'ava': 'https://tfhub.dev/google/musiq/ava/1',
    }

    model_handle = NAME_TO_HANDLE[selected_model]
    model = hub.load(model_handle)
    #predict_fn = model.signatures['serving_default']

    print(f'loaded model {selected_model} ({model_handle})')

    def load_image_from_url(img_url):
        """Returns an image with shape [1, height, width, num_channels]."""
        user_agent = {'User-agent': 'Colab Sample (https://tensorflow.org)'}
        response = requests.get(img_url, headers=user_agent)
        image_bytes = BytesIO(response.content)
        image = Image.open(image_bytes)
        return image, response.content

    def load_image_to_tf(img_path):
        image = Image.open(img_path)
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        return img_byte_arr

    def show_image(image, title=''):
        image_size = image.size
        plt.imshow(image)
        plt.axis('on')
        plt.title(title)
        plt.show()

    # image_url = 'https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEgr0DKaAoO6qTrJo3hXP8UM3D4AB8gQeNI22Q2QphBVGgn-5v84tjhH3ZWTlGtlUoPdlcx54dM93Qi04MuN7eBbj9WlT8Qxy6B2Us4kcn_53FH28MnTtGCzMPhjCVGIgXRL8ZEMeO-7iue7sNEGxBtgx2bI-eKDQAondM8Dfjb1FaybFgUQji4UU9-0vQ/s1024/image9.png'  # @param {type: 'string'}
    from collections import OrderedDict

    #image, image_bytes = load_image_from_url(image_url)
    def model_get_dict(folder, model_name):
        model_handle = NAME_TO_HANDLE[model_name]
        model = hub.load(model_handle)
        predict_fn = model.signatures['serving_default']

        files = os.listdir(folder)
        images = [f for f in files if f.endswith('.png')]
        names = ['gt{:03}.png', 'sample{:03}_low_res.png', 'sample{:03}.png']
        res_dict = {}
        num_samples = len(glob.glob(os.path.join(folder, 'gt*.png')))
        print(f'WARNING evaluating {num_samples} images')
        for i in range(num_samples):
            for im_name in names:
                img_path = os.path.join(folder, im_name.format(i))
                image_bytes = load_image_to_tf(img_path)
                prediction = predict_fn(tf.constant(image_bytes))
                res_dict[im_name.format(i)] = prediction['output_0'].numpy()
                res_dict[im_name.format(999)] = res_dict.get(im_name.format(999), 0) + prediction['output_0'].numpy()#.item()

        for im_name in names:
            res_dict[im_name.format(999)] /= num_samples
            print('MEAN FOR ', im_name, res_dict[im_name.format(999)])
        return res_dict

    test_fname = '230813_1200_concat_n03_1.2m'
    def process_folders(folders):
        for test_fname in folders:
            folder = f'/data1/erez/Documents/sidd/diffusion_coco_storage/230803_1653_basecond_Nlvl_L14norm/{test_fname}/save/'
            #res_dict = model_get_dict(folder, 'koniq')
            #print(res_dict)
            total_dict = {}
            for k in NAME_TO_HANDLE.keys():
                total_dict[k] = model_get_dict(folder, k)
            #torch.save(total_dict, os.path.join(folder, f'musiq_{test_fname}.pt'))
            with open(os.path.join(folder, f'musiq_{test_fname}.yaml'), 'w') as outfile:
                yaml.dump(total_dict, outfile, indent=4)

    #process_folders(['230813_1135_cond_n02_1.2m', '230813_1142_cond_n03_1.2m', '230813_1308_lora_cond_s21_13m'])
    #process_folders(['230919_1547_lora_cond_real21correct'])
    # p1 = '/data1/erez/Documents/sidd/diffusion_coco_storage/230803_1923_basecat_Nlvl_L14n/230813_1149_concat_n02_1.2m/save/gt001.png'
    # p2 = '/data1/erez/Documents/sidd/diffusion_coco_storage/230803_1923_basecat_Nlvl_L14n/230813_1149_concat_n02_1.2m/save/sample001_low_res.png'
    # p3 = '/data1/erez/Documents/sidd/diffusion_coco_storage/230803_1923_basecat_Nlvl_L14n/230813_1149_concat_n02_1.2m/save/sample001.png'
    #
    # # img_byte_arr = BytesIO()
    # # image.save(img_byte_arr, format='PNG')
    # # img_byte_arr = img_byte_arr.getvalue()
    # from tensorflow.python.client import device_lib
    # print(device_lib.list_local_devices())
    #
    #
    # image_bytes=load_image_to_tf(p1)
    # prediction = predict_fn(tf.constant(image_bytes))
    # print("predicted MOS: ", prediction)
    # image_bytes=load_image_to_tf(p2)
    # prediction = predict_fn(tf.constant(image_bytes))
    # print("p2 predicted MOS: ", prediction)
    # image_bytes=load_image_to_tf(p3)
    # prediction = predict_fn(tf.constant(image_bytes))
    # print("p2 predicted MOS: ", prediction)

    #show_image(image)


def metrics_orig_env():
    import lpips
    from guided_diffusion.glide.ssim import ssim
    from functools import partial


    device = torch.device('cuda')
    mseloss = torch.nn.MSELoss()
    loss_fn = lpips.LPIPS(net='alex').to(device)
    lpips_f = partial(loss_fn, normalize=True)
    ssim_f = partial(ssim, val_range=1)
    totensor = torchvision.transforms.ToTensor()

    # image_url = 'https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEgr0DKaAoO6qTrJo3hXP8UM3D4AB8gQeNI22Q2QphBVGgn-5v84tjhH3ZWTlGtlUoPdlcx54dM93Qi04MuN7eBbj9WlT8Qxy6B2Us4kcn_53FH28MnTtGCzMPhjCVGIgXRL8ZEMeO-7iue7sNEGxBtgx2bI-eKDQAondM8Dfjb1FaybFgUQji4UU9-0vQ/s1024/image9.png'  # @param {type: 'string'}
    from collections import OrderedDict

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

    # def ssim_get_dict(folder):
    #     print('ssim')
    #     from guided_diffusion.glide.ssim import ssim
    #
    #     val_names = ['sample{:03}_low_res.png', 'sample{:03}.png']
    #     gt_name = 'gt{:03}.png'
    #     res_dict = {}
    #     num_samples = 8
    #     for i in range(num_samples):
    #         gt_image = os.path.join(folder, gt_name.format(i))
    #         gt_image = Image.open(gt_image)
    #         gt_image = totensor(gt_image).unsqueeze(0).to(device)
    #
    #
    #         for im_name in val_names:
    #             img_path = os.path.join(folder, im_name.format(i))
    #             image = Image.open(img_path)
    #             image = totensor(image).unsqueeze(0).to(device)
    #             loss = ssim(image, gt_image, val_range=1)
    #             #print(loss.shape)
    #             #loss = loss.mean()
    #             res_dict[im_name.format(i)] = loss.item()
    #             res_dict[im_name.format(999)] = res_dict.get(im_name.format(999), 0) + loss.item()
    #
    #     for im_name in val_names:
    #         res_dict[im_name.format(999)] /= num_samples
    #         print(im_name, res_dict[im_name.format(999)])
    #     return res_dict


    test_fname = '230813_1200_concat_n03_1.2m'

    mode = 'cond_s21all_mix'
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
        process_folders = ['231106_1311_concat30_n04_1.2m', '231106_1310_concat30_n01_1.2m']
        process_folders = ['231113_1253_baseconcat_s21all']
    elif mode == 'cat_s21all':  # for s21all:
        process_folders = ['231009_1539_lora_concat_s21all_13m']
    elif mode == 'cond_s21all':
        process_folders = ['231009_1510_lora_cond_s21all_13m']
    elif mode == 'cond_s21all_mix':
        process_folders = ['240418_1409_loracond_s21all13m_mixcap80']
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
        from DISTS_pytorch import DISTS
        dists_f = DISTS().to(device)
        total_dict = {}
        use_metrics = {'ssim': ssim_f, 'lpips': lpips_f, 'dists': dists_f, 'mse': mseloss}

        for metric_name, metric in use_metrics.items():
            print(metric_name)
            metric_dict = metric_get_dict(folder, metric_func=metric)
            total_dict.update({f'{metric_name}_{k}':v for k,v in metric_dict.items()})

        #total_dict = metric_get_dict(folder, metric_func=dists)

        #print('ssim dict: ', total_dict)
        #torch.save(total_dict, os.path.join(folder, f'ssim_{test_fname}.pt'))
        if 'mse_sample999.png' in total_dict:
            total_dict['psnr'] = -10 * torch.log10(torch.tensor(total_dict['mse_sample999.png'])).item()
            print('psnr', total_dict['psnr'])

        with open(os.path.join(folder, f'metrics_{test_fname}.yaml'), 'w') as outfile:
            yaml.dump(total_dict, outfile, indent=4)

def compute_raw_psnr():
    # Used for  n2v evaluation.
    mse = torch.nn.MSELoss()
    from guided_diffusion.process_raw2rgb_torch import process
    from guided_diffusion.saving_imgs_utils import save_img, tensor2img

    def get_metric_dict(folder, metric_func):

        im_name = 'sample{:03}.pt'
        gt_name = 'gt{:03}.pt'
        res_dict = {}
        num_samples = len(glob.glob(os.path.join(folder, 'gt*.pt')))
        print(f'WARNING evaluating {num_samples} images')
        for i in range(num_samples):

            gt_image_p = os.path.join(folder, gt_name.format(i))
            gt_image = torch.load(gt_image_p)
            # gt_image = totensor(gt_image).unsqueeze(0).to(device)

            img_path = os.path.join(folder, im_name.format(i))
            image = torch.load(img_path)
            # image = totensor(image).unsqueeze(0).to(device)
            loss = metric_func(image, gt_image)
            # print(loss.shape)
            loss = loss.mean()
            res_dict[im_name.format(i)] = loss.item()  # numpy()
            res_dict[im_name.format(999)] = res_dict.get(im_name.format(999), 0) + loss.item()
            #rgb = process(gt_image)
            save_img(tensor2img(gt_image, min_max=(0, 1)), gt_image_p.replace('.pt', '.png'))
            save_img(tensor2img(image, min_max=(0, 1)), img_path.replace('.pt', '.png'))

        res_dict[im_name.format(999)] /= num_samples
        print(im_name, res_dict[im_name.format(999)], type(res_dict[im_name.format(999)]))

        res_dict['psnr_total'] = float( -10 * np.log10(res_dict[im_name.format(999)]))
        print(type(res_dict[im_name.format(999)]))
        print('psnr_total: ', res_dict['psnr_total'])


        return res_dict

    main_path = '/home/erez/PycharmProjects/raw_dn_related/CycleISP/results/coco/coco_raw_s21/'
    #main_path = '/home/erez/PycharmProjects/raw_dn_related/deep-image-prior/results/coco/coco_raw_s21/'
    main_path = '/home/erez/PycharmProjects/raw_dn_related/n2v/models/n2v_coco_2210/s21/'
    main_path = '/home/erez/PycharmProjects/raw_dn_related/n2v/models/n2v_coco_2210/save04/'
    #main_path = '/home/erez/PycharmProjects/raw_dn_related/n2v/models/n2v_coco/save01/'
    #main_path = '/home/erez/PycharmProjects/raw_dn_related/n2v/models/n2v_coco_2210cp2/save1/'
    main_path = '/home/erez/PycharmProjects/raw_dn_related/n2v/models/n2v_coco_2210/realcam_s21stat/'
    main_path = '/home/erez/PycharmProjects/raw_dn_related/n2v/models/n2v_coco_2210/realcam_set8/'


    total_dict = {}
    print('test_folder PSNR: ', main_path)
    metric_dict = get_metric_dict(main_path, metric_func=mse)
    total_dict.update({f'mse_{k}':v for k,v in metric_dict.items()})
    #for k,v in metric_dict.items()
    #print('ssim dict: ', total_dict)
    #torch.save(total_dict, os.path.join(main_path, f'raw_psnr_{test_fname}.pt'))
    with open(os.path.join(main_path, f'raw_psnr.yaml'), 'w') as outfile:
        yaml.dump(total_dict, outfile, indent=4)

    #mse_loss += mse(sample_cp, gt_imgs) * sample_cp.shape[0] / 4  # due to dynamic range
    #avg_tot_loss = mse_loss / samples_counter

def clip_score_orig_env(runfolder=None):
    import lpips
    from guided_diffusion.glide.ssim import ssim
    from functools import partial


    device = torch.device('cuda')
    clipscore = 1
    clip_model = clip_util.clip_model_wrap2(model_name='ViT-L/14@336px', device=device)# ViT-L/14@336px

    totensor = torchvision.transforms.ToTensor()

    # image_url = 'https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEgr0DKaAoO6qTrJo3hXP8UM3D4AB8gQeNI22Q2QphBVGgn-5v84tjhH3ZWTlGtlUoPdlcx54dM93Qi04MuN7eBbj9WlT8Qxy6B2Us4kcn_53FH28MnTtGCzMPhjCVGIgXRL8ZEMeO-7iue7sNEGxBtgx2bI-eKDQAondM8Dfjb1FaybFgUQji4UU9-0vQ/s1024/image9.png'  # @param {type: 'string'}
    from collections import OrderedDict
    mode='val'
    clip_embd_data_path = f'pretrained_models/cococap/clip_embd_L14_{mode}.pt'
    clip_cache = torch.load(clip_embd_data_path, map_location='cpu')

    with open(f'pretrained_models/cococap/clip_caps_val_end_at_30.yaml', 'r') as s:
        captions_data = yaml.load(s, yaml.SafeLoader)

    #image, image_bytes = load_image_from_url(image_url)
    def metric_get_dict(folder, metric_func: partial):
        #print('lpips')
        #files = os.listdir(folder)
        #images = [f for f in files if f.endswith('.png')]
        val_names = ['sample{:03}_low_res.png', 'sample{:03}.png']
        val_names = ['sample{:03}.png']
        # val_names = ['sample{:03}_low_res.png']
        gt_name = 'gt{:03}.png'
        res_dict = {}
        num_samples = len(glob.glob(os.path.join(folder, 'gt*.png')))
        print(f'WARNING evaluating {num_samples} images')
        for i in range(num_samples):
            img_clip_embd = clip_cache.get(i, None)[0]
            img_clip_embd = captions_data.get(i, None)[0]
            # print(img_clip_embd)

            gt_image = os.path.join(folder, gt_name.format(i))
            gt_image = Image.open(gt_image)
            gt_imagept = totensor(gt_image).unsqueeze(0).to(device)


            for im_name in val_names:
                img_path = os.path.join(folder, im_name.format(i))
                image = Image.open(img_path)
                loss = metric_func.get_cosine(image, img_clip_embd)
                #image = totensor(image).unsqueeze(0).to(device)
                #loss = metric_func(image, gt_image)
                #print(loss.shape)
                loss = loss.mean()
                res_dict[im_name.format(i)] = loss.item()# numpy()
                res_dict[im_name.format(999)] = res_dict.get(im_name.format(999), 0) + loss.item()

        for im_name in val_names:
            res_dict[im_name.format(999)] /= num_samples
            print(im_name, res_dict[im_name.format(999)])
        return res_dict

    test_fname = '230813_1200_concat_n03_1.2m'

    mode = runfolder or 'cond_s21all_mix'
    if mode == 'cond':
        process_folders = ['230813_1135_cond_n02_1.2m', '230813_1142_cond_n03_1.2m', '230813_1308_lora_cond_s21_13m', '230813_1225_basecond_s21']
    elif mode == 'cond_30':
        # process_folders = ['231026_1425_cond30_n03_1.2m', '231026_1504_cond30_n02_1.2m']
        process_folders = ['231106_1411_cond30_n04_1.2m', '231106_1410_cond30_n01_1.2m']
        process_folders = ['231113_1219_basecond_s21all'] + ['231026_1425_cond30_n03_1.2m', '231026_1504_cond30_n02_1.2m']
    elif mode == 'cat':
        process_folders = ['230813_1149_concat_n02_1.2m', '230813_1200_concat_n03_1.2m', '230813_1318_lora_concat_s21_13m', '230813_1242_baseconcat_s21']
    elif mode == 'cat_30':
        process_folders = ['231026_1553_concat30_n03_1.2m', '231026_1642_concat30_n02_1.2mgh']
        process_folders = ['231106_1311_concat30_n04_1.2m', '231106_1310_concat30_n01_1.2m']
        process_folders = ['231113_1253_baseconcat_s21all'] + ['231026_1553_concat30_n03_1.2m', '231026_1642_concat30_n02_1.2mgh'] + ['231106_1311_concat30_n04_1.2m', '231106_1310_concat30_n01_1.2m']
    elif mode == 'cat_s21all':  # for s21all:
        process_folders = ['231009_1539_lora_concat_s21all_13m']
    elif mode == 'cond_s21all':
        process_folders = ['231009_1510_lora_cond_s21all_13m']
    elif mode == 'cond_s21all_mix':
        process_folders = ['240418_1409_loracond_s21all13m_mixcap80']
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
        total_dict = {}
        use_metrics = {'clip_score': clip_model}

        for metric_name, metric in use_metrics.items():
            print(metric_name)
            metric_dict = metric_get_dict(folder, metric_func=metric)
            total_dict.update({f'{metric_name}_{k}':v for k,v in metric_dict.items()})

        #total_dict = metric_get_dict(folder, metric_func=dists)

        #print('ssim dict: ', total_dict)
        #torch.save(total_dict, os.path.join(folder, f'ssim_{test_fname}.pt'))
        if 'clip_score_sample999.png' in total_dict:
            print('clip_score_sample999.png', total_dict['clip_score_sample999.png'])

        with open(os.path.join(folder, f'clip_score336px_{test_fname}.yaml'), 'w') as outfile:
            yaml.dump(total_dict, outfile, indent=4)

def metrics_orig_env_and_raw_psnr():
    import lpips
    from guided_diffusion.glide.ssim import ssim
    from functools import partial

    from guided_diffusion.saving_imgs_utils import save_img, tensor2img


    device = torch.device('cuda')
    mseloss = torch.nn.MSELoss()
    loss_fn = lpips.LPIPS(net='alex').to(device)
    lpips_f = partial(loss_fn, normalize=True)
    ssim_f = partial(ssim, val_range=1)
    totensor = torchvision.transforms.ToTensor()

    # image_url = 'https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEgr0DKaAoO6qTrJo3hXP8UM3D4AB8gQeNI22Q2QphBVGgn-5v84tjhH3ZWTlGtlUoPdlcx54dM93Qi04MuN7eBbj9WlT8Qxy6B2Us4kcn_53FH28MnTtGCzMPhjCVGIgXRL8ZEMeO-7iue7sNEGxBtgx2bI-eKDQAondM8Dfjb1FaybFgUQji4UU9-0vQ/s1024/image9.png'  # @param {type: 'string'}
    from collections import OrderedDict

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

    def get_raw_metric_dict(folder, metric_func):

        im_name = 'sample{:03}.pt'
        gt_name = 'gt{:03}.pt'
        res_dict = {}
        num_samples = len(glob.glob(os.path.join(folder, 'gt*.pt')))
        print(f'WARNING evaluating {num_samples} images')
        for i in range(num_samples):

            gt_image_p = os.path.join(folder, gt_name.format(i))
            gt_image = torch.load(gt_image_p)
            # gt_image = totensor(gt_image).unsqueeze(0).to(device)

            img_path = os.path.join(folder, im_name.format(i))
            image = torch.load(img_path)
            # image = totensor(image).unsqueeze(0).to(device)
            image = (image+1) / 2
            gt_image = (gt_image+1) / 2
            loss = metric_func(image, gt_image)
            # print(loss.shape)
            loss = loss.mean()
            res_dict[im_name.format(i)] = loss.item()  # numpy()
            res_dict[im_name.format(999)] = res_dict.get(im_name.format(999), 0) + loss.item()
            #rgb = process(gt_image)
            #save_img(tensor2img(gt_image, min_max=(0, 1)), gt_image_p.replace('.pt', '.png'))
            #save_img(tensor2img(image, min_max=(0, 1)), img_path.replace('.pt', '.png'))

        res_dict[im_name.format(999)] /= num_samples
        print(im_name, res_dict[im_name.format(999)], type(res_dict[im_name.format(999)]))

        # res_dict['psnr_total'] = float( -10 * np.log10(res_dict[im_name.format(999)]))
        # print(type(res_dict[im_name.format(999)]))
        # print('psnr_total: ', res_dict['psnr_total'])


        return res_dict


    test_fname = '230813_1200_concat_n03_1.2m'

    mode = 'cond_s21all_x20'
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
        process_folders = ['231106_1311_concat30_n04_1.2m', '231106_1310_concat30_n01_1.2m']
        process_folders = ['231113_1253_baseconcat_s21all']
    elif mode == 'cat_s21all':  # for s21all:
        process_folders = ['231009_1539_lora_concat_s21all_13m']
    elif mode == 'cond_s21all':
        process_folders = ['231009_1510_lora_cond_s21all_13m']
    elif mode == 'cond_s21all_mix':
        process_folders = ['240418_1409_loracond_s21all13m_mixcap80']
    elif mode == 'cond_s21all_x20':
        process_folders = ['240710_1513_x20_loracond_s21all', '240712_1548_x20_basecond_s21all']
    elif mode == 'cat_s21all_x20':
        process_folders = ['240711_1435_x20_loracat_fix_s21all', '240711_1257_x20_loracat_s21all']
    else:
        process_folders = ['']

    #process_folders = process_folders_cat


    for test_fname in process_folders:
        # cat: 230803_1923_basecat_Nlvl_L14n/ # cond: 230803_1653_basecond_Nlvl_L14norm
        if 'cond' in mode:
            folder = f'/data1/erez/Documents/sidd/diffusion_coco_storage/230803_1653_basecond_Nlvl_L14norm/{test_fname}/save_all4/'
        if 'cat' in mode:
            folder = f'/data1/erez/Documents/sidd/diffusion_coco_storage/230803_1923_basecat_Nlvl_L14n/{test_fname}/save_all4/'
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

        print(f'mode == {mode} :: {folder}')
        from DISTS_pytorch import DISTS
        dists_f = DISTS().to(device)
        total_dict = {}
        use_metrics = {'ssim': ssim_f, 'lpips': lpips_f, 'dists': dists_f, 'mse': mseloss}

        for metric_name, metric in use_metrics.items():
            print(metric_name)
            metric_dict = metric_get_dict(folder, metric_func=metric)
            total_dict.update({f'{metric_name}_{k}':v for k,v in metric_dict.items()})

        if 'mse_sample999.png' in total_dict:
            total_dict['psnr'] = -10 * torch.log10(torch.tensor(total_dict['mse_sample999.png'])).item()
            print('psnr', total_dict['psnr'])

        # ---------
        print('test_folder PSNR: ', folder)
        metric_dict = get_raw_metric_dict(folder, metric_func=mseloss)
        total_dict.update({f'mse_RAW_{k}': v for k, v in metric_dict.items()})

        if 'mse_RAW_sample999.pt' in total_dict:
            total_dict['psnr_RAW'] = -10 * torch.log10(torch.tensor(total_dict['mse_RAW_sample999.pt'])).item()
            print('psnr_RAW', total_dict['psnr_RAW'])


        with open(os.path.join(folder, f'metrics_more_{test_fname}.yaml'), 'w') as outfile:
            yaml.dump(total_dict, outfile, indent=4)

def collecting_more_sampling():
    test_fname = '240711_1435_x20_loracat_fix_s21all'
    test_fname = '240711_1257_x20_loracat_s21all'
    test_fname = '240712_1548_x20_basecond_s21all'
    test_fname = '240713_1349_x20_mix_loracond_s21all'
    path = f'/data1/erez/Documents/sidd/diffusion_coco_storage/230803_1653_basecond_Nlvl_L14norm/{test_fname}'
    #path = f'/data1/erez/Documents/sidd/diffusion_coco_storage/230803_1923_basecat_Nlvl_L14n/{test_fname}'
    print(os.listdir(path))
    total_dict = {}
    total_count = {}

    from guided_diffusion.saving_imgs_utils import save_img, tensor2img

    out_folder = os.path.join(path, 'save_all4')
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    for i in range(20):
        save_folder = os.path.join(path, f'save{i}')
        files = glob.glob(f'{save_folder}/sample*[0-9].pt')
        files = glob.glob(f'{save_folder}/*.pt')
        files.sort()
        #print(files)
        for file in files:
            allpath, imname = os.path.split(file)
            image = torch.load(file, map_location='cpu')
            total_dict[imname] = total_dict.get(imname, torch.zeros_like(image)) + image
            total_count[imname] = total_count.get(imname, 0) + 1
            #print(total_count[imname])
        #print(total_count)

    print('number of images', len(total_dict.keys()))
    print(total_count)

    for imname, v in total_dict.items():
        if total_count[imname] != 20:
            print(f'num of image {imname} is {total_count[imname]}')
        total_dict[imname] = v / total_count[imname]
        out_file = os.path.join(out_folder, imname)
        torch.save(total_dict[imname], out_file)
        save_img(tensor2img(total_dict[imname]), out_file.replace('.pt', '.png'))



if __name__ == '__main__':
    # for 'transformers' image to paragraph BLIP use env: transformers38 (runai conda)
    #captioning_crops_save_dict()
    #musiq_eval_folders()

    #metrics_orig_env()

    # FOR n2v data: diffnoise env
    compute_raw_psnr()
    # metrics_orig_env()
    # clip_score_orig_env('cat_30')
    # clip_score_orig_env('cat_s21all')
    # clip_score_orig_env('cond_30')



    # clip_score_orig_env('n2v_03')
    # clip_score_orig_env('n2v_01')
    # clip_score_orig_env('n2v_s21')

    # compute_raw_psnr()


    collecting_more_sampling()
    metrics_orig_env_and_raw_psnr()