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

# import tensorflow as tf
# import tensorflow_hub as hub

import requests
from PIL import Image
from io import BytesIO

# runai bash erezjob7
# conda activate /storage/erez/conda/envs/tf38
# python musiq_tf.py --input_dir /storage/erez/Documents/sidd/musiq_eval/diffusion/230919_1442_lora_cat_real21correct/
 # python musiq_tf.py --input_dir /storage/erez/Documents/sidd/musiq_eval/n2v/realcam_s21stat
def musiq_eval_folders():
    # from: https://colab.research.google.com/github/google-research/google-research/blob/master/musiq/Inference_with_MUSIQ.ipynb#scrollTo=1wO1PVJqlQxr
    # use env tf38 on RUNAI for MUSIC metric

    #import matplotlib.pyplot as plt

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

def musiq_eval_all_in_folder():

    parser = argparse.ArgumentParser(description='TEST MUSIQ')
    parser.add_argument('--input_dir', default='', type=str, help='Directory of validation images')

    args = parser.parse_args()
    # from: https://colab.research.google.com/github/google-research/google-research/blob/master/musiq/Inference_with_MUSIQ.ipynb#scrollTo=1wO1PVJqlQxr
    # use env tf38 on RUNAI for MUSIC metric

    #import matplotlib.pyplot as plt
    print('Warning, going on all folders of path; ALL')
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


    def load_image_to_tf(img_path):
        image = Image.open(img_path)
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        return img_byte_arr

    #image, image_bytes = load_image_from_url(image_url)
    def model_get_dict(folder, predict_fn):
        dirname, basename = os.path.split(folder)
        files = os.listdir(folder)
        images = [f for f in files if f.endswith('.png')]
        #names = ['gt{:03}.png', 'sample{:03}_low_res.png', 'sample{:03}.png']
        res_dict = {}
        num_samples = len(images)#len(glob.glob(os.path.join(folder, 'gt*.png')))
        print(f'WARNING evaluating {num_samples} images')
        for i in range(num_samples):
            img_path = os.path.join(folder, images[i])
            image_bytes = load_image_to_tf(img_path)
            prediction = predict_fn(tf.constant(image_bytes))
            res_dict[images[i]] = float(prediction['output_0'].numpy())
            #res_dict[im_name.format(999)] = res_dict.get(im_name.format(999), 0) + prediction['output_0'].numpy()#.item()

        folders = [f for f in files if os.path.isdir(os.path.join(folder, f))]
        for f in folders:
            res_dict[f] = model_get_dict(os.path.join(folder, f), predict_fn)
        return res_dict

    total_dict = {}
    folder_path = '/data1/erez/Documents/sidd/diffusion_coco_storage/230803_1653_basecond_Nlvl_L14norm/230919_1547_lora_cond_real21correct/save/'
    folder_path = '/storage/erez/Documents/sidd/musiq_eval/cycleisp/realcam_raw_s21'
    folder_path = args.input_dir or folder_path # get path from argparser
    dirname, basename = os.path.split(folder_path)

    for model_name in NAME_TO_HANDLE.keys():
        model_handle = NAME_TO_HANDLE[model_name]
        model = hub.load(model_handle)
        predict_fn = model.signatures['serving_default']

        total_dict[model_name] = model_get_dict(folder_path, predict_fn)


    with open(os.path.join(folder_path, f'musiq_{basename}.yaml'), 'w') as outfile:
        yaml.dump(total_dict, outfile, indent=4)

    #show_image(image)
    print('DONE!')

@torch.no_grad()
def liberna():

    import requests
    from PIL import Image
    from io import BytesIO
    from diffusers import LDMSuperResolutionPipeline
    #import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "CompVis/ldm-super-resolution-4x-openimages"

    # load model and scheduler
    pipeline = LDMSuperResolutionPipeline.from_pretrained(model_id)
    pipeline = pipeline.to(device)

    # let's download an  image
    url = "https://user-images.githubusercontent.com/38061659/199705896-b48e17b8-b231-47cd-a270-4ffa5a93fa3e.png"
    response = requests.get(url)
    inimname = 'lena.png'
    low_res_img = Image.open(f'/home/erez/Documents/liberna/{inimname}').convert("RGB")
    #low_res_img.save(f"/home/erez/Documents/liberna/lena.png")
    low_res_img = low_res_img.resize((256, 256))

    # run pipeline in inference (sample random noise and denoise)
    imnum=7
    upscaled_image = pipeline(low_res_img, num_inference_steps=100, eta=1).images[0]
    upscaled_image.save(f"/home/erez/Documents/liberna/liberna{imnum}.png")

    imnum+=1
    # upscaled_image = pipeline(low_res_img, num_inference_steps=100, eta=1).images[0]
    upscaled_image.resize((512, 512))
    upscaled_image.save(f"/home/erez/Documents/liberna/liberna512{imnum}.png")
    # imnum+=1
    # upscaled_image = pipeline(low_res_img, num_inference_steps=500, eta=1).images[0]
    # upscaled_image.save(f"/home/erez/Documents/liberna/liberna{imnum}.png")

def liberna2():

    import requests
    from PIL import Image
    from io import BytesIO
    from diffusers import LDMSuperResolutionPipeline
    #import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "CompVis/ldm-super-resolution-4x-openimages"

    # load model and scheduler
    pipeline = LDMSuperResolutionPipeline.from_pretrained(model_id)
    pipeline = pipeline.to(device)

    p=7
    inimname = f'liberna{p}.png'
    low_res_img = Image.open(f'/home/erez/Documents/liberna/{inimname}').convert("RGB")
    low_res_img = low_res_img.resize((256, 256))

    # run pipeline in inference (sample random noise and denoise)
    imnum=0
    upscaled_image = pipeline(low_res_img, num_inference_steps=100, eta=1).images[0]
    upscaled_image.save(f"/home/erez/Documents/liberna/liberna{p}{imnum}.png")

    imnum+=1
    upscaled_image = pipeline(low_res_img, num_inference_steps=100, eta=1).images[0]
    upscaled_image.save(f"/home/erez/Documents/liberna/liberna{p}{imnum}.png")
    # imnum+=1
    # upscaled_image = pipeline(low_res_img, num_inference_steps=999, eta=1).images[0]
    # upscaled_image.save(f"/home/erez/Documents/liberna/liberna{p}{imnum}.png")

def liberna3():
    from diffusers import AutoPipelineForInpainting
    from diffusers.utils import load_image

    pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                                                     torch_dtype=torch.float16, variant="fp16").to("cuda")

    img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
    mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"
    inimname = 'lena.png'

    image = Image.open(f'/home/erez/Documents/liberna/{inimname}').convert("RGB")
    mask = 'lena_mask.png'
    mask_image = Image.open(f'/home/erez/Documents/liberna/{mask}').convert("RGB")
    image_m = pipe.image_processor.pil_to_numpy(mask_image)  # batch1
    # for i in range():
    #     image_m[:, :, i::10, :] = 0
    mask_image = pipe.image_processor.numpy_to_pil(image_m)[0]  # get first image
    mask_image.save(f"/home/erez/Documents/liberna/liberna_mask3.png")


    # image = load_image(img_url).resize((1024, 1024))
    # mask_image = load_image(mask_url).resize((1024, 1024))

    prompt = "a beautiful woman with long eyelashes is looking to the camera with shadows and lights on her face creating strong contrast"
    generator = torch.Generator(device="cuda").manual_seed(0)
    # lena95 with seed 0 , st0.6 steps 50
    image = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask_image,
        guidance_scale=8.0,
        num_inference_steps=50,  # steps between 15 and 30 work well for us
        strength=0.6,  # make sure to use `strength` below 1.0
        generator=generator,
    ).images[0]
    imnum=95068_50
    image = image.resize((512, 512))
    image.save(f"/home/erez/Documents/liberna/liberna_inpaine{imnum}.png")

def lopena():
    from diffusers import AutoPipelineForInpainting
    from diffusers.utils import load_image

    pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                                                     torch_dtype=torch.float16, variant="fp16").to("cuda")

    img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
    mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"
    inimname = 'lena_95b.png'

    image = Image.open(f'/home/erez/Documents/liberna/{inimname}').convert("RGB")
    mask = 'lena_mask2.png'
    mask_image = Image.open(f'/home/erez/Documents/liberna/{mask}').convert("RGB")
    image_m = pipe.image_processor.pil_to_numpy(mask_image)  # batch1
    # for i in range():
    #     image_m[:, :, i::10, :] = 0
    mask_image = pipe.image_processor.numpy_to_pil(image_m)[0]  # get first image
    mask_image.save(f"/home/erez/Documents/liberna/liberna_mask3.png")


    # image = load_image(img_url).resize((1024, 1024))
    # mask_image = load_image(mask_url).resize((1024, 1024))

    prompt = "a woman with a black shirt"
    generator = torch.Generator(device="cuda").manual_seed(42)
    # lena95 with seed 0 , st0.6 steps 50
    image = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask_image,
        guidance_scale=10.0,
        num_inference_steps=15,  # steps between 15 and 30 work well for us
        strength=0.95,  # make sure to use `strength` below 1.0
        generator=generator,
    ).images[0]
    imnum=7
    image = image.resize((512, 512))
    image.save(f"/home/erez/Documents/liberna/liberna95_shirt{imnum}.png")




if __name__ == '__main__':
    # musiq_eval_all_in_folder()
    # liberna3()
    lopena()

