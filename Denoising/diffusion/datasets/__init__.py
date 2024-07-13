
from . import dataset_sidd, dataset_cococap, dataset_real, dataset_s7_cam




def get_dataset(name):
    if name == 'sidd':
        return dataset_sidd.Dataset_PairedImage_crops_less_clip
    elif name == 'cococap':
        return dataset_cococap.Dataset_Cococap
    elif name == 'cococap_dualclip':
        return dataset_cococap.Dataset_Cococap_dualclipdata
    elif name == 'cococap_clip':
        return dataset_cococap.Dataset_Cococap_clipimg
    elif name == 'realcam':
        return dataset_real.Dataset_Realcam
    elif name == 'realcam_multiple':
        return dataset_real.Dataset_Realcam_multiple
    elif name == 's7':
        return dataset_s7_cam.Dataset_s7
    elif name == 's21':
        return dataset_s7_cam.Dataset_s21
    elif name == 's21_set_caption':
        return dataset_s7_cam.Dataset_s21_set_caption
    else:
        print(f'Warning: dataset class {name} is missing > return None..')
        return None
