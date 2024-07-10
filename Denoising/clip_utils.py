import torch
from guided_diffusion.glide import clip_util


def get_clip_cache(clip_model, index, image11):
    clip_embd = clip_model.get_img_embd(image11, minmax=(-1, 1))[0]  # single vector
    # print('shape clip:',clip_embd.shape) # 512
    # else: # todo disable
    #     clip_embd2 = self.clip_model.get_img_embd(image11.unsqueeze(0), minmax=(-1, 1))[0]  # single vector
    #     if not torch.equal(clip_embd, clip_embd2):
    #         print('not eq:', clip_embd.sum(), clip_embd2.sum())
    return clip_embd



def main():
    from Denoising.diffusion.datasets.dataset_sidd import Dataset_PairedImage_crops_less

    main_data_path_train = '/data1/erez/datasets/sidd/srgb_crops'
    main_data_path_val = '/data1/erez/datasets/sidd/srgb_crops_val'
    device = 'cuda'
    clip_model = clip_util.clip_model_wrap(device=device)
    dirty_count = 0
    clip_cache = {}
    clip_cache = torch.load('pretrained_models/clip_embd_train3.pt', map_location='cpu')
    for k, v in clip_cache.items():
        if v.dtype== torch.float32:
            clip_cache[k] = v.to(torch.float16)
    # print('total keys in clip_cache: ', len(self.clip_cache.keys()))

    # train_ds = DatasetFromFilenames_wrap(args.train_meas_filenames, args.train_orig_filenames, args.data_main_path,
    #                                 trim_len=0, clip_dataset=True)
    # train_loader = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=False,
    #                                            num_workers=0, collate_fn=None, drop_last=False)

    train_ds = Dataset_PairedImage_crops_less(main_data_path_train)
    bs=20
    loader = torch.utils.data.DataLoader(train_ds, batch_size=bs, shuffle=False, num_workers=8)
    # val_ds = Dataset_PairedImage_crops_less(main_data_path_val)
    # loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=8)
    index = 0
    for i,data in enumerate(loader):
        gt = data[0].to(device)
        if clip_cache.get(index+gt.shape[0], None) is None:
            img_clip_embd = clip_model.get_img_embd(gt, minmax=(-1, 1)) # get_clip_cache(clip_model, index, gt)
            # print(img_clip_embd.dtype)
            # clip_cache[index] = img_clip_embd.to(device='cpu', dtype=torch.float32)
            img_clip_embd = img_clip_embd.to(device='cpu') #, dtype=torch.float32)
            for b_ind in range(gt.shape[0]):
                clip_cache[index] = img_clip_embd[b_ind]
                index += 1

            if index % 1000 == 0:
                print(index, end='\r', flush=True)
            if index % 20000 == 0:
                torch.save(clip_cache, 'pretrained_models/clip_embd_train4.pt')
        else:
            if index%50000==0:
                img_clip_embd = clip_model.get_img_embd(gt[0:1], minmax=(-1, 1))[0]  # get_clip_cache(clip_model, index, gt)
                img_clip_embd = img_clip_embd.to(device='cpu', dtype=torch.float32)
                print('validation ', index, 'err', ((img_clip_embd-clip_cache[index])**2).mean(), end='\r', flush=True)
            index += gt.shape[0]
        pass
        #index +=1
    print(' END: ', index)
    torch.save(clip_cache, 'pretrained_models/clip_embd_train4.pt')

if __name__ == '__main__':
    main()