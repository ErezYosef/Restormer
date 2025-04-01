"""
Train a diffusion model on images.
"""
import sys
sys.path.append('..')
import torch
from guided_diffusion import dist_util, logger
#from guided_diffusion.image_datasets import load_data
from guided_diffusion.datasets.sidd_raw_dataset import warp_DataLoader
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    all_args_to_dict,
)
#from guided_diffusion.train_util import TrainLoop
from train_util_wrap import TrainLoop_wrap as TrainLoop
from guided_diffusion.script_util import parse_yaml
from guided_diffusion.respace_diffusion import SpacedDiffusion
from guided_diffusion.gaussian_diffusion import get_named_beta_schedule

#from diffusion.diffusions.coldmix_diffusion import ColdMixDiffusion_wrap
from image_train import create_argparser
from guided_diffusion.script_util import load_folder_path_parse
from diffusions import get_model, get_diffusion, create_model_wrap_clean # create_model_wrap
from datasets import get_dataset # create_model_wrap
from image_train import get_non_default_args

class TrainLoop_timing(TrainLoop):
    def validation_sample(self, data_to_sample=None, num_samples=8, only_first_batch=True, call_id=0, save_all=False,
                          update_logger_for_sample=False, log_images_wandb=True, post_tag_folder=''):
        self.model.eval()
        if data_to_sample is None:
            data_to_sample = self.val_dataset
        image_size = self.model.image_size
        # Local setup
        clip_denoised = True

        all_images = []
        mse = torch.nn.MSELoss()

        # loss_fn = self.loss_fn
        mse_loss, ssim_loss = 0, 0

        # CUDA event setup for precise timing

        for batch_counter, sample_condition_data in enumerate(data_to_sample):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            print(sample_condition_data[0].shape)
            model_kwargs = {}
            batch_size = sample_condition_data[0].shape[0]
            sample_fn = self.diffusion.p_sample_loop # if not args.use_ddim else self.diffusion.ddim_sample_loop)
            gt_imgs, data_dict = sample_condition_data

            # Ensure tensors and model are on the correct device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # self.model = self.model.to(device)

            # Start timing
            torch.cuda.synchronize()  # Synchronize before starting timing
            start_event.record()


            gt_imgs = gt_imgs.to(dtype=torch.float32, device=dist_util.dev())
            # Start timing
            torch.cuda.synchronize()  # Synchronize before starting timing
            start_event.record()


            sample, x_T_end = sample_fn(
                self.model,
                (batch_size, 4, image_size, image_size),
                clip_denoised=clip_denoised,
                **self.diffusion._adapt_kwargs_inputs_for_sampling(data_dict)
            )
            # End timing
            end_event.record()
            torch.cuda.synchronize()  # Wait for the timing to finish
            # Calculate elapsed time in milliseconds
            elapsed_time_ms = start_event.elapsed_time(end_event)
            print(f"Sampling time for batch {batch_counter + 1}: {elapsed_time_ms:.2f} ms, img shape: {sample.shape}")


            # Copy from image sample code:
            sample_cp = sample.clone()
            # sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            # sample = sample.permute(0, 2, 3, 1)
            # sample = sample.contiguous()
            #
            # gathered_samples = [torch.zeros_like(sample) for _ in range(dist.get_world_size())]
            # dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            # break

        # dist.barrier()
        #logger.log("sampling complete")
        self.model.train()
        print(self.totstep, end='\r')

def main():
    args = create_argparser().parse_args()
    non_default_args = get_non_default_args(args)
    args = parse_yaml(args, non_default_args=non_default_args)

    dist_util.setup_dist()
    loaded_folder_name = load_folder_path_parse(args)
    args.resume_checkpoint = args.model_path
    print(args)
    logger.configure(args=args, loaded_folder_name=loaded_folder_name)

    logger.log(f'\n\t'.join(f'{k} = {v}' for k, v in vars(args).items()))
    logger.log("creating model and diffusion...")
    #model, diffusion = create_model_and_diffusion(
    #    **args_to_dict(args, model_and_diffusion_defaults().keys()))
    print('pass1')
    # if args.set_seed is not None:
    #     torch.manual_seed(args.set_seed)
    #     print(f'seed sets to : {args.set_seed}')
    model_class = get_model(args.model_type) #ConcatModel_wrappret_class(UNetModel) #ConcatModelConv
    model = create_model_wrap_clean(model_class=model_class, **all_args_to_dict(args))
    betas = get_named_beta_schedule(args.noise_schedule, args.diffusion_steps)
    diffusion_args = all_args_to_dict(args)
    diffusion_args['betas'] = betas
    diffusion_class = get_diffusion(args.diffusion_type) #diffusions.get_diffusion(args.diffusion_type)  # ColdMix or BaseDiffusion
    diffusion = SpacedDiffusion(diffusion_class, **diffusion_args)
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log(f"creating data loader... dir: {args.data_dir}")

    #from dataset_sidd import Dataset_PairedImage_crops_less_clip as Dataset_class
    # last change: >> from Denoising.diffusion.datasets.dataset_real import Dataset_Realcam as Dataset_class
    Dataset_class = get_dataset(args.dataset_type) #ConcatModel_wrappret_class(UNetModel) #ConcatModelConv

    #train_ds = Dataset_PairedImage(args.main_data_path, cropsize=args.image_size, random_crop=True, phase='train', val_percent=args.val_percent)
    train_ds = Dataset_class(main_path_dataset=args.main_data_path_train, mode='train', **diffusion_args)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                               collate_fn=None, drop_last=True)
    infinite_train_loader = warp_DataLoader(train_loader)

    #val_ds = Dataset_PairedImage(args.main_data_path, cropsize=args.image_size, phase='val', val_percent=args.val_percent)
    val_ds = Dataset_class(main_path_dataset=args.main_data_path_val, mode='val', **diffusion_args)

    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False,
                                             num_workers=args.num_workers, batch_sampler=getattr(val_ds, 'batch_loader', None))

    logger.log("training...")
    train_loop = TrainLoop_timing(
        model=model,
        diffusion=diffusion,
        train_data=infinite_train_loader,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        val_dataset=val_loader,
        batches_accumulate_grads=args.batches_accumulate_grads,
        test_dataset=None,
        islora=getattr(args, 'islora', False),
        lora_checkpoint=getattr(args, 'lora_checkpoint', None) if getattr(args, 'islora', False) else None

    )
    train_loop.log_step()
    train_loop.validation_sample(only_first_batch=True, save_all=args.save_all_samples,
                                 update_logger_for_sample=False, log_images_wandb=False, post_tag_folder='')
    logger.dumpkvs()


if __name__ == "__main__":
    main()
