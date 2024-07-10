import guided_diffusion.diffusions as diffusions
import torch
from guided_diffusion import dist_util

BaseDiffusion = diffusions.get_diffusion('base_diffusion')

class BaseDiffusion_wrap(BaseDiffusion):

    def training_losses(self, model, x_start, t, model_kwargs=None):
        #print('start training losses step')
        if 'lq' not in model_kwargs:
            raise ValueError('Should be for base diffusion!')
        model_kwargs['low_res'] = model_kwargs['lq'] * 2 - 1  # concat the iamge to the diffusion input
        return super().training_losses(model, x_start, t, model_kwargs)

    @torch.no_grad()
    def p_sample_loop(self, model, shape, *args, **kwargs):
        #print('start p_sample_loop step')
        kwargs['model_kwargs']['low_res'] = kwargs['model_kwargs']['lq'] * 2 - 1
        kwargs['model_kwargs'].pop('lq')
        additional_args = {'low_res': kwargs['model_kwargs']['low_res']}
        shape = list(shape)

        shape[-1] = kwargs['model_kwargs']['low_res'].shape[-1]
        shape[-2] = kwargs['model_kwargs']['low_res'].shape[-2]
        # kwargs['diffusion_start_point'] = 3
        return super().p_sample_loop(model, shape, *args, **kwargs), additional_args

    def _adapt_kwargs_inputs_for_sampling(self, data_dict):
        # adapting sampling input args of 'p_sample_loop' (based on implementation).
        ret_dict = {}
        model_kwargs = {'lq': data_dict['lq'].to(dtype=torch.float32, device=dist_util.dev())}
        if 'clip_condition' in data_dict:
            model_kwargs['clip_condition'] = data_dict['clip_condition'].to(dtype=torch.float32, device=dist_util.dev())
        ret_dict['model_kwargs'] = model_kwargs
        ret_dict['noise'] = None
        #print('adapted')

        return ret_dict


class BaseDiffusion_dvir(BaseDiffusion):
    #@torch.no_grad()
    def initial_recon(self, model, measurements, x_start_shape_verification=None):
        m = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        m.restormer.to(measurements.device)
        init_recon = m.restormer(measurements)
        # m.separable_model.to(dist_util.dev())
        # if x_start_shape_verification is not None:
        #     assert x_T_end.shape == x_start_shape_verification, f'{x_T_end.shape} not equal to {x_start_shape_verification}'
        return (init_recon * 2 - 1).contiguous().to(dist_util.dev()) # .detach() todo keep


    def training_losses(self, model, x_start, t, model_kwargs=None):
        #print('start training losses step')
        if 'lq' not in model_kwargs:
            raise ValueError('lq Should be for dvir diffusion!')
        # model_kwargs['low_res'], rgb_x_T_end = self.process_measurments(model, model_kwargs['x_T_end_meas'], x_start.shape)
        model_kwargs['low_res'] = model_kwargs['lq'] * 2 - 1  # concat the iamge to the diffusion input
        x_learn = x_start - self.initial_recon(model, model_kwargs.pop('lq')) # perform dvir
        return super().training_losses(model, x_learn, t, model_kwargs)

    @torch.no_grad()
    def p_sample_loop(self, model, shape, *args, **kwargs):
        #print('start p_sample_loop step')
        kwargs['model_kwargs']['low_res'] = kwargs['model_kwargs']['lq'] * 2 - 1

        recon = self.initial_recon(model, kwargs['model_kwargs'].pop('lq'))
        # kwargs['diffusion_start_point'] = 3 # todo short sampling
        additional_args = {'low_res': kwargs['model_kwargs']['low_res'],
                           'initial_recon': recon}

        return recon + super().p_sample_loop(model, shape, *args, **kwargs), additional_args # (recon, )

    def _adapt_kwargs_inputs_for_sampling(self, data_dict):
        # adapting sampling inputs of 'p_sample_loop' based on implementation
        ret_dict = {}
        model_kwargs = {'lq': data_dict['lq'].to(dtype=torch.float32, device=dist_util.dev())}
        if 'clip_condition' in data_dict:
            model_kwargs['clip_condition'] = data_dict['clip_condition'].to(dtype=torch.float32, device=dist_util.dev())
        ret_dict['model_kwargs'] = model_kwargs
        ret_dict['noise'] = None
        #print('adapted')

        return ret_dict

class BaseDiffusion_dvir_freeze(BaseDiffusion_dvir):
    @torch.no_grad()
    def initial_recon(self, *args, **kwargs):
        return super().initial_recon(*args, **kwargs).detach()
'''
class BaseDiffusion_condition(BaseDiffusion):
    @torch.no_grad()
    def initial_recon(self, model, measurements):
        m = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        m.restormer.to(measurements.device)
        init_recon = m.restormer(measurements)
        # m.separable_model.to(dist_util.dev())
        # if x_start_shape_verification is not None:
        #     assert x_T_end.shape == x_start_shape_verification, f'{x_T_end.shape} not equal to {x_start_shape_verification}'
        return (init_recon * 2 - 1).contiguous().detach().to(dist_util.dev()) # .detach() todo keep

    def training_losses(self, model, x_start, t, model_kwargs=None):
        #print('start training losses step')
        if 'lq' not in model_kwargs:
            raise ValueError('lq Should be for dvir diffusion!')
        model_kwargs['low_res'] = model_kwargs['lq'] * 2 - 1

        x_learn = x_start - self.initial_recon(model, model_kwargs.pop('lq'))
        return super().training_losses(model, x_learn, t, model_kwargs)

    @torch.no_grad()
    def p_sample_loop(self, model, shape, *kargs, **kwargs):
        #print('start p_sample_loop step')
        kwargs['model_kwargs']['low_res'] = kwargs['model_kwargs']['lq'] * 2 - 1

        recon = self.initial_recon(model, kwargs['model_kwargs'].pop('lq'))
        # kwargs['diffusion_start_point'] = 3 # todo short sampling
        x_T_end_tuple = (recon, )
        return recon + super().p_sample_loop(model, shape, *kargs, **kwargs), x_T_end_tuple

    def _adapt_kwargs_inputs_for_sampling(self, data_dict):
        # adapting sampling inputs of 'p_sample_loop' based on implementation
        ret_dict = {}
        model_kwargs = {'lq': data_dict['lq'].to(dtype=torch.float32, device=dist_util.dev())}
        if 'clip_condition' in data_dict:
            model_kwargs['clip_condition'] = data_dict['clip_condition'].to(dtype=torch.float32, device=dist_util.dev())
        ret_dict['model_kwargs'] = model_kwargs
        ret_dict['noise'] = None
        #print('adapted')

        return ret_dict
'''