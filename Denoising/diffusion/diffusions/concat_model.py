import torch
from guided_diffusion.diffusions.lora import lora_model_for_diffusion

def ConcatModel_wrap(model_class):
    class ConcatModel_wrapper(lora_model_for_diffusion, model_class):
        """
        A wrapper that performs concatenation.

        Expects an extra kwarg `low_res` to condition on a low-resolution image.
        """
        # def __init__(self, image_size, in_channels, *args, **kwargs):
        #     #print(image_size, in_channels)
        #     super().__init__(image_size, in_channels * 2, *args, **kwargs)

        def forward(self, x, timesteps, low_res=None, **kwargs):
            #_, _, new_height, new_width = x.shape
            # upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
            x = torch.cat([x, low_res], dim=1)
            #print('sshape', x.shape)
            return super().forward(x, timesteps, **kwargs)
    return ConcatModel_wrapper

'''
from guided_diffusion.unet import timestep_embedding
def ConcatModel_cond_wrap(model_class):
    base_model = ConcatModel_wrap(model_class)
    class ConcatModel_wrapper(base_model):
        def forward(self, x, timesteps, low_res=None, y=None, **kwargs):
            x = torch.cat([x, low_res], dim=1)
            if self.num_classes is not None and y is None:
                print(f"not specified y if the model is class-conditional, {self.num_classes}")

            hs = []
            emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

            if self.num_classes is not None:
                #assert y.shape[0] == x.shape[0], f'{y.shape} != {x.shape}'
                emb = emb #+ self.label_emb(y) # todo think

            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb)
                hs.append(h)
            h = self.middle_block(h, emb)
            for module in self.output_blocks:
                h = torch.cat([h, hs.pop()], dim=1)
                h = module(h, emb)
            h = h.type(x.dtype)
            return self.out(h)

    return ConcatModel_wrapper
'''