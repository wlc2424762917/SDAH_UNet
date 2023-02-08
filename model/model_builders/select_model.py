"""
# --------------------------------------------
# define training model
# --------------------------------------------
"""


def define_Model(opt):
    model = opt['model']

    if model == 'unets':
        from model.model_builders.model_unet_2D import UNet_2D as M

    elif model == 'UNETR_s':
        pass

    elif model == 'tpsn':
        pass

    elif model == 'swinunet':
        from model.model_builders.model_swinUNet import swin_unet as M

    elif model == 'csunet':
        pass

    elif model == 'unet2022':
        from model.model_builders.model_UNet2022 import UNet_2022 as M

    elif model == 'sdaut_parallel_conv':
        from model.model_builders.model_sdaut_parallel_conv import sdaut_parallel_conv as M

    else:
        raise NotImplementedError('Model [{:s}] is not defined.'.format(model))

    m = M(opt)

    print('Training model [{:s}] is created.'.format(m.__class__.__name__))
    return m
