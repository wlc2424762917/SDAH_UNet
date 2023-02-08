import functools
import torch
import torchvision.models
from torch.nn import init
from model.networks.generic_UNet import *

"""
# --------------------------------------------
# select the network of seg, D 
# --------------------------------------------
"""


# --------------------------------------------
# define segmentation network
# --------------------------------------------
def define_netSeg(opt):
    opt_net = opt['netSeg']
    net_type = opt_net['net_type']

    if net_type == 'unets':
        from model.networks.generic_UNet import Generic_UNet as net
        netSeg = net(input_channels=opt_net["in_chans"],
                     base_num_features=opt_net["base_num_features"],
                     num_classes=opt_net["out_chans"],
                     num_pool=opt_net["num_pool"],
                     num_conv_per_stage=2,  # 2 as default
                     feat_map_mul_on_downscale=2,  # 2 as default
                     conv_op=nn.Conv2d,
                     norm_op=nn.BatchNorm2d,
                     norm_op_kwargs=None,
                     dropout_op=None,
                     dropout_op_kwargs=None,
                     nonlin=nn.LeakyReLU,
                     nonlin_kwargs=None,
                     deep_supervision=False,
                     dropout_in_localization=False,
                     final_nonlin=None,
                     pool_op_kernel_sizes=opt_net['pool_op_kernel_sizes'],
                     conv_kernel_sizes=opt_net['conv_kernel_sizes'],
                     upscale_logits=False,
                     convolutional_pooling=False,
                     convolutional_upsampling=False,
                     max_num_features=960,
                     basic_block=ConvDropoutNormNonlin,
                     seg_output_use_bias=False)

    elif net_type == 'swinunet':
        from model.networks.SwinUNet import SwinTransformerSys as net
        netSeg = net(img_size=opt_net['img_size'],
                     patch_size=opt_net['patch_size'],
                     in_chans=opt_net['in_chans'],
                     num_classes=opt_net['out_chans'],
                     embed_dim=opt_net['embed_dim'],
                     depths=opt_net['depths'],
                     num_heads=opt_net['num_heads'],
                     window_size=opt_net['window_size'],
                     mlp_ratio=opt_net['mlp_ratio'],
                     qkv_bias=True,
                     qk_scale=None,
                     drop_rate=0.,
                     drop_path_rate=0.1,
                     ape=False,
                     patch_norm=True,
                     use_checkpoint=False)

    elif net_type == 'unet2022':
        print(net_type)
        from model.networks.UNet2022 import unet2022 as net
        config = {}
        netSeg = net(
            num_input_channels=opt_net['in_chans'],
            embedding_dim=opt_net['embed_dim'],
            num_heads=opt_net['num_heads'],
            num_classes=opt_net['out_chans'],
            depths=opt_net['depths'],
            crop_size=opt_net['img_size'],
            convolution_stem_down=opt_net['patch_size'],
            window_size=opt_net['window_size'],
            deep_supervision=False,
            conv_op=nn.Conv2d
        )

    elif net_type == 'sdaut_parallel_conv':
        print(net_type)
        from model.networks.network_sdaut_parallel_conv import unet2022 as net
        netSeg = net(
            num_input_channels=opt_net['in_chans'],
            embedding_dim=opt_net['embed_dim'],
            types=opt_net['type'],
            num_heads=opt_net['num_heads'],
            n_groups=opt_net['num_groups'],
            num_classes=opt_net['out_chans'],
            depths=opt_net['depths'],
            crop_size=opt_net['img_size'],
            convolution_stem_down=opt_net['patch_size'],
            window_size=opt_net['window_size'],
            deep_supervision=False,
            conv_op=nn.Conv2d
        )
    else:
        raise NotImplementedError('netSeg [{:s}] is not found.'.format(net_type))

    # ----------------------------------------
    # initialize weights
    # ----------------------------------------
    if opt['is_train']:
        print(opt['is_train'])
        init_weights(netSeg,
                     init_type=opt_net['init_type'],
                     init_bn_type=opt_net['init_bn_type'],
                     gain=opt_net['init_gain'])

    return netSeg


"""
# --------------------------------------------
# weights initialization
# --------------------------------------------
"""


def init_weights(net, init_type='xavier_uniform', init_bn_type='uniform', gain=1):
    """
    #
    # Args:
    #   init_type:
    #       default, none: pass init_weights
    #       normal; normal; xavier_normal; xavier_uniform;
    #       kaiming_normal; kaiming_uniform; orthogonal
    #   init_bn_type:
    #       uniform; constant
    #   gain:
    #       0.2
    """

    def init_fn(m, init_type='xavier_uniform', init_bn_type='uniform', gain=1):
        classname = m.__class__.__name__

        if classname.find('Conv') != -1 or classname.find('Linear') != -1:

            if init_type == 'normal':
                init.normal_(m.weight.data, 0, 0.1)
                m.weight.data.clamp_(-1, 1).mul_(gain)

            elif init_type == 'uniform':
                init.uniform_(m.weight.data, -0.2, 0.2)
                m.weight.data.mul_(gain)

            elif init_type == 'xavier_normal':
                init.xavier_normal_(m.weight.data, gain=gain)
                m.weight.data.clamp_(-1, 1)

            elif init_type == 'xavier_uniform':
                init.xavier_uniform_(m.weight.data, gain=gain)

            elif init_type == 'kaiming_normal':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data.clamp_(-1, 1).mul_(gain)

            elif init_type == 'kaiming_uniform':
                init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data.mul_(gain)

            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)

            else:
                raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_type))

            if m.bias is not None:
                m.bias.data.zero_()

        elif classname.find('BatchNorm2d') != -1:

            if init_bn_type == 'uniform':  # preferred
                if m.affine:
                    init.uniform_(m.weight.data, 0.1, 1.0)
                    init.constant_(m.bias.data, 0.0)
            elif init_bn_type == 'constant':
                if m.affine:
                    init.constant_(m.weight.data, 1.0)
                    init.constant_(m.bias.data, 0.0)
            else:
                raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_bn_type))

    if init_type not in ['default', 'none']:
        print('Initialization method [{:s} + {:s}], gain is [{:.2f}]'.format(init_type, init_bn_type, gain))
        fn = functools.partial(init_fn, init_type=init_type, init_bn_type=init_bn_type, gain=gain)
        net.apply(fn)
    else:
        print('Pass this initialization! Initialization was done during network defination!')
