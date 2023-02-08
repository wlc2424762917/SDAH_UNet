'''
# -----------------------------------------
Model
UNet2022 m.1.0
# -----------------------------------------
'''
import sys
sys.path.append("/content/drive/MyDrive/Segmentation_framework")
import argparse
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.optim import Adam

from model.model_builders.select_network import define_netSeg
from model.model_builders.model_base import ModelBase
from model.loss.dice_loss import *
from common_utils.regularizers import regularizer_orth, regularizer_clip
from common_utils.utils_swinmr import *
from common_utils.utils_image import *

import matplotlib.pyplot as plt
import einops
import wandb
from math import ceil
import copy


class UNet_2022(ModelBase):
    def __init__(self, opt):
        super(UNet_2022, self).__init__(opt)
        # ------------------------------------
        # define network
        # ------------------------------------
        self.opt_train = self.opt['train']    # training option
        self.opt_dataset = self.opt['datasets']
        self.netSeg = define_netSeg(opt)
        self.netSeg = self.model_to_device(self.netSeg)
        if opt['rank'] == 0:
            wandb.watch(self.netSeg)

    """
    # ----------------------------------------
    # Preparation before training with data
    # Save model during training
    # ----------------------------------------
    """

    # ----------------------------------------
    # initialize training
    # ----------------------------------------
    def init_train(self):
        self.load()                           # load model
        self.netSeg.train()                     # set training mode,for BN
        self.define_loss()                    # define loss
        self.define_optimizer()               # define optimizer
        self.load_optimizers()                # load optimizer
        self.define_scheduler()               # define scheduler
        self.log_dict = OrderedDict()         # log

    # ----------------------------------------
    # load pre-trained segNet
    # ----------------------------------------
    def load(self):
        load_path_netSeg = self.opt['path']['pretrained_netSeg']
        if load_path_netSeg is not None:
            print("load path:", load_path_netSeg)
            #self.load_from(load_path_segNet)
            self.load_network(load_path_netSeg, self.netSeg, strict=self.opt_train['netSeg_param_strict'],
                              param_key='params')
    # ----------------------------------------
    # load optimizer
    # ----------------------------------------
    def load_optimizers(self):
        load_path_optimizerSeg = self.opt['path']['pretrained_optimizer_netSeg']
        if load_path_optimizerSeg is not None and self.opt_train['optimizer_netSeg_reuse']:
            print('Loading optimizer_segNet [{:s}] ...'.format(load_path_optimizerSeg))
            self.load_optimizer(load_path_optimizerSeg, self.Seg_optimizer)

    # ----------------------------------------
    # save model / optimizer(optional)
    # ----------------------------------------
    def save(self, iter_label):
        self.save_network(self.save_dir, self.netSeg, 'Seg', iter_label)
        if self.opt_train['Seg_optimizer_reuse']:
            self.save_optimizer(self.save_dir, self.Seg_optimizer, 'optimizerSeg', iter_label)

        # ----------------------------------------
        # define loss
        # ----------------------------------------
    def define_loss(self):
        self.dice_loss_weights = self.opt['train']['dice_loss_weights']
        self.bceloss = nn.BCELoss().to(self.device)
        if self.opt_train["loss_weights"] is not None:
            loss_weights = self.opt_train["loss_weights"]
            bc_weights = torch.FloatTensor(loss_weights).to(self.device)
            self.celoss = nn.CrossEntropyLoss(weight=bc_weights).to(self.device)
            self.diceloss = DiceLoss(loss_weights).to(self.device)
        else:
            self.celoss = nn.CrossEntropyLoss().to(self.device)
            # print(self.dice_loss_weights)
            self.diceloss = DiceLoss().to(self.device)

    def get_total_loss(self):
        # first one hot encoding gt_mask
        # self.pred_mask = torch.softmax(self.pred_mask, dim=1) crossEntropy dosesnt require prior softamx
        gt_seg_one_hot = mask_to_onehot(self.pred_seg, self.gt_seg)
        assert self.pred_seg.shape == gt_seg_one_hot.shape, "pred and gt size don't match."
        B, C, H, W = self.pred_seg.shape
        G_lossfn_type = self.opt_train['lossfn_type']
        self.alpha = self.opt_train['alpha']
        self.beta = self.opt_train['beta']
        self.loss_ce = self.celoss(self.pred_seg, torch.squeeze(self.gt_seg, 1).long())
        self.loss_dice = self.diceloss(self.pred_seg.view(B, C, -1), gt_seg_one_hot.view(B, C, -1))

        if G_lossfn_type == 'bce':
            return self.alpha * self.loss_bce
        if G_lossfn_type == 'dice':
            return self.beta * self.loss_dice
        if G_lossfn_type == 'bce_dice':
            return self.alpha * self.loss_bce + self.beta * self.loss_dice
        if G_lossfn_type == 'ce_dice':
            return self.alpha * self.loss_ce + self.beta * self.loss_dice
        else:
            raise NotImplementedError('Loss type [{:s}] is not found.'.format(G_lossfn_type))

    # ----------------------------------------
    # define optimizer
    # ----------------------------------------
    def define_optimizer(self):
        Seg_optim_params = []
        for k, v in self.netSeg.named_parameters():
            if v.requires_grad:
                Seg_optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))
        self.Seg_optimizer = Adam(Seg_optim_params, lr=self.opt_train['Seg_optimizer_lr'], weight_decay=0)

    # ----------------------------------------
    # define scheduler, only "MultiStepLR"
    # ----------------------------------------
    def define_scheduler(self):
        self.schedulers.append(lr_scheduler.MultiStepLR(self.Seg_optimizer,
                                                        self.opt_train['Seg_scheduler_milestones'],
                                                        self.opt_train['Seg_scheduler_gamma']
                                                        ))

    """
    # ----------------------------------------
    # Optimization during training with data
    # Testing/evaluation
    # ----------------------------------------
    """

    # ----------------------------------------
    # feed L/H data
    # ----------------------------------------
    def feed_data(self, data):
        self.image = data['src_image'].to(self.device)
        self.gt_seg = data['gt_seg'].to(self.device)

    def feed_3D_data(self, data):
        self.images = data['src_image'].to(self.device)
        self.gt_segs = data['gt_seg'].to(self.device)
        self.data_name = data['name'][0]
        self.gt_segs_3d = []
        self.pred_segs_3d = []

    def feed_slice_from_3D(self, idx_slice):
        self.image = self.images[:, idx_slice:(idx_slice + 1), :, :]
        self.gt_seg = self.gt_segs[:, idx_slice:(idx_slice + 1), :, :]

    # ----------------------------------------
    # prediction
    # ----------------------------------------
    def netSeg_forward(self):
        self.pred_seg = self.netSeg(self.image)

    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step):
        self.Seg_optimizer.zero_grad()
        self.netSeg_forward()

        total_loss = self.get_total_loss()
        total_loss.backward()

        # ------------------------------------
        # clip_grad
        # ------------------------------------
        # `clip_grad_norm` helps prevent the exploding gradient problem.
        Seg_optimizer_clipgrad = self.opt_train['Seg_optimizer_clipgrad'] if self.opt_train[
            'Seg_optimizer_clipgrad'] else 0
        if Seg_optimizer_clipgrad > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.opt_train['Seg_optimizer_clipgrad'],
                                           norm_type=2)

        self.Seg_optimizer.step()

        # ------------------------------------
        # regularizer
        # ------------------------------------
        Seg_regularizer_orthstep = self.opt_train['Seg_regularizer_orthstep'] if self.opt_train[
            'Seg_regularizer_orthstep'] else 0
        if Seg_regularizer_orthstep > 0 and current_step % Seg_regularizer_orthstep == 0 and current_step % \
                self.opt['train']['checkpoint_save'] != 0:
            self.netSeg.apply(regularizer_orth)
        Seg_regularizer_clipstep = self.opt_train['G_regularizer_clipstep'] if self.opt_train[
            'G_regularizer_clipstep'] else 0
        if Seg_regularizer_clipstep > 0 and current_step % Seg_regularizer_clipstep == 0 and current_step % \
                self.opt['train']['checkpoint_save'] != 0:
            self.netSeg.apply(regularizer_clip)

        # ------------------------------------
        # record log
        # ------------------------------------
        G_lossfn_type = self.opt_train['lossfn_type']
        self.log_dict['loss_total'] = total_loss.item()
        if G_lossfn_type == 'bce':
            self.log_dict['loss_bce'] = self.loss_bce.item()
        if G_lossfn_type == 'dice':
            self.log_dict['loss_dice'] = self.loss_dice.item()
        if G_lossfn_type == 'bce_dice':
            self.log_dict['loss_bce'] = self.loss_bce.item()
            self.log_dict['loss_dice'] = self.loss_dice.item()

    # ----------------------------------------
    # test / inference
    # ----------------------------------------
    def test(self):
        self.netSeg.eval()
        with torch.no_grad():
            self.netSeg_forward()
        self.netSeg.train()

    def test_use_tile(self, patch_size):
        self.netSeg.eval()
        with torch.no_grad():
            self.netSeg_predict_2D_tiled(patch_size)
        self.netSeg.train()

    # ----------------------------------------
    # get log_dict
    # ----------------------------------------
    def current_log(self):
        return self.log_dict

    # ----------------------------------------
    # get image, segmentation gt, segmentation prediction
    # ----------------------------------------
    def current_visuals(self, use_tile=False):
        if use_tile:
            out_dict = OrderedDict()
            H, W = self.pred_seg.shape
            self.gt_seg = mask_to_onehot(torch.zeros(1, 4, H, W).to(self.device), self.gt_seg)
            out_dict['src_image'] = self.image.detach()[0].float().cpu().numpy()
            out_dict['gt_seg'] = self.gt_seg.detach()[0].float().cpu().numpy()
            out_dict['pred_seg'] = torch.squeeze(self.pred_seg, dim=1).detach().float().cpu().numpy()
            return out_dict
        out_dict = OrderedDict()
        self.gt_seg = mask_to_onehot(self.pred_seg, self.gt_seg)
        out_dict['src_image'] = np.clip(self.image.detach()[0].float().cpu().numpy(), 0, 1)
        out_dict['gt_seg'] = self.gt_seg.detach()[0].float().cpu().numpy()
        pred_seg = torch.softmax(self.pred_seg, dim=1)
        out_dict['pred_seg'] = pred_seg.detach()[0].float().cpu().numpy()
        return out_dict

    # ----------------------------------------
    # get image, segmentation gt, segmentation prediction batch
    # ----------------------------------------
    def current_results(self):
        out_dict = OrderedDict()
        out_dict['src_image'] = np.clip(self.gt_image.detach().float().cpu().numpy(), 0, 1)
        out_dict['gt_seg'] = np.clip(self.gt_mask.detach().float().cpu().numpy(), 0, 1)
        out_dict['pred_seg'] = np.clip(self.pred_mask.detach().float().cpu().numpy(), 0, 1)
        return out_dict

    def collect_for_3d_volumn(self, use_tile=False):
        output = self.current_visuals(use_tile)
        self.gt_segs_3d.append(output['gt_seg'])
        self.pred_segs_3d.append(output['pred_seg'])
        # print(output['pred_mask'])

    def current_visuals_3d(self, use_tile=False):
        # slice, class, h, w --> class, h, w, slice
        out_dict = OrderedDict()
        print(np.array(self.pred_segs_3d).shape)
        print(np.array(self.gt_segs_3d).shape)
        if use_tile:
          out_dict['gt_segs_3d'] = np.array(self.gt_segs_3d).transpose(1, 2, 3, 0)
          out_dict['pred_segs_3d'] = np.array(self.pred_segs_3d).transpose(1, 2, 0)
        else:
          out_dict['gt_segs_3d'] = np.array(self.gt_segs_3d).transpose(1, 2, 3, 0)
          out_dict['pred_segs_3d'] = np.array(self.pred_segs_3d).transpose(1, 2, 3, 0)
        return out_dict

    def currnet_data_name(self):
        return self.data_name

    """
    # ----------------------------------------
    # Information of netG
    # ----------------------------------------
    """

    # ----------------------------------------
    # print network
    # ----------------------------------------
    def print_network(self):
        msg = self.describe_network(self.netSeg)
        print(msg)

    # ----------------------------------------
    # print params
    # ----------------------------------------
    def print_params(self):
        msg = self.describe_params(self.netSeg)
        print(msg)

    # ----------------------------------------
    # network information
    # ----------------------------------------
    def info_network(self):
        msg = self.describe_network(self.netSeg)
        return msg

    # ----------------------------------------
    # params information
    # ----------------------------------------
    def info_params(self):
        msg = self.describe_params(self.netSeg)
        return msg

    def load_from(self, load_path_Seg):
        pretrained_path = load_path_Seg
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model" not in pretrained_dict:
                print("---start load pretrained model by splitting---")
                pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.netSeg.load_state_dict(pretrained_dict, strict=False)
                print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained model of swin encoder---")

            model_dict = self.netSeg.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)

            if self.opt['use_pretrain_weight'] == 'all':
                for k, v in pretrained_dict.items():
                    if "layers." in k:
                        current_layer_num = 3 - int(k[7:8])
                        current_k = "layers_up." + str(current_layer_num) + k[8:]
                        full_dict.update({current_k: v})
            elif self.opt['use_pretrain_weight'] == 'enc':
                pass
            elif self.opt['use_pretrain_weight'] == 'no':
                pass
            else:
                raise "Wrong Type!"

            new_full_dict = OrderedDict()
            for k, v in full_dict.items():
                name = 'module.' + k
                new_full_dict[name] = v

            for k in list(new_full_dict.keys()):
                if k in model_dict:
                    if new_full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k, v.shape, model_dict[k].shape))
                        del new_full_dict[k]

            msg = self.netSeg.load_state_dict(new_full_dict, strict=False)
            print(msg)
        else:
            print("none pretrain")

    # ----------------------------------------
    # load the state_dict of the network
    # ----------------------------------------
    def load_network(self, load_path, network, strict=True, param_key='params'):
        network = self.get_bare_model(network)
        if strict:
            state_dict = torch.load(load_path)
            if param_key in state_dict.keys():
                state_dict = state_dict[param_key]
            network.load_state_dict(state_dict, strict=strict)
        else:
            state_dict_old = torch.load(load_path)
            if param_key in state_dict_old.keys():
                state_dict_old = state_dict_old[param_key]
            state_dict = network.state_dict()
            for ((key_old, param_old), (key, param)) in zip(state_dict_old.items(), state_dict.items()):
                state_dict[key] = param_old
            network.load_state_dict(state_dict, strict=True)
            del state_dict_old, state_dict


if __name__ == '__main__':
    from common_utils import utils_option as option
    from model.model_builders.select_model import define_Model


    json_path = r"C:\Users\wlc\PycharmProjects\Segmentation_framework\options\train_unet2022_ACDC.json"
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)

    opt = option.parse(parser.parse_args().opt, is_train=True)

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    init_iter_segNet, init_path_segNet = option.find_last_checkpoint(opt['path']['models'], net_type='Seg')
    opt['path']['pretrained_netSeg'] = init_path_segNet
    init_iter_optimizer_segNet, init_path_optimizer_segNet = option.find_last_checkpoint(opt['path']['models'],
                                                                                         net_type='optimizer_segNet')
    opt['path']['pretrained_optimizer_netSeg'] = init_path_optimizer_segNet
    current_step = max(init_iter_segNet, init_iter_optimizer_segNet)
