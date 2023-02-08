import os.path
import sys

sys.path.append("/home/lichao/segmentation_framework")

import math
import argparse
import random
import numpy as np
import logging
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
import SimpleITK as sitk
from common_utils import utils_logger
from common_utils import utils_image as util
from common_utils import utils_option as option
from common_utils.utils_dist import get_dist_info, init_dist
import os
from data.select_dataset import define_Dataset
from model.model_builders.select_model import define_Model
from tensorboardX import SummaryWriter
import time
import sys
import wandb
from collections import OrderedDict
import csv
import cv2
from medpy import metric
import seg_metrics.seg_metrics as sg


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def main(json_path=''):
    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option json file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)
    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt['dist'] = parser.parse_args().dist
    wandb.init()
    # ----------------------------------------
    # distributed settings
    # ----------------------------------------main_train_unets.py
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()
    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    init_iter_netSeg, init_path_netSeg = option.find_last_checkpoint(opt['path']['models'], net_type='Seg')
    opt['path']['pretrained_netSeg'] = init_path_netSeg
    init_iter_optimizer_netSeg, init_path_optimizer_netSeg = option.find_last_checkpoint(opt['path']['models'],
                                                                                         net_type='optimizerSeg')
    opt['path']['pretrained_optimizer_netSeg'] = init_path_optimizer_netSeg
    print("init_path_netSeg", init_path_netSeg)
    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    if opt['rank'] == 0:
        option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    if opt['rank'] == 0:
        logger_name = 'train'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name + '.log'))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))

    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # ----------------------------------------
    # choose if use tile to predict
    # ----------------------------------------
    use_tile = opt['netSeg']['use_tile']
    print(use_tile)

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for test
    # ----------------------------------------
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'test':
            # print(dataset_opt)
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=False)

    model = define_Model(opt)
    model.init_train()

    with torch.no_grad():
        test_results = OrderedDict()
        for idx_class in range(dataset_opt['num_class'] + 1):
            test_results[f'{idx_class}'] = OrderedDict()
            test_results[f'{idx_class}']['dice'] = []
            test_results[f'{idx_class}']['Hausdorff'] = []

        metrics_list = []
        metrics_list_p = []

        save_dir = opt['path']['test_image_save_dir']
        util.mkdir(save_dir)

        with open(os.path.join(save_dir, 'dice_results.csv'), 'w') as dsc:
            writer_dice = csv.writer(dsc)
            writer_dice.writerow(['Class/Method', 'ET', 'TC', 'WT'])

        with open(os.path.join(save_dir, 'hd_results.csv'), 'w') as hd:
            writer_hd = csv.writer(hd)
            writer_hd.writerow(['Class/Method', 'ET', 'TC', 'WT'])

        with open(os.path.join(save_dir, 'hd95_results.csv'), 'w') as hd95:
            writer_hd95 = csv.writer(hd95)
            writer_hd95.writerow(['Class/Method', 'ET', 'TC', 'WT'])

        with open(os.path.join(save_dir, 'msd_results.csv'), 'w') as msd:
            writer_msd = csv.writer(msd)
            writer_msd.writerow(['Class/Method', 'ET', 'TC', 'WT'])

        for idx, test_data in enumerate(test_loader):
            gt_masks = []
            pred_masks = []
            pred_masks_post_processed = []
            model.feed_3D_data(test_data)
            data_name = model.currnet_data_name()
            for idx_slice in range(test_data['src_image'].shape[1]):
                model.feed_slice_from_4D(idx_slice)
                if use_tile == True:
                    patch_size = (dataset_opt["H_size"], dataset_opt["W_size"])
                    model.test_use_tile(patch_size)
                else:
                    model.test()
                model.collect_for_3d_volumn(use_tile)

                # save image
                output_slice = model.current_visuals(use_tile)
                gt_image = output_slice['src_image']
                gt_mask = output_slice['gt_seg']
                pred_mask = output_slice['pred_seg']
                if use_tile:  # argmax already
                    pred_mask_dense = pred_mask
                else:
                    pred_mask_dense = np.argmax(pred_mask, 0)
                classes = [i for i in range(1, dataset_opt['num_class'] + 1)]
                pred_mask_dense_raw = pred_mask_dense.copy()
                pred_mask_post_processed, removed_part, remaining_part = model.remove_all_but_the_largest_connected_component(
                    pred_mask_dense_raw, classes)
                gt_mask_dense = np.argmax(gt_mask, 0)
                pred_mask_dense255 = (pred_mask_dense - pred_mask_dense.min()) / (
                        pred_mask_dense.max() - pred_mask_dense.min()) * 255
                gt_mask_dense255 = (gt_mask_dense - gt_mask_dense.min()) / (
                        gt_mask_dense.max() - gt_mask_dense.min()) * 255
                pred_mask_post_processed255 = (pred_mask_post_processed - pred_mask_post_processed.min()) / (
                        pred_mask_post_processed.max() - pred_mask_post_processed.min()) * 255
                gt_masks.append(gt_mask_dense)
                pred_masks.append(pred_mask_dense)
                pred_masks_post_processed.append(pred_mask_post_processed)

                gt_image255 = (gt_image * 255.0).round().astype(np.uint8)  # float32 to uint8
                util.mkdir(os.path.join(save_dir, 'src_image'))
                util.mkdir(os.path.join(save_dir, 'gt_seg'))
                util.mkdir(os.path.join(save_dir, 'pred_seg'))
                util.mkdir(os.path.join(save_dir, 'pred_seg_p'))
                cv2.imwrite(
                    os.path.join(save_dir, 'src_image', 'src_image_{}_{:03d}.png'.format(data_name, idx_slice + 1)),
                    gt_image255[0, :, :])
                cv2.imwrite(os.path.join(save_dir, 'gt_seg', 'gt_seg_{}_{:03d}.png'.format(data_name, idx_slice + 1)),
                            gt_mask_dense255[:, :])
                cv2.imwrite(
                    os.path.join(save_dir, 'pred_seg', 'pred_seg_{}_{:03d}.png'.format(data_name, idx_slice + 1)),
                    pred_mask_dense255[:, :])
                cv2.imwrite(
                    os.path.join(save_dir, 'pred_seg_p', 'pred_seg_p_{}_{:03d}.png'.format(data_name, idx_slice + 1)),
                    pred_mask_post_processed255[:, :])
                mkdir(os.path.join(save_dir, 'pred_seg_np'))
                mkdir(os.path.join(save_dir, 'gt_seg_np'))
                mkdir(os.path.join(save_dir, 'pred_seg_p_np'))
                np.save(
                    os.path.join(save_dir, 'pred_seg_np', 'pred_seg_np_{}_{:03d}.npy'.format(data_name, idx_slice + 1)),
                    pred_mask_dense[:, :])
                np.save(os.path.join(save_dir, 'pred_seg_p_np',
                                     'pred_seg_p_np_{}_{:03d}.npy'.format(data_name, idx_slice + 1)),
                        pred_mask_post_processed[:, :])
                np.save(os.path.join(save_dir, 'gt_seg_np', 'gt_seg_np_{}_{:03d}.npy'.format(data_name, idx_slice + 1)),
                        gt_mask_dense[:, :])

            # evaluation
            # one-hot pred --> multilabel dense pred
            pred_masks = np.array(pred_masks)
            gt_masks = np.array(gt_masks)
            pred_masks_post_processed = np.array(pred_masks_post_processed)

            csv_file = os.path.join(save_dir, 'metrics{}.csv'.format(idx))

            # ET--3
            labels = [0, 3]
            metrics = sg.write_metrics(labels=labels[1:],  # exclude background if needed
                                       gdth_img=gt_masks,
                                       pred_img=pred_masks,
                                       csv_file=csv_file,
                                       metrics=['dice', 'msd', 'hd', 'hd95'])
            print(metrics)
            metrics_list.append(metrics)

            # TC
            pred_masks[pred_masks == 3] = 1
            gt_masks[gt_masks == 3] = 1
            labels = [0, 1]
            metrics = sg.write_metrics(labels=labels[1:],  # exclude background if needed
                                       gdth_img=gt_masks,
                                       pred_img=pred_masks,
                                       csv_file=csv_file,
                                       metrics=['dice', 'msd', 'hd', 'hd95'])
            print(metrics)
            metrics_list.append(metrics)

            # WT
            pred_masks[pred_masks == 1] = 2
            gt_masks[gt_masks == 1] = 2
            labels = [0, 2]
            metrics = sg.write_metrics(labels=labels[1:],  # exclude background if needed
                                       gdth_img=gt_masks,
                                       pred_img=pred_masks,
                                       csv_file=csv_file,
                                       metrics=['dice', 'msd', 'hd', 'hd95'])
            print(metrics)
            metrics_list.append(metrics)

        dice = [[], [], []]
        hd = [[], [], []]
        hd95 = [[], [], []]
        for i in range(len(metrics_list)):
            cur_metric = metrics_list[i][0]
            label = cur_metric['label'][0] - 1
            dice[label].append(cur_metric['dice'][0])
            hd[label].append(cur_metric['hd'][0])
            hd95[label].append(cur_metric['hd95'][0])

        print(dice)

        for i in range(len(dice[0])):
            with open(os.path.join(save_dir, 'dice_results.csv'), 'a') as f:
                writer = csv.writer(f)
                writer.writerow(['UNet2022DA',
                                 dice[0][i], dice[1][i], dice[2][i]])

            with open(os.path.join(save_dir, 'hd_results.csv'), 'a') as f:
                writer = csv.writer(f)
                writer.writerow(['UNet2022DA',
                                 hd[0][i], hd[1][i], hd[2][i]])

            with open(os.path.join(save_dir, 'hd_results.csv'), 'a') as f:
                writer = csv.writer(f)
                writer.writerow(['UNet2022DA',
                                 hd95[0][i], hd95[1][i], hd95[2][i]])

        dice_std = []
        hd_std = []
        hd95_std = []
        for i in range(3):
            cur_dice = np.array(dice[i])
            cur_hd = np.array(hd[i])
            cur_hd95 = np.array(hd95[i])
            dice_std.append(np.std(cur_dice))
            hd_std.append(np.std(cur_hd))
            hd95_std.append(np.std(cur_hd95))

        dice_avg = []
        hd_avg = []
        hd95_avg = []
        for i in range(3):
            cur_dice = np.array(dice[i])
            cur_hd = np.array(hd[i])
            cur_hd95 = np.array(hd95[i])
            dice_avg.append(np.mean(cur_dice))
            hd_avg.append(np.mean(cur_hd))
            hd95_avg.append(np.mean(cur_hd95))

        print("dice_avg:{}".format(dice_avg))
        print("dice_std:{}".format(dice_std))
        print("hd_avg:{}".format(hd_avg))
        print("hd_std:{}".format(hd_std))
        print("hd95_avg:{}".format(hd95_avg))
        print("hd95_std:{}".format(hd95_std))


if __name__ == '__main__':
    ############################## PENDING ##############################

    torch.cuda.empty_cache()
    ############################## TRAINING ##############################
    # Yanglab gpu2&3
    main("/home/lichao/segmentation_framework/options/sdaut_parallel_conv_BraTS/train_sdaut_parallel_conv_kkkk_BraTS.json")
    # main("/home/lichao/segmentation_framework/options/train_sdaut_parallel_conv_kkkk_BraTS_DSC_main.json")
    ############################## TRAINED ##############################



