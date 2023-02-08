import sys
sys.path.append("/home/lichao/segmentation_framework")
import os.path
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
    import wandb

    wandb.init(project="sdaut_parallel_conv", entity="wlc")
    wandb.config.update(opt)

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
    init_iter_optimizer_netSeg, init_path_optimizer_netSeg = option.find_last_checkpoint(opt['path']['models'],  net_type='optimizerSeg')
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
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
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
        dice_result = OrderedDict()
        hd_result = OrderedDict()
        hd95_result = OrderedDict()
        msd_result = OrderedDict()

        dice_result['RA'] = []
        dice_result['MYO'] = []
        dice_result['RV'] = []

        hd_result['RA'] = []
        hd_result['MYO'] = []
        hd_result['RV'] = []

        hd95_result['RA'] = []
        hd95_result['MYO'] = []
        hd95_result['RV'] = []

        msd_result['RA'] = []
        msd_result['MYO'] = []
        msd_result['RV'] = []

        save_dir = opt['path']['test_image_save_dir']
        util.mkdir(save_dir)

        with open(os.path.join(save_dir, 'dice_results.csv'), 'w') as dsc:
            writer_dice = csv.writer(dsc)
            writer_dice.writerow(['Class/Method', 'RA', 'MYO', 'RV'])

        with open(os.path.join(save_dir, 'hd_results.csv'), 'w') as hd:
            writer_hd = csv.writer(hd)
            writer_hd.writerow(['Class/Method', 'RA', 'MYO', 'RV'])

        with open(os.path.join(save_dir, 'hd95_results.csv'), 'w') as hd95:
            writer_hd95 = csv.writer(hd95)
            writer_hd95.writerow(['Class/Method', 'RA', 'MYO', 'RV'])

        with open(os.path.join(save_dir, 'msd_results.csv'), 'w') as msd:
            writer_msd = csv.writer(msd)
            writer_msd.writerow(['Class/Method', 'RA', 'MYO', 'RV'])

        metrics_list = []
        metrics_list_p = []
        for idx, test_data in enumerate(test_loader):
                gt_masks = []
                pred_masks = []
                pred_masks_post_processed = []
                model.feed_3D_data(test_data)
                data_name = model.currnet_data_name()
                for idx_slice in range(test_data['src_image'].shape[1]):
                    model.feed_slice_from_3D(idx_slice)
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
                    classes = [i for i in range(1, dataset_opt['num_class']+1)]
                    pred_mask_dense_raw = pred_mask_dense.copy()
                    pred_mask_post_processed, removed_part, remaining_part = model.remove_all_but_the_largest_connected_component(pred_mask_dense_raw, classes)
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
                    cv2.imwrite(os.path.join(save_dir, 'src_image', 'src_image_{}_{:03d}.png'.format(data_name, idx_slice + 1)), gt_image255[0, :, :])
                    cv2.imwrite(os.path.join(save_dir, 'gt_seg', 'gt_seg_{}_{:03d}.png'.format(data_name, idx_slice + 1)), gt_mask_dense255[:, :])
                    cv2.imwrite(os.path.join(save_dir, 'pred_seg', 'pred_seg_{}_{:03d}.png'.format(data_name, idx_slice + 1)), pred_mask_dense255[:, :])
                    cv2.imwrite(os.path.join(save_dir, 'pred_seg_p', 'pred_seg_p_{}_{:03d}.png'.format(data_name, idx_slice + 1)), pred_mask_post_processed255[:, :])
                    mkdir(os.path.join(save_dir, 'pred_seg_np'))
                    mkdir(os.path.join(save_dir, 'gt_seg_np'))
                    mkdir(os.path.join(save_dir, 'pred_seg_p_np'))
                    np.save(os.path.join(save_dir, 'pred_seg_np', 'pred_seg_np_{}_{:03d}.npy'.format(data_name, idx_slice + 1)), pred_mask_dense[:, :])
                    np.save(os.path.join(save_dir, 'pred_seg_p_np', 'pred_seg_p_np_{}_{:03d}.npy'.format(data_name, idx_slice + 1)), pred_mask_post_processed[:, :])
                    np.save(os.path.join(save_dir, 'gt_seg_np', 'gt_seg_np_{}_{:03d}.npy'.format(data_name, idx_slice + 1)), gt_mask_dense[:, :])

                # evaluation
                # one-hot pred --> multilabel dense pred
                pred_masks = np.array(pred_masks)
                gt_masks = np.array(gt_masks)
                pred_masks_post_processed = np.array(pred_masks_post_processed)

                labels = [0, 1, 2, 3]
                csv_file = os.path.join(save_dir, 'metrics{}.csv'.format(idx))
                metrics = sg.write_metrics(labels=labels[1:],  # exclude background if needed
                                           gdth_img=gt_masks,
                                           pred_img=pred_masks,
                                           csv_file=csv_file,
                                           metrics=['dice', 'msd', 'hd', 'hd95'])
                print(metrics)
                metrics_list.append(metrics)
                metrics = metrics[0]
                with open(os.path.join(save_dir, 'dice_results.csv'), 'a') as dice:
                    writer = csv.writer(dice)
                    writer.writerow(['UNet',
                                     metrics['dice'][0], metrics['dice'][1], metrics['dice'][2]])

                with open(os.path.join(save_dir, 'hd_results.csv'), 'a') as hd:
                    writer = csv.writer(hd)
                    writer.writerow(['UNet',
                                     metrics['hd'][0], metrics['hd'][1], metrics['hd'][2]])

                with open(os.path.join(save_dir, 'hd95_results.csv'), 'a') as hd95:
                    writer = csv.writer(hd95)
                    writer.writerow(['UNet',
                                     metrics['hd95'][0], metrics['hd95'][1], metrics['hd95'][2]])

                with open(os.path.join(save_dir, 'msd_results.csv'), 'a') as msd:
                    writer = csv.writer(msd)
                    writer.writerow(['UNet',
                                     metrics['msd'][0], metrics['msd'][1], metrics['msd'][2]])

                csv_file = os.path.join(save_dir, 'metrics_p{}.csv'.format(idx))
                metrics_p = sg.write_metrics(labels=labels[1:],  # exclude background if needed
                                           gdth_img=gt_masks,
                                           pred_img=pred_masks_post_processed,
                                           csv_file=csv_file,
                                           metrics=['dice', 'msd', 'hd', 'hd95'])
                print(metrics_p)
                metrics_list_p.append(metrics_p)

        dice = [[], [], []]
        hd = [[], [], []]
        hd95 = [[], [], []]
        for i in range(len(metrics_list)):
            cur_metric = metrics_list[i][0]
            for i in range(len(labels) - 1):
                dice[i].append(cur_metric['dice'][i])
                hd[i].append(cur_metric['hd'][i])
                hd95[i].append(cur_metric['hd95'][i])

        dice_std = []
        hd_std = []
        hd95_std = []
        for i in range(len(labels) - 1):
            cur_dice = np.array(dice[i])
            cur_hd = np.array(hd[i])
            cur_hd95 = np.array(hd95[i])
            dice_std.append(np.std(cur_dice))
            hd_std.append(np.std(cur_hd))
            hd95_std.append(np.std(cur_hd95))

        dice_avg = []
        hd_avg = []
        hd95_avg = []
        for i in range(len(labels) - 1):
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

        dice = [[], [], []]
        hd = [[], [], []]
        hd95 = [[], [], []]
        for i in range(len(metrics_list_p)):
            cur_metric = metrics_list_p[i][0]
            # print(cur_metric['dice'][i])
            for i in range(len(labels) - 1):
                dice[i].append(cur_metric['dice'][i])
                hd[i].append(cur_metric['hd'][i])
                hd95[i].append(cur_metric['hd95'][i])

        dice_std = []
        hd_std = []
        hd95_std = []
        for i in range(len(labels) - 1):
            cur_dice = np.array(dice[i])
            cur_hd = np.array(hd[i])
            cur_hd95 = np.array(hd95[i])
            dice_std.append(np.std(cur_dice))
            hd_std.append(np.std(cur_hd))
            hd95_std.append(np.std(cur_hd95))

        dice_avg = []
        hd_avg = []
        hd95_avg = []
        for i in range(len(labels) - 1):
            cur_dice = np.array(dice[i])
            cur_hd = np.array(hd[i])
            cur_hd95 = np.array(hd95[i])
            dice_avg.append(np.mean(cur_dice))
            hd_avg.append(np.mean(cur_hd))
            hd95_avg.append(np.mean(cur_hd95))

        print("dice_avg_p:{}".format(dice_avg))
        print("dice_std_p:{}".format(dice_std))
        print("hd_avg_p:{}".format(hd_avg))
        print("hd_std_p:{}".format(hd_std))
        print("hd95_avg_p:{}".format(hd95_avg))
        print("hd95_std_p:{}".format(hd95_std))


if __name__ == '__main__':
    main("/home/lichao/segmentation_framework/options/UNet_2D_ACDC_ED.json")
    # main("/home/lichao/segmentation_framework/options/sdaut_parallel_conv_ACDC/train_sdaut_parallel_conv_ACDC_ED_kkss.json")
    # main("/home/lichao/segmentation_framework/options/sdaut_parallel_conv_ACDC/train_sdaut_parallel_conv_ACDC_ED_ssss.json")