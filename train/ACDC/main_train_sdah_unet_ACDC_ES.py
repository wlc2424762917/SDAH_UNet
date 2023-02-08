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
import os


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

    # set seed
    print("opt", opt['manual_seed'])
    seed = opt['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()
    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    print(opt['path']['models'])
    init_iter_netSeg, init_path_netSeg = option.find_last_checkpoint(opt['path']['models'], net_type='Seg')
    opt['path']['pretrained_netSeg'] = init_path_netSeg
    init_iter_optimizer_netSeg, init_path_optimizer_netSeg = option.find_last_checkpoint(opt['path']['models'],
                                                                                         net_type='optimizerSeg')
    opt['path']['pretrained_optimizer_netSeg'] = init_path_optimizer_netSeg
    current_step = init_iter_netSeg
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

    # tensorbordX log
    logger_tensorboard = SummaryWriter(os.path.join(opt['path']['log']))

    # wandb logger
    import wandb

    wandb.init(project="UNet2022_comparison_ACDC", entity="wlc")
    wandb.config.update(opt)

    wandb.define_metric("TRAIN/step")
    wandb.define_metric('TRAIN/Learning Rate', step_metric="TRAIN/step")
    wandb.define_metric('TRAIN LOSS/loss_total', step_metric="TRAIN/step")
    wandb.define_metric('TRAIN LOSS/loss_ce', step_metric="TRAIN/step")
    wandb.define_metric('TRAIN LOSS/loss_dice', step_metric="TRAIN/step")
    wandb.define_metric("VAL/step")
    for idx_class in range(opt['datasets']['test']['num_class']):
        wandb.define_metric(f'VAL METRICS/Hausdorff Class {idx_class}', step_metric="VAL/step")
        wandb.define_metric(f'VAL METRICS/Dice Class {idx_class}', step_metric="VAL/step")

    # choose if use tile to predict
    # ----------------------------------------
    use_tile = opt['netSeg']['use_tile']
    print("use_tile: ", use_tile)

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            # print(len(train_set))
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            if opt['rank'] == 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            if opt['dist']:
                train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'], drop_last=True,
                                                   seed=seed)
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'] // opt['num_gpu'],
                                          shuffle=False,
                                          num_workers=dataset_opt['dataloader_num_workers'] // opt['num_gpu'],
                                          drop_last=True,
                                          pin_memory=False,
                                          sampler=train_sampler)
            else:
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'],
                                          shuffle=dataset_opt['dataloader_shuffle'],
                                          num_workers=dataset_opt['dataloader_num_workers'],
                                          drop_last=True,
                                          pin_memory=False)

        elif phase == 'test':
            print(dataset_opt)
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=0,
                                     drop_last=False, pin_memory=False)

            with open(os.path.join(opt['path']['log'], 'val_results_ave.csv'), 'w') as cf:
                writer = csv.writer(cf)
                writer.writerow(['METHOD', 'CLASS', 'IoU', 'IoU_STD', 'Dice', 'Dice_STD',
                                 'avgHausdorff', 'avgHausdorff_STD', 'Hausdorff', 'Hausdorff_STD'])
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    model = define_Model(opt)

    model.init_train()
    if opt['rank'] == 0:
        logger.info(model.info_params())

    # early_stopping = utils_early_stopping.EarlyStopping(patience=opt['train']['early_stopping_num'])
    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''

    for epoch in range(100000):  # keep running
        print(f"epoch {epoch}:")
        for i, train_data in enumerate(train_loader):
            current_step += 1

            # if current_step == 1:
            #     time1 = time.time()
            # -------------------------------
            # 1) update learning rate
            # -------------------------------
            model.update_learning_rate(current_step)

            # -------------------------------
            # 2) feed patch pairs
            # -------------------------------
            model.feed_data(train_data)

            # -------------------------------
            # 3) optimize parameters
            # -------------------------------
            model.optimize_parameters(current_step)

            # -------------------------------
            # 4) training information
            # -------------------------------
            if current_step % opt['train']['checkpoint_print'] == 0 and opt['rank'] == 0:
                logs = model.current_log()  # such as loss
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step,
                                                                          model.current_learning_rate())
                for k, v in logs.items():  # merge log information into message
                    message += '{:s}: {:.3e} '.format(k, v)
                # message += 'Time Used: {:.8e} '.format(time_used_100_steps)
                logger.info(message)

                # record train loss
                log_wandb = {'TRAIN/step': current_step, 'TRAIN/Learning Rate': model.current_learning_rate(),
                             'TRAIN LOSS/loss_total': logs['loss_total']}
                if 'loss_ce' in logs.keys():
                    log_wandb['TRAIN LOSS/loss_ce'] = logs['loss_ce']
                if 'loss_dice' in logs.keys():
                    log_wandb['TRAIN LOSS/loss_dice'] = logs['loss_dice']
                wandb.log(log_wandb)

            # -------------------------------
            # 5) save model
            # -------------------------------
            if current_step % opt['train']['checkpoint_save'] == 0 and opt['rank'] == 0:
                logger.info('Saving the model.')
                model.save(current_step)

            # # -------------------------------
            # # 6) testing
            # # -------------------------------
            # if current_step % opt['train']['checkpoint_test'] == 0 and opt['rank'] == 0:
            # if current_step % 1 and opt['rank'] == 0:
            if current_step % 5000 == 1:
                with torch.no_grad():
                    test_results = OrderedDict()
                    for idx_class in range(dataset_opt['num_class'] + 1):
                        test_results[f'{idx_class}'] = OrderedDict()
                        test_results[f'{idx_class}']['iou'] = []
                        test_results[f'{idx_class}']['dice'] = []
                        test_results[f'{idx_class}']['avgHausdorff'] = []
                        test_results[f'{idx_class}']['Hausdorff'] = []

                    for idx, test_data in enumerate(test_loader):
                        print(f'testing image{test_data["name"]}')
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

                        output = model.current_visuals_3d(use_tile)
                        gt_segs = output['gt_segs_3d']
                        pred_segs = output['pred_segs_3d']

                        # evaluation
                        # one-hot pred --> multilabel dense pred
                        if use_tile:
                            pred_segs_dense = pred_segs
                        else:
                            pred_segs_dense = np.squeeze(np.argmax(pred_segs, 0))
                        # print(np.max(prec_masks_dense))
                        pred_masks_cliped = np.clip(pred_segs, 0, 1)
                        gt_segs_dense = np.squeeze(np.argmax(gt_segs, 0))
                        # print(gt_masks_dense.shape)
                        for idx_class in range(opt['datasets']['test']['num_class'] + 1):
                            # iou = util.get_iou_3d(prec_masks[idx_class, :, :, :], gt_masks[idx_class, :, :, :])
                            iou = 0
                            intersection = np.sum((pred_segs_dense == idx_class) * (gt_segs_dense == idx_class))
                            union = np.sum(pred_segs_dense == idx_class) + np.sum(gt_segs_dense == idx_class)
                            dice = (2 * intersection + 1e-4) / (union + 1e-4)
                            msg = '<idx_class：{:3d}, intersec:{:3f}, Pred:{:3f}, GT:{:3f}, Dice:{:3f}> ' \
                                .format(idx_class, intersection, np.sum(pred_segs_dense == idx_class),
                                        np.sum(gt_segs_dense == idx_class), dice)
                            logger.info(msg)

                            Hausdorff, avgHausdorff = 0, 0
                            avgHausdorff = util.get_hd_medpy(pred_segs_dense == idx_class, gt_segs_dense == idx_class)

                            test_results[f'{idx_class}']['iou'].append(iou)
                            test_results[f'{idx_class}']['dice'].append(dice)
                            test_results[f'{idx_class}']['avgHausdorff'].append(avgHausdorff)
                            test_results[f'{idx_class}']['Hausdorff'].append(Hausdorff)

                    # evaluation and logger
                    log_wandb = {'VAL/step': current_step, }
                    for idx_class in range(opt['datasets']['test']['num_class'] + 1):
                        # summarize psnr/ssim
                        ave_iou = np.mean(test_results[f'{idx_class}']['iou'])
                        std_iou = np.std(test_results[f'{idx_class}']['iou'], ddof=1)
                        ave_dice = np.mean(test_results[f'{idx_class}']['dice'])
                        std_dice = np.std(test_results[f'{idx_class}']['dice'], ddof=1)
                        ave_avgHausdorff = np.mean(test_results[f'{idx_class}']['avgHausdorff'])
                        std_avgHausdorff = np.std(test_results[f'{idx_class}']['avgHausdorff'], ddof=1)
                        ave_Hausdorff = np.mean(test_results[f'{idx_class}']['Hausdorff'])
                        std_Hausdorff = np.std(test_results[f'{idx_class}']['Hausdorff'], ddof=1)

                        print(
                            'Validation Step:{}\nClass {:d}\n-- IoU {} ({})\n-- Dice {} ({})\n--avgHausdorff {} ({})\n-- Hausdorff Dice {} ({})'
                            .format(current_step, idx_class, ave_iou, std_iou, ave_dice, std_dice, ave_avgHausdorff,
                                    std_avgHausdorff, ave_Hausdorff, std_Hausdorff))

                        with open(os.path.join(os.path.join(opt['path']['log']), f'val_results_ave.csv'), 'a') as cf:
                            writer = csv.writer(cf)
                            writer.writerow(
                                ['UNET2022DA_ACDC', idx_class, ave_iou, std_iou, ave_dice, std_dice, ave_avgHausdorff,
                                 std_avgHausdorff, ave_Hausdorff, std_Hausdorff])

                        log_wandb[f'VAL METRICS/Class {idx_class}/IoU'] = ave_iou
                        log_wandb[f'VAL METRICS/Class {idx_class}/Dice'] = ave_dice
                        log_wandb[f'VAL METRICS/Class {idx_class}/avgHausdorff'] = ave_avgHausdorff
                        log_wandb[f'VAL METRICS/Class {idx_class}/Hausdorff'] = ave_Hausdorff

                    wandb.log(log_wandb)

    print("Training Stop")


if __name__ == '__main__':
    ############################## PENDING ##############################

    torch.cuda.empty_cache()
    ############################## TRAINING ##############################
    # Yanglab gpu2&3
    main("/home/lichao/segmentation_framework/options/sdaut_parallel_conv_ACDC/train_sdaut_parallel_conv_ACDC_ES_kkkk.json")
    ############################## TRAINED ##############################



