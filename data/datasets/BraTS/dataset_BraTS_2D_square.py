'''
# -----------------------------------------
Data Loader
BraTS
1 ,2, 3, 4
# -----------------------------------------
'''
import sys
sys.path.append("/home/lichao/segmentation_framework")
import math
import random

import numpy as np
import torch
import torch.utils.data as data
import common_utils.utils_image as util
from skimage.measure import block_reduce
import os
from scipy.ndimage.interpolation import zoom


class Dataset_BraTS_2D(data.Dataset):
    '''
    # -----------------------------------------
    # Get src/gt for src_image-to-seg_image mapping.
    # Both "paths_src" and "paths_gt" are needed.
    # -----------------------------------------
    '''

    def __init__(self, opt):
        super(Dataset_BraTS_2D, self).__init__()
        print('Get src/gt_seg for image-to-image mapping. Both "paths_src" and "paths_gt_seg" are needed.')
        self.opt = opt
        self.n_channels = self.opt['n_channels']
        self.patch_size_H = self.opt['H_size']
        self.patch_size_W = self.opt['W_size']
        self.is_mini_dataset = self.opt['is_mini_dataset']
        self.mini_dataset_prec = self.opt['mini_dataset_prec']
        # ------------------------------------
        # get the path of src/gt_seg
        # ------------------------------------
        self.paths_src = util.get_image_paths(opt['dataroot_src'])
        assert self.paths_src, 'Error: Raw src path is empty.'
        self.paths_gt = util.get_image_paths(opt['dataroot_gt'])
        assert self.paths_gt, 'Error: Raw gt path is empty.'
        self.resize_method = self.opt['resize_method']

    def __getitem__(self, index):
        # if train, get slice pair
        # ------------------------------------
        if self.opt['phase'] == 'train':
            src_path = self.paths_src[index]
            gt_path = self.paths_gt[index]
            file_name = os.path.basename(src_path)
            # print(file_name)
            img_src = self.load_images(src_path, imgType='src')
            img_gt = self.load_images(gt_path, imgType='gt')
            H, W, _ = img_src.shape

            # --------------------------------
            # random crop
            # --------------------------------
            if self.resize_method == "random_crop":
                if H < self.patch_size_H or W < self.patch_size_W:
                    img_gt = np.pad(img_gt, ((0, (-H) % self.patch_size_H), (0, (-W) % self.patch_size_W), (0, 0)))
                    img_src = np.pad(img_src, ((0, (-H) % self.patch_size_H), (0, (-W) % self.patch_size_W), (0, 0)))

                while 1:
                    rnd_h = random.randint(0, max(0, H - self.patch_size_H))
                    rnd_w = random.randint(0, max(0, W - self.patch_size_W))
                    patch_gt = img_gt[rnd_h:rnd_h + self.patch_size_H, rnd_w:rnd_w + self.patch_size_W, :]
                    patch_src = img_src[rnd_h:rnd_h + self.patch_size_H, rnd_w:rnd_w + self.patch_size_W, :]
                    # print(np.clip(patch_gt, 0, 1).shape, np.max(np.clip(patch_gt, 0, 1)), np.sum(np.clip(patch_gt, 0, 1)))
                    pixels = np.sum(np.clip(img_gt, 0, 1))
                    if np.sum(np.clip(patch_gt, 0, 1)) > 1/5 * pixels:  #
                        break
            # --------------------------------
            # central crop
            # --------------------------------
            elif self.resize_method == "central_crop":
                if H < self.patch_size_H or W < self.patch_size_W:
                    img_gt = np.pad(img_gt, ((0, (-H) % self.patch_size_H), (0, (-W) % self.patch_size_W), (0, 0)))
                    img_src = np.pad(img_src, ((0, (-H) % self.patch_size_H), (0, (-W) % self.patch_size_W), (0, 0)))
                H, W = self.patch_size_H, self.patch_size_W
                patch_gt = img_gt[H // 2 - self.patch_size_H // 2:H // 2 + self.patch_size_H // 2,
                           W // 2 - self.patch_size_W // 2:W // 2 + self.patch_size_W // 2, :]
                patch_src = img_src[H // 2 - self.patch_size_H // 2:H // 2 + self.patch_size_H // 2,
                            W // 2 - self.patch_size_W // 2:W // 2 + self.patch_size_W // 2, :]

            # --------------------------------
            # down sample
            # --------------------------------
            elif self.resize_method == "down_sample":
                img_gt = np.pad(img_gt, ((0, (-H) % self.patch_size_H), (0, (-W) % self.patch_size_W), (0, 0)))
                img_src = np.pad(img_src, ((0, (-H) % self.patch_size_H), (0, (-W) % self.patch_size_W), (0, 0)))
                patch_gt = block_reduce(img_gt,
                                        block_size=(math.ceil(H / self.patch_size_H), math.ceil(W / self.patch_size_W), 1),
                                        func=np.mean)
                patch_src = block_reduce(img_src, block_size=(
                math.ceil(H / self.patch_size_H), math.ceil(W / self.patch_size_W), 1),
                                         func=np.mean)

            # --------------------------------
            # resample
            # --------------------------------
            elif self.resize_method == "resample":
                zoom_scale_H = self.patch_size_H/H
                zoom_scale_W = self.patch_size_W/W
                patch_gt = zoom(img_gt, zoom=[zoom_scale_H, zoom_scale_W, 1], order=0)
                patch_src = zoom(img_src, zoom=[zoom_scale_H, zoom_scale_W, 1], order=0)
            else:
                patch_gt = img_gt
                patch_src = img_src
            # --------------------------------
            # augmentation - flip and/or rotate
            # --------------------------------
            mode = random.randint(0, 7)
            patch_gt, patch_src = util.augment_img(patch_gt, mode=mode), util.augment_img(patch_src, mode=mode)

            # --------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # --------------------------------
            img_gt, img_src = util.uint2tensor3_seg(patch_gt), util.uint2tensor3(patch_src)

        # ------------------------------------
        # if test, ##fixing##
        # ------------------------------------
        elif self.opt['phase'] == 'test':
            src_path = self.paths_src[index]
            gt_path = self.paths_gt[index]
            file_name = os.path.basename(src_path)
            img_src = self.load_volumes(src_path, imgType='src')
            img_gt = self.load_volumes(gt_path, imgType='gt')
            C, H, W, _ = img_src.shape

            # --------------------------------
            # central crop
            # --------------------------------

            if self.resize_method == "central_crop":
                if H < self.patch_size_H or W < self.patch_size_W:
                    img_gt = np.pad(img_gt, ((0, (-H) % self.patch_size_H), (0, (-W) % self.patch_size_W), (0, 0)))
                    img_src = np.pad(img_src, ((0, (-H) % self.patch_size_H), (0, (-W) % self.patch_size_W), (0, 0)))
                H, W = self.patch_size_H, self.patch_size_W
                patch_gt = img_gt[:, H // 2 - self.patch_size_H // 2:H // 2 + self.patch_size_H // 2,
                           W // 2 - self.patch_size_W // 2:W // 2 + self.patch_size_W // 2, :]
                patch_src = img_src[:, H // 2 - self.patch_size_H // 2:H // 2 + self.patch_size_H // 2,
                            W // 2 - self.patch_size_W // 2:W // 2 + self.patch_size_W // 2, :]

            else:
                patch_gt = img_gt
                patch_src = img_src
            # --------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # --------------------------------
            img_gt, img_src = util.uint3tensor4_seg(patch_gt), util.uint3tensor4(patch_src)
            # img_gt = torch.squeeze(img_gt)
            # print(img_src.shape)
            # print(img_gt.shape)
        return {'gt_seg': img_gt, 'src_image': img_src, 'src_path': src_path, 'name': file_name}

    def __len__(self):
        return len(self.paths_src)

    def load_images(self, path, imgType):
        # load src & gt
        img = np.load(path).astype(np.float32)
        # # 0 ~ 255
        if imgType == 'src':
            # reshape is probably wrong here
            # img = img.reshape((img.shape[1], img.shape[2], 4))
            img = np.transpose(img, (1, 2, 0))
        elif len(img.shape) == 2:
            img = img.reshape((img.shape[0], img.shape[1], 1))
        else:
            img = np.transpose(img, (1, 2, 0))
        return img

    def load_volumes(self, path, imgType):
        # load src & gt
        img = np.load(path).astype(np.float32)
        return img


# local test
if __name__ == "__main__":
    opt = {}

    # opt['dataroot_src'] = '/media/NAS02/SynapseMultiorganSegmentation/Data/RawData/Trainingset/src'
    # opt['dataroot_gt'] = '/media/NAS02/SynapseMultiorganSegmentation/Data/RawData/Trainingset/gt'

    opt['dataroot_src'] = "/media/NAS02/BraTS2020/Trainingset/src"
    opt['dataroot_gt'] = "/media/NAS02/BraTS2020/Trainingset/gt"
    opt['dataroot_pkl'] = "/media/NAS02/BraTS2020/Trainingset/properties"

    opt['dataroot_src'] = "/media/NAS02/BraTS2020/Testingset/src"
    opt['dataroot_gt'] = "/media/NAS02/BraTS2020/Testingset/gt"
    opt['dataroot_pkl'] = "/media/NAS02/BraTS2020/Testingset/properties"

    opt['n_channels'] = 1
    opt['H_size'] = 224
    opt['W_size'] = 224

    # reconstruction settings --------------
    opt['is_noise'] = False
    opt['noise_level'] = 0
    opt['noise_var'] = 0
    # ---------------------------------------

    opt['is_mini_dataset'] = False
    opt['mini_dataset_prec'] = None
    opt['phase'] = 'test'
    opt['resize_method'] = "random_central_crop"

    test_dataset = Dataset_BraTS_2D(opt)

    idx = random.randint(0, 100)
    sample = test_dataset.__getitem__(4)
    print(sample['name'])
    print(torch.max(sample['gt_seg']))
    print(sample['gt_seg'].shape)
    print(sample['src_image'].shape)

    data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2,
                                              shuffle=True)
    for i, train_data in enumerate(data_loader):
        print(train_data['src_image'].shape)