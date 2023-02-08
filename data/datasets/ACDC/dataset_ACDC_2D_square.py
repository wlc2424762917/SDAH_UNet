'''
# -----------------------------------------
Data Loader
mmWHS
mapping label value [0, 205, 420, 500, 550, 600, 820, 850] to [0, 1, 2, 3, 4, 5, 6, 7]
[Myo, LA, LV, RA, RV, AO, PA]
# -----------------------------------------
'''
import math
import random

import numpy as np
import torch
import torch.utils.data as data
import common_utils.utils_image as util
from skimage.measure import block_reduce
import os
from scipy.ndimage.interpolation import zoom


class Dataset_ACDC_2D(data.Dataset):
    '''
    # -----------------------------------------
    # Get src/gt for src_image-to-seg_image mapping.
    # Both "paths_src" and "paths_gt" are needed.
    # -----------------------------------------
    '''

    def __init__(self, opt):
        super(Dataset_ACDC_2D, self).__init__()
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
        # print(len(self.paths_src))
        # print(len(self.paths_gt))
        if self.is_mini_dataset:

            index = list(range(0, len(self.paths_H)))
            index_chosen = random.sample(index, round(self.mini_dataset_prec * len(self.paths_H)))
            self.paths_H_new = []
            for i in index_chosen:
                self.paths_H_new.append(self.paths_H[i])
            self.paths_H = self.paths_H_new

        # ------------------------------------
        # get mask
        # ------------------------------------

        # for reconstruction
        # self.mask = define_Mask(self.opt)

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
            # central crop + random crop
            # --------------------------------
            elif self.resize_method == "random_central_crop":
                if H < self.patch_size_H or W < self.patch_size_W:
                    img_gt = np.pad(img_gt, ((0, (-H) % self.patch_size_H), (0, (-W) % self.patch_size_W), (0, 0)))
                    img_src = np.pad(img_src,
                                     ((0, (-H) % self.patch_size_H), (0, (-W) % self.patch_size_W), (0, 0)))
                if random.randint(0, 9) < 6:  # central_crop 70%
                    H, W = self.patch_size_H, self.patch_size_W
                    patch_gt = img_gt[H // 2 - self.patch_size_H // 2:H // 2 + self.patch_size_H // 2,
                               W // 2 - self.patch_size_W // 2:W // 2 + self.patch_size_W // 2, :]
                    patch_src = img_src[H // 2 - self.patch_size_H // 2:H // 2 + self.patch_size_H // 2,
                                W // 2 - self.patch_size_W // 2:W // 2 + self.patch_size_W // 2, :]
                else:  # random_crop 30%
                    rnd_h = random.randint(0, max(0, H - self.patch_size_H))
                    rnd_w = random.randint(0, max(0, W - self.patch_size_W))
                    patch_gt = img_gt[rnd_h:rnd_h + self.patch_size_H, rnd_w:rnd_w + self.patch_size_W, :]
                    patch_src = img_src[rnd_h:rnd_h + self.patch_size_H, rnd_w:rnd_w + self.patch_size_W, :]

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
                    if np.sum(np.clip(patch_gt, 0, 1)) > 500:
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
                self.patch_size = 512
                img_gt = np.pad(img_gt, ((0, (-H) % self.patch_size), (0, (-W) % self.patch_size), (0, 0)))
                img_src = np.pad(img_src, ((0, (-H) % self.patch_size), (0, (-W) % self.patch_size), (0, 0)))
                patch_gt = block_reduce(img_gt,
                                        block_size=(math.ceil(H / self.patch_size), math.ceil(W / self.patch_size), 1),
                                        func=np.mean)
                patch_src = block_reduce(img_src, block_size=(
                    math.ceil(H / self.patch_size), math.ceil(W / self.patch_size), 1),
                                         func=np.mean)

            # --------------------------------
            # resample
            # --------------------------------
            elif self.resize_method == "resample":
                zoom_scale_H = self.patch_size_H / H
                zoom_scale_W = self.patch_size_W / W
                patch_gt = zoom(img_gt, zoom=[zoom_scale_H, zoom_scale_W, 1], order=0)
                patch_src = zoom(img_src, zoom=[zoom_scale_H, zoom_scale_W, 1], order=0)

            else:
                patch_gt = img_gt
                patch_src = img_src
            # --------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # --------------------------------
            img_gt, img_src = util.uint2tensor3_seg(patch_gt), util.uint2tensor3(patch_src)
        return {'gt_seg': img_gt, 'src_image': img_src, 'src_path': src_path, 'name': file_name}

    def __len__(self):
        return len(self.paths_src)

    def load_images(self, path, imgType):
        # load src & gt
        img = np.load(path).astype(np.float32)
        img = np.reshape(img, (img.shape[0], img.shape[1], 1))
        # # 0 ~ 255
        if imgType == 'src':
            img = (img - img.min()) / (img.max() - img.min()) * 255

        return img

    def load_volumes(self, path, imgType):
        # load src & gt
        img = np.load(path).astype(np.float32)
        # # 0 ~ 255
        if imgType == 'src':
            img = (img - img.min()) / (img.max() - img.min()) * 255
        return img


# local test
if __name__ == "__main__":
    opt = {}

    # opt['dataroot_src'] = '/media/NAS02/SynapseMultiorganSegmentation/Data/RawData/Trainingset/src'
    # opt['dataroot_gt'] = '/media/NAS02/SynapseMultiorganSegmentation/Data/RawData/Trainingset/gt'

    opt['dataroot_src'] = r"E:\ACDC\trainingset_3D\src"
    opt['dataroot_gt'] = r"E:\ACDC\trainingset_3D\gt"
    opt['n_channels'] = 1
    opt['H_size'] = 256
    opt['W_size'] = 256

    # reconstruction settings --------------
    opt['is_noise'] = False
    opt['noise_level'] = 0
    opt['noise_var'] = 0
    # ---------------------------------------

    opt['is_mini_dataset'] = False
    opt['mini_dataset_prec'] = None
    opt['phase'] = 'train'
    opt['resize_method'] = "random_central_crop"

    test_dataset = Dataset_ACDC_2D(opt)

    idx = random.randint(0,100)
    sample = test_dataset.__getitem__(idx)
    print(sample['name'])
    print(torch.max(sample['gt_seg']))
    print(sample['gt_seg'].shape)
    import matplotlib.pyplot as plt
    plt.imshow(sample['gt_seg'][0])
    plt.show()

    data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2,
                                              shuffle=True)
    #for i, train_data in enumerate(data_loader):
    #    print(i)