'''
# -----------------------------------------
Data Loader
ACDC
[RV, MYO, LV]
# -----------------------------------------
'''
import sys
sys.path.append("/home/lichao/segmentation_framework")
## need to be refracted......
import math
import random

import numpy as np
import torch
import torch.utils.data as data
import common_utils.utils_image as util
from skimage.measure import block_reduce
import os
from scipy.ndimage.interpolation import zoom
import pickle


def get_paths_pkl(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    pkls = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
                pkl_path = os.path.join(dirpath, fname)
                pkls.append(pkl_path)
    assert pkls, '{:s} has no valid image pkl'.format(path)
    return pkls


def get_pkl_paths(dataroot):
    paths = None  # return None if dataroot is None
    if isinstance(dataroot, str):
        paths = sorted(get_paths_pkl(dataroot))
    elif isinstance(dataroot, list):
        paths = []
        for i in dataroot:
            paths += sorted(get_paths_pkl(i))
    return paths


class Dataset_ACDC_2D(data.Dataset):
    '''
    # -----------------------------------------
    # Get src/gt for src_image-to-seg_image mapping.
    # Both "paths_src" and "paths_gt" are needed.
    # -----------------------------------------
    '''

    def __init__(self, opt):
        super(Dataset_ACDC_2D, self).__init__()
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
        self.path_properties = get_pkl_paths(opt['dataroot_pkl'])
        assert self.path_properties, 'Error: Raw properties path is empty.'
        self.resize_method = self.opt['resize_method']

    def __getitem__(self, index):
        # if train, get slice pair
        # ------------------------------------
        if self.opt['phase'] == 'train':
            src_path = self.paths_src[index]
            gt_path = self.paths_gt[index]
            properties_path = self.path_properties[index]
            file_name = os.path.basename(src_path)
            img_src = self.load_images(src_path, imgType='src')
            img_gt = self.load_images(gt_path, imgType='gt')
            H, W, _ = img_src.shape

            # --------------------------------
            # zoom for swin_Transformer
            # --------------------------------
            if self.resize_method == "zoom":
                zoom_scale_H = self.patch_size_H / H
                zoom_scale_W = self.patch_size_W / W
                patch_gt = zoom(img_gt, zoom=[zoom_scale_H, zoom_scale_W, 1], order=0)
                patch_src = zoom(img_src, zoom=[zoom_scale_H, zoom_scale_W, 1], order=0)

            # --------------------------------
            # patch_sampler for Conv UNet
            # --------------------------------
            else:
                img_properties = self.load_pkl(properties_path)
                H, W, _ = img_src.shape
                foreground_classes = np.array([i for i in img_properties['class_locations'].keys() if len(img_properties['class_locations'][i]) != 0])
                if H < self.patch_size_H or W < self.patch_size_W:
                    img_gt = np.pad(img_gt, ((0, (-H) % self.patch_size_H), (0, (-W) % self.patch_size_W), (0, 0)))
                    img_src = np.pad(img_src, ((0, (-H) % self.patch_size_H), (0, (-W) % self.patch_size_W), (0, 0)))

                if len(foreground_classes) == 0:
                    # random crop
                    rnd_h = random.randint(0, max(0, H - self.patch_size_H))
                    rnd_w = random.randint(0, max(0, W - self.patch_size_W))
                    patch_gt = img_gt[rnd_h:rnd_h + self.patch_size_H, rnd_w:rnd_w + self.patch_size_W, :]
                    patch_src = img_src[rnd_h:rnd_h + self.patch_size_H, rnd_w:rnd_w + self.patch_size_W, :]
                else:
                    # random choose a class
                    selected_class = np.random.choice(foreground_classes)
                    pixels_of_that_class = img_properties['class_locations'][selected_class]
                    # random choose a pixel of that class
                    selected_pixel = pixels_of_that_class[np.random.choice(len(pixels_of_that_class))]
                    # selected pixel is center voxel. Subtract half the patch size to get lower bound.
                    img_gt = np.pad(img_gt, ((0, self.patch_size_H // 2), (0, self.patch_size_W), (0, 0)))
                    img_src = np.pad(img_src, ((0, self.patch_size_H // 2), (0, self.patch_size_W), (0, 0)))
                    h = max(0, selected_pixel[0] - self.patch_size_H // 2)
                    w = max(0, selected_pixel[1] - self.patch_size_W // 2)
                    patch_gt = img_gt[h:h + self.patch_size_H, w:w + self.patch_size_W, :]
                    patch_src = img_src[h:h + self.patch_size_H, w:w + self.patch_size_W, :]

            # --------------------------------
            # augmentation - flip and/or rotate
            # --------------------------------
            """mode = random.randint(0, 7)
            if mode > 5:
                patch_gt, patch_src = util.augment_img_no_rot(patch_gt, mode=mode), util.augment_img_no_rot(patch_src, mode=mode)
            """
            mode = random.randint(0, 5)
            patch_gt, patch_src = util.augment_img_2D(patch_gt, mode=mode), util.augment_img_2D(patch_src, mode=mode)
            # --------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # --------------------------------
            img_gt, img_src = util.uint2tensor3_seg(patch_gt), util.uint2tensor3(patch_src)

        # ------------------------------------
        # if test
        # ------------------------------------
        elif self.opt['phase'] == 'test':
            src_path = self.paths_src[index]
            gt_path = self.paths_gt[index]
            file_name = os.path.basename(src_path)
            img_src = self.load_volumes(src_path, imgType='src')
            img_gt = self.load_volumes(gt_path, imgType='gt')
            H, W, _ = img_src.shape
            # --------------------------------
            # use raw image, default setting, let model do the padding and recovering
            # --------------------------------
            patch_gt = img_gt
            patch_src = img_src

            # --------------------------------
            # zoom for swin_Transformer
            # --------------------------------
            if self.resize_method == "zoom":
                zoom_scale_H = self.patch_size_H / H
                zoom_scale_W = self.patch_size_W / W
                patch_gt = zoom(img_gt, zoom=[zoom_scale_H, zoom_scale_W, 1], order=0)
                patch_src = zoom(img_src, zoom=[zoom_scale_H, zoom_scale_W, 1], order=0)

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

    def load_pkl(self, path):
        with open(path, 'rb') as f:
            pkl = pickle.load(f)
        return pkl


# local test
if __name__ == "__main__":
    opt = {}

    # opt['dataroot_src'] = '/media/NAS02/SynapseMultiorganSegmentation/Data/RawData/Trainingset/src'
    # opt['dataroot_gt'] = '/media/NAS02/SynapseMultiorganSegmentation/Data/RawData/Trainingset/gt'

    opt['dataroot_src'] = "/media/NAS02/ACDC/EStestingset_3D/src"
    opt['dataroot_gt'] = "/media/NAS02/ACDC/EStestingset_3D/gt"
    opt['dataroot_pkl'] = "/media/NAS02/ACDC/EStrainingset/properties"
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
    opt['resize_method'] = None

    test_dataset = Dataset_ACDC_2D(opt)

    # idx = random.randint(0, 100)
    idx = 5
    sample = test_dataset.__getitem__(idx)
    print(sample['name'])
    print(torch.max(sample['src_image']))
    print(sample['src_image'].shape)
    import matplotlib.pyplot as plt
    plt.imshow(sample['gt_seg'][0])
    plt.show()

    data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                              shuffle=True)
    for i, train_data in enumerate(data_loader):
        print(i)
