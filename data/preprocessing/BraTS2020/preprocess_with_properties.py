import os
import numpy as np
from glob import glob
from skimage.transform import resize
import SimpleITK as sitk
from matplotlib import pylab as plt
import nibabel as nib
from collections import OrderedDict
from skimage.morphology import label
import pickle


def grayval2label(gray_mask):
    gray_mask = np.squeeze(gray_mask)
    gray_val_list = [0, 1, 2, 4]
    map_gray2label = {}
    for i, key in enumerate(gray_val_list):
        map_gray2label[key] = i
    label_mask = np.zeros_like(gray_mask)
    for row in range(gray_mask.shape[0]):
        for col in range(gray_mask.shape[1]):
            label_mask[row][col] = map_gray2label[gray_mask[row][col]]
    return label_mask.reshape((1, gray_mask.shape[0], gray_mask.shape[1]))


def get_image_properties(src_img, gt_seg, all_classes, class_connection):
    # we need to find out where the classes are and sample some random locations
    # let's do 10.000 samples per class
    # seed this for reproducibility!
    properties = OrderedDict()
    num_samples = 10000
    min_percent_coverage = 0.01  # at least 1% of the class voxels need to be selected, otherwise it may be too sparse
    rndst = np.random.RandomState(1234)
    class_locs = {}
    for c in all_classes:
        all_locs = np.argwhere(gt_seg == c)
        if len(all_locs) == 0:
            class_locs[c] = []
            continue
        target_num_samples = min(num_samples, len(all_locs))
        target_num_samples = max(target_num_samples, int(np.ceil(len(all_locs) * min_percent_coverage)))
        selected = all_locs[rndst.choice(len(all_locs), target_num_samples, replace=False)]
        class_locs[c] = selected
    properties['class_locations'] = class_locs

    properties['class_connection'] = class_connection
    return properties


def read_img(flair_path, t1_path, t2_path, t1ce_path, src_file_name, src_save_path, gt_file_path, gt_file_name, gt_save_path, pkl_save_path, classes=[1, 2, 3, 4]):  # for .mhd .nii .nrrd
    '''
    N*h*W
    :param full_path_filename:
    :return:*H*W
    '''
    if not os.path.exists(flair_path):
        raise FileNotFoundError
    flair_data = nib.load(flair_path).get_fdata()
    t1_data = nib.load(t1_path).get_fdata()
    t2_data = nib.load(t2_path).get_fdata()
    t1ce_data = nib.load(t1ce_path).get_fdata()

    flair_data_norm = (flair_data - flair_data.mean()) / (flair_data.std())  # case norm
    t1_data_norm = (t1_data - t1_data.mean()) / (t1_data.std()) 
    t2_data_norm = (t2_data - t2_data.mean()) / (t2_data.std()) 
    t1ce_data_norm = (t1ce_data - t1ce_data.mean()) / (t1ce_data.std()) 

    flair_data_norm = ((flair_data - flair_data.min()) / (flair_data.max() - flair_data.min())) * 255 # case norm
    t1_data_norm = ((t1_data - t1_data.min()) / (t1_data.max() - t1_data.min())) * 255
    t2_data_norm = ((t2_data - t2_data.min()) / (t2_data.max() - t2_data.min())) * 255
    t1ce_data_norm = ((t1ce_data - t1ce_data.min()) / (t1ce_data.max() - t1ce_data.min())) * 255
    print(src_file_name, " data_shape:", flair_data_norm.shape)
    print(flair_data_norm.max())
    if not os.path.exists(gt_file_path):
        raise FileNotFoundError
    mask_data = nib.load(gt_file_path).get_fdata()
    mask_data_norm = mask_data  # no case norm for gt_seg
    save_src_path_npy = mkdir(os.path.join(src_save_path, src_file_name))
    save_gt_path_npy = mkdir(os.path.join(gt_save_path, gt_file_name))
    save_properties_path_pkl = mkdir(os.path.join(pkl_save_path, gt_file_name))
    num_slices = 0
    num_slices = 0
    for slice_idx in range(0, mask_data_norm.shape[2]):
        flair_slice = flair_data_norm[:, :, slice_idx].reshape((1, 240, 240))
        t1_slice = t1_data_norm[:, :, slice_idx].reshape((1, 240, 240))
        t2_slice = t2_data_norm[:, :, slice_idx].reshape((1, 240, 240))
        t1ce_slice = t1ce_data_norm[:, :, slice_idx].reshape((1, 240, 240))
        image_slice = np.concatenate((flair_slice, t1_slice, t2_slice, t1ce_slice), axis=0)
        mask_slice = mask_data_norm[:, :, slice_idx]
        mask_slice_cliped = np.clip(mask_slice, 0, 1)
        #print(np.sum(mask_slice_cliped))
        if np.sum(mask_slice_cliped) > 0:
            mask_slice = grayval2label(mask_slice)
            class_connection = check_if_all_in_one_region(mask_slice, classes)
            #print(mask_slice.shape)
            properties = get_image_properties(image_slice, mask_slice, classes, class_connection)
            num_slices += 1
            np.save(os.path.join(save_src_path_npy, '{}_{:03d}.npy'.format(src_file_name, slice_idx)), image_slice)
            np.save(os.path.join(save_gt_path_npy, '{}_{:03d}.npy'.format(gt_file_name, slice_idx)), mask_slice)

            with open(os.path.join(save_properties_path_pkl, "{}_{:03d}.pkl".format(src_file_name, slice_idx)), 'wb') as f:
                pickle.dump(properties, f)
            num_slices += 1
    print(num_slices)


def read_dataset(flair_file_paths, t1_file_paths, t2_file_paths, t1ce_file_paths, gt_file_paths,  src_save_path, gt_save_path, pkl_save_path):
    for idx_data in range(len(flair_file_paths)):
        print('{} / {}'.format(idx_data + 1, len(flair_file_paths)))
        flair_path = flair_file_paths[idx_data]
        t1_path = t1_file_paths[idx_data]
        t2_path = t2_file_paths[idx_data]
        t1ce_path = t1ce_file_paths[idx_data]
        mask_path = gt_file_paths[idx_data]

        flair_nameext, _ = os.path.splitext(flair_path)
        flair_nameext, _ = os.path.splitext(flair_nameext)

        """t1_nameext, _ = os.path.splitext(t1_path)
        t1_nameext, _ = os.path.splitext(t1_nameext)

        t2_nameext, _ = os.path.splitext(t2_path)
        t2_nameext, _ = os.path.splitext(t2_nameext)

        t1ce_nameext, _ = os.path.splitext(t1ce_path)
        t1ce_nameext, _ = os.path.splitext(t1ce_nameext)"""

        mask_nameext, _ = os.path.splitext(mask_path)
        mask_nameext, _ = os.path.splitext(mask_nameext)

        _, flair_name = os.path.split(flair_nameext)
        src_name = flair_name[:-6]
        _, mask_name = os.path.split(mask_nameext)
        print(mask_name)
        print(src_name)
        read_img(flair_path, t1_path, t2_path, t1ce_path, src_name, src_save_path, mask_path, mask_name, gt_save_path, pkl_save_path)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def check_if_all_in_one_region(seg, all_classes):
    regions = list()
    for c in all_classes:
        regions.append(c)

    res = OrderedDict()
    for r in regions:
        new_seg = np.zeros(seg.shape)
        new_seg[seg == c] = 1
        labelmap, numlabels = label(new_seg, return_num=True)
        if numlabels != 1 or numlabels != 0:
            res[r] = False
        else:
            res[r] = True
    return res



if __name__ == '__main__':

    # path_raw = '/media/NAS02/SynapseMultiorganSegmentation/Data/RawData'
    path_raw = '/media/NAS02/BraTS2020'
    # path_raw = 'C:/Users/wlc/Documents/GitHub/Rawdata/'
    dataset_type = 'Training'  # 'train', 'val', 'test'
    path_raw_dataset_type = os.path.join(path_raw, dataset_type)

    src_path_save = mkdir(os.path.join(path_raw, f'{dataset_type}set', 'src'))
    gt_path_save = mkdir(os.path.join(path_raw, f'{dataset_type}set', 'gt'))
    pkl_path_save = mkdir(os.path.join(path_raw, f'{dataset_type}set', 'properties'))

    mask_paths = []
    t2_paths = []
    t1_paths = []
    t1ce_paths = []
    flair_paths = []

    start = 0
    for p_id in range(start, 370):
        p_id_str = str(p_id)
        path_raw_dataset_type_patient = os.path.join(path_raw_dataset_type, 'BraTS20_Training_'+p_id_str.zfill(3))
        flair_paths.extend(glob(os.path.join(path_raw_dataset_type_patient, '*flair.nii')))
        t1_paths.extend(glob(os.path.join(path_raw_dataset_type_patient, '*t1.nii')))
        t1ce_paths.extend(glob(os.path.join(path_raw_dataset_type_patient, '*t1ce.nii')))
        t2_paths.extend(glob(os.path.join(path_raw_dataset_type_patient, '*t2.nii')))
        mask_paths.extend(glob(os.path.join(path_raw_dataset_type_patient, '*seg.nii')))

    print(flair_paths)
    print(len(flair_paths))
    # print(len(t1_paths))
    # print(len(t1ce_paths))
    # print(len(t2_paths))
    # print(len(mask_paths))

    read_dataset(flair_paths, t1_paths, t2_paths, t1ce_paths, mask_paths, src_path_save, gt_path_save, pkl_path_save)
