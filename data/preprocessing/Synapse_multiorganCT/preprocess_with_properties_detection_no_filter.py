import os
import numpy as np
from glob import glob
from skimage.transform import resize
import SimpleITK as sitk
from collections import OrderedDict
from skimage.morphology import label
import skimage
import pickle
import matplotlib.pyplot as plt


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


def get_image_properties(src_img, gt_seg, all_classes, class_connection):
    # we need to find out where the classes are and sample some random locations
    # let's do 10.000 samples per class
    # seed this for reproducibility!
    properties = OrderedDict()
    num_samples = 10000
    min_percent_coverage = 0.01  # at least 1% of the class voxels need to be selected, otherwise it may be too sparse
    rndst = np.random.RandomState(1234)
    class_locs = {}
    max_h, min_h, max_w, min_w = 0, 512, 0, 512
    for c in all_classes:
        all_locs = np.argwhere(gt_seg == c)
        if len(all_locs) == 0:
            class_locs[c] = []
            continue
        if min(all_locs[:, 0]) < min_h:
            min_h = min(all_locs[:, 0])
        if min(all_locs[:, 1]) < min_w:
            min_w = min(all_locs[:, 1])
        if max(all_locs[:, 0]) > max_h:
            max_h = max(all_locs[:, 0])
        if max(all_locs[:, 1]) > max_w:
            max_w = max(all_locs[:, 1])
        print(c, '： ', len(all_locs))
        target_num_samples = min(num_samples, len(all_locs))
        target_num_samples = max(target_num_samples, int(np.ceil(len(all_locs) * min_percent_coverage)))
        selected = all_locs[rndst.choice(len(all_locs), target_num_samples, replace=False)]
        class_locs[c] = selected
    properties['class_locations'] = class_locs

    properties['class_connection'] = class_connection
    return properties, max_h, min_h, max_w, min_w

    properties['class_locations'] = class_locs
    properties['class_connection'] = class_connection

    return properties, max_hs, min_hs, max_ws, min_ws


def read_img(src_file_path, src_file_name, src_save_path, gt_file_path, gt_file_name, gt_save_path, save_properties_path_pkl_raw, dtype=sitk.sitkFloat32):  # for .mhd .nii .nrrd
    '''
    N*h*W
    :param full_path_filename:
    :return:*H*W
    '''
    # normalization
    classes_for_seg = [1, 2, 3, 4, 5, 6, 7, 8]
    if not os.path.exists(src_file_path):
        raise FileNotFoundError
    image = sitk.ReadImage(src_file_path)
    image_data = sitk.GetArrayFromImage(image)  # N*H*W
    image_data_upper = np.percentile(image_data, 99.5)
    image_data_lower = np.percentile(image_data, 0.5)
    print(image_data_upper, image_data_lower)
    image_data = np.clip(image_data, image_data_lower, image_data_upper)
    image_data_norm = (image_data - image_data.mean()) / (image_data.std())  # case norm
    image_data_norm = (image_data_norm - image_data_norm.min()) / (image_data_norm.max() - image_data_norm.min())
    if not os.path.exists(gt_file_path):
        raise FileNotFoundError

    # no norm for masks
    mask = sitk.ReadImage(gt_file_path)
    mask_data = sitk.GetArrayFromImage(mask)  # N*H*W
    mask_data_norm = mask_data  # no case norm

    # generate saving dir
    save_src_path_npy = mkdir(os.path.join(src_save_path, src_file_name))
    save_gt_path_npy = mkdir(os.path.join(gt_save_path, gt_file_name))
    save_gt_detection_path_npy = mkdir(os.path.join(gt_save_path+"_detection", gt_file_name))
    save_properties_path_pkl = mkdir(os.path.join(save_properties_path_pkl_raw, src_file_name))

    save_src_path_no_foreground_npy = mkdir(os.path.join(src_save_path+"_no_foreground", src_file_name))
    save_gt_path_no_foreground_npy = mkdir(os.path.join(gt_save_path+"_no_foreground", gt_file_name))
    save_gt_detection_path_no_foreground_npy = mkdir(os.path.join(gt_save_path+"_detection_no_foreground", gt_file_name))
    save_properties_no_foreground_path_pkl = mkdir(os.path.join(save_properties_path_pkl_raw+"_no_foreground", src_file_name))

    # generating map
    map_c = {0: 0, 1: 7, 2: 4, 3: 3, 4: 2, 5: 0, 6: 5, 7: 8, 8: 1, 9: 0, 10: 0, 11: 6, 12: 0, 13: 0}
    for slice_idx in range(0, mask_data_norm.shape[0]):
        image_slice = image_data_norm[slice_idx, :, :]
        mask_slice = mask_data_norm[slice_idx, :, :]
        for i in range(mask_slice.shape[0]):
            for j in range(mask_slice.shape[1]):
                mask_slice[i][j] = map_c[mask_slice[i][j]]
        mask_slice_cliped = np.clip(mask_slice, 0, 1)
        class_connection = check_if_all_in_one_region(mask_slice, classes_for_seg)
        properties, max_h, min_h, max_w, min_w = get_image_properties(image_slice, mask_slice, classes_for_seg, class_connection)
        # print(min_h, max_h, min_w, max_w)

        detection_mask_slice = np.zeros_like(mask_slice)
        # classes = [6, 8, 7, 11, 3, 2, 1, 10, 5, 9, 4, 12, 13]  # 按照从难到易, 从小到大
        # classes = [6, 8, 7, 11, 3, 2, 1, 4]
        classes = [5, 1, 8, 6, 3, 4, 7, 2]
        for c in classes:
            mask_slice_c = mask_slice == c
            label_mask_slice, num = label(mask_slice_c, return_num=True)
            for i in range(1, num+1):
                max_h, min_h, max_w, min_w = 0, 512, 0, 512
                all_locs = np.argwhere(label_mask_slice == i)
                if len(all_locs) == 0:
                    continue
                if min(all_locs[:, 0]) < min_h:
                    min_h = min(all_locs[:, 0])
                if min(all_locs[:, 1]) < min_w:
                    min_w = min(all_locs[:, 1])
                if max(all_locs[:, 0]) > max_h:
                    max_h = max(all_locs[:, 0])
                if max(all_locs[:, 1]) > max_w:
                    max_w = max(all_locs[:, 1])

                detection_mask_slice[min_h:max_h, min_w:max_w] = c

        if np.sum(mask_slice_cliped) > 0:
            np.save(os.path.join(save_src_path_npy, '{}_{:03d}.npy'.format(src_file_name, slice_idx)), image_slice)
            np.save(os.path.join(save_gt_path_npy, '{}_{:03d}.npy'.format(gt_file_name, slice_idx)), mask_slice)
            np.save(os.path.join(save_gt_detection_path_npy, '{}_{:03d}.npy'.format(gt_file_name, slice_idx)), detection_mask_slice)
            with open(os.path.join(save_properties_path_pkl, "{}_{:03d}.pkl".format(src_file_name, slice_idx)),
                      'wb') as f:
                pickle.dump(properties, f)
            print(image_slice.shape)


def read_dataset(src_file_paths, src_save_path, gt_file_paths, gt_save_path, pkl_save_path):
    for idx_data in range(len(src_file_paths)):
        print('{} / {}'.format(idx_data + 1, len(src_file_paths)))
        img_path = src_file_paths[idx_data]
        mask_path = gt_file_paths[idx_data]
        img_nameext, _ = os.path.splitext(img_path)
        img_nameext, _ = os.path.splitext(img_nameext)
        mask_nameext, _ = os.path.splitext(mask_path)
        mask_nameext, _ = os.path.splitext(mask_nameext)
        _, img_name = os.path.split(img_nameext)
        _, mask_name = os.path.split(mask_nameext)
        print(mask_name)
        print(img_name)
        read_img(img_path, img_name, src_save_path, mask_path, mask_name, gt_save_path, pkl_save_path)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


if __name__ == '__main__':

    path_raw = '/media/NAS02/Synapse_multi_organ'
    #path_raw = r"E:\Synapse_multi_organ"
    # path_raw = 'C:/Users/wlc/Documents/GitHub/Rawdata/'
    dataset_type = 'Training'  # 'train', 'val', 'test'
    path_raw_dataset_type = os.path.join(path_raw, dataset_type)
    """print('########## PROCESS {} DATASET ##########')

    print(f'########## PROCESS DATASET: {dataset_type}; raw_data_img ##########')
    paths = glob(os.path.join(path_raw_dataset_type, 'img', '*.nii.gz'))
    path_save = mkdir(os.path.join(path_raw, f'{dataset_type}set', 'src'))
    read_dataset(paths, path_save, label_or_img=0)

    print(f'########## PROCESS DATASET: {dataset_type}; raw_data_label ##########')
    paths = glob(os.path.join(path_raw_dataset_type, 'label', '*.nii.gz'))
    path_save = mkdir(os.path.join(path_raw, f'{dataset_type}set', 'gt'))
    read_dataset(paths, path_save, label_or_img=1)"""

    image_paths = glob(os.path.join(path_raw_dataset_type, 'img', '*.nii.gz'))
    mask_paths = glob(os.path.join(path_raw_dataset_type, 'label', '*.nii.gz'))

    pkl_path_save = mkdir(os.path.join(path_raw, f'{dataset_type}set_2stage_all', 'pkl'))
    src_path_save = mkdir(os.path.join(path_raw, f'{dataset_type}set_2stage_all', 'src'))
    gt_path_save = mkdir(os.path.join(path_raw, f'{dataset_type}set_2stage_all', 'gt'))
    read_dataset(image_paths, src_path_save, mask_paths, gt_path_save, pkl_path_save)

    # print(f'########## PROCESS DATASET: {dataset_type}; ACUQ: AXFLAIR ##########')
    # paths_axflair = glob(os.path.join(path_raw_dataset_type, 'file_brain_AXFLAIR_*.h5'))
    # path_save = mkdir(os.path.join(path_fastMRI, 'FastMRIBrainFLAIRnPI_256', f'{dataset_type}set'))
    # read_dataset(paths_axflair, path_save)
    #
    # print(f'########## PROCESS DATASET: {dataset_type}; ACUQ: AXT1 ##########')
    # paths_t1 = glob(os.path.join(path_raw_dataset_type, 'file_brain_AXT1_*.h5'))
    # path_save = mkdir(os.path.join(path_fastMRI, 'FastMRIBrainT1nPI_256', f'{dataset_type}set'))
    # read_dataset(paths_t1, path_save)
    #
    # print(f'########## PROCESS DATASET: {dataset_type}; ACUQ: AXT1POST ##########')
    # paths_t1post = glob(os.path.join(path_raw_dataset_type, 'file_brain_AXT1POST_*.h5'))
    # path_save = mkdir(os.path.join(path_fastMRI, 'FastMRIBrainT1POSTnPI_256', f'{dataset_type}set'))
    # read_dataset(paths_t1post, path_save)
    #
    # print(f'########## PROCESS DATASET: {dataset_type}; ACUQ: AXT1PRE ##########')
    # paths_t1pre = glob(os.path.join(path_raw_dataset_type, 'file_brain_AXT1PRE_*.h5'))
    # path_save = mkdir(os.path.join(path_fastMRI, 'FastMRIBrainT1PREnPI_256', f'{dataset_type}set'))
    # read_dataset(paths_t1pre, path_save)

    # print(f'########## PROCESS DATASET: {dataset_type}; ACUQ: AXT2 ##########')
    # paths_t2 = glob(os.path.join(path_raw_dataset_type, 'file_brain_AXT2_*.h5'))
    # path_save = mkdir(os.path.join(path_fastMRI, 'FastMRIBrainT2nPI_256', f'{dataset_type}set'))
    # read_dataset(paths_t2, path_save)