import os
import numpy as np
from glob import glob
from skimage.transform import resize
import SimpleITK as sitk
from collections import OrderedDict
from skimage.morphology import label
import pickle


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


def read_img(src_file_path, src_file_name, src_save_path, gt_file_path, gt_file_name, gt_save_path,
             save_properties_path_pkl_raw, dtype=sitk.sitkFloat32):  # for .mhd .nii .nrrd
    '''
    N*h*W
    :param full_path_filename:
    :return:*H*W
    '''

    classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    if not os.path.exists(src_file_path):
        raise FileNotFoundError
    image = sitk.ReadImage(src_file_path)
    image_data = sitk.GetArrayFromImage(image)  # N*H*W
    image_data_upper = np.percentile(image_data, 99.5)
    image_data_lower = np.percentile(image_data, 0.5)
    print(image_data_upper, image_data_lower)
    image_data = np.clip(image_data, image_data_lower, image_data_upper)
    image_data_norm = (image_data - image_data.mean()) / (image_data.std())
    image_data_norm = (image_data - image_data.min()) / (image_data.max() - image_data.min())  # case norm

    if not os.path.exists(gt_file_path):
        raise FileNotFoundError
    mask = sitk.ReadImage(gt_file_path)
    mask_data = sitk.GetArrayFromImage(mask)  # N*H*W
    mask_data_norm = mask_data  # no case norm
    # print(src_file_name, " data_shape:", image_data_norm.shape)
    save_src_path_npy = mkdir(os.path.join(src_save_path, src_file_name))
    save_gt_path_npy = mkdir(os.path.join(gt_save_path, gt_file_name))
    save_src_path_no_foreground_npy = mkdir(os.path.join(src_save_path + "_no_foreground", src_file_name))
    save_gt_path_no_foreground_npy = mkdir(os.path.join(gt_save_path + "_no_foreground", gt_file_name))
    save_properties_path_pkl = mkdir(os.path.join(save_properties_path_pkl_raw, src_file_name))
    save_properties_no_foreground_path_pkl = mkdir(
        os.path.join(save_properties_path_pkl_raw + "_no_foreground", src_file_name))

    image_slice = image_data_norm[:, :, :]
    mask_slice = mask_data_norm[:, :, :]
    mask_slice_cliped = np.clip(mask_slice, 0, 1)

    class_connection = check_if_all_in_one_region(mask_slice, classes)
    properties = get_image_properties(image_slice, mask_slice, classes, class_connection)

    if np.sum(mask_slice_cliped) > 100:
        print(image_slice.shape)
        np.save(os.path.join(save_src_path_npy, '{}.npy'.format(src_file_name)), image_slice)
        np.save(os.path.join(save_gt_path_npy, '{}.npy'.format(gt_file_name)), mask_slice)
        with open(os.path.join(save_properties_path_pkl, "{}.pkl".format(src_file_name)),
                  'wb') as f:
            pickle.dump(properties, f)

    else:
        np.save(os.path.join(save_src_path_no_foreground_npy, '{}.npy'.format(src_file_name + "_no_foreground")),
                image_slice)
        np.save(os.path.join(save_gt_path_no_foreground_npy, '{}.npy'.format(gt_file_name + "_no_foreground")),
                mask_slice)
        with open(
                os.path.join(save_properties_no_foreground_path_pkl, "{}.pkl".format(src_file_name + "_no_foreground")),
                'wb') as f:
            pickle.dump(properties, f)


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
        print(pkl_save_path)
        read_img(img_path, img_name, src_save_path, mask_path, mask_name, gt_save_path, pkl_save_path)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


if __name__ == '__main__':
    # path_raw = '/media/NAS02/SynapseMultiorganSegmentation/Data/RawData'
    path_raw = "/media/NAS02/Synapse_multi_organ"
    # path_raw = 'C:/Users/wlc/Documents/GitHub/Rawdata/'
    dataset_type = 'Testing'  # 'train', 'val', 'test'
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

    pkl_path_save = mkdir(os.path.join(path_raw, f'{dataset_type}set_3D', 'pkl'))
    src_path_save = mkdir(os.path.join(path_raw, f'{dataset_type}set_3D', 'src'))
    gt_path_save = mkdir(os.path.join(path_raw, f'{dataset_type}set_3D', 'gt'))
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