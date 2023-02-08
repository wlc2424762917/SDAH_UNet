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


def read_img(src_file_path, src_file_name, src_save_path, gt_file_path, gt_file_name, gt_save_path, pkl_save_path, classes=[1,2,3]):  # for .mhd .nii .nrrd
    '''
    N*h*W
    :param full_path_filename:
    :return:*H*W
    '''
    if not os.path.exists(src_file_path):
        raise FileNotFoundError
    image_data = nib.load(src_file_path).get_fdata()
    # image = sitk.Cast(sitk.RescaleIntensity(image), dtype)  # to [0, 255]
    # image_data = sitk.GetArrayFromImage(image)  # N*H*W
    image_data_norm = (image_data - image_data.min()) / (image_data.max() - image_data.min())  # case norm
    # print(src_file_name, " data_shape:", image_data_norm.shape)
    if not os.path.exists(gt_file_path):
        raise FileNotFoundError
    mask_data = nib.load(gt_file_path).get_fdata()
    mask_data_norm = mask_data  # no case norm for gt_seg
    save_src_path_npy = mkdir(os.path.join(src_save_path, src_file_name))
    save_gt_path_npy = mkdir(os.path.join(gt_save_path, gt_file_name))
    save_properties_path_pkl = mkdir(os.path.join(pkl_save_path, gt_file_name))
    num_slices = 0
    for slice_idx in range(0, mask_data_norm.shape[2]):
        image_slice = image_data_norm[:, :, slice_idx]
        mask_slice = mask_data_norm[:, :, slice_idx]
        class_connection = check_if_all_in_one_region(mask_slice, classes)
        print(mask_slice.shape)
        properties = get_image_properties(image_slice, mask_slice, classes, class_connection)
        num_slices += 1
        np.save(os.path.join(save_src_path_npy, '{}_{:03d}.npy'.format(src_file_name, slice_idx)), image_slice)
        np.save(os.path.join(save_gt_path_npy, '{}_{:03d}.npy'.format(gt_file_name, slice_idx)), mask_slice)

        with open(os.path.join(save_properties_path_pkl, "{}_{:03d}.pkl".format(src_file_name, slice_idx)), 'wb') as f:
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
        read_img(img_path, img_name, src_save_path, mask_path, mask_name, gt_save_path, pkl_save_path)


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
        if numlabels != 1:
            res[r] = False
        else:
            res[r] = True
    return res


if __name__ == '__main__':

    # path_raw = '/media/NAS02/SynapseMultiorganSegmentation/Data/RawData'
    path_raw = r'E:\ACDC'
    # path_raw = 'C:/Users/wlc/Documents/GitHub/Rawdata/'
    dataset_type = 'training'  # 'train', 'val', 'test'
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
    ED_src_path_save = mkdir(os.path.join(path_raw, f'ED{dataset_type}set', 'src'))
    ED_gt_path_save = mkdir(os.path.join(path_raw, f'ED{dataset_type}set', 'gt'))
    ED_pkl_path_save = mkdir(os.path.join(path_raw, f'ED{dataset_type}set', 'properties'))
    ES_src_path_save = mkdir(os.path.join(path_raw, f'ES{dataset_type}set', 'src'))
    ES_gt_path_save = mkdir(os.path.join(path_raw, f'ES{dataset_type}set', 'gt'))
    ES_pkl_path_save = mkdir(os.path.join(path_raw, f'ES{dataset_type}set', 'properties'))
    mask_paths = []
    ED_mask_paths = []
    ES_mask_paths = []
    image_paths = []
    ED_image_paths = []
    ES_image_paths = []
    oriImg_paths = []
    for p_id in range(100):
        p_id_str = str(p_id+1)
        path_raw_dataset_type_patient = os.path.join(path_raw_dataset_type, 'patient'+p_id_str.zfill(3))
        oriImg_paths.extend(glob(os.path.join(path_raw_dataset_type_patient, '*4d.nii.gz')))
        mask_paths.extend(glob(os.path.join(path_raw_dataset_type_patient, '*gt.nii.gz')))
        ED_mask_paths.extend(glob(os.path.join(path_raw_dataset_type_patient, '*frame01_gt.nii.gz')))
        image_paths.extend(list(set(glob(os.path.join(path_raw_dataset_type_patient, '*.nii.gz'))) - set(mask_paths) - set(oriImg_paths)))
        ED_image_paths.extend(glob(os.path.join(path_raw_dataset_type_patient, '*frame01.nii.gz')))
    ES_mask_paths.extend(list(set(mask_paths) - set(ED_mask_paths)))
    ES_mask_paths.sort()
    ES_image_paths.extend(list(set(image_paths) - set(ED_image_paths)))
    ES_image_paths.sort()
    read_dataset(ES_image_paths, ES_src_path_save, ES_mask_paths, ES_gt_path_save, ES_pkl_path_save)
    read_dataset(ED_image_paths, ED_src_path_save, ED_mask_paths, ED_gt_path_save, ED_pkl_path_save)

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