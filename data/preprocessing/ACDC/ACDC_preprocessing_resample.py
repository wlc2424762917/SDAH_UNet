import os
import numpy as np
from glob import glob
from skimage.transform import resize
import SimpleITK as sitk
from matplotlib import pylab as plt
import nibabel as nib
from scipy.ndimage.interpolation import zoom


"""def read_img(file_path, file_name, save_path, label_or_img, dtype=sitk.sitkFloat32):  # for .mhd .nii .nrrd
    '''
    N*h*W
    :param full_path_filename:
    :return:*H*W
    '''
    print(label_or_img)
    if not os.path.exists(file_path):
        raise FileNotFoundError
    image = sitk.ReadImage(file_path)
    if label_or_img == 0:
        image = sitk.Cast(sitk.RescaleIntensity(image), dtype)  # to [0, 255]
    data = sitk.GetArrayFromImage(image)  # N*H*W
    if label_or_img == 0:
        data_norm = (data - data.min()) / (data.max() - data.min())  # case norm
    else:
        data_norm = data
    print(file_name, " data_shape:", data_norm.shape)
    save_path_npy = mkdir(os.path.join(save_path, file_name))
    # save_path_png = mkdir(os.path.join(save_path, 'png'))
    for slice_idx in range(70, data_norm.shape[0]):
        slice = data_norm[slice_idx, :, :]
        # print(slice.shape)
        # if (9 < slice_idx) and (slice_idx < 30):
        np.save(os.path.join(save_path_npy, '{}_{:03d}.npy'.format(file_name, slice_idx)), slice)
        # cv2.imwrite(os.path.join(save_path_png, '{}_{:03d}.png'.format(data_name, slice_idx)), slice * 255)"""


def read_img(src_file_path, src_file_name, src_save_path, gt_file_path, gt_file_name, gt_save_path, dtype=sitk.sitkFloat32):  # for .mhd .nii .nrrd
    '''
    N*h*W
    :param full_path_filename:
    :return:*H*W
    '''
    if not os.path.exists(src_file_path):
        raise FileNotFoundError
    image_data = nib.load(src_file_path).get_fdata()
    print(image_data.max())
    # image = sitk.Cast(sitk.RescaleIntensity(image), dtype)  # to [0, 255]
    # image_data = sitk.GetArrayFromImage(image)  # N*H*W
    image_data_norm = (image_data - image_data.min()) / (image_data.max() - image_data.min())  # case norm
    # print(src_file_name, " data_shape:", image_data_norm.shape)
    if not os.path.exists(gt_file_path):
        raise FileNotFoundError
    mask_data = nib.load(gt_file_path).get_fdata()
    mask_data_norm = mask_data  # no case norm
    # print(src_file_name, " data_shape:", image_data_norm.shape)
    save_src_path_npy = mkdir(os.path.join(src_save_path, src_file_name))
    save_gt_path_npy = mkdir(os.path.join(gt_save_path, gt_file_name))
    # save_path_png = mkdir(os.path.join(save_path, 'png'))
    num_slices = 0
    for slice_idx in range(0, mask_data_norm.shape[2]):
        image_slice = image_data_norm[:, :, slice_idx]
        mask_slice = mask_data_norm[:, :, slice_idx]
        mask_slice_cliped = np.clip(mask_slice, 0, 1)
        print(mask_slice_cliped.shape)
        # print(mask_slice_cliped.shape)
        # print(np.sum(mask_slice_cliped), mask_slice_cliped.shape[0] * mask_slice_cliped.shape[1])
        if np.sum(mask_slice_cliped) > 200:
            num_slices += 1
            zoom_scale_H = 224 / image_slice.shape[0]
            zoom_scale_W = 256 / image_slice.shape[1]
            mask_slice = zoom(mask_slice, zoom=[zoom_scale_H, zoom_scale_W], order=0)
            image_slice = zoom(image_slice, zoom=[zoom_scale_H, zoom_scale_W], order=0)
            np.save(os.path.join(save_src_path_npy, '{}_{:03d}.npy'.format(src_file_name, slice_idx)), image_slice)
            np.save(os.path.join(save_gt_path_npy, '{}_{:03d}.npy'.format(gt_file_name, slice_idx)), mask_slice)
            # cv2.imwrite(os.path.join(save_path_png, '{}_{:03d}.png'.format(data_name, slice_idx)), slice * 255)
    print(num_slices)


def read_dataset(src_file_paths, src_save_path, gt_file_paths, gt_save_path):
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
        read_img(img_path, img_name, src_save_path, mask_path, mask_name, gt_save_path)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


if __name__ == '__main__':

    #path_raw = '/media/NAS02/SynapseMultiorganSegmentation/Data/RawData'
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
    src_path_save = mkdir(os.path.join(path_raw, f'{dataset_type}set_M', 'src'))
    gt_path_save = mkdir(os.path.join(path_raw, f'{dataset_type}set_M', 'gt'))
    mask_paths = []
    image_paths = []
    oriImg_paths = []
    for p_id in range(100):
        p_id_str = str(p_id+1)
        path_raw_dataset_type_patient = os.path.join(path_raw_dataset_type, 'patient'+p_id_str.zfill(3))
        oriImg_paths.extend(glob(os.path.join(path_raw_dataset_type_patient, '*4d.nii.gz')))
        mask_paths.extend(glob(os.path.join(path_raw_dataset_type_patient, '*gt.nii.gz')))
        image_paths.extend(list(set(glob(os.path.join(path_raw_dataset_type_patient, '*.nii.gz'))) - set(mask_paths) - set(oriImg_paths)))
    print(image_paths)
    read_dataset(image_paths, src_path_save, mask_paths, gt_path_save)

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