from matplotlib import pylab as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import rotate
from nibabel import nifti1
from nibabel.viewers import OrthoSlicer3D
example_filename = r"E:\ACDC\training\patient001\patient001_frame01.nii.gz"

img = nib.load(example_filename)
print(img)
print(img.header['db_name'])   # 输出头信息

# 由文件本身维度确定，可能是3维，也可能是4维
width, height, queue = img.dataobj.shape
# OrthoSlicer3D(img.dataobj).show()


def rotate_img(img, angle, bg_patch=(5,5)):
    assert len(img.shape) <= 3, "Incorrect image shape"
    print(img.shape)
    rgb = len(img.shape) == 3
    if rgb:
        bg_color = np.mean(img[:bg_patch[0], :bg_patch[1], :], axis=(0,1))
    else:
        bg_color = np.mean(img[:bg_patch[0], :bg_patch[1]])
    img = rotate(img, angle, reshape=False)
    mask = [img <= 0, np.any(img <= 0, axis=-1)][rgb]
    img[mask] = bg_color
    return img


num = 1
for i in range(0, queue, 2):

    img_arr = img.dataobj[:, :, i]

    img_arr = rotate_img(img, i*30)
    print(set(img_arr.reshape(-1)))
    plt.subplot(5, 4, num)
    plt.imshow(img_arr, cmap='gray')
    num += 1

plt.show()