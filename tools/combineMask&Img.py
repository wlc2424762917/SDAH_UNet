import os
from PIL import Image
import numpy as np


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def gray2rgb(gray, color_dict):
    """
    convert gray image into RGB image
    :param gray: single channel image with numpy type
    :param color_dict: color map
    :return:  rgb image
    """
    # 1：创建新图像容器
    rgb_image = np.zeros(shape=(*gray.shape, 3))
    # 2： 遍历每个像素点
    for i in range(rgb_image.shape[0]):
        for j in range(rgb_image.shape[1]):
            if gray[i, j] >= 1:
                # print(gray[i,j])
                rgb_image[i, j, :] = color_dict[list(color_dict.keys())[gray[i,j] - 1]]
    return rgb_image.astype(np.uint8)


def combine_mask_and_img(root_path_img, root_path_mask, output_path):
    mkdir(output_path)
    color_dict = {"yellow": [255, 255, 0],
                  "red": [238, 0, 0],
                  "pink": [255, 192, 203],
                  "blue": [0, 153, 255],
                  "blueViolet": [138, 43, 226],
                  "green": [0, 128, 0],
                  "orange": [255, 102, 0],
                  "Aqua": [0, 255, 255],
                  "ligtGreen": [144, 238, 144],
                  "midnightBlue": [25, 25, 112],
                  "wheat": [245, 222, 179],
                  "Ivory": [255, 255, 240],
                  "black": [0, 0, 0],
                  }
    img_list = os.listdir(root_path_img)
    label_list = os.listdir(root_path_mask)
    img_list = sorted(img_list)
    label_list = sorted(label_list)
    for num, img_label in enumerate(zip(img_list, label_list)):
        img = Image.open(os.path.join(root_path_img, img_label[0]))
        img = img.convert("RGB")
        label = np.load(os.path.join(root_path_mask, img_label[1]))
        label = gray2rgb(np.array(label).astype(int), color_dict)
        label = Image.fromarray(label)
        # label.show()
        fin = Image.blend(img, label, 0.5)
        # fin.show()
        fin.save(os.path.join(output_path, img_label[0]))

"""WHS_root_path_background = r"C:/Users/wlc/Documents/GitHub/MRI_Recon/mmWHS_MR_seg/test_new/GTImage"
WHS_root_path_gt = r"C:/Users/wlc/Documents/GitHub/MRI_Recon/mmWHS_MR_seg/test_new/GTMaskMultiOrganNp"
WHS_output_path_gt_img = r"C:/Users/wlc/Documents/GitHub/MRI_Recon/mmWHS_MR_seg/test_new/GTMask+GTImage_MultiOrgan/"

WHS_root_path_pred = r"C:/Users/wlc/Documents/GitHub/MRI_Recon/mmWHS_MR_seg/test_new/PredMaskMultiOrganNP/"
WHS_output_path_pred_img = r"C:/Users/wlc/Documents/GitHub/MRI_Recon/mmWHS_MR_seg/test_new/PredMask+GTImage_MultiOrgan/"

combine_mask_and_img(WHS_root_path_background, WHS_root_path_gt, WHS_output_path_gt_img)
combine_mask_and_img(WHS_root_path_background, WHS_root_path_pred, WHS_output_path_pred_img)"""

"""WHS_root_path_background = r"D:\MRI_Recon_ACDC\GTImage"
WHS_root_path_gt = r"D:\MRI_Recon_ACDC\GTMaskMultiOrganNp"
WHS_output_path_gt_img = r"D:\MRI_Recon_ACDC\GTMask+GTImage"

WHS_root_path_pred = r"D:\MRI_Recon_ACDC\PredMaskMultiOrganNp"
WHS_output_path_pred_img = r"D:\MRI_Recon_ACDC\PredMask+GTImage"

combine_mask_and_img(WHS_root_path_background, WHS_root_path_gt, WHS_output_path_gt_img)
combine_mask_and_img(WHS_root_path_background, WHS_root_path_pred, WHS_output_path_pred_img)"""

root_path_background = r"D:\Segmentation_framework_results_random_central_tiled\src_image"
root_path_gt = r"D:\Segmentation_framework_results_random_central_tiled\gt_seg_np"
output_path_gt_img = r"D:\Segmentation_framework_results_random_central_tiled\GTMask+GTImage"

root_path_pred = r"D:\Segmentation_framework_results_random_central_tiled\pred_seg_np"
output_path_pred_img = r"D:\Segmentation_framework_results_random_central_tiled\PredMask+GTImage"

combine_mask_and_img(root_path_background, root_path_gt, output_path_gt_img)
combine_mask_and_img(root_path_background, root_path_pred, output_path_pred_img)