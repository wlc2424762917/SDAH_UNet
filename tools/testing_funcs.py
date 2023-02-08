from matplotlib import pylab as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import rotate
from nibabel import nifti1
from nibabel.viewers import OrthoSlicer3D
from collections import OrderedDict
from skimage.morphology import label

example_filename = r"E:\ACDC\EStrainingset_L\src\patient011_frame08\patient011_frame08_005.npy"

img = np.load(example_filename)
print(img.shape)


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


#img = rotate_img(img, 30)
#plt.imshow(img)

pkl_path = r"E:\ACDC\trainingset\properties\patient001_frame01_gt\patient001_frame01_002.pkl"

import pickle
with open (pkl_path, 'rb') as f:
  pkl = pickle.load(f)

#print(pkl)


img_path = r"E:\ACDC\EDtrainingset\gt\patient002_frame01_gt\patient002_frame01_gt_003.npy"
seg = np.load(img_path)

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

print(check_if_all_in_one_region(seg, [1,2,3]))


def remove_all_but_the_largest_connected_component(image: np.ndarray, for_which_classes: list,
                                                   minimum_valid_object_size: dict = None):
    """
    removes all but the largest connected component, individually for each class
    :param image:
    :param for_which_classes: can be None. Should be list of int. Can also be something like [(1, 2), 2, 4].
    Here (1, 2) will be treated as a joint region, not individual classes (example LiTS here we can use (1, 2)
    to use all foreground classes together)
    :param minimum_valid_object_size: Only objects larger than minimum_valid_object_size will be removed. Keys in
    minimum_valid_object_size must match entries in for_which_classes
    :return:
    """
    if for_which_classes is None:
        for_which_classes = np.unique(image)
        for_which_classes = for_which_classes[for_which_classes > 0]

    assert 0 not in for_which_classes, "cannot remove background"
    largest_removed = {}
    kept_size = {}
    for c in for_which_classes:
        if isinstance(c, (list, tuple)):
            c = tuple(c)  # otherwise it cant be used as key in the dict
            mask = np.zeros_like(image, dtype=bool)
            for cl in c:
                mask[image == cl] = True
        else:
            mask = image == c
        # get labelmap and number of objects
        lmap, num_objects = label(mask.astype(int), return_num=True)
        print(lmap, num_objects)

        # collect object sizes
        object_sizes = {}
        for object_id in range(1, num_objects + 1):
            object_sizes[object_id] = (lmap == object_id).sum()

        largest_removed[c] = None
        kept_size[c] = None

        if num_objects > 0:
            # we always keep the largest object. We could also consider removing the largest object if it is smaller
            # than minimum_valid_object_size in the future but we don't do that now.
            maximum_size = max(object_sizes.values())
            kept_size[c] = maximum_size

            for object_id in range(1, num_objects + 1):
                # we only remove objects that are not the largest
                # if object_sizes[object_id] != maximum_size:
                if object_sizes[object_id] < 0.2 * maximum_size:
                    # we only remove objects that are smaller than minimum_valid_object_size
                    remove = True
                    if minimum_valid_object_size is not None:
                        remove = object_sizes[object_id] < minimum_valid_object_size[c]
                    if remove:
                        plt.imshow(lmap == object_id)
                        image[(lmap == object_id) & mask] = 0
                        if largest_removed[c] is None:
                            largest_removed[c] = object_sizes[object_id]
                        else:
                            largest_removed[c] = max(largest_removed[c], object_sizes[object_id])
    return image, largest_removed, kept_size

image = np.load(r"D:\Segmentation_framework_results_ED_central_central\pred_seg_np\pred_seg_np_patient008_frame01.npy_007.npy")
image_p, largest_removed, kept_size = remove_all_but_the_largest_connected_component(image, for_which_classes=[1,2,3])
print(image.shape)
print(np.sum((image_p == image).astype(int)))
plt.imshow(image)
print(largest_removed)
print(kept_size)
