#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import os
import pickle
from glob import glob

import numpy as np
from statsmodels.iolib import load_pickle

from utils import get_pool_and_conv_props
from model.networks.generic_UNet import Generic_UNet
# from nnunet.paths import *
import json


class ExperimentPlanner2D():
    def __init__(self, plans_path):
        super(ExperimentPlanner2D, self).__init__()
        self.plan = None
        self.plan_fname = os.path.join(plans_path, "nnUNetPlans" + "_plans_2D.pkl")

        self.unet_base_num_features = 32
        self.unet_max_num_features = 1024
        self.unet_featuremap_min_edge_length = 16
        self.unet_max_numpool = 6
        self.unet_min_batch_size = 2
        self.preprocessor_name = "PreprocessorFor2D"
        self.conv_per_stage = 2
        self.batch_size_covers_max_percent_of_dataset = 0.05  # all samples in the batch together cannot cover more
        # than 5% of the entire dataset

    def save_my_plans(self):
        with open(self.plan_fname, 'wb') as f:
            pickle.dump(self.plan, f)
            # f.write(json.dumps(self.plan, indent=2))

    def load_my_plans(self):
        self.plan = load_pickle(self.plan_fname)
        return plan

    def get_median_size(self, path_raw_dataset_type):
        counter_H = {}
        counter_W = {}

        patient_paths = glob(os.path.join(path_raw_dataset_type, 'patient*'))
        img_paths = []
        for patient_path in patient_paths:
            img_paths = glob(os.path.join(patient_path, '*.npy'))
            for img_path in img_paths:
                img = np.load(img_path)
                H = img.shape[0]
                W = img.shape[1]
                if H not in counter_H:
                    counter_H[H] = 0
                counter_H[H] += 1
                if W not in counter_W:
                    counter_W[W] = 0
                counter_W[W] += 1
        median_H = max(counter_H, key=counter_H.get)
        median_W = max(counter_W, key=counter_W.get)
        return [median_H, median_W]

    def get_class_connection(self, path_raw_dataset_type):
        final_class_connection = {}

        patient_paths = glob(os.path.join(path_raw_dataset_type, 'patient*'))
        for patient_path in patient_paths:
            pkl_paths = glob(os.path.join(patient_path, '*.pkl'))
            for pkl_path in pkl_paths:
                pkl = load_pickle(pkl_path)
                class_connection = pkl['class_connection']
                print(class_connection)
                if len(final_class_connection) == 0:
                    final_class_connection = class_connection
                else:
                    for key in final_class_connection:
                        final_class_connection[key] &= class_connection[key]
        return final_class_connection

    def get_properties_for_UNet_2D(self, current_spacing, original_spacing, median_shape, num_cases,
                                 num_modalities, num_classes):  # currently, for 2D image, consider current_spacing == original_spacing

        dataset_num_pixels = np.prod(median_shape, dtype=np.int64) * num_cases
        input_patch_size = median_shape

        network_numpool, net_pool_kernel_sizes, net_conv_kernel_sizes, input_patch_size, \
        shape_must_be_divisible_by = get_pool_and_conv_props(current_spacing, input_patch_size,
                                                             self.unet_featuremap_min_edge_length,
                                                             self.unet_max_numpool)
        print(input_patch_size, shape_must_be_divisible_by)
        estimated_gpu_ram_consumption = Generic_UNet.compute_approx_vram_consumption(input_patch_size,
                                                                                     network_numpool,
                                                                                     self.unet_base_num_features,
                                                                                     self.unet_max_num_features,
                                                                                     num_modalities, num_classes,
                                                                                     net_pool_kernel_sizes,
                                                                                     conv_per_stage=self.conv_per_stage)
        print(estimated_gpu_ram_consumption)
        batch_size = int(np.floor(Generic_UNet.use_this_for_batch_size_computation_2D /
                                  estimated_gpu_ram_consumption))
        # batch_size = 2
        if batch_size < self.unet_min_batch_size:
            raise RuntimeError("This framework is not made to process patches this large. We will add patch-based "
                               "2D networks later. Sorry for the inconvenience")

        # check if batch size is too large (more than 5 % of dataset)
        max_batch_size = np.round(self.batch_size_covers_max_percent_of_dataset * dataset_num_pixels /
                                  np.prod(input_patch_size, dtype=np.int64)).astype(int)
        batch_size = max(1, min(batch_size, max_batch_size))

        plan = {
            'batch_size': batch_size,
            'num_pool_per_axis': network_numpool,
            'base_num_features' : self.unet_base_num_features,
            'patch_size': input_patch_size,
            'median_patient_size': median_shape,
            'current_spacing': current_spacing,
            'original_spacing': original_spacing,
            'pool_op_kernel_sizes': net_pool_kernel_sizes,
            'conv_kernel_sizes': net_conv_kernel_sizes,
            'do_dummy_2D_data_aug': False,
            'shape_must_be_divisible_by': shape_must_be_divisible_by
        }
        self.plan = plan
        return plan


if __name__ == "__main__":
    planer = ExperimentPlanner2D(r"C:\Users\wlc\PycharmProjects\Segmentation_framework\model\network_plans")
    median_shape = planer.get_median_size(path_raw_dataset_type=r"E:\ACDC\EDtrainingset\src")
    class_connection = planer.get_class_connection(path_raw_dataset_type=r"E:\ACDC\EDtrainingset\properties")
    print(min(median_shape))
    print(class_connection)
    plan = planer.get_properties_for_UNet_2D(current_spacing=np.array([1, 1]),
                                             original_spacing=np.array([1, 1]),
                                             median_shape=np.array(median_shape),
                                             num_cases=100,
                                             num_modalities=1,
                                             num_classes=4)
    planer.save_my_plans()
    plan = planer.load_my_plans()
    print(plan)

    # get information from heuristic parameters plan
    plan_file_name = r"C:\Users\wlc\PycharmProjects\Segmentation_framework\model\network_plans\nnUNetPlans_plans_2D.pkl"
    plan = load_pickle(plan_file_name)
    pool_op_kernel_sizes = plan["pool_op_kernel_sizes"]
    conv_kernel_sizes = plan["conv_kernel_sizes"]
    batch_size = int(plan["batch_size"])
    base_num_features = int(plan["base_num_features"])
    num_pool = max(int(plan["num_pool_per_axis"][0]), int(plan["num_pool_per_axis"][1]))
    patch_size = plan["patch_size"]
    shape_must_be_divisible_by = [int(plan['shape_must_be_divisible_by'][0]), int(plan['shape_must_be_divisible_by'][1])]
    print(shape_must_be_divisible_by)
    H = int(patch_size[0])
    W = int(patch_size[1])

    # get opt json file template
    with open(r"C:\Users\wlc\PycharmProjects\Segmentation_framework\options\template.json", 'r') as fp:
        train_ACDC_plan = json.load(fp)
        # print('data: ', train_ACDC_plan)
        # print('data_type: ', type(train_ACDC_plan))

    # write heuristic parameters into the template
    train_ACDC_plan['datasets']['train']['H_size'] = H
    train_ACDC_plan['datasets']['train']['W_size'] = W
    train_ACDC_plan['datasets']['train']['dataloader_batch_size'] = batch_size
    train_ACDC_plan['datasets']['test']['H_size'] = H
    train_ACDC_plan['datasets']['test']['W_size'] = W
    train_ACDC_plan['netSeg']['pool_op_kernel_sizes'] = pool_op_kernel_sizes
    train_ACDC_plan['netSeg']['conv_kernel_sizes'] = conv_kernel_sizes
    train_ACDC_plan['netSeg']['num_pool'] = num_pool
    train_ACDC_plan['netSeg']['base_num_features'] = base_num_features  # use default setting
    train_ACDC_plan['netSeg']['shape_must_be_divisible_by'] = shape_must_be_divisible_by  # use default setting

    # save as opt json file
    plan_fname = r"C:\Users\wlc\PycharmProjects\Segmentation_framework\options\UNet_2D_ACDC_ED_crop.json"
    with open(plan_fname, 'w') as f:
        # pickle.dump(self.plan, f)
        json.dump(train_ACDC_plan, f, ensure_ascii=False, sort_keys=True, indent=2)

    print("done")
