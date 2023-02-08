import os
import pickle
import numpy as np
from statsmodels.iolib import load_pickle
import json

# get information from heuristic parameters plan
plan_file_name = r"C:\Users\wlc\PycharmProjects\Segmentation_framework\model\network_plans\nnUNetPlans_plans_2D.pkl"
plan = load_pickle(plan_file_name)
pool_op_kernel_sizes = plan["pool_op_kernel_sizes"]
conv_kernel_sizes = plan["conv_kernel_sizes"]
batch_size = int(plan["batch_size"])
base_num_features = int(plan["base_num_features"])
num_pool = max(int(plan["num_pool_per_axis"][0]), int(plan["num_pool_per_axis"][1]))
patch_size = plan["patch_size"]
H = int(patch_size[0])
W = int(patch_size[1])

# get opt json file template
with open(r"C:\Users\wlc\PycharmProjects\Segmentation_framework\options\template.json", 'r')as fp:
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

# save as opt json file
plan_fname = r"C:\Users\wlc\PycharmProjects\Segmentation_framework\options\UNet_2D_ACDC.json"
with open(plan_fname, 'w') as f:
    # pickle.dump(self.plan, f)
    json.dump(train_ACDC_plan, f, ensure_ascii=False, sort_keys=True, indent=2)

print("done")