# 数据增强
# 1. 随机旋转
# 2. 放大
# 3. 塑性形变

import torch
import SimpleITK as sitk
from torch.utils.data import dataset
import os
import csv
import random

sample_path = r'D:\Projects\OrgansSegment\Data\Sample'
image_path = os.path.join(sample_path, 'image')
label_path = os.path.join(sample_path, 'label')

sample_slices = 48  # 48*256*256 for a sample


class Dataset(dataset):
    def __init__(self, csv_path):
        if 'train' in csv_path:
            self.is_training = True
        else:
            self.is_training = False

        csv_reader = csv.reader(open(csv_path, 'r'))
        self.sample_ids = [row[0] for row in csv_reader]

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, item):
        id = self.sample_ids[item]

        # 读取
        image = sitk.ReadImage(os.path.join(image_path, "%image04d.nii" % id))
        label = sitk.ReadImage(os.path.join(label_path, "%label04d.nii" % id))
        img_array = sitk.GetArrayFromImage(image)
        lbl_array = sitk.GetArrayFromImage(label)

        # 随机采样连续的48张slices
        start_slice = random.randint(0, img_array.shape[0] - sample_slices)
        end_slice = start_slice + sample_slices - 1
        img_array = img_array[start_slice:end_slice+1, :, :]
        lbl_array = lbl_array[start_slice:end_slice+1, :, :]

        # 数据增强
        if self.is_training:

