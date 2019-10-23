# 数据增强
# 1. 随机旋转
# 2. 放大
# 3. 塑性形变

import torch
import SimpleITK as sitk
import scipy.ndimage as ndimage
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import os
import csv
import random
import numpy as np

sample_path = r'D:\Projects\OrgansSegment\Data\Sample'
image_path = os.path.join(sample_path, 'image')
label_path = os.path.join(sample_path, 'label')

sample_slices = 48  # 48*256*256 for a sample
lower = -350


def produceRandomlyDeformedImage(sitkImage, sitkLabel, numcontrolpoints, stdDef):
    transfromDomainMeshSize=[numcontrolpoints]*sitkImage.GetDimension()

    tx = sitk.BSplineTransformInitializer(sitkImage, transfromDomainMeshSize)

    params = tx.GetParameters()

    paramsNp=np.asarray(params,dtype=float)
    paramsNp = paramsNp + np.random.randn(paramsNp.shape[0])*stdDef

    paramsNp[0:int(len(params)/3)]=0 #remove z deformations! The resolution in z is too bad

    params=tuple(paramsNp)
    tx.SetParameters(params)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(sitkImage)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(tx)

    resampler.SetInterpolator(sitk.sitkLinear)
    outimgsitk = resampler.Execute(sitkImage)

    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    outlabsitk = resampler.Execute(sitkLabel)

    return sitk.GetArrayFromImage(outimgsitk), sitk.GetArrayFromImage(outlabsitk)


class MyDataset(Dataset):
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
        id = int(self.sample_ids[item])

        # 读取
        image = sitk.ReadImage(os.path.join(image_path, "image%04d.nii" % id))
        label = sitk.ReadImage(os.path.join(label_path, "label%04d.nii" % id))
        img_array = sitk.GetArrayFromImage(image)
        lbl_array = sitk.GetArrayFromImage(label)

        # 数据增强
        if self.is_training:
            # 以0.5概率进行B样条形变
            bspline = False
            if random.uniform(0, 1) >= 0.7:
                img_array, lbl_array = produceRandomlyDeformedImage(image, label, 2, random.uniform(5, 15))
                bspline = True
            # 以0.5的概率在5度的范围内随机旋转
            # 角度为负数是顺时针旋转，角度为正数是逆时针旋转
            if random.uniform(0, 1) >= 0.6:
                angle = random.uniform(-5, 5)
                img_array = ndimage.rotate(img_array, angle, axes=(1, 2), reshape=False, cval=lower)
                lbl_array = ndimage.rotate(lbl_array, angle, axes=(1, 2), reshape=False, cval=0)

            # 有0.5的概率不进行任何修修改，剩下0.5随机挑选0.8-0.5大小的patch放大到256*256
            if random.uniform(0, 1) >= 0.7:
                img_array, lbl_array = self.zoom(img_array, lbl_array, patch_size=random.uniform(0.5, 0.8))

        # 随机采样连续的48张slices
        if (not bspline) or img_array.shape[0] < sample_slices+4:
            start_slice = random.randint(0, img_array.shape[0] - sample_slices)
        else:  # b样条形变可能会造成边界插值误差
            start_slice = random.randint(2, img_array.shape[0] - sample_slices - 2)
        end_slice = start_slice + sample_slices - 1
        img_array = img_array[start_slice:end_slice+1, :, :]
        lbl_array = lbl_array[start_slice:end_slice+1, :, :]

        # 处理完毕，将array转换为tensor
        img_array = torch.FloatTensor(img_array).unsqueeze(0)
        lbl_array = torch.FloatTensor(lbl_array)

        return img_array, lbl_array

    def zoom(self, ct_array, seg_array, patch_size):
        length = int(256 * patch_size)

        x1 = int(random.uniform(0, 255 - length))
        y1 = int(random.uniform(0, 255 - length))

        x2 = x1 + length
        y2 = y1 + length

        ct_array = ct_array[:, x1:x2 + 1, y1:y2 + 1]
        seg_array = seg_array[:, x1:x2 + 1, y1:y2 + 1]

        with torch.no_grad():
            ct_array = torch.FloatTensor(ct_array).unsqueeze(dim=0).unsqueeze(dim=0)
            ct_array = Variable(ct_array)
            ct_array = F.upsample(ct_array, (sample_slices, 256, 256), mode='trilinear').squeeze().detach().numpy()

            seg_array = torch.FloatTensor(seg_array).unsqueeze(dim=0).unsqueeze(dim=0)
            seg_array = Variable(seg_array)
            seg_array = F.upsample(seg_array, (sample_slices, 256, 256), mode='nearest').squeeze().detach().numpy()

            return ct_array, seg_array


if __name__ == "__main__":
    train_ds = MyDataset('D:/Projects/OrgansSegment/Data/data_preprocess/train_info.csv')

    # 测试代码
    train_dl = DataLoader(train_ds, 3, True)
    for index, (ct, seg) in enumerate(train_dl):

        print(index, ct.size(), seg.size())
        print('----------------')


