import os
import SimpleITK as sitk
import torch
import numpy as np
import csv

sample_path = r'D:\Projects\OrgansSegment\Data\Sample'
image_path = os.path.join(sample_path, 'image')
label_path = os.path.join(sample_path, 'label')
slice = 48


def sample_predict(net, ct_path):
    """
    predict sample segmentation
    :param net: model
    :param ct_path: sample after downsample
    :return: segmentation array (D, 256, 256)
    """
    net.eval()
    ct = sitk.ReadImage(ct_path)
    ct_array = sitk.GetArrayFromImage(ct)  # (D, 256, 256)

    # 切块取样
    cut_flag = False
    start_slice = 0
    end_slice = start_slice + slice
    ct_array_list = []

    while end_slice <= ct_array.shape[0]:
        ct_array_list.append(ct_array[start_slice:end_slice, :, :])
        start_slice = end_slice
        end_slice = start_slice + slice

    # 当无法整除的时候反向取最后一个block
    if end_slice is not ct_array.shape[0]:
        cut_flag = True
        count = ct_array.shape[0] - start_slice
        ct_array_list.append(ct_array[-slice:, :, :])

    # 预测
    outputs_list = []
    with torch.no_grad():
        for ct_sample in ct_array_list:
            ct_tensor = torch.FloatTensor(ct_sample).cuda()
            ct_tensor = ct_tensor.unsqueeze(dim=0)
            ct_tensor = ct_tensor.unsqueeze(dim=0)  # (1, 1, 48, 256, 256)

            output = net(ct_tensor)
            output = output.squeeze()  # (9, 48, 256, 256)

            outputs_list.append(output.cpu().detached().numpy())
            del output

    # 拼接
    pred_seg = np.concatenate(outputs_list[0:-1], axis=1)
    if cut_flag:
        pred_seg = np.concatenate([pred_seg, outputs_list[-1][:, -count, :, :]], axis=1)
    else:
        pred_seg = np.concatenate([pred_seg, outputs_list[-1]], axis=1)

    return pred_seg


def dataset_accuracy(net, csv_path):
    file = open(csv_path, 'r')
    lines = csv.reader(file)
    for line in lines:
        seg_path = os.path.join(label_path, line[2])
        ct_path = os.path.join(image_path, line[2].replace('label', 'image'))

        pred_seg = sample_predict(net, ct_path)
        target = sitk.ReadImage(seg_path)
        target_array = sitk.GetArrayFromImage(target)







