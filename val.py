import os
import SimpleITK as sitk
import torch
import numpy as np
import csv
from utils import accuracy
from model.vnet import get_net
import scipy.ndimage as ndimage

organs_name = ['spleen', 'left kidney', 'gallbladder', 'esophagus',
               'liver', 'stomach', 'pancreas', 'duodenum']

TCIA_Path = r'D:\Projects\OrgansSegment\Data\TCIA\label_tcia_multiorgan'
BTCV_Path = r'D:\Projects\OrgansSegment\Data\BTCV\label_btcv_multiorgan'

sample_path = r'D:\Projects\OrgansSegment\Data\Sample'
image_path = os.path.join(sample_path, 'image')
label_path = os.path.join(sample_path, 'label')
slice = 48

file = open('./csv_files/prediction.csv', 'w')
csv_writer = csv.writer(file)
csv_writer.writerow([' ']+organs_name)

model_dir = './module/net90-0.797-0.644.pth'

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

            outputs_list.append(output.cpu().detach().numpy())
            del output

    # 拼接
    pred_seg = np.concatenate(outputs_list[0:-1], axis=1)  # (9, D, 256, 256)
    if cut_flag:
        pred_seg = np.concatenate([pred_seg, outputs_list[-1][:, -count:, :, :]], axis=1)
    else:
        pred_seg = np.concatenate([pred_seg, outputs_list[-1]], axis=1)

    return pred_seg


def save_seg(pred_seg, info, accs):
    prediction = np.argmax(pred_seg, axis=1).squeeze()  # (D, 256, 256)

    dataset = info[1]
    label = info[2]
    if dataset == 'TCIA':
        seg = sitk.ReadImage(os.path.join(TCIA_Path, label))
    elif dataset == 'BTCV':
        seg = sitk.ReadImage(os.path.join(BTCV_Path, label))

    # 插值
    seg_array = sitk.GetArrayFromImage(seg)
    pred_unsample = ndimage.zoom(prediction, (seg_array.shape[0]/prediction.shape[0], 2, 2), order=0)

    # 保存
    pred = sitk.GetImageFromArray(pred_unsample)

    pred.SetDirection(seg.GetDirection())
    pred.SetOrigin(seg.GetOrigin())
    pred.SetSpacing(seg.GetSpacing())

    sitk.WriteImage(pred, os.path.join('./prediction', dataset+'-'+label))
    del pred
    csv_writer.writerow([dataset+'-'+label]+accs)



def dataset_accuracy(net, csv_path, save=False):
    file = open(csv_path, 'r')
    lines = csv.reader(file)
    mean_acc = []
    orgs_acc = []
    for line in lines:
        seg_path = os.path.join(label_path, "label%04d.nii" % int(line[0]))
        ct_path = os.path.join(image_path, "image%04d.nii" % int(line[0]))

        pred_seg = sample_predict(net, ct_path)

        pred_seg = np.expand_dims(pred_seg, axis=0)  # (1, 9, D, 256, 256)

        target = sitk.ReadImage(seg_path)
        target_array = sitk.GetArrayFromImage(target)
        target_array = np.expand_dims(target_array, axis=0)  # (1, D, 256, 256)

        accs, acc = accuracy(pred_seg, target_array)

        if save:
            save_seg(pred_seg, line, accs)

        orgs_acc.append(accs)
        mean_acc.append(acc)

    return np.mean(orgs_acc, axis=0), np.mean(mean_acc)



if __name__ == "__main__":
    # model
    net = get_net(False)
    net = torch.nn.DataParallel(net).cuda()
    net.load_state_dict(torch.load(model_dir))
    net.eval()

    test_org_acc, test_mean_acc = dataset_accuracy(net, 'csv_files/test_info.csv', save=True)
    print(' '.join(["%s:%.3f" % (i, j) for i, j in zip(organs_name, test_org_acc)]))








