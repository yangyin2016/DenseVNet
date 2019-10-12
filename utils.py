import torch
import numpy as np
from skimage import measure

organs_index = [1, 3, 4, 5, 6, 7, 11, 14]
num_organ = len(organs_index)  # 8

orgs_size = {1: 246174.52222222224, 3: 161109.5842696629, 4: 25727.5, 5: 12812.022471910112,
             6: 1521912.1555555556, 7: 381291.64444444445, 11: 67835.83333333333, 14: 85104.07777777778}


def accuracy(output, target):
    """
    :param output: (B, 9, D, 256, 256)
    :param target: (B, D, 256, 256)
    :return: dice
    """
    organs_dice = []
    prediction = np.argmax(output, axis=1)  # (B, D, 256, 256)
    for idx, organ_idx in enumerate(organs_index):
        pred_temp = (prediction == (idx + 1)) + .0
        target_temp = (target == organ_idx) + .0
        # target中不含该organ
        if (target_temp == 0).all():
            organs_dice.append('None')
        else:
            dice = 2 * np.sum(pred_temp * target_temp, axis=(1, 2, 3)) / \
                   (np.sum(pred_temp ** 2, axis=(1, 2, 3))
                    + np.sum(target_temp ** 2, axis=(1, 2, 3)) + 1e-5)
            organs_dice.append(np.mean(dice))

    return organs_dice, np.mean(list(set(organs_dice).difference(['None'])))


def post_process(input):
    """
    分割结果的后处理：保留最大的且>10%organ_size的连通区域
    :param input: (9, D, 256, 256)
    :return: (9, D, 256, 256)
    """
    pred_seg = np.argmax(input, axis=0)  # (D, 256, 256)
    output = np.zeros(input.shape)
    for id in range(input.shape[0]):
        org_seg = (pred_seg == id) + .0
        labels, num = measure.label(org_seg, return_num=True)
        regions = measure.regionprops(labels)
        regions_area = [regions[i].area for i in range(num)]
        label_num = regions_area.index(max(regions_area)) + 1  # 不会计算background(0)
        org_seg[labels == label_num] = 1
        org_seg[labels != label_num] = 0
        output[id, :, :, :] = org_seg

    return output


if __name__ == "__main__":
    a = torch.randint(1, 10, (6, 9, 48, 256, 256))
    b = torch.randint(1, 10, (6, 48, 256, 256))
    _, r = accuracy(a.numpy(), b.numpy())
    print(r)
