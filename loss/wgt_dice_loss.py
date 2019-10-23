"""
label smooth
class weight
"""

import torch
import torch.nn as nn

organs_index = [1, 3, 4, 5, 6, 7, 11, 14]
num_organ = len(organs_index) # 8
organ_weight = [1.0, 2.0, 4.0, 5.0, 1.0, 3.0, 4.0, 4.0]


class WgtDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_stage1, pred_stage2, target):
        """
        计算多类别平均dice loss
        :param pred_stage1: 阶段一的输出 (B, 9, 48, 256, 256)
        :param pred_stage2: 阶段二的输出 (B, 9, 48, 256, 256)
        :param target: 金标准 (B, 48, 256, 256)
        :return: loss
        """
        # 首先将金标准拆开
        organs_target = torch.zeros(target.size(0), num_organ, 48, 256, 256)
        for idx, organ_idx in enumerate(organs_index):
            organs_target[:, idx, :, :, :] = (target == organ_idx) + .0

        organs_target = organs_target.cuda() # (B, 8, 48, 256, 256)

        # 计算第一阶段的loss
        loss_stage1 = 0.0
        for idx in range(1, num_organ+1):
            pred_temp = pred_stage1[:, idx, :, :, :]
            pred_temp = 0.9-torch.relu(0.9-pred_temp)
            target_temp = organs_target[:, idx - 1, :, :, :]
            org_dice = 2 * (torch.sum(pred_temp * target_temp, [1, 2, 3]) + 1e-6) / \
                           (torch.sum(pred_temp.pow(2), [1, 2, 3])
                            + torch.sum(target_temp.pow(2), [1, 2, 3]) + 1e-6)
            org_loss = organ_weight[idx-1] * (1 - org_dice)
            loss_stage1 += org_loss

        loss_stage1 /= num_organ

        # 计算第二阶段的loss
        loss_stage2 = 0.0
        for idx in range(1, num_organ+1):
            pred_temp = pred_stage2[:, idx, :, :, :]
            pred_temp = 0.9-torch.relu(0.9-pred_temp)
            target_temp = organs_target[:, idx - 1, :, :, :]
            org_dice = 2 * (torch.sum(pred_temp * target_temp, [1, 2, 3]) + 1e-6) / \
                           (torch.sum(pred_temp.pow(2), [1, 2, 3])
                            + torch.sum(target_temp.pow(2), [1, 2, 3]) + 1e-6)
            org_loss = organ_weight[idx-1] * (1 - org_dice)
            loss_stage2 += org_loss

        loss_stage2 /= num_organ

        # total loss
        dice_loss = loss_stage1 + loss_stage2

        return dice_loss.mean()
