import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

organs_index = [1, 3, 4, 5, 6, 7, 11, 14]
organs_weight = [2.0, 2.0, 6.0, 10.0, 1.0, 2.0, 4.0, 4.0]
num_organ = len(organs_index) # 8


class FocalDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(reduce=False)

    def _loss(self, pred, target):
        # 首先将金标准拆开
        organs_target = torch.zeros(target.size(0), num_organ, 48, 256, 256)
        for idx, organ_idx in enumerate(organs_index):
            organs_target[:, idx, :, :, :] = (target == organ_idx) + .0

        organs_target = organs_target.cuda()  # (B, 8, 48, 256, 256)
        target = organs_target.argmax(dim=1)  # (B, 48, 256, 256)

        # dice loss
        dice_loss_sum = 0.0
        for idx in range(1, num_organ + 1):
            pred_temp = pred[:, idx, :, :, :]
            pred_temp = 0.9 - torch.relu(0.9 - pred_temp)
            target_temp = organs_target[:, idx - 1, :, :, :]

            org_dice = 2 * (torch.sum(pred_temp * target_temp, [1, 2, 3]) + 1e-6) / \
                       (torch.sum(pred_temp, [1, 2, 3])
                        + torch.sum(target_temp, [1, 2, 3]) + 1e-6)
            dice_loss = 1 - org_dice

            dice_loss_sum += dice_loss
        dice_loss_sum /= num_organ
        dice_loss_sum = dice_loss_sum.mean()

        # focal loss
        focal_loss = self.loss(pred, target)
        exponential = (1 - F.softmax(pred, dim=1).max(dim=1)[0]) ** 2
        focal_loss_sum = (focal_loss * exponential).mean()

        return dice_loss_sum + focal_loss_sum

    def forward(self, pred_stage1, pred_stage2, target):
        """
        计算多类别平均dice loss
        :param pred_stage1: 阶段一的输出 (B, 9, 48, 256, 256)
        :param pred_stage2: 阶段二的输出 (B, 9, 48, 256, 256)
        :param target: 金标准 (B, 48, 256, 256)
        :return: loss
        """

        # 计算第一阶段的loss
        loss_stage1 = self._loss(pred_stage1, target)

        # 计算第二阶段的loss
        loss_stage2 = self._loss(pred_stage2, target)

        return loss_stage1 + loss_stage2
