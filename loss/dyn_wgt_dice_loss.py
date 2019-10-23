import torch
import torch.nn as nn
import numpy as np

organs_index = [1, 3, 4, 5, 6, 7, 11, 14]
organs_init_weight = [1, 2, 4, 5, 1, 3, 4, 4]
num_organ = len(organs_index) # 8


class DynWgtDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def _dice(self, pred, target):
        dices = 0.0
        orgs_dices = []
        for idx in range(1, num_organ + 1):
            pred_temp = pred[:, idx, :, :, :]
            pred_temp = 0.9 - torch.relu(0.9 - pred_temp)
            target_temp = target[:, idx - 1, :, :, :]
            dice = (2 * torch.sum(pred_temp * target_temp, [1, 2, 3]) + 1e-6) / \
                   (torch.sum(pred_temp.pow(2), [1, 2, 3])
                    + torch.sum(target_temp.pow(2), [1, 2, 3]) + 1e-6)
            orgs_dices.append(dice)

        orgs_dices_scalar = np.array([torch.mean(i).item() for i in orgs_dices])
        max_org_dice = max(orgs_dices_scalar)
        orgs_weight = np.array([organs_init_weight[i] * min(10.0, max_org_dice / (orgs_dices_scalar[i] + 1e-6)) for i in range(num_organ)])

        for idx, dice in enumerate(orgs_dices):
            dices += orgs_weight[idx] * dice

        dices /= num_organ
        return dices

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
        dice_stage1 = self._dice(pred_stage1, organs_target)

        # 计算第二阶段的loss
        dice_stage2 = self._dice(pred_stage2, organs_target)

        # total loss
        dice_loss = 2 - (dice_stage1 + dice_stage2)

        return dice_loss.mean()
