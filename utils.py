import torch
import numpy as np

organs_index = [1, 3, 4, 5, 6, 7, 11, 14]
num_organ = len(organs_index)  # 8


def accuracy(output, target):
    """
    :param output: (B, 9, 48, 256, 256)
    :param target: (B, 48, 256, 256)
    :return: dice
    """
    organs_dice = []
    prediction = np.argmax(output, axis=1)  # (B, 48, 256, 256)
    for idx, organ_idx in enumerate(organs_index):
        pred_temp = (prediction == (idx+1)) + .0
        target_temp = (target == organ_idx) + .0
        dice = 2 * np.sum(pred_temp * target_temp, axis=(1, 2, 3)) / \
                           (np.sum(pred_temp**2, axis=(1, 2, 3))
                            + np.sum(target_temp**2, axis=(1, 2, 3)) + 1e-5)
        dice /= num_organ
        organs_dice.append(np.mean(dice))

    return organs_dice, np.mean(organs_dice)


if __name__ == "__main__":
    a = torch.randint(1, 10, (1, 9, 48, 256, 256))
    b = torch.randint(1, 10, (1, 48, 256, 256))
    _, r = accuracy(a.numpy(), b.numpy())
    print(r)
