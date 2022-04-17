from typing import Tuple

import torch
import torch.nn as nn

# device = torch.device(config.device)

# class_weight = torch.Tensor(0.79724, 0.09561, 0.0521, 0.03626, 0.00592, 0.002900, 0.002818, 0.002124, 0.002859, 0.002154)

class MixupLoss:
    def __init__(self, reduction: str):
        # self.class_weight = torch.FloatTensor([0.79724, 0.09561, 0.0521, 0.03626, 0.00592, 0.002900, 0.002818, 0.002124, 0.002859, 0.002154])
        # self.class_weight = torch.FloatTensor([0.026, 0.0789, 0.0526, 0.0526, 0.01315, 0.01315, 0.01315, 0.01315, 0.01315, 0.01315]).cuda('cuda:1')
        self.loss_func = nn.CrossEntropyLoss(weight=None, reduction=reduction)

    def __call__(
            self, predictions: torch.Tensor,
            targets: Tuple[torch.Tensor, torch.Tensor, float]) -> torch.Tensor:
        targets1, targets2, lam = targets
        return lam * self.loss_func(predictions, targets1) + (
            1 - lam) * self.loss_func(predictions, targets2)


if __name__ == '__main__':
    loss = MixupLoss(reduction='mean')