from typing import Callable, Tuple

import torch
import torch.nn as nn
import yacs.config

from .cutmix import CutMixLoss
from .mixup import MixupLoss
from .ricap import RICAPLoss
from .dual_cutout import DualCutoutLoss
from .label_smoothing import LabelSmoothingLoss


# weight = torch.FloatTensor([0.026, 0.0789, 0.0526, 0.0526, 0.01315, 0.01315, 0.01315, 0.01315, 0.01315, 0.01315]).cuda('cuda:1')

def create_loss(config: yacs.config.CfgNode) -> Tuple[Callable, Callable]:
    if config.augmentation.use_mixup:
        train_loss = MixupLoss(reduction='mean')
    elif config.augmentation.use_ricap:
        train_loss = RICAPLoss(reduction='mean')
    elif config.augmentation.use_cutmix:
        train_loss = CutMixLoss(reduction='mean')
    elif config.augmentation.use_label_smoothing:
        train_loss = LabelSmoothingLoss(config, reduction='mean')
    elif config.augmentation.use_dual_cutout:
        train_loss = DualCutoutLoss(config, reduction='mean')
    else:
        train_loss = nn.CrossEntropyLoss(reduction='mean')
        # train_loss = nn.CrossEntropyLoss(weight=weight, reduction='mean')
    val_loss = nn.CrossEntropyLoss(reduction='mean')
    return train_loss, val_loss
