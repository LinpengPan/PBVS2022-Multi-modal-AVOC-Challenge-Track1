# -*- coding: utf-8 -*-
# @Time    : 
# @Author  : Gongzhe Li
import numpy as np
import torch
import torch.nn.functional as F
import yacs.config


def onehot_encoding(label: torch.Tensor, n_classes: int) -> torch.Tensor:
	return torch.zeros(label.size(0), n_classes).to(label.device).scatter_(
		1, label.view(-1, 1), 1)


def cross_entropy_loss(data: torch.Tensor, target: torch.Tensor,
                       reduction: str) -> torch.Tensor:
	logp = F.log_softmax(data, dim=1)
	loss = torch.sum(-logp * target, dim=1)
	if reduction == 'none':
		return loss
	elif reduction == 'mean':
		return loss.mean()
	elif reduction == 'sum':
		return loss.sum()
	else:
		raise ValueError(
            '`reduction` must be one of \'none\', \'mean\', or \'sum\'.')






# class CE_with_prior:
# 	def __init__(self):
# 		self.log_prior




'''
def CE_with_prior(one_hot_label, logits, prior, tau=1.0):
    param: one_hot_label
    param: logits
    param: prior: real data distribution obtained by statistics
    param: tau: regulator, default is 1
    return: loss
    log_prior = K.constant(np.log(prior + 1e-8))

    # align dim 
    for _ in range(K.ndim(logits) - 1):     
        log_prior = K.expand_dims(log_prior, 0)

    logits = logits + tau * log_prior
    loss = K.categorical_crossentropy(one_hot_label, logits, from_logits=True)

    return loss
'''







