# -*- coding: utf-8 -*-
"""
 Author       :LinpengPan
 Date         :2022/3/9  11:00
 Filename     :can.py
 E-mail       :linpengpan@qq.com
 usage        :
 """
import numpy as np

__all__ = ["CAN", "PrepareCAN"]


def entropy(probs, base):
    """
    has tested
    :param probs:
    :param base:
    :return:
    """
    probs = sanitycheck_probs(probs)
    logProbs = np.log(probs) / np.log(base)
    return -(logProbs * probs).sum()


def sanitycheck_probs(probs):
    # check if there is any 0 values, if so, add a eps to the position
    probs = np.array(probs)
    probs = probs + (probs == 0) * 1e-16
    return probs


def topkEntory(probs, k):
    assert k > 1
    best_k = np.sort(sanitycheck_probs(probs))[-k:]
    best_k = best_k / np.sum(best_k)  # 原论文可没提到需要进行归一化
    return entropy(best_k, k)


class CAN:

    def __init__(self, alpha, d, delta_q, A0, B0):
        """

        :param alpha:
        :param d:
        :param q: 对角型先验矩阵
        """
        self.alpha = alpha
        self.d = d
        self.delta_q = delta_q
        self.A0 = np.array(A0)
        self.B0 = np.array(B0)

    def alterNorm(self, A0, b):
        # row Norm
        b = np.reshape(b, [1, -1])
        L = np.concatenate((A0, b), axis=0)
        L = L ** self.alpha
        deltaS = np.diag(np.squeeze(np.transpose(L) @ np.ones((L.shape[0], 1)), axis=1))
        Sd = L @ np.linalg.inv(deltaS)

        # col Norm
        deltaL = np.diag(np.squeeze(Sd @ self.delta_q @ np.ones((self.delta_q.shape[0], 1)), axis=1))
        Ld = np.linalg.inv(deltaL) @ Sd @ self.delta_q

        return Ld

    def reAdjusted(self):
        B0Adjusted = []
        for row in self.B0:
            for i in range(self.d):
                Li = self.alterNorm(self.A0, row)
            B0Adjusted.append(Li[-1, :])
        return np.array(B0Adjusted)


class PrepareCAN:

    def __init__(self, A, names, thre=0.9):
        """

        :param A: 样本的概率分布
        :param names: 样本的名字，和A按照行进行对应
        :param thre: 分割混淆等级的阈值
        """
        self.A = np.array(A)
        self.thre = thre
        assert 0 <= thre <= 1
        self.k_max = self.A.shape[1]
        self.names = names
        assert self.A.shape[0] == len(names)

    def splitIndex(self):
        n_samples, n_labels = self.A.shape

        confused_ids = set()
        confidence_ids = []
        for k in range(2, self.k_max + 1):
            for i in range(n_samples):
                ent = topkEntory(self.A[i, :], k)
                if ent >= self.thre:
                    confused_ids.add(i)
        confidence_ids = [i for i in range(n_samples) if i not in confused_ids]
        confused_ids = list(confused_ids)
        return confused_ids, confidence_ids

    def filter_logits(self, probs, targets, selected_ids):
        return probs[selected_ids, :], [targets[i] for i in selected_ids]

    def split(self):
        B0_id, A0_id = self.splitIndex()
        A0, A0_names = self.filter_logits(self.A, self.names, A0_id)
        B0, B0_names = self.filter_logits(self.A, self.names, B0_id)
        return A0, A0_names, B0, B0_names


