# --------------------------------------------------------
# Mask
# --------------------------------------------------------
import torch.nn as nn


class Mask(nn.Module):
    """
    mask的基本信息
    """
    def __init__(self):
        super(Mask, self).__init__()

    def forward(self, z_f, x_f):
        raise NotImplementedError

    def template(self, template):
        raise NotImplementedError

    def track(self, search):
        raise NotImplementedError

    def param_groups(self, start_lr, feature_mult=1):
        """
        过滤掉不符合条件的元素
        :param start_lr:
        :param feature_mult:
        :return:
        """
        params = filter(lambda x:x.requires_grad, self.parameters())
        params = [{'params': params, 'lr': start_lr * feature_mult}]
        return params
