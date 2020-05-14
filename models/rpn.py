# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import torch.nn as nn
import torch.nn.functional as F


class RPN(nn.Module):
    """
    RPN网络的基本信息
    """
    def __init__(self):
        "初始化"
        super(RPN, self).__init__()

    def forward(self, z_f, x_f):
        "前向传播"
        raise NotImplementedError

    def template(self, template):
        "模板信息"
        raise NotImplementedError

    def track(self, search):
        "跟踪信息"
        raise NotImplementedError

    def param_groups(self, start_lr, feature_mult=1, key=None):
        # 若key为空，返回需要进行梯度更新的参数
        if key is None:
            params = filter(lambda x:x.requires_grad, self.parameters())
        # 否则返回key中需要更新的参数
        else:
            params = [v for k, v in self.named_parameters() if (key in k) and v.requires_grad]
        params = [{'params': params, 'lr': start_lr * feature_mult}]
        return params


def conv2d_dw_group(x, kernel):
    """
    模板与待搜索图像之间的相关，并变换维度
    :param x:
    :param kernel:
    :return:
    """
    # 获得batch size 和 channel
    batch, channel = kernel.shape[:2]
    # 进行维度重构
    x = x.view(1, batch*channel, x.size(2), x.size(3))  # 1 * (b*c) * k * k
    kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))  # (b*c) * 1 * H * W
    # 计算图像和模板的相似度，使用分组卷积。
    out = F.conv2d(x, kernel, groups=batch*channel)
    # 维度重构
    out = out.view(batch, channel, out.size(2), out.size(3))
    return out

# 计算模板与搜索图像的关系，目标置信度和检测位置
class DepthCorr(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, kernel_size=3):
        super(DepthCorr, self).__init__()
        # 调整图层的不对称性特征 adjust layer for asymmetrical features
        # 对模板进行处理
        self.conv_kernel = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                )
        # 对搜索图像进行处理
        self.conv_search = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                )
        # 网络输出位置：位置，类别等
        self.head = nn.Sequential(
                nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, out_channels, kernel_size=1)
                )

    def forward_corr(self, kernel, input):
        # 计算模板和搜索图像的特征
        kernel = self.conv_kernel(kernel)
        input = self.conv_search(input)
        # 计算模板与搜索图像检测相关性
        feature = conv2d_dw_group(input, kernel)
        return feature

    def forward(self, kernel, search):
        """
        前向传播
        :param kernel: 模板
        :param search: 搜索的图像
        :return:
        """
        feature = self.forward_corr(kernel, search)
        out = self.head(feature)
        return out
