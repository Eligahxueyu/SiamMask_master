# --------------------------------------------------------
# 目标跟踪器参数设置
# --------------------------------------------------------
from __future__ import division
from utils.anchors import Anchors


# 跟踪器配置参数
class TrackerConfig(object):
    penalty_k = 0.09
    window_influence = 0.39
    lr = 0.38
    seg_thr = 0.3  # 分割阈值 for mask
    windowing = 'cosine'  # 对于较大的位移进行惩罚 to penalize large displacements [cosine/uniform]
    # Params from the network architecture, have to be consistent with the training
    exemplar_size = 127  # 跟踪目标模板的大小 input z size
    instance_size = 255  # 跟踪实例的大小 input x size (search region)
    total_stride = 8  #
    out_size = 63  # for mask
    base_size = 8
    score_size = (instance_size-exemplar_size)//total_stride+1+base_size
    context_amount = 0.5  # 跟踪目标的周边信息比例 context amount for the exemplar
    ratios = [0.33, 0.5, 1, 2, 3]  # anchors宽高比
    scales = [8, ]  # 尺度,即anchor的大小
    anchor_num = len(ratios) * len(scales)  # anchor的个数
    round_dight = 0  #
    anchor = []

    def update(self, newparam=None, anchors=None):
        """
        更新参数
        :param newparam: 新的参数
        :param anchors: anchors的参数
        :return:
        """
        # 新的参数直接添加到配置中
        if newparam:
            for key, value in newparam.items():
                setattr(self, key, value)
        # 添加anchors的参数
        if anchors is not None:
            # 若anchors是字典形式的将其转换为Anchors
            if isinstance(anchors, dict):
                anchors = Anchors(anchors)
            # 更新到config中
            if isinstance(anchors, Anchors):
                self.total_stride = anchors.stride
                self.ratios = anchors.ratios
                self.scales = anchors.scales
                self.round_dight = anchors.round_dight
        self.renew()

    def renew(self):
        """
        更新配置信息
        :return:
        """
        # 分类尺寸
        self.score_size = (self.instance_size - self.exemplar_size) // self.total_stride + 1 + self.base_size
        # anchor数目
        self.anchor_num = len(self.ratios) * len(self.scales)




