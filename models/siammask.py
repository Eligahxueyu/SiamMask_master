# --------------------------------------------------------
# SiamMask定义了网络的主要的模块
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.anchors import Anchors


class SiamMask(nn.Module):
    """
    主要用来定义siamMask网络的框架，及其主要模块
    """
    def __init__(self, anchors=None, o_sz=63, g_sz=127):
        super(SiamMask, self).__init__()
        self.anchors = anchors  # anchor_cfg anchors中的配置信息
        self.anchor_num = len(self.anchors["ratios"]) * len(self.anchors["scales"])  # anchor的数目
        self.anchor = Anchors(anchors)  # anchor
        self.features = None  # 特征提取网络模型
        self.rpn_model = None  # rpn网络模型
        self.mask_model = None  # 图像分割的网络模型
        self.o_sz = o_sz  # 输入尺寸
        self.g_sz = g_sz  # 输出尺寸
        self.upSample = nn.UpsamplingBilinear2d(size=[g_sz, g_sz])  # 2d数据的双线性插值

        self.all_anchors = None

    def set_all_anchors(self, image_center, size):
        """
        初始化anchors（该方法未使用）
        :param image_center: 图像中心
        :param size:
        :return:
        """
        # cx,cy,w,h
        if not self.anchor.generate_all_anchors(image_center, size):
            return
        all_anchors = self.anchor.all_anchors[1]  # cx, cy, w, h
        self.all_anchors = torch.from_numpy(all_anchors).float().cuda()
        self.all_anchors = [self.all_anchors[i] for i in range(4)]

    def feature_extractor(self, x):
        """
        特征提取
        :param x:输入数据
        :return:数据特征
        """
        return self.features(x)

    def rpn(self, template, search):
        """
        rpn网络
        :param template: 模板
        :param search: 搜索图像
        :return:
        """
        # 预测分类和位置结果
        pred_cls, pred_loc = self.rpn_model(template, search)
        return pred_cls, pred_loc

    def mask(self, template, search):
        """
        分割预测结果
        :param template: 模板
        :param search: 待搜索图像
        :return: 掩膜结果
        """
        pred_mask = self.mask_model(template, search)
        return pred_mask

    def _add_rpn_loss(self, label_cls, label_loc, lable_loc_weight, label_mask, label_mask_weight,
                      rpn_pred_cls, rpn_pred_loc, rpn_pred_mask):
        """
        rpn损失函数
        """
        # 分类的损失结果（交叉熵损失）
        rpn_loss_cls = select_cross_entropy_loss(rpn_pred_cls, label_cls)
        # 回归的损失结果
        rpn_loss_loc = weight_l1_loss(rpn_pred_loc, label_loc, lable_loc_weight)
        # 分割的损失结果和准确率
        rpn_loss_mask, iou_m, iou_5, iou_7 = select_mask_logistic_loss(rpn_pred_mask, label_mask, label_mask_weight)

        return rpn_loss_cls, rpn_loss_loc, rpn_loss_mask, iou_m, iou_5, iou_7

    def run(self, template, search, softmax=False):
        '''
        构建网络
        :param template: 模板
        :param search: 待搜索图像
        :param softmax:
        :return:
        '''
        # 提取模板特征
        template_feature = self.feature_extractor(template)
        # 提取图像特征
        search_feature = self.feature_extractor(search)
        # 预测结果
        rpn_pred_cls, rpn_pred_loc = self.rpn(template_feature, search_feature)
        rpn_pred_mask = self.mask(template_feature, search_feature)  # (b, 63*63, w, h)
        # 利用softmax进行分类
        if softmax:
            rpn_pred_cls = self.softmax(rpn_pred_cls)
        return rpn_pred_cls, rpn_pred_loc, rpn_pred_mask, template_feature, search_feature

    def softmax(self, cls):
        """
        softmax
        :param cls:
        :return:
        """
        # 获取cls的结果，及其对应的anchor的大小
        b, a2, h, w = cls.size()
        # 维度变换
        cls = cls.view(b, 2, a2//2, h, w)
        # 高维转置
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        # 对softmax结果求以4为底的对数
        cls = F.log_softmax(cls, dim=4)
        return cls


    def forward(self, input):
        """
        torch中正向传递的算法，所有的子函数将覆盖函数
        :param input: dict of input with keys of:
                'template': [b, 3, h1, w1], input template image.输入的模板图像
                'search': [b, 3, h2, w2], input search image.待搜索图像
                'label_cls':[b, max_num_gts, 5] or None(self.training==False),
                                     each gt contains x1,y1,x2,y2,class.
        :return: dict of loss, predict, accuracy 损失 预测结果 准确率
        """
        # 参数设置
        template = input['template']
        search = input['search']
        if self.training:
            label_cls = input['label_cls']
            label_loc = input['label_loc']
            lable_loc_weight = input['label_loc_weight']
            label_mask = input['label_mask']
            label_mask_weight = input['label_mask_weight']

        # 运行网络
        rpn_pred_cls, rpn_pred_loc, rpn_pred_mask, template_feature, search_feature = \
            self.run(template, search, softmax=self.training)

        outputs = dict()
        # 预测结果
        outputs['predict'] = [rpn_pred_loc, rpn_pred_cls, rpn_pred_mask, template_feature, search_feature]

        if self.training:
            # 损失函数
            rpn_loss_cls, rpn_loss_loc, rpn_loss_mask, iou_acc_mean, iou_acc_5, iou_acc_7 = \
                self._add_rpn_loss(label_cls, label_loc, lable_loc_weight, label_mask, label_mask_weight,
                                   rpn_pred_cls, rpn_pred_loc, rpn_pred_mask)
            # 输出损失函数和精度结果
            outputs['losses'] = [rpn_loss_cls, rpn_loss_loc, rpn_loss_mask]
            outputs['accuracy'] = [iou_acc_mean, iou_acc_5, iou_acc_7]

        return outputs

    def template(self, z):
        """
        用于处理模板图像
        :param z: 跟踪目标的模板
        :return: 模板的分类和回归结果
        """
        self.zf = self.feature_extractor(z)
        cls_kernel, loc_kernel = self.rpn_model.template(self.zf)
        return cls_kernel, loc_kernel

    def track(self, x, cls_kernel=None, loc_kernel=None, softmax=False):
        """
        目标跟踪
        :param x:
        :param cls_kernel:
        :param loc_kernel:
        :param softmax:
        :return:
        """
        # 特征提取
        xf = self.feature_extractor(x)
        # 跟踪结果
        rpn_pred_cls, rpn_pred_loc = self.rpn_model.track(xf, cls_kernel, loc_kernel)
        if softmax:
            rpn_pred_cls = self.softmax(rpn_pred_cls)
        # 返回目标跟踪的位置和分类结果
        return rpn_pred_cls, rpn_pred_loc


def get_cls_loss(pred, label, select):
    """
    计算分类的损失
    :param pred: 预测结果
    :param label: 真实结果
    :param select: 预测位置
    :return:
    """
    # 预测位置为0个，返回0
    if select.nelement() == 0: return pred.sum()*0.
    # 获取预测结果和真实结果
    pred = torch.index_select(pred, 0, select)
    label = torch.index_select(label, 0, select)
    # 计算最大似然函数
    return F.nll_loss(pred, label)


def select_cross_entropy_loss(pred, label):
    """
    交叉熵损失
    :param pred: 预测值
    :param label: 标签值（真实值）
    :return: 返回正负类的损失值
    """
    # 将预测数据展成...*2的形式
    pred = pred.view(-1, 2)
    # 将标签值展成一维形式
    label = label.view(-1)
    # 指明标签值
    # GPU
    # pos = Variable(label.data.eq(1).nonzero().squeeze()).cuda()
    # neg = Variable(label.data.eq(0).nonzero().squeeze()).cuda()
    pos = Variable(label.data.eq(1).nonzero().squeeze())
    neg = Variable(label.data.eq(0).nonzero().squeeze())
    # 计算正负样本的分类损失
    loss_pos = get_cls_loss(pred, label, pos)
    loss_neg = get_cls_loss(pred, label, neg)
    return loss_pos * 0.5 + loss_neg * 0.5


def weight_l1_loss(pred_loc, label_loc, loss_weight):
    """
    smooth L1 损失
    :param pred_loc: [b, 4k, h, w]
    :param label_loc: [b, 4k, h, w]
    :param loss_weight:  [b, k, h, w]
    :return: loc loss value
    """
    # 预测位置的中心坐标和大小
    b, _, sh, sw = pred_loc.size()
    # 变换维度：
    pred_loc = pred_loc.view(b, 4, -1, sh, sw)
    # 计算预测与真实值之间的差值
    diff = (pred_loc - label_loc).abs()
    # 计算梯度
    diff = diff.sum(dim=1).view(b, -1, sh, sw)
    # 损失
    loss = diff * loss_weight
    return loss.sum().div(b)


def select_mask_logistic_loss(p_m, mask, weight, o_sz=63, g_sz=127):
    """
    计算图像分割分支的损失函数及精度信息
    :param p_m:预测的分割结果
    :param mask: 掩膜真实结果
    :param weight:
    :param o_sz:模板的大小
    :param g_sz:图像的大小
    :return:
    """
    weight = weight.view(-1)
    pos = Variable(weight.data.eq(1).nonzero().squeeze())
    if pos.nelement() == 0: return p_m.sum() * 0, p_m.sum() * 0, p_m.sum() * 0, p_m.sum() * 0
    # 维度转换
    p_m = p_m.permute(0, 2, 3, 1).contiguous().view(-1, 1, o_sz, o_sz)
    p_m = torch.index_select(p_m, 0, pos)
    # 2d升采样
    p_m = nn.UpsamplingBilinear2d(size=[g_sz, g_sz])(p_m)
    p_m = p_m.view(-1, g_sz * g_sz)
    # 对掩膜的真实结果进行处理
    mask_uf = F.unfold(mask, (g_sz, g_sz), padding=32, stride=8)
    mask_uf = torch.transpose(mask_uf, 1, 2).contiguous().view(-1, g_sz * g_sz)

    mask_uf = torch.index_select(mask_uf, 0, pos)
    # 计算损失函数
    loss = F.soft_margin_loss(p_m, mask_uf)
    # 计算精度
    iou_m, iou_5, iou_7 = iou_measure(p_m, mask_uf)
    # 返回结果
    return loss, iou_m, iou_5, iou_7


def iou_measure(pred, label):
    """
    iou计算
    :param pred: 预测值
    :param label: 真实值
    :return: iou平均值，iou>0.5的比例，iou>0.7的比例
    """
    # pred中大于0的置为1
    pred = pred.ge(0)
    # 将pred中等于1的与label中为1的相加
    mask_sum = pred.eq(1).add(label.eq(1))
    # mask_sum中为2的表示交
    intxn = torch.sum(mask_sum == 2, dim=1).float()
    # mask_sum中大于0的表示并
    union = torch.sum(mask_sum > 0, dim=1).float()
    # 交并比
    iou = intxn/union
    return torch.mean(iou), (torch.sum(iou > 0.5).float()/iou.shape[0]), (torch.sum(iou > 0.7).float()/iou.shape[0])
    

if __name__ == "__main__":
    p_m = torch.randn(4, 63*63, 25, 25)
    cls = torch.randn(4, 1, 25, 25) > 0.9
    mask = torch.randn(4, 1, 255, 255) * 2 - 1

    loss = select_mask_logistic_loss(p_m, mask, cls)
    print(loss)
