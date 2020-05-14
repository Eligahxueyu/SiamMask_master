# --------------------------------------------------------
# 学习率lr更新
# --------------------------------------------------------
from __future__ import division
import numpy as np
import math
from torch.optim.lr_scheduler import _LRScheduler
import matplotlib.pyplot as plt


class LRScheduler(_LRScheduler):
    """
    学习率更新策略
    """
    def __init__(self, optimizer, last_epoch=-1):
        # 若不存在"lr_spaces"返回异常，lr_spaces是lr更新序列
        if 'lr_spaces' not in self.__dict__:
            raise Exception('lr_spaces must be set in "LRSchduler"')
        super(LRScheduler, self).__init__(optimizer, last_epoch)

    def get_cur_lr(self):
        """
        获取当前epoch的学习率
        :return:
        """
        return self.lr_spaces[self.last_epoch]

    def get_lr(self):
        """
        定义学习率的更新策略
        :return:
        """
        epoch = self.last_epoch
        # 返回当前epoch优化器中学习率
        return [self.lr_spaces[epoch] * pg['initial_lr'] / self.start_lr for pg in self.optimizer.param_groups]

    def __repr__(self):
        """
        返回学习率更新的序列
        :return:
        """
        return "({}) lr spaces: \n{}".format(self.__class__.__name__, self.lr_spaces)


class LogScheduler(LRScheduler):
    """
    指数式更新学习率
    """
    def __init__(self, optimizer, start_lr=0.03, end_lr=5e-4, epochs=50, last_epoch=-1, **kwargs):
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.epochs = epochs
        # 指明初始值和终值，依据epochs生成学习率
        self.lr_spaces = np.logspace(math.log10(start_lr), math.log10(end_lr), epochs)

        super(LogScheduler, self).__init__(optimizer, last_epoch)


class StepScheduler(LRScheduler):
    """
    步进式的更新学习率
    """
    def __init__(self, optimizer, start_lr=0.01, end_lr=None, step=10, mult=0.1, epochs=50, last_epoch=-1, **kwargs):
        """
        初始化
        :param optimizer: 优化器
        :param start_lr: 初始lr
        :param end_lr: 终值lr
        :param step: 经过step个epoch更新学习率
        :param mult: 更新参数gamma
        :param epochs:
        :param last_epoch: 起始epoch
        :param kwargs:
        """

        # 若end_lr不为None
        if end_lr is not None:
            if start_lr is None:
                # 根据end_lr求解start_lr，multshi
                start_lr = end_lr / (mult ** (epochs // step))
            else:  # for warm up policy
                # 计算mult
                mult = math.pow(end_lr/start_lr, 1. / (epochs // step))
        self.start_lr = start_lr
        # 得到学习率的序列
        self.lr_spaces = self.start_lr * (mult**(np.arange(epochs) // step))
        self.mult = mult
        # 没过step个epoch更新学习率
        self._step = step

        super(StepScheduler, self).__init__(optimizer, last_epoch)


class MultiStepScheduler(LRScheduler):
    """
    多步长更新学习率
    """
    def __init__(self, optimizer, start_lr=0.01, end_lr=None, steps=[10,20,30,40], mult=0.5, epochs=50, last_epoch=-1, **kwargs):
        """
        :param optimizer: 优化器
        :param start_lr: 起始学习率
        :param end_lr: 终值学习率
        :param steps: 学习率进行更新的步长序列
        :param mult: 更新参数
        :param epochs:
        :param last_epoch: 起始epoch
        :param kwargs:
        """
        if end_lr is not None:
            if start_lr is None:
                # 计算start_lr
                start_lr = end_lr / (mult ** (len(steps)))
            else:
                # 计算mult
                mult = math.pow(end_lr/start_lr, 1. / len(steps))
        self.start_lr = start_lr
        # 获取lr_spaces
        self.lr_spaces = self._build_lr(start_lr, steps, mult, epochs)
        self.mult = mult
        self.steps = steps

        super(MultiStepScheduler, self).__init__(optimizer, last_epoch)

    def _build_lr(self, start_lr, steps, mult, epochs):
        """
        计算学习率列表
        :param start_lr:
        :param steps:
        :param mult:
        :param epochs:
        :return:
        """
        lr = [0] * epochs
        lr[0] = start_lr
        for i in range(1, epochs):
            lr[i] = lr[i-1]
            # 若i在steps中则修改学习率，否则学习率不变
            if i in steps:
                lr[i] *= mult
        return np.array(lr, dtype=np.float32)


class LinearStepScheduler(LRScheduler):
    """
    线性更新学习率
    """
    def __init__(self, optimizer, start_lr=0.01, end_lr=0.005, epochs=50, last_epoch=-1, **kwargs):
        self.start_lr = start_lr
        self.end_lr = end_lr
        # 生成学习率序列
        self.lr_spaces = np.linspace(start_lr, end_lr, epochs)

        super(LinearStepScheduler, self).__init__(optimizer, last_epoch)


class CosStepScheduler(LRScheduler):
    """
    cos式的更新学习率
    """
    def __init__(self, optimizer, start_lr=0.01, end_lr=0.005, epochs=50, last_epoch=-1, **kwargs):
        self.start_lr = start_lr
        self.end_lr = end_lr
        # 获取学习率
        self.lr_spaces = self._build_lr(start_lr, end_lr, epochs)

        super(CosStepScheduler, self).__init__(optimizer, last_epoch)

    def _build_lr(self, start_lr, end_lr, epochs):
        """
        创建学习率列表
        :param start_lr: 开始学习率
        :param end_lr: 终值学习率
        :param epochs: epoch
        :return:
        """
        # 将epochs转换为浮点型数据
        index = np.arange(epochs).astype(np.float32)
        # 更新学习率
        lr = end_lr + (start_lr - end_lr) * (1. + np.cos(index * np.pi/ epochs)) * 0.5
        return lr.astype(np.float32)


class WarmUPScheduler(LRScheduler):
    """
    将不同的学习率更新方式进行连接
    """
    def __init__(self, optimizer, warmup, normal, epochs=50, last_epoch=-1):
        warmup = warmup.lr_spaces # [::-1]
        normal = normal.lr_spaces
        # 将两种更新方式进行连接
        self.lr_spaces = np.concatenate([warmup, normal])
        self.start_lr = normal[0]

        super(WarmUPScheduler, self).__init__(optimizer, last_epoch)

# 学习率更新方式集合
LRs = {
    'log': LogScheduler,
    'step': StepScheduler,
    'multi-step': MultiStepScheduler,
    'linear': LinearStepScheduler,
    'cos': CosStepScheduler}


def _build_lr_scheduler(optimizer, cfg, epochs=50, last_epoch=-1):
    """
    根据配置信息完成学习率更新
    :param optimizer:
    :param cfg:
    :param epochs:
    :param last_epoch:
    :return:
    """
    # 默认为按LOG方式进行更新
    if 'type' not in cfg:
        cfg['type'] = 'log'
    # 若更新方式不在LRs中则返回异常
    if cfg['type'] not in LRs:
        raise Exception('Unknown type of LR Scheduler "%s"'%cfg['type'])
    # 返回学习率结果
    return LRs[cfg['type']](optimizer, last_epoch=last_epoch, epochs=epochs, **cfg)


def _build_warm_up_scheduler(optimizer, cfg, epochs=50, last_epoch=-1):
    """
    根据配置信息，按照warm_up方式完成学习率更新
    :param optimizer:
    :param cfg:
    :param epochs:
    :param last_epoch:
    :return:
    """
    # 获取第一种更新方式的epoch
    warmup_epoch = cfg['warmup']['epoch']
    # 构建学习率更新列表
    # 将学习率增加
    sc1 = _build_lr_scheduler(optimizer, cfg['warmup'], warmup_epoch, last_epoch)
    # 学习率下降
    sc2 = _build_lr_scheduler(optimizer, cfg, epochs - warmup_epoch, last_epoch)
    # 返回连接后的结果
    return WarmUPScheduler(optimizer, sc1, sc2, epochs, last_epoch)


def build_lr_scheduler(optimizer, cfg, epochs=50, last_epoch=-1):
    """
    将上述两种方法进行整合
    :param optimizer:
    :param cfg:
    :param epochs:
    :param last_epoch:
    :return:
    """
    # 若配置信息中含有"warmup"
    if 'warmup' in cfg:
        return _build_warm_up_scheduler(optimizer, cfg, epochs, last_epoch)
    else:
    # 否则
        return _build_lr_scheduler(optimizer, cfg, epochs, last_epoch)


if __name__ == '__main__':

    import torch.nn as nn
    from torch.optim import SGD
    # 模型搭建
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv = nn.Conv2d(10, 10, kernel_size=3)
    # 模型实例化
    net = Net().parameters()
    # 优化器
    optimizer = SGD(net, lr=0.01)

    # step更新机制
    step = {
            'type': 'step',
            'start_lr': 0.01,
            'step': 10,
            'mult': 0.1
            }
    lr = build_lr_scheduler(optimizer, step)
    print("test1")
    print(lr)
    plt.plot(lr.lr_spaces)
    plt.grid()
    plt.title("StepScheduler")
    plt.xlabel("epochs")
    plt.ylabel("lr")
    plt.show()

    # Linear更新机制
    step = {
        'type': 'linear',
        'start_lr': 0.01,
        'end_lr':0.005
    }
    lr = build_lr_scheduler(optimizer, step)
    print("test1")
    print(lr)
    plt.plot(lr.lr_spaces)
    plt.grid()
    plt.title("linear")
    plt.xlabel("epochs")
    plt.ylabel("lr")
    plt.show()

    # log更新机制
    log = {
            'type': 'log',
            'start_lr': 0.03,
            'end_lr': 5e-4,
            }
    lr = build_lr_scheduler(optimizer, log)

    print(lr)
    plt.plot(lr.lr_spaces)
    plt.grid()
    plt.title("logScheduler")
    plt.xlabel("epochs")
    plt.ylabel("lr")
    plt.show()

    # multi-step更新机制
    log = {
            'type': 'multi-step',
            "start_lr": 0.01,
            "mult": 0.1,
            "steps": [10, 15, 20]
            }
    lr = build_lr_scheduler(optimizer, log)
    print(lr)
    plt.plot(lr.lr_spaces)
    plt.grid()
    plt.title("MultiStepScheduler")
    plt.xlabel("epochs")
    plt.ylabel("lr")
    plt.show()

    # cos更新机制
    cos = {
            "type": 'cos',
            'start_lr': 0.01,
            'end_lr': 0.0005,
            }
    lr = build_lr_scheduler(optimizer, cos)
    print(lr)
    plt.plot(lr.lr_spaces)
    plt.grid()
    plt.title("CosScheduler")
    plt.xlabel("epochs")
    plt.ylabel("lr")
    plt.show()

    # warmup更新机制，先上升后下降
    step = {
            'type': 'step',
            'start_lr': 0.001,
            'end_lr': 0.03,
            'step': 1,
            }

    warmup = log.copy()
    warmup['warmup'] = step
    warmup['warmup']['epoch'] = 10
    lr = build_lr_scheduler(optimizer, warmup, epochs=55)
    print(lr)
    plt.plot(lr.lr_spaces)
    plt.grid()
    plt.title("WarmupScheduler")
    plt.xlabel("epochs")
    plt.ylabel("lr")
    plt.show()

    lr.step()
    print(lr.last_epoch)

    lr.step(5)
    print(lr.last_epoch)


