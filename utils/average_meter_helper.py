# --------------------------------------------------------
# 计算和存储指标数据
# --------------------------------------------------------
import numpy as np


class Meter(object):
    "指标数据"
    def __init__(self, name, val, avg):
        # 名称
        self.name = name
        # 值
        self.val = val
        # 平均值
        self.avg = avg

    def __repr__(self):
        return "{name}: {val:.6f} ({avg:.6f})".format(
            name=self.name, val=self.val, avg=self.avg
        )

    def __format__(self, *tuples, **kwargs):
        return self.__repr__()


class AverageMeter(object):
    """计算平均值和当前值并进行存储"""
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        # 重置
        self.val = {}
        self.sum = {}
        self.count = {}

    def update(self, batch=1, **kwargs):
        # 参数更新
        val = {}
        for k in kwargs:
            # 遍历参数
            val[k] = kwargs[k] / float(batch)
        self.val.update(val)
        for k in kwargs:
            # 计算sum和count
            if k not in self.sum:
                self.sum[k] = 0
                self.count[k] = 0
            self.sum[k] += kwargs[k]
            self.count[k] += batch

    def __repr__(self):
        s = ''
        for k in self.sum:
            s += self.format_str(k)
        return s

    def format_str(self, attr):
        # 格式输出
        return "{name}: {val:.6f} ({avg:.6f}) ".format(
                    name=attr,
                    val=float(self.val[attr]),
                    avg=float(self.sum[attr]) / self.count[attr])

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return super(AverageMeter, self).__getattr__(attr)
        if attr not in self.sum:
            # logger.warn("invalid key '{}'".format(attr))
            print("invalid key '{}'".format(attr))
            return Meter(attr, 0, 0)
        return Meter(attr, self.val[attr], self.avg(attr))

    def avg(self, attr):
        return float(self.sum[attr]) / self.count[attr]


class IouMeter(object):
    """计算和存储IOU指标数据"""
    def __init__(self, thrs, sz):
        "初始化"
        self.sz = sz
        self.iou = np.zeros((sz, len(thrs)), dtype=np.float32)
        self.thrs = thrs
        self.reset()

    def reset(self):
        # 重置
        self.iou.fill(0.)
        self.n = 0

    def add(self, output, target):
        '添加交并比'
        if self.n >= len(self.iou):
            return
        target, output = target.squeeze(), output.squeeze()
        # 计算交并比
        for i, thr in enumerate(self.thrs):
            pred = output > thr
            mask_sum = (pred == 1).astype(np.uint8) + (target > 0).astype(np.uint8)
            # 并
            intxn = np.sum(mask_sum == 2)
            # 交
            union = np.sum(mask_sum > 0)
            if union > 0:
                # 交并比
                self.iou[self.n, i] = intxn / union
            elif union == 0 and intxn == 0:
                # 交并比为1
                self.iou[self.n, i] = 1
        self.n += 1

    def value(self, s):
        nb = max(int(np.sum(self.iou > 0)), 1)
        iou = self.iou[:nb]

        def is_number(s):
            "判断是否为数值"
            try:
                float(s)
                return True
            except ValueError:
                return False
        if s == 'mean':
            # 均值
            res = np.mean(iou, axis=0)
        elif s == 'median':
            # 中位数
            res = np.median(iou, axis=0)
        elif is_number(s):
            # 均值
            res = np.sum(iou > float(s), axis=0) / float(nb)
        return res


if __name__ == '__main__':
    avg = AverageMeter()
    avg.update(time=1.1, accuracy=.99)
    avg.update(time=1.0, accuracy=.90)

    print(avg)
    print(avg.sum)
    print(avg.time)
    print(avg.time.avg)
    print(avg.time.val)
    print(avg.SS)



