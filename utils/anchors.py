# --------------------------------------------------------
# anchor处理帮助类
# --------------------------------------------------------
import numpy as np
import math
from utils.bbox_helper import center2corner, corner2center


class Anchors:
    """
    anchors类
    """
    def __init__(self, cfg):
        self.stride = 8  # anchors的范围
        self.ratios = [0.33, 0.5, 1, 2, 3]  # anchors的宽高比
        self.scales = [8]  # anchor的尺度
        self.round_dight = 0  # 兼容python2和python3的数据的舍入
        self.image_center = 0  # 基础锚点的中心在原点
        self.size = 0
        self.anchor_density = 1  # anchor的密度，即每隔几个像素产生锚点

        self.__dict__.update(cfg)

        self.anchor_num = len(self.scales) * len(self.ratios) * (self.anchor_density**2)  # anchor的数目
        self.anchors = None  # 某一像素点的anchor,维度为（anchor_num*4）in single position (anchor_num*4)
        self.all_anchors = None  # 所有像素点的anchor，维度为（2*(4*anchor_num*h*w)）：其中包含两种数据格式的锚点表示方法：[x1, y1, x2, y2]和[cx, cy, w, h]：in all position 2*(4*anchor_num*h*w)
        self.generate_anchors()

    def generate_anchors(self):
        """
        生成anchor
        :return:
        """
        # 生成全零数组存储锚点
        self.anchors = np.zeros((self.anchor_num, 4), dtype=np.float32)
        # 生成anchor的大小
        size = self.stride * self.stride
        count = 0
        # 用检测区域的长度除以步长得到生成anchor的点
        anchors_offset = self.stride / self.anchor_density
        # 计算生成anchor的点相对于原点的偏移
        anchors_offset = np.arange(self.anchor_density)*anchors_offset
        anchors_offset = anchors_offset - np.mean(anchors_offset)
        # 利用meshgrid生成x，y方向的偏移值
        x_offsets, y_offsets = np.meshgrid(anchors_offset, anchors_offset)
        # 遍历生成锚点的点，生成对应的anchor
        for x_offset, y_offset in zip(x_offsets.flatten(), y_offsets.flatten()):
            # 遍历宽高比
            for r in self.ratios:
                # 生成anchor的宽高
                if self.round_dight > 0:
                    ws = round(math.sqrt(size*1. / r), self.round_dight)
                    hs = round(ws * r, self.round_dight)
                else:
                    ws = int(math.sqrt(size*1. / r))
                    hs = int(ws * r)
                # 根据anchor的尺寸生成anchor
                for s in self.scales:
                    w = ws * s
                    h = hs * s
                    self.anchors[count][:] = [-w*0.5+x_offset, -h*0.5+y_offset, w*0.5+x_offset, h*0.5+y_offset][:]
                    count += 1


    def generate_all_anchors(self, im_c, size):
        """
        生成整幅图像的anchors
        :param im_c:图像的中心点
        :param size:图像的尺寸
        :return:
        """
        if self.image_center == im_c and self.size == size:
            return False
        # 更新config中的内容
        self.image_center = im_c
        self.size = size
        # anchor0 的xy 坐标，即 x 和 y 对称。
        a0x = im_c - size // 2 * self.stride
        # 生成anchor0的坐标
        ori = np.array([a0x] * 4, dtype=np.float32)
        # 以图像中心点为中心点的anchor
        zero_anchors = self.anchors + ori
        # 获取anchor0的坐标
        x1 = zero_anchors[:, 0]
        y1 = zero_anchors[:, 1]
        x2 = zero_anchors[:, 2]
        y2 = zero_anchors[:, 3]

        x1, y1, x2, y2 = map(lambda x: x.reshape(self.anchor_num, 1, 1), [x1, y1, x2, y2])
        cx, cy, w, h = corner2center([x1, y1, x2, y2])
        # disp_x是[1, 1, size]，disp_y是[1, size, 1]
        disp_x = np.arange(0, size).reshape(1, 1, -1) * self.stride
        disp_y = np.arange(0, size).reshape(1, -1, 1) * self.stride
        # 得到整幅图像中anchor中心点的坐标
        cx = cx + disp_x
        cy = cy + disp_y

        # 通过广播生成整幅图像的anchor broadcast
        zero = np.zeros((self.anchor_num, size, size), dtype=np.float32)
        cx, cy, w, h = map(lambda x: x + zero, [cx, cy, w, h])
        x1, y1, x2, y2 = center2corner([cx, cy, w, h])
        # 以中心点坐标,宽高和左上角、右下角坐标两种方式存储anchors
        self.all_anchors = np.stack([x1, y1, x2, y2]), np.stack([cx, cy, w, h])
        return True


if __name__ == '__main__':
    anchors = Anchors(cfg={'stride':16, 'anchor_density': 2})
    anchors.generate_all_anchors(im_c=255//2, size=(255-127)//16+1+8)
    print(anchors.all_anchors)
    # a = 1

