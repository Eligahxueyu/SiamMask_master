# --------------------------------------------------------
# 矩形框处理帮助
# --------------------------------------------------------
import numpy as np
from collections import namedtuple
# 定义类型Corner: 左上角坐标和右下角坐标
Corner = namedtuple('Corner', 'x1 y1 x2 y2')
BBox = Corner
# 定义类型Center：中心点坐标和宽高
Center = namedtuple('Center', 'x y w h')


def corner2center(corner):
    """
    左上角右下角坐标转换为中心坐标，宽高
    :param corner: Corner or np.array 4*N
    :return: Center or 4 np.array N
    """
    # 判断输入数据是否为Corner
    if isinstance(corner, Corner):
        # 获取坐标数据
        x1, y1, x2, y2 = corner
        # 计算中心点坐标和宽高
        return Center((x1 + x2) * 0.5, (y1 + y2) * 0.5, (x2 - x1), (y2 - y1))
    else:
        # 获取坐标
        x1, y1, x2, y2 = corner[0], corner[1], corner[2], corner[3]
        # 计算中心点坐标
        x = (x1 + x2) * 0.5
        y = (y1 + y2) * 0.5
        # 计算宽高
        w = x2 - x1
        h = y2 - y1
        return x, y, w, h


def center2corner(center):
    """
    中心坐标，宽高转换为左上角右下角坐标
    :param center: Center or np.array 4*N
    :return: Corner or np.array 4*N
    """
    # 判断数据是否为Center
    if isinstance(center, Center):
        # 获取坐标数据和宽高
        x, y, w, h = center
        # 计算Corner
        return Corner(x - w * 0.5, y - h * 0.5, x + w * 0.5, y + h * 0.5)
    else:
        # 获取数据
        x, y, w, h = center[0], center[1], center[2], center[3]
        # 左上角坐标
        x1 = x - w * 0.5
        y1 = y - h * 0.5
        # 右下角坐标
        x2 = x + w * 0.5
        y2 = y + h * 0.5
        return x1, y1, x2, y2


def cxy_wh_2_rect(pos, sz):
    """
    转换矩形框的表示方式
    :param pos: 矩形框中心点坐标
    :param sz: 矩形框大小：宽高
    :return: 矩形框的左上角坐标，宽，高
    """
    return np.array([pos[0]-sz[0]/2, pos[1]-sz[1]/2, sz[0], sz[1]])  # 0-index


def get_axis_aligned_bbox(region):
    """
    将目标区域其最小外接矩形的形式：中心点坐标和宽，高的形式
    :param region:
    :return:中心点坐标，宽，高
    """
    nv = region.size
    # 若region是四角坐标，可能不平行于图像
    if nv == 8:
        # 计算中心点坐标
        cx = np.mean(region[0::2])
        cy = np.mean(region[1::2])
        # 计算外接矩形的左上角坐标和右下角坐标
        x1 = min(region[0::2])
        x2 = max(region[0::2])
        y1 = min(region[1::2])
        y2 = max(region[1::2])
        # 求L2范数
        # 平行四边形面积
        A1 = np.linalg.norm(region[0:2] - region[2:4]) * np.linalg.norm(region[2:4] - region[4:6])
        # 外接矩形面积
        A2 = (x2 - x1) * (y2 - y1)
        s = np.sqrt(A1 / A2)
        # 求宽和高
        w = s * (x2 - x1) + 1
        h = s * (y2 - y1) + 1
    # region 是左上角坐标和宽高
    else:
        x = region[0]
        y = region[1]
        w = region[2]
        h = region[3]
        # 中心点坐标
        cx = x+w/2
        cy = y+h/2

    return cx, cy, w, h


def aug_apply(bbox, param, shape, inv=False, rd=False):
    """
    对矩形进行增强 apply augmentation
    :param bbox: original bbox in image
    :param param: augmentation param, shift/scale
    :param shape: image shape, h, w, (c)
    :param inv: inverse
    :param rd: round bbox
    :return: bbox(, param)
        bbox: augmented bbox
        param: real augmentation param
    """
    if not inv:
        # 获取中心坐标
        center = corner2center(bbox)
        original_center = center

        real_param = {}
        # 矩形缩放
        if 'scale' in param:
            # 获取缩放比例
            scale_x, scale_y = param['scale']
            imh, imw = shape[:2]
            # 获取宽高
            h, w = center.h, center.w
            # 计算比例
            scale_x = min(scale_x, float(imw) / w)
            scale_y = min(scale_y, float(imh) / h)

            # center.w *= scale_x
            # center.h *= scale_y
            # 计算中心
            center = Center(center.x, center.y, center.w * scale_x, center.h * scale_y)
        # 获取目标框（x1, y1, x2, y2 ）
        bbox = center2corner(center)
        # 矩形平移
        if 'shift' in param:
            tx, ty = param['shift']
            x1, y1, x2, y2 = bbox
            imh, imw = shape[:2]
            # 获取平移距离
            tx = max(-x1, min(imw - 1 - x2, tx))
            ty = max(-y1, min(imh - 1 - y2, ty))

            bbox = Corner(x1 + tx, y1 + ty, x2 + tx, y2 + ty)

        if rd:
            bbox = Corner(*map(round, bbox))

        current_center = corner2center(bbox)
        # 缩放和平移参数
        real_param['scale'] = current_center.w / original_center.w, current_center.h / original_center.h
        real_param['shift'] = current_center.x - original_center.x, current_center.y - original_center.y

        return bbox, real_param
    else:
        # 矩形框缩放
        if 'scale' in param:
            scale_x, scale_y = param['scale']
        else:
            scale_x, scale_y = 1., 1.
        # 平移
        if 'shift' in param:
            tx, ty = param['shift']
        else:
            tx, ty = 0, 0
        # 中心点坐标
        center = corner2center(bbox)
        center = Center(center.x - tx, center.y - ty, center.w / scale_x, center.h / scale_y)
        return center2corner(center)


def IoU(rect1, rect2):
    # 计算IOU
    x1, y1, x2, y2 = rect1[0], rect1[1], rect1[2], rect1[3]
    tx1, ty1, tx2, ty2 = rect2[0], rect2[1], rect2[2], rect2[3]
    # 获取坐标的最大值和最小值
    xx1 = np.maximum(tx1, x1)
    yy1 = np.maximum(ty1, y1)
    xx2 = np.minimum(tx2, x2)
    yy2 = np.minimum(ty2, y2)
    # 获取交的宽高
    ww = np.maximum(0, xx2 - xx1)
    hh = np.maximum(0, yy2 - yy1)
    # rect1的面积
    area = (x2-x1) * (y2-y1)
    # rect2的面积
    target_a = (tx2-tx1) * (ty2 - ty1)
    # 交
    inter = ww * hh
    # 交并比
    overlap = inter / (area + target_a - inter)
    return overlap
