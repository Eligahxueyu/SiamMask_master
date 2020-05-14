# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
from __future__ import division
import argparse
import logging
import numpy as np
import cv2
from PIL import Image
from os import makedirs
from os.path import join, isdir, isfile

from utils.log_helper import init_log, add_file_handler
from utils.load_helper import load_pretrain
from utils.bbox_helper import get_axis_aligned_bbox, cxy_wh_2_rect
from utils.benchmark_helper import load_dataset, dataset_zoo

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from utils.anchors import Anchors
from utils.tracker_config import TrackerConfig

from utils.config_helper import load_config
# from utils.pyvotkit.region import vot_overlap, vot_float2str
# 在目标分割中将某一像素作为目标的阈值
thrs = np.arange(0.3, 0.5, 0.05)
# 参数信息配置
parser = argparse.ArgumentParser(description='Test SiamMask')
parser.add_argument('--arch', dest='arch', default='', choices=['Custom',],
                    help='architecture of pretrained model')
parser.add_argument('--config', dest='config', required=True, help='hyper-parameter for SiamMask')
parser.add_argument('--resume', default='', type=str, required=True,
                    metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--mask', action='store_true', help='whether use mask output')
parser.add_argument('--refine', action='store_true', help='whether use mask refine output')
parser.add_argument('--dataset', dest='dataset', default='VOT2018', choices=dataset_zoo,
                    help='datasets')
parser.add_argument('-l', '--log', default="log_test.txt", type=str, help='log file')
parser.add_argument('-v', '--visualization', dest='visualization', action='store_true',
                    help='whether visualize result')
parser.add_argument('--save_mask', action='store_true', help='whether use save mask for davis')
parser.add_argument('--gt', action='store_true', help='whether use gt rect for davis (Oracle)')
parser.add_argument('--video', default='', type=str, help='test special video')
parser.add_argument('--cpu', action='store_true', help='cpu mode')
parser.add_argument('--debug', action='store_true', help='debug mode')


def to_torch(ndarray):
    '''
    将数据转换为TorcH tensor
    :param ndarray: ndarray
    :return: torch中的tensor
    '''
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def im_to_torch(img):
    '''
    将图像转换为torch中的tensor
    :param img: 输入图像
    :return: 输出张量
    '''
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = to_torch(img).float()
    return img


def get_subwindow_tracking(im, pos, model_sz, original_sz, avg_chans, out_mode='torch'):
    """
    获取跟踪目标的信息(图像窗口)
    :param im:跟踪的模板图像
    :param pos:目标位置
    :param model_sz:模型要求输入的目标尺寸
    :param original_sz: 扩展后的目标尺寸
    :param avg_chans:图像的平均值
    :param out_mode: 输出模式
    :return:
    """
    if isinstance(pos, float):
        # 目标中心点坐标
        pos = [pos, pos]
    # 目标的尺寸
    sz = original_sz
    # 图像尺寸
    im_sz = im.shape
    # 扩展背景后边界到中心的距离
    c = (original_sz + 1) / 2
    # 判断目标是否超出图像边界，若超出边界则对图像进行填充
    context_xmin = round(pos[0] - c)
    context_xmax = context_xmin + sz - 1
    context_ymin = round(pos[1] - c)
    context_ymax = context_ymin + sz - 1
    left_pad = int(max(0., -context_xmin))
    top_pad = int(max(0., -context_ymin))
    right_pad = int(max(0., context_xmax - im_sz[1] + 1))
    bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))
    # 图像填充使得图像的原点发生变化，计算填充后图像块的坐标
    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad

    # zzp: a more easy speed version
    r, c, k = im.shape
    # 若进行填充需对目标位置重新赋值
    if any([top_pad, bottom_pad, left_pad, right_pad]):
        # 生成与填充后图像同样大小的全零数组
        te_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad, k), np.uint8)
        # 对原图像区域进行赋值
        te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
        # 将填充区域赋值为图像的均值
        if top_pad:
            te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
        if bottom_pad:
            te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
        if left_pad:
            te_im[:, 0:left_pad, :] = avg_chans
        if right_pad:
            te_im[:, c + left_pad:, :] = avg_chans
        # 根据填充结果修改目标的位置
        im_patch_original = te_im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]
    else:
        im_patch_original = im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]
    # 若跟踪目标块的尺寸与模型输入尺寸不同，则利用opencv修改图像尺寸
    if not np.array_equal(model_sz, original_sz):
        im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))
    else:
        im_patch = im_patch_original
    # cv2.imshow('crop', im_patch)
    # cv2.waitKey(0)
    # 若输出模式是Torch，则将其通道调换，否则直接输出im_patch
    return im_to_torch(im_patch) if out_mode in 'torch' else im_patch


def generate_anchor(cfg, score_size):
    """
    生成锚点：anchor
    :param cfg: anchor的配置信息
    :param score_size:分类的评分结果
    :return:生成的anchor
    """
    # 初始化anchor
    anchors = Anchors(cfg)
    # 得到生成的anchors
    anchor = anchors.anchors
    # 得到每一个anchor的左上角和右下角坐标
    x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
    # 将anchor转换为中心点坐标和宽高的形式
    anchor = np.stack([(x1+x2)*0.5, (y1+y2)*0.5, x2-x1, y2-y1], 1)
    # 获取生成anchor的范围
    total_stride = anchors.stride
    # 获取锚点的个数
    anchor_num = anchor.shape[0]
    # 将对锚点组进行广播，并设置其坐标。
    anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
    # 加上ori偏移后，xx和yy以图像中心为原点
    ori = - (score_size // 2) * total_stride
    xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                         [ori + total_stride * dy for dy in range(score_size)])
    xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
             np.tile(yy.flatten(), (anchor_num, 1)).flatten()
    # 获取anchor
    anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
    return anchor

def siamese_init(im, target_pos, target_sz, model, hp=None, device='cpu'):
    """
    初始化跟踪器，根据目标的信息构建state 字典
    :param im: 当前处理的图像
    :param target_pos: 目标的位置
    :param target_sz: 目标的尺寸
    :param model: 训练好的网络模型
    :param hp: 超参数
    :param device: 硬件信息
    :return: 跟踪器的state字典数据
    """

    # 初始化state字典
    state = dict()
    # 设置图像的宽高
    state['im_h'] = im.shape[0]
    state['im_w'] = im.shape[1]
    # 配置跟踪器的相关参数
    p = TrackerConfig()
    # 对参数进行更新
    p.update(hp, model.anchors)
    # 更新参数
    p.renew()
    # 获取网络模型
    net = model
    # 根据网络参数对跟踪器的参数进行更新，主要是anchors
    p.scales = model.anchors['scales']
    p.ratios = model.anchors['ratios']
    p.anchor_num = model.anchor_num
    # 生成锚点
    p.anchor = generate_anchor(model.anchors, p.score_size)
    # 图像的平均值
    avg_chans = np.mean(im, axis=(0, 1))
    # 根据设置的上下文比例，输入z 的宽高及尺寸
    wc_z = target_sz[0] + p.context_amount * sum(target_sz)
    hc_z = target_sz[1] + p.context_amount * sum(target_sz)
    s_z = round(np.sqrt(wc_z * hc_z))
    # 初始化跟踪目标 initialize the exemplar
    z_crop = get_subwindow_tracking(im, target_pos, p.exemplar_size, s_z, avg_chans)
    # 将其转换为Variable可在pythorch中进行反向传播
    z = Variable(z_crop.unsqueeze(0))
    # 专门处理模板
    net.template(z.to(device))
    # 设置使用的惩罚窗口
    if p.windowing == 'cosine':
        # 利用hanning窗的外积生成cosine窗口
        window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))
    elif p.windowing == 'uniform':
        window = np.ones((p.score_size, p.score_size))
    # 每一个anchor都有一个对应的惩罚窗口
    window = np.tile(window.flatten(), p.anchor_num)
    # 将信息更新到state字典中
    state['p'] = p
    state['net'] = net
    state['avg_chans'] = avg_chans
    state['window'] = window
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    return state


def siamese_track(state, im, mask_enable=False, refine_enable=False, device='cpu', debug=False):
    """
    对目标进行跟踪
    :param state:目标状态
    :param im:跟踪的图像帧
    :param mask_enable:是否进行掩膜
    :param refine_enable:是否进行特征的融合
    :param device:硬件信息
    :param debug: 是否进行debug
    :return:跟踪目标的状态 state字典
    """
    # 获取目标状态
    p = state['p']
    net = state['net']
    avg_chans = state['avg_chans']
    window = state['window']
    target_pos = state['target_pos']
    target_sz = state['target_sz']
    # 包含周边信息的跟踪框的宽度，高度，尺寸
    wc_x = target_sz[1] + p.context_amount * sum(target_sz)
    hc_x = target_sz[0] + p.context_amount * sum(target_sz)
    s_x = np.sqrt(wc_x * hc_x)
    # 模板模型输入框尺寸与跟踪框的比例
    scale_x = p.exemplar_size / s_x
    # 使用与模板分支相同的比例得到检测区域
    d_search = (p.instance_size - p.exemplar_size) / 2
    pad = d_search / scale_x
    s_x = s_x + 2 * pad
    # 对检测框进行扩展，包含周边信息
    crop_box = [target_pos[0] - round(s_x) / 2, target_pos[1] - round(s_x) / 2, round(s_x), round(s_x)]
    # 若进行debug
    if debug:
        # 复制图片
        im_debug = im.copy()
        # 产生crop_box
        crop_box_int = np.int0(crop_box)
        # 将其绘制在图片上
        cv2.rectangle(im_debug, (crop_box_int[0], crop_box_int[1]),
                      (crop_box_int[0] + crop_box_int[2], crop_box_int[1] + crop_box_int[3]), (255, 0, 0), 2)
        # 图片展示
        cv2.imshow('search area', im_debug)
        cv2.waitKey(0)

    # extract scaled crops for search region x at previous target position
    # 将目标位置按比例转换为要跟踪的目标
    x_crop = Variable(get_subwindow_tracking(im, target_pos, p.instance_size, round(s_x), avg_chans).unsqueeze(0))
    # 调用网络进行目标跟踪
    if mask_enable:
        # 进行目标分割
        score, delta, mask = net.track_mask(x_crop.to(device))
    else:
        # 只进行目标追踪，不进行分割
        score, delta = net.track(x_crop.to(device))
    # 目标框回归结果(将其转成4*...的样式)
    delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1).data.cpu().numpy()
    # 目标分类结果（将其转成2*...的样式）
    score = F.softmax(score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0), dim=1).data[:,
            1].cpu().numpy()
    # 计算目标框的中心点坐标，delta[0],delta[1],以及宽delta[2]和高delta[3],这里变量不是很明确。
    delta[0, :] = delta[0, :] * p.anchor[:, 2] + p.anchor[:, 0]
    delta[1, :] = delta[1, :] * p.anchor[:, 3] + p.anchor[:, 1]
    delta[2, :] = np.exp(delta[2, :]) * p.anchor[:, 2]
    delta[3, :] = np.exp(delta[3, :]) * p.anchor[:, 3]

    def change(r):
        """
        将r和1/r逐位比较取最大值
        :param r:
        :return:
        """
        return np.maximum(r, 1. / r)

    def sz(w, h):
        """
        计算等效边长
        :param w: 宽
        :param h: 高
        :return: 等效边长
        """
        pad = (w + h) * 0.5
        sz2 = (w + pad) * (h + pad)
        return np.sqrt(sz2)

    def sz_wh(wh):
        """
        计算等效边长
        :param wh: 宽高的数组
        :return: 等效边长
        """
        pad = (wh[0] + wh[1]) * 0.5
        sz2 = (wh[0] + pad) * (wh[1] + pad)
        return np.sqrt(sz2)

    # 尺寸惩罚 size penalty
    target_sz_in_crop = target_sz*scale_x
    s_c = change(sz(delta[2, :], delta[3, :]) / (sz_wh(target_sz_in_crop)))  # scale penalty
    r_c = change((target_sz_in_crop[0] / target_sz_in_crop[1]) / (delta[2, :] / delta[3, :]))  # ratio penalty
    # p.penalty_k超参数
    penalty = np.exp(-(r_c * s_c - 1) * p.penalty_k)
    # 对分类结果进行惩罚
    pscore = penalty * score

    # cos window (motion model)
    # 窗口惩罚：按一定权值叠加一个窗分布值
    pscore = pscore * (1 - p.window_influence) + window * p.window_influence
    # 获取最优权值的索引
    best_pscore_id = np.argmax(pscore)
    # 将最优的预测结果映射回原图
    pred_in_crop = delta[:, best_pscore_id] / scale_x
    # 计算lr
    lr = penalty[best_pscore_id] * score[best_pscore_id] * p.lr  # lr for OTB
    # 计算目标的位置和尺寸：根据预测偏移得到目标位置和尺寸
    res_x = pred_in_crop[0] + target_pos[0]
    res_y = pred_in_crop[1] + target_pos[1]

    res_w = target_sz[0] * (1 - lr) + pred_in_crop[2] * lr
    res_h = target_sz[1] * (1 - lr) + pred_in_crop[3] * lr
    # 目标的位置和尺寸
    target_pos = np.array([res_x, res_y])
    target_sz = np.array([res_w, res_h])

    # for Mask Branch
    # 若进行分割
    if mask_enable:
        # 获取最优预测结果的位置索引：np.unravel_index：将平面索引或平面索引数组转换为坐标数组的元组
        best_pscore_id_mask = np.unravel_index(best_pscore_id, (5, p.score_size, p.score_size))
        delta_x, delta_y = best_pscore_id_mask[2], best_pscore_id_mask[1]
        # 是否进行特征融合
        if refine_enable:
            # 调用track_refine，运行 Refine 模块，由相关特征图上 1×1×256 的特征向量与检测下采样前的特征图得到目标掩膜
            mask = net.track_refine((delta_y, delta_x)).to(device).sigmoid().squeeze().view(
                p.out_size, p.out_size).cpu().data.numpy()
        else:
            # 不进行融合时直接生成掩膜数据
            mask = mask[0, :, delta_y, delta_x].sigmoid(). \
                squeeze().view(p.out_size, p.out_size).cpu().data.numpy()

        def crop_back(image, bbox, out_sz, padding=-1):
            """
            对图像进行仿射变换
            :param image: 图像
            :param bbox:
            :param out_sz: 输出尺寸
            :param padding: 是否进行扩展
            :return: 仿射变换后的结果
            """
            # 构造变换矩阵
            # 尺度系数
            a = (out_sz[0] - 1) / bbox[2]
            b = (out_sz[1] - 1) / bbox[3]
            # 平移量
            c = -a * bbox[0]
            d = -b * bbox[1]
            mapping = np.array([[a, 0, c],
                                [0, b, d]]).astype(np.float)
            # 进行仿射变换
            crop = cv2.warpAffine(image, mapping, (out_sz[0], out_sz[1]),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=padding)
            return crop
        # 检测区域框长度与输入模型的大小的比值：缩放系数
        s = crop_box[2] / p.instance_size
        # 预测的模板区域框
        sub_box = [crop_box[0] + (delta_x - p.base_size / 2) * p.total_stride * s,
                   crop_box[1] + (delta_y - p.base_size / 2) * p.total_stride * s,
                   s * p.exemplar_size, s * p.exemplar_size]
        # 缩放系数
        s = p.out_size / sub_box[2]
        # 背景框
        back_box = [-sub_box[0] * s, -sub_box[1] * s, state['im_w'] * s, state['im_h'] * s]
        # 仿射变换
        mask_in_img = crop_back(mask, back_box, (state['im_w'], state['im_h']))
        # 得到掩膜结果
        target_mask = (mask_in_img > p.seg_thr).astype(np.uint8)
        # 根据cv2的版本查找轮廓
        if cv2.__version__[-5] == '4':
            # opencv4中返回的参数只有两个，其他版本有四个
            contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        else:
            _, contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # 获取轮廓的面积
        cnt_area = [cv2.contourArea(cnt) for cnt in contours]
        if len(contours) != 0 and np.max(cnt_area) > 100:
            # 获取面积最大的轮廓
            contour = contours[np.argmax(cnt_area)]  # use max area polygon
            # 转换为...*2的形式
            polygon = contour.reshape(-1, 2)
            # pbox = cv2.boundingRect(polygon)  # Min Max Rectangle
            # 得到最小外接矩形后找到该矩形的四个顶点
            prbox = cv2.boxPoints(cv2.minAreaRect(polygon))  # Rotated Rectangle

            # box_in_img = pbox
            # 获得跟踪框
            rbox_in_img = prbox
        else:  # empty mask
            # 根据预测的目标位置和尺寸得到location
            location = cxy_wh_2_rect(target_pos, target_sz)
            # 得到跟踪框的四个顶点
            rbox_in_img = np.array([[location[0], location[1]],
                                    [location[0] + location[2], location[1]],
                                    [location[0] + location[2], location[1] + location[3]],
                                    [location[0], location[1] + location[3]]])
    # 得到目标的位置和尺寸
    target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
    target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
    target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
    target_sz[1] = max(10, min(state['im_h'], target_sz[1]))
    # 更新state对象
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    state['score'] = score[best_pscore_id]
    state['mask'] = mask_in_img if mask_enable else []
    state['ploygon'] = rbox_in_img if mask_enable else []
    return state


def track_vot(model, video, hp=None, mask_enable=False, refine_enable=False, device='cpu'):
    """
    对目标进行追踪
    :param model: 训练好的模型
    :param video: 视频数据
    :param hp: 超参数
    :param mask_enable: 是否生成掩膜，默认为False
    :param refine_enable: 是否使用融合后的模型
    :param device:硬件信息
    :return:目标跟丢次数，fps
    """
    # 记录目标框及其状态
    regions = []  # result and states[1 init / 2 lost / 0 skip]
    # 获取要处理的图像，和真实值groundtruth
    image_files, gt = video['image_files'], video['gt']
    # 设置相关参数：初始帧，终止帧，目标丢失次数，toc
    start_frame, end_frame, lost_times, toc = 0, len(image_files), 0, 0
    # 遍历要处理的图像
    for f, image_file in enumerate(image_files):
        # 读取图像
        im = cv2.imread(image_file)
        tic = cv2.getTickCount()
        # 若为初始帧图像
        if f == start_frame:  # init
            # 获取目标区域的位置：中心点坐标，宽，高
            cx, cy, w, h = get_axis_aligned_bbox(gt[f])
            # 目标位置
            target_pos = np.array([cx, cy])
            # 目标大小
            target_sz = np.array([w, h])
            # 初始化跟踪器
            state = siamese_init(im, target_pos, target_sz, model, hp, device)  # init tracker
            # 将目标框转换为：左上角坐标，宽，高的形式
            location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            # 若数据集是VOT，在regions中添加1，否则添加gt[f]，第一帧目标的真实位置
            regions.append(1 if 'VOT' in args.dataset else gt[f])
        # 非初始帧数据
        elif f > start_frame:  # tracking
            # 进行目标追踪
            state = siamese_track(state, im, mask_enable, refine_enable, device, args.debug)  # track
            # 若进行掩膜处理
            if mask_enable:
                # 将跟踪结果铺展开
                location = state['ploygon'].flatten()
                # 获得掩码
                mask = state['mask']
            # 不进行掩膜处理
            else:
                # 将目标框表示形式转换为：左上角坐标，宽，高的形式
                location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
                # 掩膜为空
                mask = []
            # 如果是VOT数据，计算交叠程度，其他数据默认交叠为1
            if 'VOT' in args.dataset:
                # 目标的真实位置
                gt_polygon = ((gt[f][0], gt[f][1]), (gt[f][2], gt[f][3]),
                              (gt[f][4], gt[f][5]), (gt[f][6], gt[f][7]))
                # 若进行掩膜处理
                if mask_enable:
                    # 预测结果为：
                    pred_polygon = ((location[0], location[1]), (location[2], location[3]),
                                    (location[4], location[5]), (location[6], location[7]))
                # 若不进行掩膜
                else:
                    # 预测结果为：
                    pred_polygon = ((location[0], location[1]),
                                    (location[0] + location[2], location[1]),
                                    (location[0] + location[2], location[1] + location[3]),
                                    (location[0], location[1] + location[3]))
                # 计算两个目标之间的交叠程度
                b_overlap = vot_overlap(gt_polygon, pred_polygon, (im.shape[1], im.shape[0]))
            else:
                b_overlap = 1
            # 如果跟踪框和真实框有交叠，添加跟踪结果中
            if b_overlap:
                regions.append(location)
            # 如果跟丢，则记录跟丢次数，五帧后重新进行目标初始化
            else:  # lost
                regions.append(2)
                lost_times += 1
                start_frame = f + 5  # skip 5 frames
        # 其他帧数据跳过(比如小于初始帧的数据)
        else:  # skip
            regions.append(0)
        # 计算跟踪时间
        toc += cv2.getTickCount() - tic
        # 如果进行显示并且跳过丢失的帧数据
        if args.visualization and f >= start_frame:  # visualization (skip lost frame)
            # 复制原图像的副本
            im_show = im.copy()
            # 如果帧数为0，销毁窗口
            if f == 0: cv2.destroyAllWindows()
            # 标注信息中包含第f帧的结果时：
            if gt.shape[0] > f:
                # 将标准的真实信息绘制在图像上
                if len(gt[f]) == 8:
                    cv2.polylines(im_show, [np.array(gt[f], np.int).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
                else:
                    cv2.rectangle(im_show, (gt[f, 0], gt[f, 1]), (gt[f, 0] + gt[f, 2], gt[f, 1] + gt[f, 3]), (0, 255, 0), 3)
            # 将跟踪结果绘制在图像上
            if len(location) == 8:
                # 若进行掩膜处理，将掩膜结果绘制在图像上
                if mask_enable:
                    mask = mask > state['p'].seg_thr
                    im_show[:, :, 2] = mask * 255 + (1 - mask) * im_show[:, :, 2]
                location_int = np.int0(location)
                cv2.polylines(im_show, [location_int.reshape((-1, 1, 2))], True, (0, 255, 255), 3)
            else:
                location = [int(l) for l in location]
                cv2.rectangle(im_show, (location[0], location[1]),
                              (location[0] + location[2], location[1] + location[3]), (0, 255, 255), 3)
            cv2.putText(im_show, str(f), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(im_show, str(lost_times), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(im_show, str(state['score']) if 'score' in state else '', (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow(video['name'], im_show)
            cv2.waitKey(1)
    toc /= cv2.getTickFrequency()

    # 结果保存到文本文件中 save result
    # 文件夹名称：包括模型结构、mask、refine、resume信息
    name = args.arch.split('.')[0] + '_' + ('mask_' if mask_enable else '') + ('refine_' if refine_enable else '') +\
           args.resume.split('/')[-1].split('.')[0]
    # 如果是VOT数据集
    if 'VOT' in args.dataset:
        # 构建追踪结果存储位置
        video_path = join('test', args.dataset, name,
                          'baseline', video['name'])
        # 若不存在该路径，进行创建
        if not isdir(video_path): makedirs(video_path)
        # 文本文件的路径
        result_path = join(video_path, '{:s}_001.txt'.format(video['name']))
        # 将追踪结果写入文本文件中
        # with open(result_path, "w") as fin:
        #     for x in regions:
        #         fin.write("{:d}\n".format(x)) if isinstance(x, int) else \
        #                 fin.write(','.join([vot_float2str("%.4f", i) for i in x]) + '\n')
    # 如果是OTB数据
    else:  # OTB
        # 构建存储路径
        video_path = join('test', args.dataset, name)
        # 若不存在该路径，进行创建
        if not isdir(video_path): makedirs(video_path)
        # 文本文件的路径
        result_path = join(video_path, '{:s}.txt'.format(video['name']))
        # 将追踪结果写入文本文件中
        with open(result_path, "w") as fin:
            for x in regions:
                fin.write(','.join([str(i) for i in x])+'\n')
    # 将信息写入到log文件中
    logger.info('({:d}) Video: {:12s} Time: {:02.1f}s Speed: {:3.1f}fps Lost: {:d}'.format(
        v_id, video['name'], toc, f / toc, lost_times))
    # 返回结果
    return lost_times, f / toc


def MultiBatchIouMeter(thrs, outputs, targets, start=None, end=None):
    """
    批量计算某个目标在视频（多帧图像）中的IOU
    :param thrs:阈值
    :param outputs:追踪的目标结果
    :param targets:真实的目标结果
    :param start:起止帧
    :param end:终止帧
    :return:某个目标的区域相似度
    """
    # 将追踪结果与真实结果转换为ndarray的形式
    targets = np.array(targets)
    outputs = np.array(outputs)
    # 利用标注信息获取视频的帧数
    num_frame = targets.shape[0]
    # 若未指定初始帧
    if start is None:
        # 根据目标跟踪结果确定目标ids
        object_ids = np.array(list(range(outputs.shape[0]))) + 1
    else:
        # 根据指定初始帧确定目标的ids
        object_ids = [int(id) for id in start]
    # 确定目标个数
    num_object = len(object_ids)
    # 用来存储某一目标的交并比
    res = np.zeros((num_object, len(thrs)), dtype=np.float32)
    # 计算掩膜中的最大值及其所在id(该位置认为是目标的位置)
    output_max_id = np.argmax(outputs, axis=0).astype('uint8')+1
    outputs_max = np.max(outputs, axis=0)
    # 遍历阈值
    for k, thr in enumerate(thrs):
        # 若追踪的max大于阈值， output_thr设为1，否则设为0
        output_thr = outputs_max > thr
        # 遍历追踪的目标
        for j in range(num_object):
            # 得到指定的目标
            target_j = targets == object_ids[j]
            # 确定目标所在的视频帧数
            if start is None:
                start_frame, end_frame = 1, num_frame - 1
            else:
                start_frame, end_frame = start[str(object_ids[j])] + 1, end[str(object_ids[j])] - 1
            # 交并比
            iou = []
            # 遍历帧
            for i in range(start_frame, end_frame):
                # 找到追踪结果为j的位置置为1
                pred = (output_thr[i] * output_max_id[i]) == (j+1)
                # 计算真值和追踪结果的和
                mask_sum = (pred == 1).astype(np.uint8) + (target_j[i] > 0).astype(np.uint8)
                # 计算交
                intxn = np.sum(mask_sum == 2)
                # 计算并
                union = np.sum(mask_sum > 0)
                # 计算交并比
                if union > 0:
                    iou.append(intxn / union)
                elif union == 0 and intxn == 0:
                    iou.append(1)
            # 计算目标j，阈值为k时的平均交并比
            res[j, k] = np.mean(iou)
    return res


def track_vos(model, video, hp=None, mask_enable=False, refine_enable=False, mot_enable=False, device='cpu'):
    """
    对数据进行分割并追踪
    :param model: 训练好的模型
    :param video: 视频数据
    :param hp: 超参数
    :param mask_enable: 是否生成掩膜，默认为False
    :param refine_enable: 是否使用融合后的模型
    :param mot_enable:是否进行多目标追踪
    :param device:硬件信息
    :return:区域相似度（掩膜与真值之间的IOU），fps
    """
    # 要处理的图像序列
    image_files = video['image_files']
    # 标注信息：分割中标注的内容也是图像
    annos = [np.array(Image.open(x)) for x in video['anno_files']]
    # 获取初始帧的标注信息
    if 'anno_init_files' in video:
        annos_init = [np.array(Image.open(x)) for x in video['anno_init_files']]
    else:
        annos_init = [annos[0]]
    # 如不进行多目标跟踪，则把多个实例合并为一个示例后进行跟踪
    if not mot_enable:
        # 将标注信息中大于0的置为1，存为掩膜的形式
        annos = [(anno > 0).astype(np.uint8) for anno in annos]
        annos_init = [(anno_init > 0).astype(np.uint8) for anno_init in annos_init]
    # 统计起始帧图像中的目标id
    if 'start_frame' in video:
        object_ids = [int(id) for id in video['start_frame']]
    else:
        # 若起始帧不存在，则根据初始的标注信息确定目标id
        object_ids = [o_id for o_id in np.unique(annos[0]) if o_id != 0]
        # 若目标idgeshu小于初始帧的标注个数时，说明不进行多目标追踪，合并后的标注信息作为每个目标的标准信息
        if len(object_ids) != len(annos_init):
            annos_init = annos_init*len(object_ids)
    # 统计跟踪目标个数
    object_num = len(object_ids)

    toc = 0
    # 用来存放每一帧图像的掩模信息
    pred_masks = np.zeros((object_num, len(image_files), annos[0].shape[0], annos[0].shape[1]))-1
    # 遍历每一个目标，在起止帧之间进行目标跟踪
    for obj_id, o_id in enumerate(object_ids):
        # 确定起止帧的id
        if 'start_frame' in video:
            start_frame = video['start_frame'][str(o_id)]
            end_frame = video['end_frame'][str(o_id)]
        else:
            start_frame, end_frame = 0, len(image_files)
        # 遍历每一帧图像
        for f, image_file in enumerate(image_files):
            im = cv2.imread(image_file)
            tic = cv2.getTickCount()
            # 若是起始帧图像，进行初始化
            if f == start_frame:  # init
                # 确定目标o_id的掩模
                mask = annos_init[obj_id] == o_id
                # 计算mask垂直边界的最小矩形（矩形与图像的上下边界平行）
                x, y, w, h = cv2.boundingRect((mask).astype(np.uint8))
                # 计算边界矩形的中心坐标
                cx, cy = x + w/2, y + h/2
                # 目标位置：矩形中心
                target_pos = np.array([cx, cy])
                # 目标尺寸：矩形尺寸
                target_sz = np.array([w, h])
                # 初始化跟踪器
                state = siamese_init(im, target_pos, target_sz, model, hp, device=device)  # init tracker
            # 若非起始帧图像，则执行跟踪操作
            elif end_frame >= f > start_frame:  # tracking
                # 目标跟踪
                state = siamese_track(state, im, mask_enable, refine_enable, device=device)  # track
                # 某一帧图像掩膜信息
                mask = state['mask']
            toc += cv2.getTickCount() - tic
            # 所有帧图像，更新掩膜信息
            if end_frame >= f >= start_frame:
                # 更新所有帧图像的某一目标的掩膜
                pred_masks[obj_id, f, :, :] = mask
    toc /= cv2.getTickFrequency()
    # 若标注信息与测试图像长度一致，计算区域相似度
    if len(annos) == len(image_files):
        # 批量计算IOU
        multi_mean_iou = MultiBatchIouMeter(thrs, pred_masks, annos,
                                            start=video['start_frame'] if 'start_frame' in video else None,
                                            end=video['end_frame'] if 'end_frame' in video else None)
        # 将每一目标的IOU写入到日志文件中
        for i in range(object_num):
            for j, thr in enumerate(thrs):
                logger.info('Fusion Multi Object{:20s} IOU at {:.2f}: {:.4f}'.format(video['name'] + '_' + str(i + 1), thr,
                                                                           multi_mean_iou[i, j]))
    else:
        multi_mean_iou = []
    # 保存掩膜
    if args.save_mask:
        video_path = join('test', args.dataset, 'SiamMask', video['name'])
        if not isdir(video_path): makedirs(video_path)
        pred_mask_final = np.array(pred_masks)
        pred_mask_final = (np.argmax(pred_mask_final, axis=0).astype('uint8') + 1) * (
                np.max(pred_mask_final, axis=0) > state['p'].seg_thr).astype('uint8')
        for i in range(pred_mask_final.shape[0]):
            cv2.imwrite(join(video_path, image_files[i].split('/')[-1].split('.')[0] + '.png'), pred_mask_final[i].astype(np.uint8))
    # 显示，因为是全部处理完成后进行显示，会有卡顿
    if args.visualization:
        pred_mask_final = np.array(pred_masks)
        pred_mask_final = (np.argmax(pred_mask_final, axis=0).astype('uint8') + 1) * (
                np.max(pred_mask_final, axis=0) > state['p'].seg_thr).astype('uint8')
        COLORS = np.random.randint(128, 255, size=(object_num, 3), dtype="uint8")
        COLORS = np.vstack([[0, 0, 0], COLORS]).astype("uint8")
        mask = COLORS[pred_mask_final]
        for f, image_file in enumerate(image_files):
            output = ((0.4 * cv2.imread(image_file)) + (0.6 * mask[f,:,:,:])).astype("uint8")
            cv2.imshow("mask", output)
            cv2.waitKey(1)

    logger.info('({:d}) Video: {:12s} Time: {:02.1f}s Speed: {:3.1f}fps'.format(
        v_id, video['name'], toc, f*len(object_ids) / toc))

    return multi_mean_iou, f*len(object_ids) / toc


def main():
    # 获取命令行参数信息
    global args, logger, v_id
    args = parser.parse_args()
    # 获取配置文件中配置信息：主要包括网络结构，超参数等
    cfg = load_config(args)
    # 初始化logxi信息，并将日志信息输入到磁盘文件中
    init_log('global', logging.INFO)
    if args.log != "":
        add_file_handler('global', args.log, logging.INFO)
    # 将相关的配置信息输入到日志文件中
    logger = logging.getLogger('global')
    logger.info(args)

    # setup model
    # 加载网络模型架构
    if args.arch == 'Custom':
        from custom import Custom
        model = Custom(anchors=cfg['anchors'])
    else:
        parser.error('invalid architecture: {}'.format(args.arch))
    # 加载网络模型参数
    if args.resume:
        assert isfile(args.resume), '{} is not a valid file'.format(args.resume)
        model = load_pretrain(model, args.resume)
    # 使用评估模式，将drop等激活
    model.eval()
    # 硬件信息
    device = torch.device('cuda' if (torch.cuda.is_available() and not args.cpu) else 'cpu')
    model = model.to(device)
    # 加载数据集 setup dataset
    dataset = load_dataset(args.dataset)

    # 这三种数据支持掩膜 VOS or VOT?
    if args.dataset in ['DAVIS2016', 'DAVIS2017', 'ytb_vos'] and args.mask:
        vos_enable = True  # enable Mask output
    else:
        vos_enable = False

    total_lost = 0  # VOT
    iou_lists = []  # VOS
    speed_list = []
    # 对数据进行处理
    for v_id, video in enumerate(dataset.keys(), start=1):
        if args.video != '' and video != args.video:
            continue
        # true 调用track_vos
        if vos_enable:
            # 如测试数据是['DAVIS2017', 'ytb_vos']时，会开启多目标跟踪
            iou_list, speed = track_vos(model, dataset[video], cfg['hp'] if 'hp' in cfg.keys() else None,
                                 args.mask, args.refine, args.dataset in ['DAVIS2017', 'ytb_vos'], device=device)
            iou_lists.append(iou_list)
        # False 调用track_vot
        else:
            lost, speed = track_vot(model, dataset[video], cfg['hp'] if 'hp' in cfg.keys() else None,
                             args.mask, args.refine, device=device)
            total_lost += lost
        speed_list.append(speed)

    # report final result
    if vos_enable:
        for thr, iou in zip(thrs, np.mean(np.concatenate(iou_lists), axis=0)):
            logger.info('Segmentation Threshold {:.2f} mIoU: {:.3f}'.format(thr, iou))
    else:
        logger.info('Total Lost: {:d}'.format(total_lost))

    logger.info('Mean Speed: {:.2f} FPS'.format(np.mean(speed_list)))


if __name__ == '__main__':
    main()
