from __future__ import division
import flask
import argparse
import numpy as np
import cv2
from os.path import join, isdir, isfile

from utils.load_helper import load_pretrain
import torch
from utils.config_helper import load_config
from tools.test import *


# 1 创建解析对象
parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')

# 2 添加参数
# 2.1 resume：梗概
parser.add_argument('--resume', default='SiamMask.pth', type=str,
                    metavar='PATH', help='path to latest checkpoint (default: none)')
# 2.2 config配置
parser.add_argument('--config', dest='config', default='config.json',
                    help='hyper-parameter of SiamMask in json format')
# 2.3 处理的图像的序列
parser.add_argument('--base_path', default='../../data/car', help='datasets')
# 2.4 硬件信息
parser.add_argument('--cpu', action='store_true', help='cpu mode')
# 3 解析参数
args = parser.parse_args()


def process_vedio(vedio_path, initRect):
    """
    视频处理
    :param vedio_path：视频路径
    :param initRect: 跟踪目标的初始位置
    :return:
    """

    # 1. 设置设备信息 Setup device
    # 有GPU时选择GPU，否则使用CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 默认优化运行效率
    torch.backends.cudnn.benchmark = True

    # 2. 模型设置 Setup Model
    # 2.1 将命令行参数解析出来
    cfg = load_config(args)

    # 2.2 custom是构建的网络，否则引用model中的网络结构
    from custom import Custom
    siammask = Custom(anchors=cfg['anchors'])
    # 2.3 判断是否存在模型的权重文件
    if args.resume:
        assert isfile(args.resume), 'Please download {} first.'.format(args.resume)
        siammask = load_pretrain(siammask, args.resume)
    # 在运行推断前，需要调用 model.eval() 函数，以将 dropout 层 和 batch normalization 层设置为评估模式(非训练模式).
    # to(device)将张量复制到GPU上，之后的计算将在GPU上运行
    siammask.eval().to(device)

    # 首帧跟踪目标的位置
    x, y, w, h = initRect
    print(x)
    VeryBig = 999999999  # 用于将视频框调整到最大
    Cap = cv2.VideoCapture(vedio_path)  # 设置读取摄像头
    ret, frame = Cap.read()  # 读取帧
    ims = [frame]  # 把frame放入列表格式的frame， 因为原文是将每帧图片放入列表

    im = frame
    f = 0
    target_pos = np.array([x + w / 2, y + h / 2])
    target_sz = np.array([w, h])
    state = siamese_init(im, target_pos, target_sz, siammask, cfg['hp'])  # init tracker"
    middlepath = "../data/middle.mp4"
    outpath = "../data/output.mp4"
    vediowriter = cv2.VideoWriter(middlepath, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), 10, (320, 240))
    while (True):
        tic = cv2.getTickCount()
        ret, im = Cap.read()  # 逐个提取frame
        if (ret == False):
            break;
        state = siamese_track(state, im, mask_enable=True, refine_enable=True)  # track
        location = state['ploygon'].flatten()
        mask = state['mask'] > state['p'].seg_thr
        im[:, :, 2] = (mask > 0) * 255 + (mask == 0) * im[:, :, 2]
        cv2.polylines(im, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
        vediowriter.write(im)
        cv2.imshow('SiamMask', im)
        key = cv2.waitKey(1)
        if key > 0:
            break

        f = f + 1
    vediowriter.release()

    return


if __name__ == '__main__':
    process_vedio('../data/car.mp4', [162, 121, 28, 25])
