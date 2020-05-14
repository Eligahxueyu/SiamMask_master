# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# 数据集的处理
# --------------------------------------------------------
from __future__ import division
from torch.utils.data import Dataset
import numpy as np
import json
import random
import logging
from os.path import join
from utils.bbox_helper import *
from utils.anchors import Anchors
import math
import sys
pyv = sys.version[0]
import cv2
if pyv[0] == '3':
    cv2.ocl.setUseOpenCL(False)

logger = logging.getLogger('global')


sample_random = random.Random()
sample_random.seed(123456)


class SubDataSet(object):

    def __init__(self, cfg):
        for string in ['root', 'anno']:
            if string not in cfg:
                raise Exception('SubDataSet need "{}"'.format(string))
        # 打开标注文件
        with open(cfg['anno']) as fin:
            logger.info("loading " + cfg['anno'])
            # 获取标注数据
            self.labels = self.filter_zero(json.load(fin), cfg)
            # 判断参数是否为数值
            def isint(x):
                try:
                    int(x)
                    return True
                except:
                    return False

            # add frames args into labels 将图像的描述信息加入到label标注中
            to_del = []
            # 遍历标注数据中的vedio
            for video in self.labels:
                # 遍历video中跟踪器
                for track in self.labels[video]:
                    # 得到跟踪目标的帧图像
                    frames = self.labels[video][track]
                    frames = list(map(int, filter(lambda x: isint(x), frames.keys())))
                    # 排序
                    frames.sort()
                    # 填充到labels中
                    self.labels[video][track]['frames'] = frames
                    # 若得到的帧数小于0，则进行删除
                    if len(frames) <= 0:
                        logger.info("warning {}/{} has no frames.".format(video, track))
                        to_del.append((video, track))

            # 删除不在video中的图像 delete tracks with no frames
            for video, track in to_del:
                del self.labels[video][track]

            # （ delete videos with no valid track）
            to_del = []
            # 判断video中是否包含跟踪目标的图像
            for video in self.labels:
                if len(self.labels[video]) <= 0:
                    logger.info("warning {} has no tracks".format(video))
                    to_del.append(video)
            # 删除不包含跟踪目标的video
            for video in to_del:
                del self.labels[video]
            # 将包含标注信息的video赋值给videos
            self.videos = list(self.labels.keys())

            logger.info(cfg['anno'] + " loaded.")

        # 默认参数 default args
        self.root = "/"
        self.start = 0
        # 标注数量
        self.num = len(self.labels)
        self.num_use = self.num
        self.frame_range = 100
        self.mark = "vid"
        # 图片格式
        self.path_format = "{}.{}.{}.jpg"
        # 掩膜格式
        self.mask_format = "{}.{}.m.png"
        self.pick = []
        # input args
        self.__dict__.update(cfg)
        # 是否包含掩膜
        self.has_mask = self.mark in ['coco', 'ytb_vos']
        self.num_use = int(self.num_use)
        # 打乱shuffle
        self.shuffle()

    def filter_zero(self, anno, cfg):
        "过滤跟踪框为0的标注数据"
        name = cfg.get('mark', '')
        # 过滤为0的跟踪框后的结果
        out = {}
        # 记录跟踪框的个数
        tot = 0
        # 记录跟踪框不为0的个数
        new = 0
        # 记录跟踪框为0的个数
        zero = 0
        # 遍历标注数据
        for video, tracks in anno.items():
            new_tracks = {}
            # 遍历跟踪数据
            for trk, frames in tracks.items():
                new_frames = {}
                # 遍历帧数据得到跟踪框
                for frm, bbox in frames.items():
                    tot += 1
                    # bbox的表示方式，获取其宽高
                    if len(bbox) == 4:
                        x1, y1, x2, y2 = bbox
                        w, h = x2 - x1, y2 - y1
                    else:
                        w, h = bbox
                    # 判断跟踪框bbox是否为0
                    if w == 0 or h == 0:
                        logger.info('Error, {name} {video} {trk} {bbox}'.format(**locals()))
                        zero += 1
                        continue
                    new += 1
                    new_frames[frm] = bbox

                if len(new_frames) > 0:
                    new_tracks[trk] = new_frames
            # 记录跟踪框不为0的标注数据
            if len(new_tracks) > 0:
                out[video] = new_tracks
        return out

    def log(self):
        "日志信息"
        logger.info('SubDataSet {name} start-index {start} select [{select}/{num}] path {format}'.format(
            name=self.mark, start=self.start, select=self.num_use, num=self.num, format=self.path_format
        ))

    def shuffle(self):
        "打乱数据"
        # 要进行打乱的列表
        lists = list(range(self.start, self.start + self.num))
        m = 0
        pick = []
        # 对数据进行分段打乱
        while m < self.num_use:
            sample_random.shuffle(lists)
            pick += lists
            m += self.num
        # 数据打乱后的结果
        self.pick = pick[:self.num_use]
        return self.pick

    def get_image_anno(self, video, track, frame):
        "获取图像及标注信息"
        frame = "{:06d}".format(frame)
        # 图像路径
        image_path = join(self.root, video, self.path_format.format(frame, track, 'x'))
        # 图像标注
        image_anno = self.labels[video][track][frame]
        # 掩膜路径
        mask_path = join(self.root, video, self.mask_format.format(frame, track))
        return image_path, image_anno, mask_path

    def get_positive_pair(self, index):
        "正样本"
        # 获取video，track的相关信息
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = random.choice(list(video.keys()))
        track_info = video[track]
        frames = track_info['frames']
        # 存在hard
        if 'hard' not in track_info:
            template_frame = random.randint(0, len(frames)-1)
            # 获取搜索范围
            left = max(template_frame - self.frame_range, 0)
            right = min(template_frame + self.frame_range, len(frames)-1) + 1
            search_range = frames[left:right]
            # 获取模板和搜索图像
            template_frame = frames[template_frame]
            search_frame = random.choice(search_range)
        else:
            # 搜索图像的帧数
            search_frame = random.choice(track_info['hard'])
            # 范围
            left = max(search_frame - self.frame_range, 0)
            right = min(search_frame + self.frame_range, len(frames)-1) + 1  # python [left:right+1) = [left:right]
            template_range = frames[left:right]
            # 获取模板和搜索图像
            template_frame = random.choice(template_range)
            search_frame = frames[search_frame]
        # 返回模板帧和图像帧
        return self.get_image_anno(video_name, track, template_frame), \
               self.get_image_anno(video_name, track, search_frame)

    def get_random_target(self, index=-1):
        "获取任意目标"
        if index == -1:
            index = random.randint(0, self.num-1)
        # 获取videoh和track信息
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = random.choice(list(video.keys()))
        track_info = video[track]
        # 获取图像帧数
        frames = track_info['frames']
        frame = random.choice(frames)
        # 返回图像帧
        return self.get_image_anno(video_name, track, frame)


def crop_hwc(image, bbox, out_sz, padding=(0, 0, 0)):
    """
    对图像进行裁剪
    :param image: 要裁剪的图像
    :param bbox:
    :param out_sz:
    :param padding:
    :return:
    """
    # 将图像bbox进行裁剪
    bbox = [float(x) for x in bbox]
    a = (out_sz-1) / (bbox[2]-bbox[0])
    b = (out_sz-1) / (bbox[3]-bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float)
    # 反射变换
    crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    return crop

"""
图像增强
"""
class Augmentation:
    def __init__(self, cfg):
        # 默认参数 default args
        self.shift = 0
        # 尺度变化
        self.scale = 0
        self.blur = 0  # False
        self.resize = False
        self.rgbVar = np.array([[-0.55919361,  0.98062831, - 0.41940627],
            [1.72091413,  0.19879334, - 1.82968581],
            [4.64467907,  4.73710203, 4.88324118]], dtype=np.float32)
        self.flip = 0
        # # 本证向量
        # self.eig_vec = np.array([
        #     [0.4009, 0.7192, -0.5675],
        #     [-0.8140, -0.0045, -0.5808],
        #     [0.4203, -0.6948, -0.5836],
        # ], dtype=np.float32)
        # # 本征值
        # self.eig_val = np.array([[0.2175, 0.0188, 0.0045]], np.float32)
        self.__dict__.update(cfg)

    @staticmethod
    def random():
        return random.random() * 2 - 1.0
    # 图像模糊
    def blur_image(self, image):
        # 核函数
        def rand_kernel():
            size = np.random.randn(1)
            size = int(np.round(size)) * 2 + 1
            if size < 0: return None
            if random.random() < 0.5: return None
            size = min(size, 45)
            kernel = np.zeros((size, size))
            c = int(size/2)
            # 随机数
            wx = random.random()
            # 生成核函数
            kernel[:, c] += 1. / size * wx
            kernel[c, :] += 1. / size * (1-wx)
            return kernel

        kernel = rand_kernel()
        # 图像模糊
        if kernel is not None:
            image = cv2.filter2D(image, -1, kernel)
        return image

    def __call__(self, image, bbox, size, gray=False, mask=None):
        # 灰度图像
        if gray:
            # 将彩图转换为灰度图
            grayed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # 生成三个通道都是灰度值的彩图
            image = np.zeros((grayed.shape[0], grayed.shape[1], 3), np.uint8)
            image[:, :, 0] = image[:, :, 1] = image[:, :, 2] = grayed
        # 图像尺寸
        shape = image.shape
        # 在图像中心生成bbox
        crop_bbox = center2corner((shape[0]//2, shape[1]//2, size-1, size-1))

        param = {}
        # 图像平移
        if self.shift:
            param['shift'] = (Augmentation.random() * self.shift, Augmentation.random() * self.shift)
        # 图像尺度变换
        if self.scale:
            param['scale'] = ((1.0 + Augmentation.random() * self.scale), (1.0 + Augmentation.random() * self.scale))
        # 矩形框也要相应的进行平移和尺度变换
        crop_bbox, _ = aug_apply(Corner(*crop_bbox), param, shape)
        # 获取bbox的左上角坐标
        x1 = crop_bbox.x1
        y1 = crop_bbox.y1
        # 获取bbox
        bbox = BBox(bbox.x1 - x1, bbox.y1 - y1,
                    bbox.x2 - x1, bbox.y2 - y1)
        # bbox尺度变换
        if self.scale:
            scale_x, scale_y = param['scale']
            bbox = Corner(bbox.x1 / scale_x, bbox.y1 / scale_y, bbox.x2 / scale_x, bbox.y2 / scale_y)
        # 裁剪图像
        image = crop_hwc(image, crop_bbox, size)
        if not mask is None:
            # mask也要进行裁剪
            mask = crop_hwc(mask, crop_bbox, size)
        # 偏置
        offset = np.dot(self.rgbVar, np.random.randn(3, 1))
        offset = offset[::-1]  # bgr 2 rgb
        offset = offset.reshape(3)
        # 图像减去偏置
        image = image - offset
        # 图像模糊
        if self.blur > random.random():
            image = self.blur_image(image)
        # 图像缩放
        if self.resize:
            imageSize = image.shape[:2]
            # 缩放比例
            ratio = max(math.pow(random.random(), 0.5), 0.2)  # 25 ~ 255
            rand_size = (int(round(ratio*imageSize[0])), int(round(ratio*imageSize[1])))
            # 缩放
            image = cv2.resize(image, rand_size)
            # 缩放
            image = cv2.resize(image, tuple(imageSize))
        # 图像翻转
        if self.flip and self.flip > Augmentation.random():
            # 图像翻转
            image = cv2.flip(image, 1)
            # 掩膜翻转
            mask = cv2.flip(mask, 1)
            width = image.shape[1]
            # bbox进行变换
            bbox = Corner(width - 1 - bbox.x2, bbox.y1, width - 1 - bbox.x1, bbox.y2)

        return image, bbox, mask


class AnchorTargetLayer:
    # 计算anchor的类别，RPN的返回结果，IOU
    def __init__(self, cfg):
        # 设置阈值
        self.thr_high = 0.6
        self.thr_low = 0.3
        # 正负样本
        self.negative = 16
        self.rpn_batch = 64
        self.positive = 16

        self.__dict__.update(cfg)

    def __call__(self, anchor, target, size, neg=False, need_iou=False):
        # 获取anchor的数目
        anchor_num = anchor.anchors.shape[0]
        # anchor的类别
        cls = np.zeros((anchor_num, size, size), dtype=np.int64)
        cls[...] = -1  # -1 ignore 0 negative 1 positive
        # rpn的输出结果
        delta = np.zeros((4, anchor_num, size, size), dtype=np.float32)
        delta_weight = np.zeros((anchor_num, size, size), dtype=np.float32)
        # 计算值的集合和数量
        def select(position, keep_num=16):
            num = position[0].shape[0]
            if num <= keep_num:
                return position, num
            slt = np.arange(num)
            np.random.shuffle(slt)
            slt = slt[:keep_num]
            return tuple(p[slt] for p in position), keep_num
        # 负样本
        if neg:
            # 获取左右位置
            l = size // 2 - 3
            r = size // 2 + 3 + 1
            # 将cls置为0
            cls[:, l:r, l:r] = 0
            # 获取样本的索引和总数
            neg, neg_num = select(np.where(cls == 0), self.negative)
            cls[:] = -1
            cls[neg] = 0
            # 返回结果
            if not need_iou:
                # 分类结果，rpn预测的位置，惩罚系数
                return cls, delta, delta_weight
            else:
                overlap = np.zeros((anchor_num, size, size), dtype=np.float32)
                return cls, delta, delta_weight, overlap
        # bbox表示方式转换
        tcx, tcy, tw, th = corner2center(target)
        # 获取anchor的坐标
        anchor_box = anchor.all_anchors[0]
        anchor_center = anchor.all_anchors[1]
        x1, y1, x2, y2 = anchor_box[0], anchor_box[1], anchor_box[2], anchor_box[3]
        cx, cy, w, h = anchor_center[0], anchor_center[1], anchor_center[2], anchor_center[3]

        # delta RPN网络预测的位置
        delta[0] = (tcx - cx) / w
        delta[1] = (tcy - cy) / h
        delta[2] = np.log(tw / w)
        delta[3] = np.log(th / h)

        # IoU 交并比
        overlap = IoU([x1, y1, x2, y2], target)
        # 获取正负样本
        pos = np.where(overlap > self.thr_high)
        neg = np.where(overlap < self.thr_low)
        # 获取正负样本及其索引
        pos, pos_num = select(pos, self.positive)
        neg, neg_num = select(neg, self.rpn_batch - pos_num)
        # 正样本置为1
        cls[pos] = 1
        # 设置惩罚系数
        delta_weight[pos] = 1. / (pos_num + 1e-6)
        # 负样本为0
        cls[neg] = 0
        # 返回结果
        if not need_iou:
            # 分类结果，rpn预测的位置，惩罚系数
            return cls, delta, delta_weight
        else:
            return cls, delta, delta_weight, overlap

"""
获取数据集
"""
class DataSets(Dataset):
    def __init__(self, cfg, anchor_cfg, num_epoch=1):
        super(DataSets, self).__init__()
        global logger
        logger = logging.getLogger('global')

        # anchors
        self.anchors = Anchors(anchor_cfg)
        # size
        self.template_size = 127
        self.origin_size = 127
        self.search_size = 255
        self.size = 17
        self.base_size = 0
        self.crop_size = 0
        # 根据配置文件更新参数
        if 'template_size' in cfg:
            self.template_size = cfg['template_size']
        if 'origin_size' in cfg:
            self.origin_size = cfg['origin_size']
        if 'search_size' in cfg:
            self.search_size = cfg['search_size']
        if 'base_size' in cfg:
            self.base_size = cfg['base_size']
        if 'size' in cfg:
            self.size = cfg['size']

        if (self.search_size - self.template_size) / self.anchors.stride + 1 + self.base_size != self.size:
            raise Exception("size not match!")  # TODO: calculate size online
        if 'crop_size' in cfg:
            self.crop_size = cfg['crop_size']
        self.template_small = False
        if 'template_small' in cfg and cfg['template_small']:
            self.template_small = True
        # 生成anchor
        self.anchors.generate_all_anchors(im_c=self.search_size//2, size=self.size)
        if 'anchor_target' not in cfg:
            cfg['anchor_target'] = {}
        # 生成anchor的信息：cls,reg,mask
        self.anchor_target = AnchorTargetLayer(cfg['anchor_target'])

        # data sets
        if 'datasets' not in cfg:
            raise(Exception('DataSet need "{}"'.format('datasets')))

        self.all_data = []
        start = 0
        self.num = 0
        for name in cfg['datasets']:
            dataset = cfg['datasets'][name]
            dataset['mark'] = name
            dataset['start'] = start
            # 加载数据
            dataset = SubDataSet(dataset)
            dataset.log()
            self.all_data.append(dataset)
            # 数据数量
            start += dataset.num  # real video number
            # 打乱的数据数量
            self.num += dataset.num_use  # the number used for subset shuffle

        # 数据增强data augmentation
        aug_cfg = cfg['augmentation']
        self.template_aug = Augmentation(aug_cfg['template'])
        self.search_aug = Augmentation(aug_cfg['search'])
        self.gray = aug_cfg['gray']
        self.neg = aug_cfg['neg']
        self.inner_neg = 0 if 'inner_neg' not in aug_cfg else aug_cfg['inner_neg']

        self.pick = None  # list to save id for each img
        if 'num' in cfg:  # number used in training for all dataset
            self.num = int(cfg['num'])
        self.num *= num_epoch
        self.shuffle()

        self.infos = {
                'template': self.template_size,
                'search': self.search_size,
                'template_small': self.template_small,
                'gray': self.gray,
                'neg': self.neg,
                'inner_neg': self.inner_neg,
                'crop_size': self.crop_size,
                'anchor_target': self.anchor_target.__dict__,
                'num': self.num // num_epoch
                }
        logger.info('dataset informations: \n{}'.format(json.dumps(self.infos, indent=4)))

    def imread(self, path):
        # 数据读取
        img = cv2.imread(path)
        if self.origin_size == self.template_size:
            # 返回图像
            return img, 1.0

        def map_size(exe, size):
            return int(round(((exe + 1) / (self.origin_size + 1) * (size+1) - 1)))
        # 尺寸调整
        nsize = map_size(self.template_size, img.shape[1])
        # 调整图像大小
        img = cv2.resize(img, (nsize, nsize))
        # 返回图像和缩放比例
        return img, nsize / img.shape[1]

    def shuffle(self):
        "打乱"
        pick = []
        m = 0
        # 获取数据
        while m < self.num:
            p = []
            for subset in self.all_data:
                sub_p = subset.shuffle()
                p += sub_p
            # 打乱数据
            sample_random.shuffle(p)
            # 将打乱的结果进行拼接
            pick += p
            m = len(pick)
        # 将打乱的结果赋值给pick
        self.pick = pick
        logger.info("shuffle done!")
        logger.info("dataset length {}".format(self.num))

    def __len__(self):
        return self.num

    def find_dataset(self, index):
        "查找数据"
        for dataset in self.all_data:
            if dataset.start + dataset.num > index:
                # 返回索引范围内的数据
                return dataset, index - dataset.start

    def __getitem__(self, index, debug=False):
        # 在打乱的结果中找到索引
        index = self.pick[index]
        # 查找得到数据
        dataset, index = self.find_dataset(index)
        # 灰度图
        gray = self.gray and self.gray > random.random()
        # 负样本
        neg = self.neg and self.neg > random.random()
        # 负样本
        if neg:
            # 获取template
            template = dataset.get_random_target(index)
            # 根据设置，从数据生成负样本或随机选择负样本
            if self.inner_neg and self.inner_neg > random.random():
                search = dataset.get_random_target()
            else:
                search = random.choice(self.all_data).get_random_target()
        else:
            # 获得正样本对
            template, search = dataset.get_positive_pair(index)
        # 裁剪图像的中央大小为size的部分
        def center_crop(img, size):
            # 获取图像的形状
            shape = img.shape[1]
            # 若为size,则直接返回
            if shape == size: return img
            # 否则，裁剪中央位置为size大小的图像
            c = shape // 2
            l = c - size // 2
            r = c + size // 2 + 1
            return img[l:r, l:r]
        # 读取模板图像
        template_image, scale_z = self.imread(template[0])
        # 若设置为小模板时，则从模板图像中进行裁剪
        if self.template_small:
            template_image = center_crop(template_image, self.template_size)
        # 读取待搜索图像
        search_image, scale_x = self.imread(search[0])
        # 若存在掩膜并且不是负样本数据
        if dataset.has_mask and not neg:
            # 读取掩膜数据
            search_mask = (cv2.imread(search[2], 0) > 0).astype(np.float32)
        else:
            # 掩膜数据用全零数组替代
            search_mask = np.zeros(search_image.shape[:2], dtype=np.float32)
        # 若裁剪size大于0，对搜索图像和掩膜进行裁剪
        if self.crop_size > 0:
            search_image = center_crop(search_image, self.crop_size)
            search_mask = center_crop(search_mask, self.crop_size)
        # 根据图像大小生成bbox，shape是模板图像中bbox的形状
        def toBBox(image, shape):
            # 图像的大小
            imh, imw = image.shape[:2]
            # 获取shape的宽高
            if len(shape) == 4:
                w, h = shape[2]-shape[0], shape[3]-shape[1]
            else:
                w, h = shape
            # 扩展比例
            context_amount = 0.5
            # 模板尺寸
            exemplar_size = self.template_size  # 127
            # 获取宽高
            wc_z = w + context_amount * (w+h)
            hc_z = h + context_amount * (w+h)
            # 等效边长
            s_z = np.sqrt(wc_z * hc_z)
            # 比例
            scale_z = exemplar_size / s_z
            # 宽高
            w = w*scale_z
            h = h*scale_z
            # 中心点坐标
            cx, cy = imw//2, imh//2
            bbox = center2corner(Center(cx, cy, w, h))
            return bbox
        # 生成模板图像和待搜索图像中的bbox
        template_box = toBBox(template_image, template[1])
        search_box = toBBox(search_image, search[1])
        # 模板数据增强
        template, _, _ = self.template_aug(template_image, template_box, self.template_size, gray=gray)
        # 待搜索图像的数据增强
        search, bbox, mask = self.search_aug(search_image, search_box, self.search_size, gray=gray, mask=search_mask)

        # def draw(image, box, name):
        #     image = image.copy()
        #     x1, y1, x2, y2 = map(lambda x: int(round(x)), box)
        #     cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0))
        #     cv2.imwrite(name, image)
        #
        # if debug:
        #     draw(template_image, template_box, "debug/{:06d}_ot.jpg".format(index))
        #     draw(search_image, search_box, "debug/{:06d}_os.jpg".format(index))
        #     draw(template, _, "debug/{:06d}_t.jpg".format(index))
        #     draw(search, bbox, "debug/{:06d}_s.jpg".format(index))
        # 生成anchor对应的信息
        cls, delta, delta_weight = self.anchor_target(self.anchors, bbox, self.size, neg)
        if dataset.has_mask and not neg:
            # 掩膜图像
            mask_weight = cls.max(axis=0, keepdims=True)
        else:
            mask_weight = np.zeros([1, cls.shape[1], cls.shape[2]], dtype=np.float32)
        # 模板和搜索图像
        template, search = map(lambda x: np.transpose(x, (2, 0, 1)).astype(np.float32), [template, search])
        # 掩膜结果
        mask = (np.expand_dims(mask, axis=0) > 0.5) * 2 - 1  # 1*H*W
        # 返回结果
        return template, search, cls, delta, delta_weight, np.array(bbox, np.float32), \
               np.array(mask, np.float32), np.array(mask_weight, np.float32)

