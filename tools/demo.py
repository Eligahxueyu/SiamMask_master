# --------------------------------------------------------
# demo.py
# --------------------------------------------------------
import io
import json
import flask
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from tools.test import *
import glob

# 1 创建解析对象
parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')

# 2 添加参数
# 2.1 resume：梗概
parser.add_argument('--resume', default='SiamMask.pth', type=str,
                    metavar='PATH',help='path to latest checkpoint (default: none)')
# 2.2 config配置
parser.add_argument('--config', dest='config', default='config.json',
                    help='hyper-parameter of SiamMask in json format')
# 2.3 处理的图像的序列
parser.add_argument('--base_path', default='../data/car', help='datasets')
# 2.4 硬件信息
parser.add_argument('--cpu', action='store_true', help='cpu mode')
# 3 解析参数
args = parser.parse_args()

writer = None

if __name__ == '__main__':
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

    # 3. 读取图片序列 Parse Image file
    img_files = sorted(glob.glob(join(args.base_path, '*.jp*')))
    ims = [cv2.imread(imf) for imf in img_files]

    # 4. 选择目标区域 Select ROI
    cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty("SiamMask", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # 5. 将目标框转换为矩形左上角坐标，宽 高的形式
    try:
        init_rect = cv2.selectROI('SiamMask', ims[0], False, False)
        x, y, w, h = init_rect
        print(x,y,w,h)
    except:
        exit()

    toc = 0
    # 6. 遍历所有的图片
    for f, im in enumerate(ims):
        # 用于记时：初始的记时周期
        tic = cv2.getTickCount()
        # 初始化
        if f == 0:  # init
            # 目标位置
            target_pos = np.array([x + w / 2, y + h / 2])
            # 目标大小
            target_sz = np.array([w, h])
            # 目标追踪初始化
            state = siamese_init(im, target_pos, target_sz, siammask, cfg['hp'], device=device)  # init tracker
        # 目标跟踪
        elif f > 0:  # tracking
            # 目标追踪,进行state的更新
            state = siamese_track(state, im, mask_enable=True, refine_enable=True, device=device)  # track
            # 确定目标位置
            location = state['ploygon'].flatten()
            # 生成目标分割的掩码
            mask = state['mask'] > state['p'].seg_thr
            # 将掩码信息显示在图像上
            im[:, :, 2] = (mask > 0) * 255 + (mask == 0) * im[:, :, 2]
            # 绘制跟踪目标的位置信息
            cv2.polylines(im, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
            cv2.imshow('SiamMask', im)
            key = cv2.waitKey(1)
            if key > 0:
                break
        # 用于记时，获取每一张图片最终的记时周期，并进行统计
        toc += cv2.getTickCount() - tic
    # 获取全部图片的处理时间
    toc /= cv2.getTickFrequency()
    # 计算fps
    fps = f / toc
    print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)'.format(toc, fps))
