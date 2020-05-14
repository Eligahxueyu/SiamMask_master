# --------------------------------------------------------
# 基础网络的训练
# --------------------------------------------------------
import argparse
import logging
import os
import cv2
import shutil
import time
import json
import math
import torch
from torch.utils.data import DataLoader

from utils.log_helper import init_log, print_speed, add_file_handler, Dummy
from utils.load_helper import load_pretrain, restore_from
from utils.average_meter_helper import AverageMeter

from datasets.siam_mask_dataset import DataSets

from utils.lr_helper import build_lr_scheduler
from tensorboardX import SummaryWriter

from utils.config_helper import load_config
from torch.utils.collect_env import get_pretty_env_info

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='PyTorch Tracking SiamMask Training')

parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--clip', default=10.0, type=float,
                    help='gradient clip value')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', default='',
                    help='use pre-trained model')
parser.add_argument('--config', dest='config', required=True,
                    help='hyperparameter of SiamMask in json format')
parser.add_argument('--arch', dest='arch', default='', choices=['Custom',],
                    help='architecture of pretrained model')
parser.add_argument('-l', '--log', default="log.txt", type=str,
                    help='log file')
parser.add_argument('-s', '--save_dir', default='snapshot', type=str,
                    help='save dir')
parser.add_argument('--log-dir', default='board', help='TensorBoard log dir')


best_acc = 0.


def collect_env_info():
    """
    环境信息
    :return:
    """
    env_str = get_pretty_env_info()
    env_str += "\n        OpenCV ({})".format(cv2.__version__)
    return env_str


def build_data_loader(cfg):
    """
    获取数据集
    :param cfg:
    :return:
    """
    logger = logging.getLogger('global')

    logger.info("build train dataset")  # train_dataset
    # 获取训练集数据，包含数据增强的内容
    train_set = DataSets(cfg['train_datasets'], cfg['anchors'], args.epochs)
    # 对数据进行打乱处理
    train_set.shuffle()

    # 获取验证集数据，若为配置验证集数据则使用训练集数据替代
    logger.info("build val dataset")  # val_dataset
    if not 'val_datasets' in cfg.keys():
        cfg['val_datasets'] = cfg['train_datasets']
    val_set = DataSets(cfg['val_datasets'], cfg['anchors'])
    val_set.shuffle()
    # DataLoader是Torch内置的方法，它允许使用多线程加速数据的读取
    train_loader = DataLoader(train_set, batch_size=args.batch, num_workers=args.workers,
                              pin_memory=True, sampler=None)
    val_loader = DataLoader(val_set, batch_size=args.batch, num_workers=args.workers,
                            pin_memory=True, sampler=None)

    logger.info('build dataset done')
    return train_loader, val_loader


def build_opt_lr(model, cfg, args, epoch):
    """
    获取优化方法和学习率
    :param model:
    :param cfg:
    :param args:
    :param epoch:
    :return:
    """
    # 获取要训练的网络
    backbone_feature = model.features.param_groups(cfg['lr']['start_lr'], cfg['lr']['feature_lr_mult'])
    if len(backbone_feature) == 0:
        # 获取要训练的rpn网络的参数
        trainable_params = model.rpn_model.param_groups(cfg['lr']['start_lr'], cfg['lr']['rpn_lr_mult'], 'mask')
    else:
        # 获取基础网络，rpn和mask网络的训练参数
        trainable_params = backbone_feature + \
                           model.rpn_model.param_groups(cfg['lr']['start_lr'], cfg['lr']['rpn_lr_mult']) + \
                           model.mask_model.param_groups(cfg['lr']['start_lr'], cfg['lr']['mask_lr_mult'])
    # 随机梯度下降算法优化
    optimizer = torch.optim.SGD(trainable_params, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # 获取学习率
    lr_scheduler = build_lr_scheduler(optimizer, cfg['lr'], epochs=args.epochs)
    # 更新学习率
    lr_scheduler.step(epoch)
    # 返回优化器和学习率
    return optimizer, lr_scheduler


def main():
    """
    基础网络的训练
    :return:
    """
    global args, best_acc, tb_writer, logger
    args = parser.parse_args()
    # 初始化日志信息
    init_log('global', logging.INFO)

    if args.log != "":
        add_file_handler('global', args.log, logging.INFO)
    # 获取log信息
    logger = logging.getLogger('global')
    logger.info("\n" + collect_env_info())
    logger.info(args)
    # 获取配置信息
    cfg = load_config(args)
    logger.info("config \n{}".format(json.dumps(cfg, indent=4)))

    if args.log_dir:
        tb_writer = SummaryWriter(args.log_dir)
    else:
        tb_writer = Dummy()

    # 构建数据集
    train_loader, val_loader = build_data_loader(cfg)
    # 加载训练网络
    if args.arch == 'Custom':
        from custom import Custom
        model = Custom(pretrain=True, anchors=cfg['anchors'])
    else:
        exit()
    logger.info(model)
    # 加载预训练网络
    if args.pretrained:
        model = load_pretrain(model, args.pretrained)

    # GPU版本
    # model = model.cuda()
    # dist_model = torch.nn.DataParallel(model, list(range(torch.cuda.device_count()))).cuda()
    # 网络模型
    dist_model = torch.nn.DataParallel(model)
    # 模型参数的更新比例
    if args.resume and args.start_epoch != 0:
        model.features.unfix((args.start_epoch - 1) / args.epochs)
    # 获取优化器和学习率的更新策略
    optimizer, lr_scheduler = build_opt_lr(model, cfg, args, args.start_epoch)
    # optionally resume from a checkpoint 加载模型
    if args.resume:
        assert os.path.isfile(args.resume), '{} is not a valid file'.format(args.resume)
        model, optimizer, args.start_epoch, best_acc, arch = restore_from(model, optimizer, args.resume)
        # GPU
        # dist_model = torch.nn.DataParallel(model, list(range(torch.cuda.device_count()))).cuda()
        dist_model = torch.nn.DataParallel(model)

    logger.info(lr_scheduler)

    logger.info('model prepare done')
    # 模型训练
    train(train_loader, dist_model, optimizer, lr_scheduler, args.start_epoch, cfg)


def train(train_loader, model, optimizer, lr_scheduler, epoch, cfg):
    """
    模型训练
    :param train_loader:训练数据
    :param model:
    :param optimizer:
    :param lr_scheduler:
    :param epoch:
    :param cfg:
    :return:
    """

    global tb_index, best_acc, cur_lr, logger
    # 获取当前的学习率
    cur_lr = lr_scheduler.get_cur_lr()
    logger = logging.getLogger('global')
    #
    avg = AverageMeter()
    model.train()
    # GPU
    #  model = model.cuda()
    end = time.time()

    def is_valid_number(x):
        return not(math.isnan(x) or math.isinf(x) or x > 1e4)

    num_per_epoch = len(train_loader.dataset) // args.epochs // args.batch
    print("num_per_epoch",num_per_epoch)
    start_epoch = epoch
    epoch = epoch
    # 获取每个batch的输入
    for iter, input in enumerate(train_loader):
        if epoch != iter // num_per_epoch + start_epoch:  # next epoch
            epoch = iter // num_per_epoch + start_epoch
            # 创建存储路径
            if not os.path.exists(args.save_dir):  # makedir/save model
                os.makedirs(args.save_dir)
            # 存储训练结果
            save_checkpoint({
                    'epoch': epoch,
                    'arch': args.arch,
                    'state_dict': model.module.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                    'anchor_cfg': cfg['anchors']
                }, False,
                os.path.join(args.save_dir, 'checkpoint_e%d.pth' % (epoch)),
                os.path.join(args.save_dir, 'best.pth'))

            if epoch == args.epochs:
                return
            # 更新优化器和学习方法
            if model.module.features.unfix(epoch/args.epochs):
                logger.info('unfix part model.')
                optimizer, lr_scheduler = build_opt_lr(model.module, cfg, args, epoch)
            # 获取当前学习率
            lr_scheduler.step(epoch)
            cur_lr = lr_scheduler.get_cur_lr()

            logger.info('epoch:{}'.format(epoch))
        # 更新日志
        tb_index = iter
        if iter % num_per_epoch == 0 and iter != 0:
            for idx, pg in enumerate(optimizer.param_groups):
                logger.info("epoch {} lr {}".format(epoch, pg['lr']))
                tb_writer.add_scalar('lr/group%d' % (idx+1), pg['lr'], tb_index)

        data_time = time.time() - end
        avg.update(data_time=data_time)
        # 输入数据
        x = {
            # GPU
            # 'cfg': cfg,
            # 'template': torch.autograd.Variable(input[0]).cuda(),
            # 'search': torch.autograd.Variable(input[1]).cuda(),
            # 'label_cls': torch.autograd.Variable(input[2]).cuda(),
            # 'label_loc': torch.autograd.Variable(input[3]).cuda(),
            # 'label_loc_weight': torch.autograd.Variable(input[4]).cuda(),
            # 'label_mask': torch.autograd.Variable(input[6]).cuda(),
            # 'label_mask_weight': torch.autograd.Variable(input[7]).cuda(),
            'cfg': cfg,
            'template': torch.autograd.Variable(input[0]),
            'search': torch.autograd.Variable(input[1]),
            'label_cls': torch.autograd.Variable(input[2]),
            'label_loc': torch.autograd.Variable(input[3]),
            'label_loc_weight': torch.autograd.Variable(input[4]),
            'label_mask': torch.autograd.Variable(input[6]),
            'label_mask_weight': torch.autograd.Variable(input[7]),
        }
        # 输出数据
        outputs = model(x)

        # 计算损失函数
        rpn_cls_loss, rpn_loc_loss, rpn_mask_loss = torch.mean(outputs['losses'][0]), torch.mean(outputs['losses'][1]), torch.mean(outputs['losses'][2])
        # 计算精度
        mask_iou_mean, mask_iou_at_5, mask_iou_at_7 = torch.mean(outputs['accuracy'][0]), torch.mean(outputs['accuracy'][1]), torch.mean(outputs['accuracy'][2])
        # 获取分类，回归和分割所占的比例
        cls_weight, reg_weight, mask_weight = cfg['loss']['weight']
        # 计算总损失
        loss = rpn_cls_loss * cls_weight + rpn_loc_loss * reg_weight + rpn_mask_loss * mask_weight
        # 将梯度置零
        optimizer.zero_grad()
        # 反向传播
        loss.backward()

        if cfg['clip']['split']:
            torch.nn.utils.clip_grad_norm_(model.module.features.parameters(), cfg['clip']['feature'])
            torch.nn.utils.clip_grad_norm_(model.module.rpn_model.parameters(), cfg['clip']['rpn'])
            torch.nn.utils.clip_grad_norm_(model.module.mask_model.parameters(), cfg['clip']['mask'])
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)  # gradient clip

        if is_valid_number(loss.item()):
            optimizer.step()

        siammask_loss = loss.item()

        batch_time = time.time() - end
        # 参数更新
        avg.update(batch_time=batch_time, rpn_cls_loss=rpn_cls_loss, rpn_loc_loss=rpn_loc_loss,
                   rpn_mask_loss=rpn_mask_loss, siammask_loss=siammask_loss,
                   mask_iou_mean=mask_iou_mean, mask_iou_at_5=mask_iou_at_5, mask_iou_at_7=mask_iou_at_7)
        # 参数写入tensorboard
        tb_writer.add_scalar('loss/cls', rpn_cls_loss, tb_index)
        tb_writer.add_scalar('loss/loc', rpn_loc_loss, tb_index)
        tb_writer.add_scalar('loss/mask', rpn_mask_loss, tb_index)
        tb_writer.add_scalar('mask/mIoU', mask_iou_mean, tb_index)
        tb_writer.add_scalar('mask/AP@.5', mask_iou_at_5, tb_index)
        tb_writer.add_scalar('mask/AP@.7', mask_iou_at_7, tb_index)
        end = time.time()
        # 日志输出
        if (iter + 1) % args.print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}] lr: {lr:.6f}\t{batch_time:s}\t{data_time:s}'
                        '\t{rpn_cls_loss:s}\t{rpn_loc_loss:s}\t{rpn_mask_loss:s}\t{siammask_loss:s}'
                        '\t{mask_iou_mean:s}\t{mask_iou_at_5:s}\t{mask_iou_at_7:s}'.format(
                        epoch+1, (iter + 1) % num_per_epoch, num_per_epoch, lr=cur_lr, batch_time=avg.batch_time,
                        data_time=avg.data_time, rpn_cls_loss=avg.rpn_cls_loss, rpn_loc_loss=avg.rpn_loc_loss,
                        rpn_mask_loss=avg.rpn_mask_loss, siammask_loss=avg.siammask_loss, mask_iou_mean=avg.mask_iou_mean,
                        mask_iou_at_5=avg.mask_iou_at_5,mask_iou_at_7=avg.mask_iou_at_7))
            print_speed(iter + 1, avg.batch_time.avg, args.epochs * num_per_epoch)


def save_checkpoint(state, is_best, filename='checkpoint.pth', best_file='model_best.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_file)


if __name__ == '__main__':
    main()
