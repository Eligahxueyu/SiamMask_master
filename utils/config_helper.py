# --------------------------------------------------------
# 配置文件处理帮助
# --------------------------------------------------------
import json
from os.path import exists


def proccess_loss(cfg):
    """
    解析配置文件中的loss函数
    :param cfg:
    :return:
    """
    # 回归
    if 'reg' not in cfg:
        # 默认为L1Loss
        cfg['reg'] = {'loss': 'L1Loss'}
    else:
        if 'loss' not in cfg['reg']:
            cfg['reg']['loss'] = 'L1Loss'
    # 分类
    if 'cls' not in cfg:
        cfg['cls'] = {'split': True}
    # cls, reg, mask损失的比重
    cfg['weight'] = cfg.get('weight', [1, 1, 36])


def add_default(conf, default):
    # 默认设置
    default.update(conf)
    return default


def load_config(args):
    """
    加载命令行中指定的配置文件中的信息
    :param args:命令行解析结果
    :return:json配置文件中的信息
    """
    # 断言命令行中是否包含config,若包含则对其进行解析
    assert exists(args.config), '"{}" not exists'.format(args.config)
    config = json.load(open(args.config))

    # deal with network 网络结构
    if 'network' not in config:
        print('Warning: network lost in config. This will be error in next version')

        config['network'] = {}

        if not args.arch:
            raise Exception('no arch provided')
    args.arch = config['network']['arch']

    # deal with loss 损失函数
    if 'loss' not in config:
        config['loss'] = {}

    proccess_loss(config['loss'])

    # deal with lr 学习率
    if 'lr' not in config:
        config['lr'] = {}
    default = {
            'feature_lr_mult': 1.0,
            'rpn_lr_mult': 1.0,
            'mask_lr_mult': 1.0,
            'type': 'log',
            'start_lr': 0.03
            }
    default.update(config['lr'])
    config['lr'] = default

    # clip 命令行中的参数，是否进行裁剪
    if 'clip' in config or 'clip' in args.__dict__:
        if 'clip' not in config:
            config['clip'] = {}
        config['clip'] = add_default(config['clip'],
                {'feature': args.clip, 'rpn': args.clip, 'split': False})
        if config['clip']['feature'] != config['clip']['rpn']:
            config['clip']['split'] = True
        if not config['clip']['split']:
            args.clip = config['clip']['feature']

    return config

