# --------------------------------------------------------
# 模型加载帮助类
# --------------------------------------------------------

import torch
import logging
logger = logging.getLogger('global')


def check_keys(model, pretrained_state_dict):
    "模型检查"
    # 预训练模型中的keys
    ckpt_keys = set(pretrained_state_dict.keys())
    # 原始模型中的keys
    model_keys = set(model.state_dict().keys())
    # 预训练模型和原始模型包含的keys
    used_pretrained_keys = model_keys & ckpt_keys
    # 只在预训练模型中的keys
    unused_pretrained_keys = ckpt_keys - model_keys
    # 只在原始模型中的keys
    missing_keys = model_keys - ckpt_keys
    # 预训练模型丢失的keys大于0
    if len(missing_keys) > 0:
        logger.info('[Warning] missing keys: {}'.format(missing_keys))
        logger.info('missing keys:{}'.format(len(missing_keys)))
    # 只在原始模型中的keys
    if len(unused_pretrained_keys) > 0:
        logger.info('[Warning] unused_pretrained_keys: {}'.format(unused_pretrained_keys))
        logger.info('unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    logger.info('used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters share common prefix 'module.' '''
    logger.info('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


# 加载模型的权重文件
def load_pretrain(model, pretrained_path):
    logger.info('load pretrained model from {}'.format(pretrained_path))
    # 加载预训练模型
    if not torch.cuda.is_available():
        # CPU
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        # GPU
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    # 去除前置网络
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')

    try:
        # 模型检测
        check_keys(model, pretrained_dict)
    except:
        logger.info('[Warning]: using pretrain as features. Adding "features." as prefix')
        new_dict = {}
        for k, v in pretrained_dict.items():
            k = 'features.' + k
            new_dict[k] = v
        pretrained_dict = new_dict
        check_keys(model, pretrained_dict)
    # 加载模型
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def restore_from(model, optimizer, ckpt_path):
    "模型恢复"
    logger.info('restore from {}'.format(ckpt_path))
    device = torch.cuda.current_device()
    # 加载权重文件
    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage.cuda(device))
    epoch = ckpt['epoch']
    best_acc = ckpt['best_acc']
    arch = ckpt['arch']
    ckpt_model_dict = remove_prefix(ckpt['state_dict'], 'module.')
    check_keys(model, ckpt_model_dict)
    # 加载模型
    model.load_state_dict(ckpt_model_dict, strict=False)
    # 模型检查
    check_keys(optimizer, ckpt['optimizer'])
    # 加载优化器
    optimizer.load_state_dict(ckpt['optimizer'])
    return model, optimizer, epoch, best_acc, arch
