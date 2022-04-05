import pickle
from pathlib import Path

import torch
import random
import numpy as np


def root_dir():
    return (Path(__file__) / '..' / '..').resolve()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def print_cfg_info(cfg):
    print('*--- config info ---*')
    print('config name:', cfg['lbl'])
    print('config description:', cfg['desc'])
    print('batch size:', cfg['batch_size'])
    print('device:', cfg['device'])
    print('noise size:', cfg['z_dim'])
    print('generator dimensions:', cfg['gen_dims'])
    print('discriminator dimensions:', cfg['disc_dims'])
    print('from epoch:', cfg['from_epoch'] if cfg['from_ckpt'] else 0)
    print('number of epochs:', cfg['num_epochs'])
    print('*-------------------*')


def load_data_scaler(cfg):
    if cfg['data_suffix'] == '_standard':
        with open(root_dir() / 'data/standard_scaler.pickle', 'rb') as fp:
            return pickle.load(fp)
    return None


def create_meters(names):
    return {n: AverageMeter() for n in names}


def log_meters(logger, meters, step):
    for k, m in meters.items():
        if m.avg != 0:
            logger.add_scalar(k, m.avg, step)


def reset_meters(meters):
    for m in meters.values():
        m.reset()


def state_dict2cpu(obj):
    state_dict = obj.state_dict()
    for k, v in state_dict.items():
        state_dict[k] = v.cpu()
    return state_dict


def save_ckpt(epoch, nets, optims, cfg, best_metrics: dict = None, best_name: str = None, net_vars=None):
    netG, netD = nets
    optimG, optimD = optims
    state_dict = {
        'cfg': cfg,
        'epoch': epoch,
        'net_vars': net_vars,
        'best_metrics': best_metrics,
        'netG': state_dict2cpu(netG),
        'netD': state_dict2cpu(netD),
        'optimG': optimG.state_dict(),
        'optimD': optimD.state_dict(),
    }
    if best_name is None:
        dst = root_dir() / f'ckpt/{cfg["lbl"]}/{cfg["lbl"]}_{epoch}.pth.tar'
    else:
        dst = root_dir() / f'ckpt/{cfg["lbl"]}/{best_name}.pth.tar'
    torch.save(state_dict, dst)


def load_ckpt(epoch, nets, optims, cfg, best_name=None):
    netG, netD = nets
    optimG, optimD = optims
    if best_name is None:
        src = root_dir() / f'ckpt/{cfg["lbl"]}/{cfg["lbl"]}_{epoch}.pth.tar'
    else:
        src = root_dir() / f'ckpt/{cfg["lbl"]}/{best_name}.pth.tar'
    state_dict = torch.load(src, map_location='cpu')
    if netG:
        netG.load_state_dict(state_dict['netG'])
    if netD:
        netD.load_state_dict(state_dict['netD'])
    if optimG: optimG.load_state_dict(state_dict['optimG'])
    if optimD: optimD.load_state_dict(state_dict['optimD'])
    return state_dict['epoch'], [netG, netD], [optimG, optimD], state_dict['best_metrics']


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        return self

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        return self
