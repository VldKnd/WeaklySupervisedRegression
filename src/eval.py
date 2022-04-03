import pickle
import random
from pathlib import Path
from functools import partial
from multiprocessing import Pool

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.gan import Generator
from src.metrics import compute_metrics
from src.utils import (
    set_seed,
    root_dir,
    load_ckpt,
)


def compute_stats(netG, cfg, train, test, num_steps=2000):
    netG.eval()
    if cfg['data_suffix'] == '_standard':
        with open(root_dir() / 'data/standard_scaler.pickle', 'rb') as fp:
            scaler = pickle.load(fp)

    sub_train = [np.array(random.sample(train, cfg['num_pred_rows'])) for _ in range(num_steps)]
    noise = torch.randn([num_steps, cfg['num_pred_rows'], cfg['z_dim']], device=cfg['device'])
    with torch.no_grad():
        fake = netG(noise)
    pred = fake.detach().cpu().numpy()
    if cfg['data_suffix'] == '_standard':
        pred = np.array([scaler.inverse_transform(p) for p in pred])
    pred = abs(pred)
    noise = noise.cpu().numpy()

    func_compute = partial(compute_metrics, test=test, data_scaler=None)
    with Pool(processes=cfg['num_wrks']) as pool:
        results = list(tqdm(pool.imap(func_compute, zip(pred, sub_train, noise)), total=num_steps))
    pred, noise, metrics = zip(*results)
    stats = {k: [] for k in metrics[0].keys()}
    for m in metrics:
        for k, v in m.items():
            stats[k].append(v)
    stats['pred'] = pred
    stats['noise'] = noise

    return stats


def eval(netG, cfg):
    netG.eval()
    noise = torch.randn([cfg['num_pred_rows'], cfg['z_dim']], device=cfg['device'])
    with torch.no_grad():
        fake = netG(noise)
    noise = noise.detach().cpu().numpy()
    Path(root_dir() / 'submit').mkdir(parents=True, exist_ok=True)
    pd.DataFrame(noise).to_csv(root_dir() / 'submit' / 'Noise.csv', index=False)
    pred = fake.detach().cpu().numpy()
    if cfg['data_suffix'] == '_standard':
        with open(root_dir() / 'data/standard_scaler.pickle', 'rb') as fp:
            scaler = pickle.load(fp)
        pred = scaler.inverse_transform(pred)
    pred = abs(pred)
    pd.DataFrame(pred).to_csv(root_dir() / 'submit' / 'generated_samples.csv', index=False)
    return pred


def main(cfg, epoch, best_name=None):
    set_seed(cfg['seed'])

    netG = Generator(cfg)
    _, [netG, _], _, _ = load_ckpt(epoch, [netG, None], [None, None], cfg, best_name)
    netG.to(cfg['device'], non_blocking=True)
    return netG, cfg


if __name__ == '__main__':
    import json

    cfg_name = 'default'
    with open(root_dir() / f'cfg/{cfg_name}.json') as f:
        cfg = json.load(f)
    main(cfg, cfg['num_epochs'] - 1)
