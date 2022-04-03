import pickle
import random

import torch
import numpy as np
import pandas as pd

from generator import Generator


def eval(netG, noise=None):
    netG.eval()
    with torch.no_grad():
        fake = netG(noise)
    pred = fake.detach().cpu().numpy()
    with open('standard_scaler.pickle', 'rb') as fp:
        scaler = pickle.load(fp)
    return abs(scaler.inverse_transform(pred))


def main(cfg):
    netG = Generator(cfg)
    netG.load_state_dict(torch.load('generator.pth.tar', map_location='cpu'))
    noise = torch.Tensor(pd.read_csv('noise.csv', header=None).values)
    pred = eval(netG, noise)
    fake = pd.read_csv('generated_samples.csv', header=None).values
    pd.DataFrame(pred).to_csv('output.csv', header=False, index=False)
    print(fake.shape, pred.shape, (fake - pred).mean())
  

if __name__ == '__main__':
    cfg = {
        'z_dim': 512,
        'num_pred_rows': 408,
        'gen_dims': [512, 512, 1024, 4],
    }
    main(cfg)

