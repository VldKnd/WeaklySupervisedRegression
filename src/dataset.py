import random

import torch
import pandas as pd
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from src.utils import root_dir


class MyDataset(Dataset):

    def __init__(self, cfg, mode='train'):
        self._mode = mode
        self._len = cfg['num_reps']
        self._size = cfg['mini_batch_size']
        data_suffix = cfg['data_suffix'] if mode == 'train' else ''
        file_path = f'data/{mode}_data{data_suffix}.csv'
        self._data = pd.read_csv(root_dir() / file_path, header=None).values.tolist()
        self._transforms = transforms.Compose([
            transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float)),
        ])

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> Tensor:
        if self._mode == 'train':
            data = random.sample(self._data, self._size)
        else:
            data = self._data
        return self._transforms(data)
