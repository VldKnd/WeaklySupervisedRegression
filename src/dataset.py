import math
import torch
import random
from torch import Tensor
from torch.utils.data import Dataset

class WeakDataset():

    def __init__(self, X, batch_size=64, device=torch.device("cpu")):
        self.iter = 0
        self._len_X = X.shape[0]
        self.X = torch.Tensor(X).to(device)
        self._batch_size = batch_size
        self.idxes = torch.randperm(self._len_X)
        self.max = math.ceil(self._len_X/self._batch_size)

    def __iter__(self):
        self.n = 0
        self.idxes_X = torch.randperm(self._len_X)
        return self
    
    def __len__(self):
        return self.max

    def __next__(self):
        if self.n < self.max:
            self.n += 1
            return self.X[(self.n-1)*self._batch_size:self.n*self._batch_size]
        else:
            raise StopIteration
            
    def sample(self):
        if self.iter == self.max or (self.iter+1)*self._batch_size > self._len_X:
            self.iter = 1
            self.idxes = torch.randperm(self._len_X)
            return self.X[self.idxes[(self.iter-1)*self._batch_size:self.iter*self._batch_size]]
        else:
            self.iter += 1
            return self.X[self.idxes[(self.iter-1)*self._batch_size:self.iter*self._batch_size]]
            
            
class MyDataset:

    def __init__(self, X, y, batch_size=64, device=torch.device("cpu")):
        self._len_X = X.shape[0]
        self._len_y = y.shape[0]
        self.X = torch.Tensor(X).to(device)
        if len(y.shape) == 1:
            self.y = torch.Tensor(y)[:, None].to(device)
        else:
            self.y = torch.Tensor(y).to(device)
        self._batch_size = batch_size
        self.max = math.ceil(self._len_X/self._batch_size)

    def __iter__(self):
        self.n = 0
        self.idxes = torch.randperm(self._len_X)
        return self
    
    def __len__(self):
        return self.max

    def __next__(self):
        if self.n < self.max and (self.n+1)*self._batch_size <= self._len_X:
            self.n += 1
            return (self.X[self.idxes[(self.n-1)*self._batch_size:self.n*self._batch_size]],
                    self.y[self.idxes[(self.n-1)*self._batch_size:self.n*self._batch_size]])
        else:
            raise StopIteration