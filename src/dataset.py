import math
import torch
import random
from torch import Tensor
from torch.utils.data import Dataset

class MyWeakDataset:

    def __init__(self, X, y, X_weak=None, batch_size=64):
        self._len_X = X.shape[0]
        if X_weak is not None:
            self._len_X_weak = X_weak.shape[0]
            self.X_weak = torch.Tensor(X_weak)
        else:
            self._len_X_weak = 0
            self.X_weak = None
            
        self._len_y = y.shape[0]
        
        self.X = torch.Tensor(X)
        if len(y.shape) == 1:
            self.y = torch.Tensor(y)[:, None]
        else:
            self.y = torch.Tensor(y)
        self._batch_size = batch_size
        self._batch_size_weak = batch_size*self._len_X_weak//self._len_X
        self.max = math.ceil(self._len_X/self._batch_size)

    def __iter__(self):
        self.n = 0
        self.idxes_X = torch.randperm(self._len_X)
        self.idxes_y = torch.randperm(self._len_y)
        self.idxes_X_weak = torch.randperm(self._len_X_weak)
        return self
    
    def __len__(self):
        return self.max

    def __next__(self):
        if self.n < self.max:
            self.n += 1
            if self._len_X_weak:
                return (self.X[(self.n-1)*self._batch_size:self.n*self._batch_size],
                        self.X_weak[(self.n-1)*self._batch_size_weak:self.n*self._batch_size_weak],
                        self.y[(self.n-1)*self._batch_size:self.n*self._batch_size])
            
            return (self.X[self.idxes_X],
                    self.y[self.idxes_y])
        else:
            raise StopIteration
            
class MyDataset:

    def __init__(self, X, y, batch_size=64):
        self._len_X = X.shape[0]
        self._len_y = y.shape[0]
        self.X = torch.Tensor(X)
        if len(y.shape) == 1:
            self.y = torch.Tensor(y)[:, None]
        else:
            self.y = torch.Tensor(y)
        self._batch_size = batch_size
        self.max = math.ceil(self._len_X/self._batch_size)

    def __iter__(self):
        self.n = 0
        self.idxes_X = torch.randperm(self._len_X)
        self.idxes_y = torch.randperm(self._len_y)
        return self
    
    def __len__(self):
        return self.max

    def __next__(self):
        if self.n < self.max:
            self.n += 1
            return (self.X[(self.n-1)*self._batch_size:self.n*self._batch_size],
                    self.y[(self.n-1)*self._batch_size:self.n*self._batch_size])
        else:
            raise StopIteration