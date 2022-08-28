from abc import ABC
from abc import abstractmethod
import numpy as np


class Dataset(ABC):
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass


class DataLoader:
    def __init__(self, dataset, batch_size=8, shuffle=False):
        self.bs = batch_size
        self.dataset = dataset
        self.shuffle = shuffle
        self.idx = np.arange(len(self.dataset))
        self.make_indexs()

    def make_indexs(self):
        self.index = 0
        if self.shuffle:
            np.random.shuffle(self.idx)

    def __next__(self):
        if self.index < (len(self.dataset) // self.bs):
            slc = slice(self.index * self.bs, (self.index + 1) * self.bs)
            self.index += 1
            return self.dataset[self.idx[slc]]
        else:
            self.make_indexs()
            raise StopIteration

    def __iter__(self):
        return self
