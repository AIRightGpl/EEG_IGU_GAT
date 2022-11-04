import torch
from torch.utils.data import Dataset
from typing import List

class Myset(Dataset):

    def __init__(self, x: List[torch.Tensor], y: List[int]):
        super().__init__()
        assert len(x) >= 3 and len(y) >= 3, "number of samples should be greater than 3!"
        assert len(x) == len(y), "the number of sample data should align with that of sample label!"
        self._length = len(x)
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self._length