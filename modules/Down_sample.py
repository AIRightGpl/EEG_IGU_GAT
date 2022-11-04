import torch
from torch import nn
from torch.nn import Conv2d


class Down_sample(nn.Module):

    def __init__(self, hidd_lst, n_head):

        super(Down_sample, self).__init__()
        self.Dsamp1 = Conv2d(1, 1, (1, 5), stride=(1, 2), padding=(0, 2))
        self.bsn1 = nn.BatchNorm1d()
        self.n_layers = len(hidd_lst)