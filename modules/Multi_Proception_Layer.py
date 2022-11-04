import torch
from torch import nn
import torch.nn.functional as f


class MLP(nn.Module):
    # This is an simple MLP module with activation Func ELU
    def __init__(self, n_InFeature, hidden_lst):
        super(MLP, self).__init__()

        for i in range(len(hidden_lst)):
            if i == 0:
                exec('self.L{} = nn.Linear(n_InFeature, hidden_lst[{}])'.format(i + 1, i))
            else:
                exec('self.L{} = nn.Linear(hidden_lst[{}], hidden_lst[{}])'.format(i + 1, i - 1, i))
            exec('self.bn{} = nn.BatchNorm1d(hidden_lst[{}])'.format(i + 1, i))
            exec('self.elu_{} = nn.ELU()'.format(i + 1))
        self.n_layers = len(hidden_lst)

    def forward(self, x):

        # for i in range(self.n_layers):
        #     exec('x = self.L{}(x)'.format(i + 1))  ## expect to be (batsiz, 16) after executing this but got size unchanged (batsiz,32) instead
        #     exec('x = self.bn{}(x)'.format(i + 1))
        #     exec('x = self.elu_{}(x)'.format(i + 1))

        x = self.L1(x)
        x = self.bn1(x)
        x = self.elu_1(x)
        x = self.L2(x)
        x = self.bn2(x)
        x = self.elu_2(x)
        x = self.L3(x)
        x = self.bn3(x)
        x = self.elu_3(x)

        return x
