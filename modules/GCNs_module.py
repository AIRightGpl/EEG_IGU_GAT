import torch
from torch import nn
import torch.nn.functional as f
from torch_geometric.nn import GCN
from torch_geometric.graphgym import auto_select_device


class GCNs_Net(nn.Module):

    def __init__(self, n_nodeFeatures, n_class, Ghidd_lst):
        super(GCNs_Net, self).__init__()

        # This model is built by just laying out GCN modules
        # reference DOI:  10.1109/TNNLS.2022.3202569  but no graph pooling
        # #====================================================#
        # this is just piling up GCNs

        for i in range(len(Ghidd_lst)):
            if i == 0:
                exec('self.GCN{} = GCN(n_nodeFeatures, Ghidd_lst[{}])'.format(i + 1, i))
            else:
                exec('self.GCN{} = GCN(Ghidd_lst[{}], Ghidd_lst[{}])'.format(i + 1, i - 1, i))
            exec('self.gbn{} = GraphNorm(Ghidd_lst[{}]*n_head)'.format(i + 1, i))
            exec('self.prlu_{} = nn.PReLU()'.format(i + 1))
        self.n_layers = len(Ghidd_lst)


    def forward(self):

        for i in range(self.n_layers):
            if i == 0:
                exec('x_geb = self.GCN{}(x_spars, edge_index)'.format(i + 1))
            else:
                exec('x_geb = self.GCN{}(x_geb, edge_index)'.format(i + 1))
            exec('x_geb = self.gbn{}(x_geb)'.format(i + 1))
            exec('x_geb = self.prlu_{}(x_geb)'.format(i + 1))

        return x_geb  ## static error may occur