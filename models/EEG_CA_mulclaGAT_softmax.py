import torch
from torch import nn
from torch_geometric.nn import global_mean_pool
from modules.EEG_MultiChan_filter import EEG_MCf
from modules.Multi_garph_GAT import MG_GATs


class EEG_multiGAT(nn.Module):

    # Same as EEG_GAT But split into modules for sake of pretrain

    def __init__(self, n_length, n_class=4):
        super(EEG_multiGAT, self).__init__()

        self.mcf_sequence = EEG_MCf()
        self.GATs_sequence = MG_GATs(int(n_length / 5), [32, 8, 4], n_head=4, n_class=n_class)   ## [60, 30, 15, 8] hidden_layer for 655 length
        self.probpredict = nn.Softmax() ## [16, 8, 4] hidden_layer for 655 length
        # here, mlp sequence will be replaced with a new structure like multiple GAT and softmax

    def forward(self, x, edge_index, batch):

        x = self.mcf_sequence(x)
        x = self.GATs_sequence(x, edge_index, batch)
        prob = self.probpredict(x)

        return prob, x.detach()