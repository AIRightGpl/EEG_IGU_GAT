import torch
from torch import nn
from torch_geometric.nn import global_mean_pool
from modules.EEG_MultiChan_filter import EEG_MCf
from modules.GATs_module import GATs
from modules.Multi_Proception_Layer import MLP


class EEG_GAT_ablation(nn.Module):

    # Same as EEG_GAT But split into modules for sake of pretrain

    def __init__(self, n_length):
        super(EEG_GAT_ablation, self).__init__()

        self.mcf_sequence = EEG_MCf()
        # self.GATs_sequence = GATs(int(n_length / 5), [32, 16, 8, 4], n_head=4)   ## [60, 30, 15, 8] hidden_layer for 655 length
        self.Fla = nn.Flatten()
        self.mlp_sequence = MLP(int(n_length * 64 / 5), [16, 8, 4])  ## [16, 8, 4] hidden_layer for 655 length


    def forward(self, x, edge_index, batch):

        x = self.mcf_sequence(x)
        # x, attention_weight = self.GATs_sequence(x, edge_index)
        # x_embed = global_mean_pool(x, batch=batch)
        x_embed = self.Fla(x)
        x = self.mlp_sequence(x_embed)

        return x, 0