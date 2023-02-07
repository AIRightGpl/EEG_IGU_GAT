import torch
from torch import nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_batch
from modules.EEG_MultiChanAttention import EEG_MCf
from modules.GATs_module import GATs
from modules.Multi_Proception_Layer import MLP


class EEG_GAT_moduled(nn.Module):

    # Same as EEG_GAT But split into modules for sake of pretrain

    def __init__(self, n_length, n_class=4):
        super(EEG_GAT_moduled, self).__init__()

        self.mcf_sequence = EEG_MCf()
        self.GATs_sequence = GATs(int(n_length / 5), [32, 16, 8, 4], n_head=4)   ## [60, 30, 15, 8] hidden_layer for 655 length
        self.mlp_sequence = MLP(16, [16, 8, n_class])  ## [16, 8, 4] hidden_layer for 655 length
        # here, mlp sequence will be replaced with a new structure like multiple GAT and softmax

    def forward(self, x, edge_index, batch):

        x = self.mcf_sequence(x)
        x, attention_weight = self.GATs_sequence(x, edge_index)
        embeddings = to_dense_batch(x, batch)[0].detach()
        x_gembed = global_mean_pool(x, batch=batch)
        x = self.mlp_sequence(x_gembed)

        return x, embeddings