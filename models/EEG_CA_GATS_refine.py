import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_batch
from modules.EEG_MultiChanAttention_refine import EEG_MCf
from modules.GATs_module import GATs
from modules.Channel_Attention import ChannelAttention
from modules.Multi_Proception_Layer import MLP


class EEG_GAT_moduled(nn.Module):

    # Same as EEG_GAT But split into modules for sake of pretrain

    def __init__(self, n_length, n_class=4):
        super(EEG_GAT_moduled, self).__init__()

        self.mcf_sequence = EEG_MCf(group_lst=[5, 5, 5], core_lst=[21, 15, 9])
        self.GATs_sequence = GATs(int(n_length / 5), [32, 16, 8, 4], n_head=8)   ## [60, 30, 15, 8] hidden_layer for 655
        ## [32, 16, 8, 4] 2023-03-13 try [32, 4, 4, 4] to apply residual connection
        self.gru = nn.GRU(32, 32, 2, batch_first=True)
        self.ca1 = ChannelAttention(8, shrink_ratio=4)
        self.mlp_sequence = MLP(32, [16, 8, n_class])  ## [16, 8, 4] hidden_layer for 655 length
        # here, mlp sequence will be replaced with a new structure like multiple GAT and softmax

    def forward(self, x, edge_index, batch, sequence=False):
        # This part is align to the sequence, if the input is a sequence, then it fold the sequence to batch
        if sequence:
            bz, tp, n_chan, n_leng = x.shape
            fold_x = x.reshape(bz * tp, n_chan, n_leng).type(torch.float32)
        else:
            bz, n_chan, n_leng = x.shape
            fold_x = x
        # module above fold the sequence to batch, such (15,20,64,160) to (15*20,64,160), making it easier to
        # apply convolution and mini-batch
        x = self.mcf_sequence(fold_x)
        x, attention_weight = self.GATs_sequence(x, edge_index)
        embeddings = to_dense_batch(x, batch)[0].detach()
        x_gembed = global_mean_pool(x, batch=batch)

        # features above looks like (bz*n_seq,n_channel,n_len), unfold em here
        if sequence:
            unfold_x = x_gembed.reshape(bz, tp, -1)
        else:
            unfold_x = x_gembed
        # unfold_x.shape: (15, 25, 32)
        #--------------------------------------------------------------------------------------------------------------#
        # Here optional GRU unit with intend to attract temporal features
        # x_af_gru, hn = self.gru(unfold_x)
        # x_agg = F.elu(x_af_gru + unfold_x)
        # ## x_af_gru.shape = (15, 25, 32)
        # n_bat, n_time, n_fea = x_agg.shape
        # x_agg = x_agg.reshape(n_bat, n_time, 8, 4)
        # weight_att = self.ca1(x_agg)
        # x_trs = x_agg * weight_att
        # x = F.elu(x_trs + x_agg)
        # x = x.reshape(bz, -1)
        # x = F.avg_pool1d(x, 8, 8)
        # x = self.mlp_sequence(x)
        #--------------------------------------------------------------------------------------------------------------#
        # Here is the optional just mlp for classification
        x = self.mlp_sequence(unfold_x)

        return x, embeddings