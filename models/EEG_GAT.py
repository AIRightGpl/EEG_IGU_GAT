import torch
from torch import nn
import torch.nn.functional as f
from torch_geometric.nn import GATConv, GraphNorm, global_mean_pool
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import Batch



class EEG_GAT(nn.Module):

    def __init__(self, n_nodeFeatures, n_class, Ghidd_lst, n_head=4):
        super(EEG_GAT, self).__init__()

        # This model is built upon EEGNet backbone called EEG-GAT
        # reference DOI:  10.1109/EMBC48229.2022.9871984
        # #========================================================================##
        # With three different 1-D convolution as temporal convolution to learn frequency filters and explore
        # short, medium and long term temporal dependencies.
        # #========================================================================##
        self.temp_conv1_1 = nn.Conv2d(1, 3, (1, 15), padding=(0, 7))
        self.bn1_1 = nn.BatchNorm2d(3)
        self.prlu1_1 = nn.PReLU()

        self.temp_conv2_1 = nn.Conv2d(1, 3, (1, 7), padding=(0, 3))
        self.bn2_1 = nn.BatchNorm2d(3)
        self.prlu2_1 = nn.PReLU()

        self.temp_conv3_1 = nn.Conv2d(1, 3, (1, 3), padding=(0, 1))
        self.bn3_1 = nn.BatchNorm2d(3)
        self.prlu3_1 = nn.PReLU()
        # ----------------------------------------------------------------#
        self.temp_conv1_2 = nn.Conv2d(3, 3, (1, 15), padding=(0, 7))
        self.bn1_2 = nn.BatchNorm2d(3)
        self.prlu1_2 = nn.PReLU()

        self.temp_conv2_2 = nn.Conv2d(3, 3, (1, 7), padding=(0, 3))
        self.bn2_2 = nn.BatchNorm2d(3)
        self.prlu2_2 = nn.PReLU()

        self.temp_conv3_2 = nn.Conv2d(3, 3, (1, 3), padding=(0, 1))
        self.bn3_2 = nn.BatchNorm2d(3)
        self.prlu3_2 = nn.PReLU()
        # ----------------------------------------------------------------#
        self.temp_conv1_3 = nn.Conv2d(3, 3, (1, 15), padding=(0, 7))
        self.bn1_3 = nn.BatchNorm2d(3)
        self.prlu1_3 = nn.PReLU()

        self.temp_conv2_3 = nn.Conv2d(3, 3, (1, 7), padding=(0, 3))
        self.bn2_3 = nn.BatchNorm2d(3)
        self.prlu2_3 = nn.PReLU()

        self.temp_conv3_3 = nn.Conv2d(3, 3, (1, 3), padding=(0, 1))
        self.bn3_3 = nn.BatchNorm2d(3)
        self.prlu3_3 = nn.PReLU()
        # ----------------------------------------------------------------#
        # Depth-wise separable convolution
        #
        # =======================================================================##
        self.depthAggri = nn.Conv3d(1, 1, (9, 1, 1))
        self.depthConv1 = nn.Conv2d(1, 1, (1, 5), stride=(1, 5))
        self.bnd_2 = nn.BatchNorm2d(1)
        self.prlu_d = nn.PReLU()
        # =======================================================================##
        # GAT conv * 4
        # -----------------------------------------------------------------#
        self.GAT1 = GATConv(n_nodeFeatures, Ghidd_lst[0], heads=n_head)
        self.gbn1 = GraphNorm(Ghidd_lst[0]*n_head)
        self.prlu_1 = nn.PReLU()
        # -----------------------------------------------------------------#
        self.GAT2 = GATConv(Ghidd_lst[0]*n_head, Ghidd_lst[1], heads=n_head)
        self.gbn2 = GraphNorm(Ghidd_lst[1]*n_head)
        self.prlu_2 = nn.PReLU()
        # -----------------------------------------------------------------#
        self.GAT3 = GATConv(Ghidd_lst[1]*n_head, Ghidd_lst[2], heads=n_head)
        self.gbn3 = GraphNorm(Ghidd_lst[2]*n_head)
        self.prlu_3 = nn.PReLU()
        # -----------------------------------------------------------------#
        self.GAT4 = GATConv(Ghidd_lst[2]*n_head, Ghidd_lst[3], heads=n_head)
        self.gbn4 = GraphNorm(Ghidd_lst[3]*n_head)
        self.prlu_4 = nn.PReLU()
        # =================================================================#
        self.L1 = nn.Linear(32, 16)
        self.elu1 = nn.ELU()
        self.L2 = nn.Linear(16, 8)
        self.elu2 = nn.ELU()
        self.L3 = nn.Linear(8, n_class)
        self.elu3 = nn.ELU()
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.max_pool = torch.nn.AdaptiveAvgPool1d(1)


    def forward(self, x, edge_index, batch):
        bs, channel, length = x.shape
        ori_x = x.reshape(bs, 1, channel, length).type(torch.float)
        # batch = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
        # mini_batch_graph = to_dense_adj(edge_index, batch=batch)

        x1 = self.temp_conv1_1(ori_x)
        x1 = self.bn1_1(x1)
        x1 = self.prlu1_1(x1)
        x2 = self.temp_conv2_1(ori_x)
        x2 = self.bn2_1(x2)
        x2 = self.prlu2_1(x2)
        x3 = self.temp_conv3_1(ori_x)
        x3 = self.bn3_1(x3)
        x3 = self.prlu3_1(x3)

        x1 = self.temp_conv1_2(x1)
        x1 = self.bn1_2(x1)
        x1 = self.prlu1_2(x1)
        x2 = self.temp_conv2_2(x2)
        x2 = self.bn2_2(x2)
        x2 = self.prlu2_2(x2)
        x3 = self.temp_conv3_2(x3)
        x3 = self.bn3_2(x3)
        x3 = self.prlu3_2(x3)

        x1 = self.temp_conv1_3(x1)
        x1 = self.bn1_3(x1)
        x1 = self.prlu1_3(x1)
        x2 = self.temp_conv2_3(x2)
        x2 = self.bn2_3(x2)
        x2 = self.prlu2_3(x2)
        x3 = self.temp_conv3_3(x3)
        x3 = self.bn3_3(x3)
        x3 = self.prlu3_3(x3)

        x_aggra = torch.cat([x1, x2, x3], dim=1)
        batsiz, aggchan, elechan, simptim = x_aggra.shape
        x_aggra = x_aggra.reshape(batsiz, 1, aggchan, elechan, simptim)
        x_aggra = self.depthAggri(x_aggra)
        x_aggra = torch.squeeze(x_aggra, dim=1)
        x_spars = self.depthConv1(x_aggra)
        x_spars = self.bnd_2(x_spars)
        x_spars = self.prlu_d(x_spars)
        batsiz, n_chan, elechan, simptim = x_spars.shape
        ## current x_spars' size is (5, 1, 64, 131),
        ## which is not align with GraphAttention implement (mini-batch)

        x_spars = x_spars.reshape(batsiz*n_chan*elechan, simptim)

        x_geb = self.GAT1(x_spars, edge_index)
        x_geb = self.gbn1(x_geb)
        x_geb = self.prlu_1(x_geb)
        x_geb = self.GAT2(x_geb, edge_index)
        x_geb = self.gbn2(x_geb)
        x_geb = self.prlu_2(x_geb)
        x_geb = self.GAT3(x_geb, edge_index)
        x_geb = self.gbn3(x_geb)
        x_geb = self.prlu_3(x_geb)
        x_geb, _, _ = self.GAT4(x_geb, edge_index, return_attention_weights=True)
        x_geb = self.gbn4(x_geb)
        x_geb = self.prlu_4(x_geb)

        x_embed = global_mean_pool(x_geb, batch=batch)
        x_redu = self.L1(x_embed)
        x_redu = self.elu1(x_redu)
        x_redu = self.L2(x_redu)
        x_redu = self.elu2(x_redu)
        x_redu = self.L3(x_redu)
        x_redu = self.elu3(x_redu)

        x_out = nn.Softmax(x_redu)


        return x_out



if __name__ == '__main__':
    import pickle
    from modules.dataset_conversion_module import SqlConnector
    from torch.utils.data import DataLoader
    from toolbox_lib.Graph_tool import replicate_graph_batch
    dbcon = SqlConnector('eeg_motor_MI', 'subject_id', 'trails_id', 'eeg_data')
    data1 = dbcon.attract('eeg_data', 'subject_id', '1')
    trail = pickle.loads(data1[2][0])
    device = torch.device('cuda:1' if torch.cuda.is_available() else "cpu")
    model = EEG_GAT(131, 4, [60, 30, 15, 8]).to(device)
    edge_indx = torch.tensor([[0, 1, 1, 1, 2, 2, 2, 3, 3, 63], [1, 0, 2, 3, 1, 3, 63, 1, 2, 2]], dtype=torch.long, device=device)
    model.train()
    loader = DataLoader(trail['data'], batch_size=9)
    for index, data in enumerate(loader):
        data = data.to(device)
        n_batch, _, _ = data.shape
        batch_edge_idx, batch = replicate_graph_batch(edge_indx, n_batch)
        out = model(data, batch_edge_idx, batch)