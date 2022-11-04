import torch
from torch import nn
import torch.nn.functional as f

class EEG_MCf(nn.Module):

    def __init__(self):
        super(EEG_MCf, self).__init__()

        # This model is part of EEG-GAT backbone called from EEG-GAT
        # reference DOI:  10.1109/EMBC48229.2022.9871984 (EEG-GAT)
        #                 10.1088/1741-2552/aace8c   (EEGNet)
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
        # this operate aggragate different group of
        # =======================================================================##
        self.depthAggri = nn.Conv3d(1, 1, (9, 1, 1))
        self.depthConv1 = nn.Conv2d(1, 1, (1, 5), stride=(1, 5))
        self.bnd_2 = torch.nn.BatchNorm2d(1)
        self.prlu_d = nn.PReLU()
        # ======================================================================##
        # self.L1 = nn.Linear(32, 16)
        # self.elu1 = nn.ELU()
        # self.L2 = nn.Linear(16, 8)
        # self.elu2 = nn.ELU()
        # self.L3 = nn.Linear(8, n_class)
        # self.elu3 = nn.ELU()
        # self.avg_pool = torch.nn.AdaptiveAvgPool1d(1)
        # self.max_pool = torch.nn.AdaptiveAvgPool1d(1)


    def forward(self, x):

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

        return x_spars