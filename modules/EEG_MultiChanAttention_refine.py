import torch
from torch import nn
from modules.Channel_Attention import ChannelAttention
import torch.nn.functional as f

class EEG_MCf(nn.Module):

    def __init__(self, group_lst=[5, 5, 5], core_lst=[41, 13, 7]):
        super(EEG_MCf, self).__init__()

        # This model is part of EEG-GAT backbone called from EEG-GAT
        # reference DOI:  10.1109/EMBC48229.2022.9871984 (EEG-GAT)
        #                 10.1088/1741-2552/aace8c   (EEGNet)
        # #========================================================================##
        # With three different 1-D convolution as temporal convolution to learn frequency filters and explore
        # short, medium and long term temporal dependencies.
        # #========================================================================##
        self.temp_conv1_1 = nn.Conv2d(1, group_lst[0], (1, core_lst[0]), padding=(0, int((core_lst[0] - 1) / 2)))
        self.bn1_1 = nn.BatchNorm2d(group_lst[0])
        self.prlu1_1 = nn.PReLU()

        self.temp_conv2_1 = nn.Conv2d(1, group_lst[0], (1, core_lst[1]), padding=(0, int((core_lst[1] - 1) / 2)))
        self.bn2_1 = nn.BatchNorm2d(group_lst[0])
        self.prlu2_1 = nn.PReLU()

        self.temp_conv3_1 = nn.Conv2d(1, group_lst[0], (1, core_lst[2]), padding=(0, int((core_lst[2] - 1) / 2)))
        self.bn3_1 = nn.BatchNorm2d(group_lst[0])
        self.prlu3_1 = nn.PReLU()
        # ----------------------------------------------------------------#
        self.temp_conv1_2 = nn.Conv2d(group_lst[0], group_lst[1], (1, core_lst[0]), padding=(0, int((core_lst[0] - 1) / 2)))
        self.bn1_2 = nn.BatchNorm2d(group_lst[1])
        self.prlu1_2 = nn.PReLU()

        self.temp_conv2_2 = nn.Conv2d(group_lst[0], group_lst[1], (1, core_lst[1]), padding=(0, int((core_lst[1] - 1) / 2)))
        self.bn2_2 = nn.BatchNorm2d(group_lst[1])
        self.prlu2_2 = nn.PReLU()

        self.temp_conv3_2 = nn.Conv2d(group_lst[0], group_lst[1], (1, core_lst[2]), padding=(0, int((core_lst[2] - 1) / 2)))
        self.bn3_2 = nn.BatchNorm2d(group_lst[1])
        self.prlu3_2 = nn.PReLU()
        # ----------------------------------------------------------------#
        self.temp_conv1_3 = nn.Conv2d(group_lst[1], group_lst[2], (1, core_lst[0]), padding=(0, int((core_lst[0] - 1) / 2)))
        self.bn1_3 = nn.BatchNorm2d(group_lst[2])
        self.prlu1_3 = nn.PReLU()

        self.temp_conv2_3 = nn.Conv2d(group_lst[1], group_lst[2], (1, core_lst[1]), padding=(0, int((core_lst[1] - 1) / 2)))
        self.bn2_3 = nn.BatchNorm2d(group_lst[2])
        self.prlu2_3 = nn.PReLU()

        self.temp_conv3_3 = nn.Conv2d(group_lst[1], group_lst[2], (1, core_lst[2]), padding=(0, int((core_lst[2] - 1) / 2)))
        self.bn3_3 = nn.BatchNorm2d(group_lst[2])
        self.prlu3_3 = nn.PReLU()
        # ----------------------------------------------------------------#
        # Depth-wise separable convolution
        # this operate aggregate different group of
        # =======================================================================##
        self.depthAggri = nn.Conv3d(1, 1, (group_lst[-1] * 3, 1, 1))
        self.depthConverge1 = nn.MaxPool2d((1, 5), stride=(1, 5))
        self.bnd_2 = torch.nn.BatchNorm2d(1)
        self.prlu_d = nn.PReLU()
        # ======================================================================##
        # Version 2.0, abandon the fixed-parameter(when conducting prediction),
        # adapted the Channel-Attention mechanism to arrange weight for channels
        # for multiple feature maps aggregation
        self.CA = ChannelAttention(group_lst[-1] * 3)
        self.relu = nn.ReLU()



    def forward(self, x):

        bs, channel, length = x.shape
        ori_x = x.reshape(bs, 1, channel, length).type(torch.float)
        # batch = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
        # mini_batch_graph = to_dense_adj(edge_index, batch=batch)

        x1 = self.temp_conv1_1(ori_x)
        x1 = self.bn1_1(x1)
        x1_shal = self.prlu1_1(x1)
        x2 = self.temp_conv2_1(ori_x)
        x2 = self.bn2_1(x2)
        x2_shal = self.prlu2_1(x2)
        x3 = self.temp_conv3_1(ori_x)
        x3 = self.bn3_1(x3)
        x3_shal = self.prlu3_1(x3)

        x1 = self.temp_conv1_2(x1_shal)
        x1 = self.bn1_2(x1)
        x1 = self.prlu1_2(x1)
        x2 = self.temp_conv2_2(x2_shal)
        x2 = self.bn2_2(x2)
        x2 = self.prlu2_2(x2)
        x3 = self.temp_conv3_2(x3_shal)
        x3 = self.bn3_2(x3)
        x3 = self.prlu3_2(x3)

        x1 = self.temp_conv1_3(x1)
        x1 = self.bn1_3(x1)

        x2 = self.temp_conv2_3(x2)
        x2 = self.bn2_3(x2)

        x3 = self.temp_conv3_3(x3)
        x3 = self.bn3_3(x3)

        ## res through deeper 2d convolution
        #------------------------------------------------------------------#
        x1 = self.prlu1_3(x1 + x1_shal)
        x2 = self.prlu2_3(x2 + x2_shal)
        x3 = self.prlu3_3(x3 + x3_shal)
        #------------------------------------------------------------------#

        x_aggra_ori = torch.cat([x1, x2, x3], dim=1)
        batsiz, aggchan, elechan, simptim = x_aggra_ori.shape

        # -----------------------------------------------------------------#
        # instead of making prediction with fixed channel_weights, creating
        # Channel_Attention for variable channel_weights
        # -----------------------------------------------------------------#
        # x_aggra = x_aggra.reshape(batsiz, 1, aggchan, elechan, simptim)
        # x_aggra = self.depthAggri(x_aggra)
        # x_aggra = torch.squeeze(x_aggra, dim=1)

        x_weight = self.CA(x_aggra_ori)
        x_aggra = x_aggra_ori * x_weight
        x_aggra = self.relu(x_aggra + x_aggra_ori)
        batsiz, aggchan, elechan, simptim = x_aggra.shape
        x_aggra = x_aggra.reshape(batsiz, 1, aggchan, elechan, simptim)
        x_aggra = self.depthAggri(x_aggra)
        x_aggra = torch.squeeze(x_aggra, dim=1)
        x_spars = self.depthConverge1(x_aggra)
        x_spars = self.bnd_2(x_spars)
        x_spars = self.prlu_d(x_spars)

        return x_spars