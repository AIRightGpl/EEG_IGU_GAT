import torch
from torch import nn
from torch_geometric.nn import GATConv, GraphNorm, global_mean_pool


class GATs(nn.Module):

    def __init__(self, n_nodeFeatures,  Ghidd_lst, n_head=4):

        super(GATs, self).__init__()
        for i in range(len(Ghidd_lst)):
            if i == 0:
                exec('self.GAT{} = GATConv(n_nodeFeatures, Ghidd_lst[{}], heads=n_head)'.format(i + 1, i))
            else:
                exec('self.GAT{} = GATConv(Ghidd_lst[{}]*n_head, Ghidd_lst[{}], heads=n_head)'.format(i + 1, i - 1, i))
            exec('self.gbn{} = GraphNorm(Ghidd_lst[{}]*n_head)'.format(i + 1, i))
            exec('self.prlu_{} = nn.PReLU()'.format(i + 1))
        self.n_layers = len(Ghidd_lst)


    def forward(self, x_spars, edge_index):

        batsiz, n_chan, elechan, simptim = x_spars.shape
        x_spars = x_spars.reshape(batsiz * n_chan * elechan, simptim)
        # for i in range(self.n_layers):
        #
        #     exec('x_spars = self.GAT{}(x_spars, edge_index)'.format(i + 1))
        #     exec('x_spars = self.gbn{}(x_spars)'.format(i + 1))
        #     exec('x_spars = self.prlu_{}(x_spars)'.format(i + 1))
        x_geb = self.GAT1(x_spars, edge_index)
        x_geb = self.gbn1(x_geb)
        x_geb = self.prlu_1(x_geb)
        x_geb = self.GAT2(x_geb, edge_index)
        x_geb = self.gbn2(x_geb)
        x_geb = self.prlu_2(x_geb)
        x_geb = self.GAT3(x_geb, edge_index)
        x_geb = self.gbn3(x_geb)
        x_geb = self.prlu_3(x_geb)
        x_geb, at_w = self.GAT4(x_geb, edge_index, return_attention_weights=True)
        x_geb = self.gbn4(x_geb)
        x_geb = self.prlu_4(x_geb)

        return x_geb, at_w  ## static error may occur


if __name__ == '__main__':
    device = torch.device('cuda:1' if torch.cuda.is_available() else "cpu")
    model = GATs(131, [60, 30, 15, 8]).to(device)

    print('fin')