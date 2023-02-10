import torch
from torch import nn
from torch_geometric.nn import GATConv, GraphNorm, global_mean_pool


class MG_GATs(nn.Module):

    def __init__(self, n_nodeFeatures,  Ghidd_lst, n_head=4, n_class=4):

        super(MG_GATs, self).__init__()
        for j in range(n_class):
            for i in range(len(Ghidd_lst)):
                if i == 0:
                    exec('self.GAT{}_{} = GATConv(n_nodeFeatures, Ghidd_lst[{}], heads=n_head)'.format(j + 1, i + 1, i))
                else:
                    exec('self.GAT{}_{} = GATConv(Ghidd_lst[{}]*n_head, Ghidd_lst[{}], heads=n_head)'.format(j + 1,
                                                                                                             i + 1,
                                                                                                             i - 1, i))
                exec('self.gbn{}_{} = GraphNorm(Ghidd_lst[{}]*n_head)'.format(j + 1, i + 1, i))
                exec('self.prlu{}_{} = nn.PReLU()'.format(j + 1, i + 1))
            exec('self.ds{} = nn.Conv1d(1, 1, 4, stride=4)'.format(j + 1))
            exec('self.avp{} = nn.AvgPool1d(4)'.format(j + 1))
        # self.gat = GATConv(3, 4, heads=5)
        self.n_layers = len(Ghidd_lst)
        self.n_class = n_class


    def forward(self, x_spars, edge_index, batch):

        batsiz, n_chan, elechan, simptim = x_spars.shape
        x_spars = x_spars.reshape(batsiz * n_chan * elechan, simptim)

        for i in range(self.n_class):
            for j in range(self.n_layers):
                if j == 0:
                    exec('x_geb{} = self.GAT{}_{}(x_spars, edge_index[{}])'.format(i + 1, i + 1, j + 1, i + 1))
                else:
                    exec('x_geb{} = self.GAT{}_{}(x_geb{}, edge_index[{}])'.format(i + 1, i + 1, j + 1, i + 1, i + 1))
                exec('x_geb{} = self.gbn{}_{}(x_geb{})'.format(i + 1, i + 1, j + 1, i + 1))
                exec('x_geb{} = self.prlu{}_{}(x_geb{})'.format(i + 1, i + 1, j + 1, i + 1))
            exec('x_gmped{} = global_mean_pool(x_geb{}, batch=batch[{}])'.format(i + 1, i + 1, i + 1))
            # exec('x{} = x')
            # exec('bz, n_fea = x_gmped{}.shape'.format(i + 1))
            exec('x_gmp{} = x_gmped{}.unsqueeze(1)'.format(i + 1, i + 1))
            exec('x{} = self.ds{}(x_gmp{})'.format(i + 1, i + 1, i + 1))
            exec('x{} = torch.squeeze(x{}, 1)'.format(i + 1, i + 1))
            exec('x{} = self.avp{}(x{})'.format(i + 1, i + 1, i + 1))

        for i in range(self.n_class):
            if i == 0:
                exec('x_agg = x{}'.format(i + 1)) ## static error
            else:
                exec('x_agg = torch.cat((x_agg, x{}), 1)'.format(i + 1))

        x = locals()['x_agg']  ## if not, NameError: name 'x' is not defined, it seems that exec create a new local?

        return x


if __name__ == '__main__':
    device = torch.device('cuda:1' if torch.cuda.is_available() else "cpu")
    model = MG_GATs(131, [60, 30, 15, 8]).to(device)

    print('fin')