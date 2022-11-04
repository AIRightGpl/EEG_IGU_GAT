import torch
from torch import nn
from torch_geometric.nn import global_mean_pool
from modules.EEG_MultiChan_filter import EEG_MCf
from modules.GATs_module import GATs
from modules.Multi_Proception_Layer import MLP


class EEG_GAT_moduled(nn.Module):

    # Same as EEG_GAT But split into modules for sake of pretrain

    def __init__(self, n_length, n_class=4):
        super(EEG_GAT_moduled, self).__init__()

        self.mcf_sequence = EEG_MCf()
        self.GATs_sequence = GATs(int(n_length / 5), [32, 16, 8, 4], n_head=4)   ## [60, 30, 15, 8] hidden_layer for 655 length
        self.mlp_sequence = MLP(16, [16, 8, n_class])  ## [16, 8, 4] hidden_layer for 655 length


    def forward(self, x, edge_index, batch):

        x = self.mcf_sequence(x)
        x, attention_weight = self.GATs_sequence(x, edge_index)
        x_embed = global_mean_pool(x, batch=batch)
        x = self.mlp_sequence(x_embed)

        return x, attention_weight

if __name__ == '__main__':
    import pickle
    from modules.dataset_conversion_module import SqlConnector
    from modules.Mydataset import Myset
    from torch.utils.data import DataLoader
    from toolbox_lib.Graph_tool import replicate_graph_batch

    dbcon = SqlConnector('eeg_motor_MI', 'subject_id', 'trails_id', 'eeg_data')
    data1 = dbcon.attract('eeg_data', 'subject_id', '1')
    trail = pickle.loads(data1[2][0])
    device = torch.device('cuda:1' if torch.cuda.is_available() else "cpu")
    model = EEG_GAT_moduled().to(device)
    edge_indx = torch.tensor([[0, 1, 1, 1, 2, 2, 2, 3, 3, 63], [1, 0, 2, 3, 1, 3, 63, 1, 2, 2]], dtype=torch.long,
                             device=device)
    model.train()
    loader = DataLoader(Myset(trail['data'], trail['label']), batch_size=9, shuffle=True)
    for index, data in enumerate(loader):
        eegdata = data[0].to(device)
        n_batch, _, _ = eegdata.shape
        batch_edge_idx, batch = replicate_graph_batch(edge_indx, n_batch)
        out = model(eegdata, batch_edge_idx, batch)
        print('done')
