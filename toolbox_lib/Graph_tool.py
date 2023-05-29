## Graph defination:

import torch
import torch.nn.functional as F
import os.path as osp
import math
from copy import deepcopy
from matplotlib import pyplot as plt
from copy import deepcopy
from typing import List, Union
import numpy as np
import networkx as nx
from networkx.algorithms.tree import maximum_spanning_tree

layout_BCIV2a = None


def Linear_normalize(tensor_in: torch.Tensor) -> torch.Tensor:
    tensor_out = (tensor_in - tensor_in.min()) / (tensor_in.max() - tensor_in.min())
    return tensor_out


def Initiate_regulgraph(input_channels: int = 64, node_degree: int = 12, randomseed: int = 66) -> torch.Tensor:
    rg = nx.random_graphs.random_regular_graph(node_degree, input_channels, seed=randomseed)
    adj_mat = torch.from_numpy(nx.to_numpy_array(rg))
    edge_index = torch.nonzero(adj_mat).T
    return edge_index, adj_mat


def Initiate_fullgraph(input_channels: int = 64) -> torch.Tensor:
    full = torch.full((input_channels, input_channels), 1)
    edge_index = torch.transpose(torch.nonzero(full), 1, 0)
    return edge_index, full


def Initiate_graph(lst_samples: List[torch.Tensor], pt=0.75, self_loop=True, method='proportional') -> torch.Tensor:
    assert method in ['proportional', 'above_average', 'absolute'], 'Unsupported sparse method'
    stacked = []
    for i, data in enumerate(lst_samples):
        corrmat = torch.corrcoef(data)
        corrmat = torch.abs(corrmat)
        corrmat = corrmat - torch.diag_embed(torch.diag(corrmat))
        stacked.append(corrmat)
    avecormat = torch.stack(stacked, dim=0).mean(dim=0)
    sparsed = sparsematrix(avecormat, threshold_method=method, threshold=pt)
    if self_loop:
        sparsed = sparsed + torch.eye(sparsed.shape[0])
    else:
        pass
    edge_index = torch.transpose(torch.nonzero(sparsed), 1, 0)
    return edge_index, sparsed


def Initiate_clasgraph(lst_samples: List[torch.Tensor], lst_label: List[torch.Tensor], pt=0.75, self_loop=True,
                       method='maximum_spanning') -> torch.Tensor:
    assert len(lst_samples) == len(lst_label), 'invalidate sample & label, the length is not align'
    data_dict = dict.fromkeys(set(lst_label), [])
    idx_dict = dict.fromkeys(set(lst_label))
    for i in range(len(lst_samples)):
        data_dict[lst_label[i]].append(lst_samples[i])
    for ind, key in enumerate(data_dict):
        idx_dict[key], _ = Initiate_graph(data_dict[key], pt=pt, self_loop=self_loop, method=method)

    return idx_dict


def test(func):
    def warpper():
        print('test begin')
        low_tran_mat = torch.tensor([[1, 0, 0, 0], [6, 2, 0, 0], [1, 3, 2, 0], [3, 4, 7, 1]])
        hig_tran_mat = torch.tensor([[1, 6, 1, 3], [0, 2, 3, 4], [0, 0, 2, 7], [0, 0, 0, 1]])
        stls = func(hig_tran_mat)
        print('test done')
        return stls

    return warpper()


def replicate(edge_index: torch.Tensor, batch_size: int, device, node_num: int = 64) -> torch.Tensor:
    edge_index = edge_index.to(device)  ## the setting of step may occur error, because the torch.sparse_coo_tensor
    # step = torch.max(edge_index) + 1       ## throw error of imcompetanc max-indice:63 but found 65
    step = node_num
    batch = []
    for i in range(batch_size):
        atb = edge_index + step * i
        batch.extend([i] * step)
        if i == 0:
            new = edge_index
        else:
            new = torch.cat((new, atb), dim=1)
    new = new.to(device)
    batch = torch.tensor(batch, dtype=torch.int64, device=device)

    return new, batch


def replicate_graph_batch(edge_index: Union[torch.Tensor, dict], batch_size: int, device, node_num: int = 64) -> torch.Tensor:
    if isinstance(edge_index, torch.Tensor):
        new, batch = replicate(edge_index, batch_size, device, node_num=node_num)
    elif isinstance(edge_index, dict):
        new = dict.fromkeys(edge_index)
        batch = dict.fromkeys(edge_index)
        for key in edge_index.keys():
            new[key], batch[key] = replicate(edge_index[key], batch_size, device, node_num=node_num)
    else:
        print('error in \'replicate_graph_batch\' unsupported \'edge_index\' type')
    return new, batch


def lower_trangular_matrix_2_symmetric(ltm: torch.Tensor) -> torch.Tensor:
    [a, b] = ltm.shape
    assert (a == b), "lower_triangular_matrix to be transformed to Symmetric matrix must be a square matrix"
    symmetric_mat = deepcopy(ltm)
    for k in range(a - 1):
        symmetric_mat[k, k + 1:] = ltm[k + 1:, k].transpose(-1, 0)
    return symmetric_mat


def higher_trangular_matrix_2_symmetric(htm: torch.Tensor) -> torch.Tensor:
    [a, b] = htm.shape
    assert (a == b), "lower_triangular_matrix to be transformed to Symmetric matrix must be a square matrix"
    symmetric_mat = deepcopy(htm)
    for k in range(a - 1):
        symmetric_mat[k + 1:, k] = htm[k, k + 1:].transpose(-1, 0)
    return symmetric_mat


def KeepHTM(converMat):
    [a, b] = converMat.shape
    assert (a == b), "converMat to be transformed to Symmetric matrix must be a square matrix"
    for i in range(a):
        for j in range(i):
            converMat[i, j] = 0
    output = converMat
    return output


def sort_matele(corrmat: torch.Tensor, description='symmetric'):
    ## 'corrmat' is a torch ndarray  square matrix
    ## 'description' to decide 'corrmat' whether symmetric
    ## 'output' indice where larger element ranks lower and element in diagonal ranks 0,
    ## 'n' is the number of total element that not in diagonal(depends on 'description')
    lis = []
    [a, b] = corrmat.shape
    assert (a == b), "corrmatrix must be a square matrix"
    assert (description in ['symmetric',
                            'asymmetric']), "only support matrix description:\'symmetric\', \'asymmetric\'!"
    output = torch.zeros(a, b)
    if description == 'symmetric':
        for i in range(1, a):
            for j in range(i):
                lis.append({'value': corrmat[i, j], 'indice1': i, 'indice2': j})
    elif description == 'asymmetric':
        for i in range(a):
            for j in range(b):
                if i == j:
                    lis.append({'value': 0, 'indice1': i, 'indice2': j})  ##corrmat[i, j]
                else:
                    lis.append({'value': corrmat[i, j], 'indice1': i, 'indice2': j})
    sorlis = sorted(lis, key=lambda k: k['value'], reverse=True)
    for n in range(1, len(sorlis) + 1):
        output[sorlis[n - 1]['indice1'], sorlis[n - 1]['indice2']] = n
    if description == 'symmetric':
        output = lower_trangular_matrix_2_symmetric(output)
    return output, n


def sparsematrix(corrmat: torch.Tensor, threshold_method='proportional', threshold=0.6, drop_index=0) -> torch.Tensor:
    [a, b] = corrmat.shape
    assert (a == b), "correlation matrix must be a square matrix"
    indice, n_edges = sort_matele(corrmat)
    baseMask = torch.where(drop_index < indice, torch.ones(a, b), torch.zeros(a, b))

    if threshold_method == 'proportional':
        propthres = round(n_edges * threshold)
        sparseMask = torch.where(indice <= propthres, baseMask, torch.zeros(a, b))

    elif threshold_method == 'absolute':
        absthres = threshold
        sparseMask = torch.where(absthres <= corrmat, baseMask, torch.zeros(a, b))

    elif threshold_method == 'above_average':
        absthres = torch.mean(corrmat)
        sparseMask = torch.where(absthres <= corrmat, baseMask, torch.zeros(a, b))

    elif threshold_method == 'maximum_spanning':
        formal_mat = nx.from_numpy_matrix(corrmat.numpy())
        sparseMask = maximum_spanning_tree(formal_mat)
        ## format purpose
        sparseMask = nx.adjacency_matrix(sparseMask).todense()
        sparseMask = torch.from_numpy(np.array(sparseMask))
        # print('\'Graph\' object has no attribute \'shape\'')

    return sparseMask


# if round(maxn*threshold)%2 == 0:
#     absthres = round(maxn*threshold)
# elif int(maxn*threshold) < round(maxn*threshold):
#     absthres = int(maxn*threshold)
# else:
#     absthres = math.ceil(maxn*threshold)  ## this is a way to apply propotional threshold to whole matrix, not reasonalble enough

# def update_graph(embending: Tensor, inten):
#     [indw, indl] = embending.shape
#     pass


def AdjMat2pygedge_index(AdjMat: torch._tensor) -> torch.Tensor:
    index = torch.nonzero(AdjMat)  ##from adjmat representation to pyg format edge_index
    edge_index = torch.tensor(index.t(), dtype=torch.long)
    return edge_index


def visualize_data(attr: List[str], index: torch.Tensor, color):
    G = nx.Graph()
    name = attr
    edge_index = index.tolist()
    G.add_edges_from(edge_index)
    nx.draw(G)
    plt.show()

## DEPRECATED
# class Graph_holder():
#     def __init__(self, device, method='Random Generate'):
#         assert method in ['Random Generate', 'metric sparse']
#         self.embeddings = []
#         self.device = device
#         self.method = method
#
#     def __swm_EdgeProb(self, embedd1, embedd2, para1=2, para2=0.01):
#         prob = para2 / (torch.norm(embedd1 - embedd2) ** para1 + para2)
#         return prob
#
#     def __swm_attention_EdgeProb(self, attention_wei, param_mat):
#         pass  ## current framework need to transfer constantly between cpu and device (usually 'cuda:1'). Thus, abandoned
#
#     def __metric_EdgeProb(self, embedd1, embedd2):
#         prob = torch.cosine_similarity(embedd1, embedd2, dim=0)
#         return prob
#
#     def Embedds2Adjmat(self):
#         assert self.embeddings, "embedding list should not be empty, try append before call this"
#         batchpileadj = []
#         for embeM in self.embeddings:
#             embeM = torch.squeeze(torch.tensor(embeM, dtype=torch.float32, device=self.device))
#             proadj = torch.zeros([embeM.shape[0], embeM.shape[0]], dtype=torch.float32, device=self.device)
#             for i in range(embeM.shape[0] - 1):
#                 for j in range(i + 1, embeM.shape[0]):
#                     if self.method == 'Random Generate':
#                         proadj[i, j] = self.__swm_EdgeProb(embeM[i, :], embeM[j, :])
#                     elif self.method == 'metric sparse':
#                         proadj[i, j] = self.__metric_EdgeProb(embeM[i, :], embeM[j, :])
#             batchpileadj.append(proadj)
#         expectprob = torch.stack(batchpileadj, dim=0).mean(dim=0)
#         # expectprob = HTM2SymmetricMat(expectprob)
#         if self.method == 'Random Generate':
#             randmat = torch.rand(embeM.shape[0], embeM.shape[0], dtype=torch.float32, device=self.device)
#             randmat = KeepHTM(randmat)
#             mask = expectprob.ge(randmat)
#             expectadj = higher_trangular_matrix_2_symmetric(mask.int())
#         elif self.method == 'metric sparse':
#             bimat = sparasematrix(expectprob)
#             expectadj = higher_trangular_matrix_2_symmetric(bimat.int())
#         return expectadj
#
#     def append_embeddings(self, embedding):
#         if len(embedding.shape) > 2:
#             leng = embedding.shape[0]
#             self.embeddings.extend(deepcopy(torch.tensor_split(embedding, leng, dim=0)))
#         else:
#             self.embeddings.append(deepcopy(embedding))
#
#     def clear_embeddings(self):
#         self.embeddings.clear()

class Graph_Updater():
    # Graph_updater is the Successor of the Graph_holder

    def __init__(self, device, method='rg'):
        # method here is set to 'rg' (Random Generate) for default.
        # Here the choice:
        #    method= 'rg': Random Generated method, which is based on WS (Watts-Strogatz model)model, is construct
        #                   as follows: (1)Create a circular mesh with N nodes of average in-out degree 2K. Each node is
        #                                  connected to the K nearest neighbors on either side.
        #                               (2)For each edge in the graph, reconnect the target node with probability β.
        #                                  Reconnected edges that being neither repeated nor self-looped.
        #                   After performing the first step, the graph will be a perfect circular grid. So when β = 0,
        #                   no edges are reconnected and the model degrade to circular mesh. Whereas if β = 1, then all
        #                   edges will be reconnected and the ring mesh will become a random graph.
        #                   The implementation of Random Generate to GCN (https://doi.org/10.1007/978-3-030-20351-1_52)
        #                   The probability of reconnecting nodes p_i,j = epsilon/(|hi-hj|^delta+epsilon) see reference
        #                   of paper above  (From which world is your graph. Advances in Neural Information Processing
        #                   Systems, vol. 30, pp. 1469–1479.)
        #                   # 2023.05 update: new 'rg' apply 'gm' method P(u,v)= E(u,v)^η x K(u,v)^γ
        #            'ms': Metric Sparse method, a straight-forward method based on cosine similarity and sparse method,
        #                   (like computing PLV from EEG, Here compute cosine similarity from graph embedding)
        #            'sg': Proposed Stepped Generated method, following the principles:(1) minimal wiring cost ;
        #                   (2) small-world model(maybe not WS) ; (3) with prior-knowledge from Neuron-Science
        #                   (for example: the position of each electrode, which electrode reflect the brain area that
        #                   contribute to the classification most)
        #           'EDR': Exponential Distance Rule, proposed in https://doi.org/10.1016/j.neuron.2013.07.036
        #            'gm': Reference  https://doi.org/10.1016/j.neuroimage.2015.09.041 , a more advanced model than EDR
        #                   P(u,v)= E(u,v)^η x K(u,v)^γ , which E(u,v) denotes the Euclidean distance between brain
        #                   regions u and v, K(u, v), represents an arbitrary non-geometric relationship between nodes
        #                   u and v
        assert method in ['rg', 'ms', 'sg', 'EDR', 'gm'], 'error, method only support rg, ms, sg, EDR!'
        self.meth = method
        self.dev = device
        self.edge_value = None
        self.embedding = None

    def append_edge_value(self, edge_attention_w: torch.Tensor, batch_size: int, chan_num: int = 64):
        pick = int(edge_attention_w[0].shape[1] / batch_size) - chan_num
        if self.edge_value == None:
            edge_index = edge_attention_w[0][:, :pick]
            edge_value = edge_attention_w[1][:pick, :].mean(dim=1)
            self.edge_value = torch.sparse_coo_tensor(edge_index, edge_value, (chan_num, chan_num), device=self.dev)
        else:
            edge_index = edge_attention_w[0][:, :pick]
            edge_value = edge_attention_w[1][:pick, :].mean(dim=1)
            weight_mat = torch.sparse_coo_tensor(edge_index, edge_value, (chan_num, chan_num), device=self.dev)
            if len(self.edge_value.shape) == 2:
                self.edge_value = torch.stack([self.edge_value, weight_mat], dim=0)
            else:
                self.edge_value = torch.cat([self.edge_value, weight_mat.unsqueeze(0)], dim=0)

    def append_node_embedding(self, embedding: torch.Tensor, batch_size: int):
        if self.embedding == None:
            if batch_size >= 1:
                self.embedding = torch.stack(torch.tensor_split(embedding, batch_size, dim=0)).squeeze()  ## align with mini-batch
                # which is applied in model the embedding should be (#batch*node, #hidden(embedding)) and .detach()
            else:
                self.embedding = embedding
        else:
            if batch_size >= 1:
                self.embedding = torch.cat(
                    (self.embedding, torch.stack(torch.tensor_split(embedding, batch_size, dim=0)).squeeze()))
            else:
                self.embedding = torch.cat((self.embedding, embedding))

    def clear_attr(self):
        self.embedding = None
        self.edge_value = None

    def rand_reconnect(self, probability_matrix: torch.Tensor) -> torch.Tensor:
        randmat = torch.rand(probability_matrix.shape[0], probability_matrix.shape[0],
                             dtype=torch.float32, device=self.dev)
        randmat = KeepHTM(randmat)
        mask = probability_matrix.ge(randmat)
        expect_adj = higher_trangular_matrix_2_symmetric(mask.int())
        return expect_adj

    def weight_based_sparse(self, weight_matrix, threshold=0.6):
        bimat = sparsematrix(weight_matrix, hreshold=threshold)
        expect_adj = higher_trangular_matrix_2_symmetric(bimat.int())
        return expect_adj

    def metric_edgeprob(self, embedds):
        embedd1 = torch.unsqueeze(embedds, dim=1)
        embedd2 = torch.unsqueeze(embedds, dim=0)
        prob = torch.cosine_similarity(embedd1, embedd2, dim=-1)
        return prob

    def __swm_attention_edgeprob(self, edge_value, param):
        rec_ed_v = 1.0 / edge_value
        recip_edge_value = torch.where(torch.isinf(rec_ed_v), torch.full_like(rec_ed_v, 0), rec_ed_v)
        prob = param // torch.add(recip_edge_value, param)
        # but this method is not applicable, due to edge_value is sparse thus a lot of 0 s and the reciprocal create Inf
        # so set Inf to 0 and this method will only work on the already exist edges and node-pairs, the final result
        # would be a more sparse graph
        return prob

    def swm_edgeprob(self, embedds, para1=2, para2=0.005):  ##para2=0.01(0322)  ##para2=0.005(0323)
        embedd1 = torch.unsqueeze(embedds, dim=1)
        embedd2 = torch.unsqueeze(embedds, dim=0)
        prob = para2 / (torch.norm(embedd1 - embedd2, p=2, dim=-1) ** para1 + para2)
        # prob = para2 / (torch.cdist(embedd1, embedd2))
        return prob

    def __edr_edgeprob(self, dist_mat):
        prob = 0.19 * torch.exp((-0.19) * dist_mat)
        return prob

    def __egm_edgeprob(self, edge_value, dist_mat, eta=-5, gamma=-1):
        # when eta < 0 short-range connection are favored eta > 0 long-range connection are favored
        # gamma plays the same role as eta, it controls whether probability of connection altered by embedds distance

        prob = (dist_mat ** eta) * (edge_value ** gamma)
        return prob

    def edgevalue2adj(self, prior_mat=None):
        assert self.edge_value is not None, 'edge attention values is empty, try \'append_edge_value\' before computing'
        ed_value = self.edge_value.to_dense().mean(dim=0)
        if self.meth == 'sg':
            weight_matrix = self.__swm_attention_edgeprob(ed_value, prior_mat)
        elif self.meth == 'EDR':
            weight_matrix = self.__edr_edgeprob(prior_mat)
        elif self.meth == 'gm':
            weight_matrix = self.__egm_edgeprob(ed_value, prior_mat)
        else:
            print('unsupported edge to adj method')
        expect_adj = self.rand_reconnect(weight_matrix)

        # clear the storage
        self.clear_attr()
        return expect_adj

    def embedding2adj(self, prior_mat=None, yta=-1):
        assert self.embedding is not None, 'node embedding is empty, try \'append_node_embedding\' before computing'
        embedds = self.embedding.mean(dim=0)
        if self.meth == 'ms':
            if prior_mat is None:
                weight_matrix = self.metric_edgeprob(embedds)
            else:
                recip_dis_pri = prior_mat ** yta
                recip_dis = torch.where(torch.isinf(recip_dis_pri), torch.full_like(recip_dis_pri, 1), recip_dis_pri)
                recip_dis = Linear_normalize(recip_dis)  ## linear normalize will cause bias
                recip_dis = torch.where((recip_dis == 0), torch.full_like(recip_dis, 1), recip_dis)
                weight_matrix = self.metric_edgeprob(embedds) * recip_dis
            # expect_adj = self.weight_based_sparse(weight_matrix)
            expect_adj = self.rand_reconnect(weight_matrix)
        elif self.meth == 'rg':
            if prior_mat is None:
                weight_matrix = self.swm_edgeprob(embedds)
            else:
                recip_dis_pri = prior_mat ** yta
                recip_dis = torch.where(torch.isinf(recip_dis_pri), torch.full_like(recip_dis_pri, 1), recip_dis_pri)
                recip_dis = Linear_normalize(recip_dis)   ## linear normalize will cause bias
                recip_dis = torch.where((recip_dis == 0), torch.full_like(recip_dis, 1), recip_dis)
                weight_matrix = self.swm_edgeprob(embedds) * recip_dis
                # weight_matrix = Linear_normalize(weight_matrix)
                expect_adj = self.rand_reconnect(weight_matrix)
        else:
            print('unsupported node embedding to adj method')

        # clear the storage
        self.clear_attr()
        return expect_adj

        # pro_adj = torch.zeros([ed_value.shape[0], ed_value.shape[0]], dtype=torch.float32, device=self.device)



if __name__ == '__main__':
    # corrm = np.array([[0, 6, 3, 2], [6, 0, 7, 5], [3, 7, 0, 1], [2, 5, 1, 0]])
    # # adjmat = torch.tensor([[0,1,0],[1,0,1],[0,1,0]],dtype=torch.int)
    # # testmat = torch.tensor([[1,4,2,3],[6,2,1,5],[7,8,7,9],[9,6,1,5]],dtype=torch.float32)
    # # htm = KeepHTM(testmat)
    # # edge_index = AdjMat2pygedge_index(adjmat)
    # out, num = sort_matele(corrm)
    # sparse = sparasematrix(corrm, thresmethod='proportional', threshold=0.45)
    # print('check processe')
    # stls = higher_trangular_matrix_2_symmetric()
    # print('done')
    ##
    # device = torch.device('cuda:1' if torch.cuda.is_available() else "cpu")
    # edge_idx = torch.tensor([[0, 0, 0, 1, 1, 2, 2, 3], [1, 2, 3, 0, 2, 0, 1, 0]], dtype=torch.long, device=device)
    # new, batch = replicate_graph_batch(edge_idx, 6)
    # print('done')
    ##
    # test = Initiate_regulgraph(64, 12)
    # print('done')

    from dataloader.public_109ser_loader import form_onesub_set
    trainset, trainlab, testset, testlab = form_onesub_set(9, size=160, step=20)
    edge_idx, _ = Initiate_graph(trainset, method='above_average')
    print('stop')