import torch
import torch.nn as nn
import time
from numpy import *
from toolbox_lib.Graph_tool import replicate_graph_batch
import modules.Mydataset
from .set_logger import logger
from .Graph_tool import AdjMat2pygedge_index


def train_epoch(model, trainset: torch.utils.data.DataLoader, edge_idx, lossfunc, optimizer, device,
                n_class=4, n_chan=64, sequence=False):
    model.train()
    for index, data in enumerate(trainset):
        eegdata = data[0].to(device)
        if sequence:
            bz, tp, _, _= eegdata.shape
            bz = bz * tp
        else:
            bz, _, _ = eegdata.shape
        edge_index, batch = replicate_graph_batch(edge_idx, bz, device, node_num=n_chan)
        out, attention_weight = model(eegdata, edge_index, batch, sequence=sequence)
        # batch-size=200 and edge_idx(2,896) edge_index(2,179200) attention_weight(2)(2,192000)(192000,4), cuz self-loop
        label = nn.functional.one_hot((data[1] - 1).long(), n_class).to(device)  ## 4  #torch.tensor(data[1] - 1, dtype=torch.long)
        # label = (data.y-1).long()
        loss = lossfunc(out, label.float())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return attention_weight


def test_epoch(model, testset: torch.utils.data.DataLoader, edge_idx, lossfunc, device, n_class=4, n_chan=64, flag=False
               , handle=None, sequence=False):
    model.eval()

    correct = 0
    total_count = 0
    with torch.no_grad():
        for _, data in enumerate(testset):
            eegdata = data[0].to(device)
            if sequence:
                bz, tp, _, _ = eegdata.shape
                frg_bz = bz * tp
            else:
                bz, _, _ = eegdata.shape
                frg_bz = bz
            edge_index, batch = replicate_graph_batch(edge_idx, frg_bz, device, node_num=n_chan)
            # if use_graph.device not in ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']:
            #     edge_index = use_graph.to(self.device)
            out, attribute = model(eegdata, edge_index, batch, sequence=sequence)

            # Here to make append the embeddings or edge_values
            if handle is not None and flag:
                if handle.meth in ['sg', 'EDR', 'gm']:
                    handle.append_edge_value(attribute, frg_bz)
                else:
                    handle.append_node_embedding(attribute, frg_bz)
            # GAT in pyg will compute self-loop attention weight so for 0.75 there be 3088 edges
            pred = out.argmax(dim=1)
            label = nn.functional.one_hot((data[1] - 1).long(), n_class).to(device)
            # for MSEloss  ##4
            # label = (data.y-1).long() # for crossEntropy_loss for it will automatically transfer from index to one hot
            loss = lossfunc(out, label.float())
            correct += int((pred == (data[1] - 1).to(device)).sum())  # something wrong with the acc
            total_count += bz

    return correct / total_count, loss.item()


def logging_Initiation(description: str, logroot='./log'):
    import os
    import time
    from toolbox_lib.set_logger import logger
    if not os.path.exists(logroot):
        print('no existing logroot folder in current path')
        os.makedirs(logroot)
        print('create log folder')
    date = time.strftime('%Y_%m_%d', time.localtime())
    os.makedirs(logroot + '/' + date, exist_ok=True)
    lognam = time.strftime('%H-%M-%S', time.localtime())
    curpath = logroot + '/' + date + '/' + lognam
    os.makedirs(curpath)
    train_writer = logger(logpath=curpath + '/train')
    test_writer = logger(logpath=curpath + '/test')
    train_writer.begin(description+time.strftime('%Y_%m_%d-%H_%M_%S', time.localtime()))
    test_writer.begin(description + time.strftime('%Y_%m_%d-%H_%M_%S', time.localtime()))
    train_writer.writecsv(["time", "epochs", "train_loss", "train_acc"])
    test_writer.writecsv(["time", "epochs", "test_loss", "test_acc"])
    return train_writer, test_writer


