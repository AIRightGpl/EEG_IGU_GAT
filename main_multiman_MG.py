# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
#####
# But It is part of my work flow calling BCI Competition VIa dataset and GCN module to excute RGCN method
# to Reproduce the algorithm in paper:
#   Zhang Y, Huang H. New graph-blind convolutional network for brain connectome data analysis[C]
#   International Conference on Information Processing in Medical Imaging. Springer, Cham, 2019: 669-681.


import os
import torch
## Graph and data loading




if __name__ == '__main__':
    import time
    from numpy import loadtxt
    from toolbox_lib.trainer_re import logging_Initiation, train_epoch, test_epoch
    from toolbox_lib.Graph_tool import Initiate_fullgraph, Initiate_clasgraph, Initiate_regulgraph, Graph_Updater
    from dataloader.eegmotormovement_loader import Construct_Dataset_crosssub
    from modules.Mydataset import Myset
    from torch.utils.data import DataLoader
    from models.EEG_CA_mulclaGAT_softmax import EEG_multiGAT
    ##================================================================================================================##
    # Here set the clip parameters
    clip_length = 160
    clip_step = 20
    batch_size = 200
    channels = 64

    ##================================================================================================================##
    # Here specify the device and load the model to device("EEG_GAT_moduled" is the model that devide
    device = torch.device('cuda:1' if torch.cuda.is_available() else "cpu")
    this_model = EEG_multiGAT(clip_length).to(device)

    ##================================================================================================================##
    # try to load the parameter from pretrained model
    # this_model.mcf_sequence.load_state_dict(torch.load('.pth'), strict=True)
    # this_model.mcf_sequence.load_state_dict(torch.load('.pth'), strict=True)

    ##================================================================================================================##
    # prepare the train and test dataset and create dataloader
    trai_sub_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    test_sub_list = [10, 11, 12, 13]
    trainset, trainlab, testset, testlab = Construct_Dataset_crosssub(trai_sub_list, test_sub_list, size=clip_length,
                                                                      step=clip_step)
    # different method for graph initiation
    ##------------------------------------------------------------------------------------------------------------##
    # edge_idx, _ = Initiate_graph(trainset, pt=0.75)  ## sparse rate = 0.75
    # edge_idx, _ = Initiate_fullgraph(input_channels=64)
    edge_idx = Initiate_clasgraph(trainset, trainlab, method='maximum_spanning')
    # edge_idx, adj_mat = Initiate_regulgraph(input_channels=64, node_degree=14)
    ##------------------------------------------------------------------------------------------------------------##

    dist_atr = torch.tensor(loadtxt('64chans_distmat.csv', delimiter=','), device=device)
    train_set = Myset(trainset, trainlab)
    test_set = Myset(testset, testlab)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    ##================================================================================================================##
    # initiate the logging, graph_updater and the optimizer
    tra_wtr, tes_wtr = logging_Initiation("crosssub1-9train_10-13test", logroot='./log/multi_MG')
    lossfunc = torch.nn.CrossEntropyLoss()
    optmizer = torch.optim.Adam(this_model.parameters(), lr=1e-6, weight_decay=1e-4)  # note, when initiating optimizer,
                                                                            # need to specify which parameter to apply
    best_test_acc = 0

    curr_path = './saved_multi_MG/tra' + ''.join(list(map(lambda x: str(x), trai_sub_list))) + 'tes' + ''.join(
        list(map(lambda x: str(x), test_sub_list)))
    if not os.path.exists(curr_path): os.makedirs(curr_path, exist_ok=True)
    edge_idx_saved = curr_path + '/' + 'edge_index_init.pth'
    if not os.path.exists(edge_idx_saved):
        graph = {'edge_index': edge_idx}
        torch.save(obj=graph, f=edge_idx_saved)

    ##================================================================================================================##
    # begin training, note
    for i in range(800):
        # train session, train epoch to back-propagate the grad and update parameter in both model and optimizer
        # train_epoch is the model.train() and test_epoch is in model.eval()
        flag = i % 10 == 0 and i > 0
        attention_weight = train_epoch(this_model, train_loader, edge_idx, lossfunc, optmizer, device)
        train_acc, train_loss = test_epoch(this_model, train_loader, edge_idx, lossfunc, device)

        # write train result to logging
        print("train_result - epoch:{} - time:{} - loss:{} - acc:{:.4%}\n".format(i, time.time(), train_loss,
                                                                                  train_acc))
        tra_wtr.writetxt(
            "train_result - epoch:{} - time:{} - loss:{} - acc:{:.4%}\n".format(i, time.time(), train_loss,
                                                                                train_acc))
        tra_wtr.writecsv([time.time(), i, train_loss, train_acc])

        ##------------------------------------------------------------------------------------------------------------##
        # Here attract the attention weight from the GAT process, and compute the updated-graph(whether WS from lower
        # sparse threshold through random generate method or EDR-based method
        ##------------------------------------------------------------------------------------------------------------##
        # test session, test_epoch applying to test_loader
        test_acc, test_loss = test_epoch(this_model, test_loader, edge_idx, lossfunc, device)

        # write test result to logging
        print("test_result - epoch:{} - time:{} - loss:{} - acc:{:.4%}\n".format(i, time.time(), test_loss,
                                                                                  test_acc))
        tes_wtr.writetxt("test_result - epoch:{} - time:{} - loss:{} - acc:{:.4%}\n".format(i, time.time(), test_loss,
                                                                                test_acc))
        tes_wtr.writecsv([time.time(), i, test_loss, test_acc])

        # save parameter of model

        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            tim = time.strftime("%Y-%m-%d %H_%M_%S", time.localtime())
            all_state = {'model': {'mcf': this_model.mcf_sequence.state_dict(),
                                   'm_gat': this_model.GATs_sequence.state_dict(),
                                   'pred': this_model.probpredict.state_dict()},
                         'optimizer': optmizer.state_dict(),
                         'n_epoch': i}
            torch.save(obj=all_state, f=curr_path + '/' + 'bestmodel' + '.pth')
            if flag:
                graph = {'edge_index': edge_idx}
                torch.save(obj=graph, f=curr_path + '/' + tim + 'edg_idx.pth')
                torch.save(obj=all_state, f=curr_path + '/' + tim + '.pth')

    tra_wtr.close()
    tes_wtr.close()
