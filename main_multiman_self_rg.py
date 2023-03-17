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
    from dataloader.lowlimbmotorimagery_loader import form_multsub_set
    from modules.Mydataset import Myset
    from torch.utils.data import DataLoader
    from models.EEG_CA_GATS_refine import EEG_GAT_moduled
    ##================================================================================================================##
    # Here set the clip parameters
    clip_length = 400
    clip_step = 50
    batch_size = 30
    channels = 32
    seq = True

    ##================================================================================================================##
    # Here specify the device and load the model to device("EEG_GAT_moduled" is the model that devide
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    this_model = EEG_GAT_moduled(clip_length, n_class=3).to(device)

    ##================================================================================================================##
    # try to load the parameter from pretrained model
    # this_model.mcf_sequence.load_state_dict(torch.load('.pth'), strict=True)
    # this_model.mcf_sequence.load_state_dict(torch.load('.pth'), strict=True)

    ##================================================================================================================##
    # prepare the train and test dataset and create dataloader
    trai_sub_list = [0, 3, 5, 7, 8]
    test_sub_list = [6, 1]
    trainset, trainlab, testset, testlab = form_multsub_set(trai_sub_list, test_sub_list, size=clip_length,
                                                            step=clip_step, seq=seq)
    # different method for graph initiation
    ##------------------------------------------------------------------------------------------------------------##
    # edge_idx, _ = Initiate_graph(trainset, pt=0.75)  ## sparse rate = 0.75
    edge_idx, adj_mat = Initiate_fullgraph(input_channels=32)
    # edge_idx, _ = Initiate_clasgraph(trainset, trainlab, method='maximum_spanning')
    # edge_idx, adj_mat = Initiate_regulgraph(input_channels=channels, node_degree=8)
    ##------------------------------------------------------------------------------------------------------------##

    # load EEG channel distance matrix, and apply linear scale to distance matrix to assure each element of the mat
    # within range [0, 1], Thus, P(u, v) = D(u, v) * p(u, v) ranges from 0 to 1
    dist_atr = torch.tensor(loadtxt('32chans_distmat.csv', delimiter=','), device=device)
    # dist_atr = (dist_atr - dist_atr.min()) / (dist_atr.max() - dist_atr.min())
    ##============================================================================================================##
    # We should not resize this dis_atr to 0-1, but consider the formular P(u,v)=E(u,v)^yta * K(u,v)^gamma, and
    # yta=-1, so, should not resize this to 0-1 but need to resize to 1-1^+
    dist_atr = dist_atr / (dist_atr.min())

    train_set = Myset(trainset, trainlab)
    test_set = Myset(testset, testlab)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    ##================================================================================================================##
    # initiate the logging and the optimizer
    tra_wtr, tes_wtr = logging_Initiation("cross_subject_self", logroot='./log/self_cross_rgfRGRU_lr-2')
    lossfunc = torch.nn.CrossEntropyLoss()
    optmizer = torch.optim.Adam(this_model.parameters(), lr=1e-2, weight_decay=1e-4)  # note, when initiating optimizer,
                                                                            # need to specify which parameter to apply
    best_test_acc = 0

    # Graph_Updater initiate here
    graph_base = Graph_Updater(device, method='rg')
    update_count = 0

    curr_path = './saved_self_cross_rgfRGRU_lr-2/tra' + ''.join(list(map(lambda x: str(x), trai_sub_list))) + 'tes' + ''.join(
        list(map(lambda x: str(x), test_sub_list))) + '1'
    if not os.path.exists(curr_path): os.makedirs(curr_path, exist_ok=True)
    edge_idx_saved = curr_path + '/' + 'edge_index_Ini.pth'
    graph_ini_saved = curr_path + '/' + 'graph_Ini.pth'
    if not os.path.exists(edge_idx_saved):
        graph = {'edge_index': edge_idx}
        torch.save(obj=graph, f=edge_idx_saved)
    if not os.path.exists(graph_ini_saved):
        graph_ini = {'adjacent_matrix': adj_mat}
        torch.save(obj=graph_ini, f=graph_ini_saved)

    ##================================================================================================================##
    # begin training, note
    for i in range(1200):
        # set flag for updating graph
        flag = i % 10 == 0 and i != 0

        # train session, train epoch to back-propagate the grad and update parameter in both model and optimizer
        # train_epoch is the model.train() and test_epoch is in model.eval()
        attention_weight = train_epoch(this_model, train_loader, edge_idx, lossfunc, optmizer, device, n_class=3,
                                       n_chan=channels, sequence=seq)
        train_acc, train_loss = test_epoch(this_model, train_loader, edge_idx, lossfunc, device, n_class=3,
                                           n_chan=channels, flag=flag, handle=graph_base, sequence=seq)

        # write train result to logging
        print("train_result - epoch:{} - time:{} - loss:{} - acc:{:.4%}\n".format(i, time.time(), train_loss,
                                                                                  train_acc))
        tra_wtr.writetxt(
            "train_result - epoch:{} - time:{} - loss:{} - acc:{:.4%}\n".format(i, time.time(), train_loss,
                                                                                train_acc))
        tra_wtr.writecsv([time.time(), i, train_loss, train_acc])

        ##------------------------------------------------------------------------------------------------------------##
        # test session, test_epoch applying to test_loader
        test_acc, test_loss = test_epoch(this_model, test_loader, edge_idx, lossfunc, device, n_class=3,
                                         n_chan=channels, sequence=seq)

        # write test result to logging
        print("test_result - epoch:{} - time:{} - loss:{} - acc:{:.4%}\n".format(i, time.time(), test_loss,
                                                                                  test_acc))
        tes_wtr.writetxt("test_result - epoch:{} - time:{} - loss:{} - acc:{:.4%}\n".format(i, time.time(), test_loss,
                                                                                test_acc))
        tes_wtr.writecsv([time.time(), i, test_loss, test_acc])

        # when flag=True, update the graph with embeddings and free the space of Graph_updater
        if flag:
            new_adjacent_mat = graph_base.embedding2adj(prior_mat=dist_atr)
            update_count = update_count + 1
            curr_graph = {'adj_mat': new_adjacent_mat}
            edge_idx_saved = curr_path + '/' + 'graph_No{}_acc{}.pth'.format(update_count, test_acc)
            if not os.path.exists(edge_idx_saved):
                torch.save(obj=curr_graph, f=edge_idx_saved)

            # change the edge_idx
            edge_idx = torch.transpose(torch.nonzero(new_adjacent_mat), 1, 0)

        # save parameter of model
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            tim = time.strftime("%Y-%m-%d %H_%M_%S", time.localtime())
            # all_state = {'model': {'mcf': this_model.mcf_sequence.state_dict(),
            #                        'gat': this_model.GATs_sequence.state_dict(),
            #                        'mlp': this_model.mlp_sequence.state_dict()},
            #              'optimizer': optmizer.state_dict(),
            #              'n_epoch': i}
            all_state = {'model': this_model.state_dict()}
            torch.save(obj=all_state, f=curr_path + '/' + 'bestmodel' + '.pth')
            if i % 10 == 0 and i > 0:
                torch.save(obj=all_state, f=curr_path + '/' + tim + '.pth')

    tra_wtr.close()
    tes_wtr.close()
