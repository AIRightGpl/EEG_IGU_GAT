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
    from dataloader.eegmotormovement_loader import Construct_Dataset_withinSubj, Create_TrainTest
    from dataloader.public_109ser_loader import form_onesub_set
    from modules.Mydataset import Myset
    from torch.utils.data import DataLoader
    from models.EEG_CA_GATS import EEG_GAT_moduled
    ##================================================================================================================##
    # Here set the clip parameters and dataset parameter
    clip_length = 160
    clip_step = 20
    test_size = 0.3
    batch_size = 200

    # Here specify the device
    device = torch.device('cuda:1' if torch.cuda.is_available() else "cpu")

    for n_sub in range(1, 110):
        ##============================================================================================================##
        # load the model to device "EEG_GAT_moduled" is the model for person

        this_model = EEG_GAT_moduled(clip_length).to(device)

        ##============================================================================================================##
        # try to load the parameter from pretrained model
        # this_model.mcf_sequence.load_state_dict(torch.load('.pth'), strict=True)
        # this_model.mcf_sequence.load_state_dict(torch.load('.pth'), strict=True)

        ##============================================================================================================##
        # prepare the train and test dataset and create dataloader

        # dataset, labelset = Construct_Dataset_withinSubj(n_sub, size=clip_length, step=clip_step)
        # trainset, trainlab, testset, testlab = Create_TrainTest(dataset, labelset, testsize=test_size, randstat=0)
        trainset, trainlab, testset, testlab = form_onesub_set(n_sub, size=clip_length, step=clip_step)

        # different method for graph initiation
        ##------------------------------------------------------------------------------------------------------------##
        # edge_idx, _ = Initiate_graph(trainset, pt=0.75)  ## sparse rate = 0.75
        # edge_idx, _ = Initiate_fullgraph(input_channels=64)
        # edge_idx, _ = Initiate_clasgraph(trainset, trainlab, method='maximum_spanning')
        edge_idx, adj_mat = Initiate_regulgraph(input_channels=64, node_degree=14)
        ##------------------------------------------------------------------------------------------------------------##

        # load EEG channel distance matrix, and apply linear scale to distance matrix to assure each element of the mat
        # within range [0, 1], Thus, P(u, v) = D(u, v) * p(u, v) ranges from 0 to 1
        dist_atr = torch.tensor(loadtxt('64chans_distmat.csv', delimiter=','), device=device)
        dist_atr = (dist_atr - dist_atr.min()) / (dist_atr.max() - dist_atr.min())

        # apply dataloader to dataset
        train_set = Myset(trainset, trainlab)
        test_set = Myset(testset, testlab)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

        ##============================================================================================================##
        # Initiate the logging and the optimizer
        tra_wtr, tes_wtr = logging_Initiation("subject{}testsize{}_".format(n_sub, test_size), logroot='./log/pub_rg')
        lossfunc = torch.nn.CrossEntropyLoss()
        optmizer = torch.optim.Adam(this_model.parameters(), lr=1e-5,
                                    weight_decay=1e-4)  # note, when initiating optimizer,

        # Graph_Updater initiate here
        graph_base = Graph_Updater(device, method='rg')
        update_count = 0

        # initiation for training
        best_test_acc = 0
        curr_path = './saved_rg1/subject{}testsize{}'.format(n_sub, test_size)
        if not os.path.exists(curr_path): os.makedirs(curr_path, exist_ok=True)
        edge_idx_saved = curr_path + '/' + 'edge_index_ini.pth'
        graph_ini_saved = curr_path + '/' + 'graph_Ini.pth'
        if not os.path.exists(edge_idx_saved):
            edge_coo = {'edge_index': edge_idx}
            torch.save(obj=edge_coo, f=edge_idx_saved)
        if not os.path.exists(graph_ini_saved):
            graph_ini = {'adjacent_matrix': adj_mat}
            torch.save(obj=graph_ini, f=graph_ini_saved)

        ##============================================================================================================##
        # Training process
        for i in range(500):
            # set flag for updating graph
            flag = i % 10 == 0

            # train session, train epoch to back-propagate the grad and update parameter in both model and optimizer
            # train_epoch apply model.train() and test_epoch apply model.eval()
            attention_weight = train_epoch(this_model, train_loader, edge_idx, lossfunc, optmizer, device)
            # Get train result here. When flag and handle not None, the embedding/edge_value of one epoch samples will
            # be stored in Graph_updater for computing graph
            train_acc, train_loss = test_epoch(this_model, train_loader, edge_idx, lossfunc, device,
                                               flag=flag, handle=graph_base)

            # write train result to logging
            ##--------------------------------------------------------------------------------------------------------##
            print("train_result - epoch:{} - time:{} - loss:{} - acc:{:.4%}\n".format(i, time.time(), train_loss,
                                                                                      train_acc))
            tra_wtr.writetxt(
                "train_result - epoch:{} - time:{} - loss:{} - acc:{:.4%}\n".format(i, time.time(), train_loss,
                                                                                    train_acc))
            tra_wtr.writecsv([time.time(), i, train_loss, train_acc])
            ##--------------------------------------------------------------------------------------------------------##

            # test session, test_epoch applying to test_loader
            test_acc, test_loss = test_epoch(this_model, test_loader, edge_idx, lossfunc, device)

            # write test result to logging
            ##--------------------------------------------------------------------------------------------------------##
            print("test_result - epoch:{} - time:{} - loss:{} - acc:{:.4%}\n".format(i, time.time(), test_loss,
                                                                                     test_acc))
            tes_wtr.writetxt(
                "test_result - epoch:{} - time:{} - loss:{} - acc:{:.4%}\n".format(i, time.time(), test_loss,
                                                                                   test_acc))
            tes_wtr.writecsv([time.time(), i, test_loss, test_acc])
            ##--------------------------------------------------------------------------------------------------------##

            # when flag=True, update the graph with embeddings and free the space of Graph_updater
            if flag:
                new_adjacent_mat = graph_base.embedding2adj(prior_mat=dist_atr)
                update_count = update_count + 1
                curr_graph = {'adj_mat': new_adjacent_mat}
                edge_idx_saved = curr_path + '/' + 'graph_No{}_acc{}.pth'.format(update_count, test_acc)
                if not os.path.exists(edge_idx_saved):
                    torch.save(obj=curr_graph, f=edge_idx_saved)
            # save parameter of model
            ##--------------------------------------------------------------------------------------------------------##
            if test_acc >= best_test_acc + 0.05:
                best_test_acc = test_acc
                tim = time.strftime("%Y-%m-%d %H_%M_%S", time.localtime())
                all_state = {'model': {'mcf': this_model.mcf_sequence.state_dict(),
                                       'gat': this_model.GATs_sequence.state_dict(),
                                       'mlp': this_model.mlp_sequence.state_dict()},
                             'optimizer': optmizer.state_dict(),
                             'n_epoch': i}
                torch.save(obj=all_state, f=curr_path + '/' + 'bestmodel' + '.pth')
                if flag:
                    torch.save(obj=all_state, f=curr_path + '/' + tim + '.pth')

        tra_wtr.close()
        tes_wtr.close()
