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
    from toolbox_lib.trainer_re import logging_Initiation, train_epoch, test_epoch
    from toolbox_lib.Graph_tool import Initiate_fullgraph, Initiate_clasgraph, Initiate_regulgraph
    from dataloader.lowlimbmotorimagery_loader import form_onesub_set
    from modules.Mydataset import Myset
    from torch.utils.data import DataLoader
    # from models.EEG_GAT_modules import EEG_GAT_moduled ##before 02.20
    # from models.EEG_CA_GATS import EEG_GAT_moduled ## after 02.20  ##before 02.23
    # from models.EEG_GlobalAT_pretrain import EEG_GAT_ablation
    from models.EEG_CA_GATS_refine import EEG_GAT_moduled
    from models.EEGNet import EEGNet
    ##================================================================================================================##
    # Here set the clip parameters and dataset parameter
    clip_length = 400
    clip_step = 50
    test_size = 0.3
    batch_size = 100
    channels = 32
    seq = True
    # Here specify the device
    device = torch.device('cuda:6' if torch.cuda.is_available() else "cpu")

    for n_sub in range(10):
        ##============================================================================================================##
        # load the model to device "EEG_GAT_moduled" is the model for person

        this_model = EEG_GAT_moduled(clip_length, n_class=3).to(device)
        # this_model = EEGNet(125, n_class=3).to(device)
        ##============================================================================================================##
        # try to load the parameter from pretrained model
        # this_model.mcf_sequence.load_state_dict(torch.load('.pth'), strict=True)
        # this_model.mcf_sequence.load_state_dict(torch.load('.pth'), strict=True)

        ##============================================================================================================##
        # prepare the train and test dataset and create dataloader

        trainset, trainlab, testset, testlab = form_onesub_set(n_sub, size=clip_length, step=clip_step, sequence=seq)
        # edge_idx, _ = Initiate_graph(trainset, pt=0.75)  ## sparse rate = 0.75
        edge_idx, _ = Initiate_fullgraph(input_channels=32)
        train_set = Myset(trainset, trainlab)
        test_set = Myset(testset, testlab)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

        ##============================================================================================================##
        # initiate the logging and the optimizer
        tra_wtr, tes_wtr = logging_Initiation("subject{}_".format(n_sub), logroot='./log/self_ful_sin_lr-3')
        lossfunc = torch.nn.CrossEntropyLoss()
        optmizer = torch.optim.Adam(this_model.parameters(), lr=1e-3,
                                    weight_decay=1e-4)  # note, when initiating optimizer,
        # need to specify which parameter to apply
        best_test_acc = 0

        curr_path = './saved_self_ful_sin_lr-3/subject{}'.format(n_sub)
        if not os.path.exists(curr_path): os.makedirs(curr_path, exist_ok=True)
        edge_idx_saved = curr_path + '/' + 'edge_index.pth'
        if not os.path.exists(edge_idx_saved):
            graph = {'edge_index': edge_idx}
            torch.save(obj=graph, f=edge_idx_saved)

        ##============================================================================================================##
        # begin training, note
        for i in range(500):
            # train session, train epoch to back-propagate the grad and update parameter in both model and optimizer
            # train_epoch is the model.train() and test_epoch is in model.eval()
            attention_weight = train_epoch(this_model, train_loader, edge_idx, lossfunc, optmizer, device, n_class=3,
                                           n_chan=channels, sequence=seq)
            train_acc, train_loss = test_epoch(this_model, train_loader, edge_idx, lossfunc, device, n_class=3,
                                               n_chan=channels, sequence=seq)

            # write train result to logging
            print("train_result - epoch:{} - time:{} - loss:{} - acc:{:.4%}\n".format(i, time.time(), train_loss,
                                                                                      train_acc))
            tra_wtr.writetxt(
                "train_result - epoch:{} - time:{} - loss:{} - acc:{:.4%}\n".format(i, time.time(), train_loss,
                                                                                    train_acc))
            tra_wtr.writecsv([time.time(), i, train_loss, train_acc])

            ##--------------------------------------------------------------------------------------------------------##
            # test session, test_epoch applying to test_loader
            test_acc, test_loss = test_epoch(this_model, test_loader, edge_idx, lossfunc, device, n_class=3,
                                             n_chan=channels, sequence=seq)

            # write test result to logging
            print("test_result - epoch:{} - time:{} - loss:{} - acc:{:.4%}\n".format(i, time.time(), test_loss,
                                                                                     test_acc))
            tes_wtr.writetxt(
                "test_result - epoch:{} - time:{} - loss:{} - acc:{:.4%}\n".format(i, time.time(), test_loss,
                                                                                   test_acc))
            tes_wtr.writecsv([time.time(), i, test_loss, test_acc])

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
        tes_wtr.writetxt("test_result - best - acc:{:.4%}".format(best_test_acc))
        tes_wtr.writecsv([time.time(), i, 0, best_test_acc])
        tra_wtr.close()
        tes_wtr.close()
