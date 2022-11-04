import os
import torch
from toolbox_lib.Graph_tool import Initiate_graph
from dataloader.eegmotormovement_loader import Construct_Dataset_withinSubj, Create_TrainTest
from dataloader.lowlimbmotorimagery_loader import form_onesub_set
from toolbox_lib.Graph_tool import Initiate_graph
from modules.Mydataset import Myset

test_size = 0.3


def eegMIdataset(clip_length, clip_step):
    for n_sub in range(1, 110):
        dataset, labelset = Construct_Dataset_withinSubj(n_sub, size=clip_length, step=clip_step)
        trainset, trainlab, testset, testlab = Create_TrainTest(dataset, labelset, testsize=test_size, randstat=69)
        ed_idx = Initiate_graph(trainset, pt=0.75, self_loop=True)
        train_set = Myset(trainset, trainlab)
        test_set = Myset(testset, testlab)
        curpath = './eegMI_set/{}/'.format(n_sub)
        if not os.path.exists(curpath): os.makedirs(curpath, exist_ok=True)
        torch.save(train_set, curpath + 'train.pt')
        torch.save(ed_idx, curpath + 'edgeIndex.pt')
        torch.save(test_set, curpath + 'test.pt')
    return


def LLMIdataset(clip_length, clip_step):
    for n_sub in range(10):
        trainset, trainlab, testset, testlab = form_onesub_set(n_sub, size=clip_length, step=clip_step)
        ed_idx = Initiate_graph(trainset, pt=0.75, self_loop=True)  ## sparse rate = 0.75
        train_set = Myset(trainset, trainlab)
        test_set = Myset(testset, testlab)
        curpath = './llMI_set/{}/'.format(n_sub)
        if not os.path.exists(curpath): os.makedirs(curpath, exist_ok=True)
        torch.save(train_set, curpath + 'train.pt')
        torch.save(ed_idx, curpath + 'edgeIndex.pt')
        torch.save(test_set, curpath + 'test.pt')
    return


if __name__ == '__main__':
    eegMIdataset(400, 10)
    LLMIdataset(400, 50)