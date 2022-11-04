import copy

import numpy as np
import torch
from torch.utils import data
import scipy.io as scio
import torch.nn.functional as F
import h5py


class Dataset(data.Dataset):
    def __init__(self, dataset_choice, train_or_test, human, clip_length, step, with_remix, logger_setup=None):

        self.with_remix = with_remix

        EEG_train = []
        # A_train = []
        Label_train = []
        Human_list = []

        for i in range(len(human)):
            # channel_title = meta_EEG['title'][0][human[i]].__array__()
            Human_list.append(scio.loadmat('./Dataset/public_data/2023_single/{}.mat'.format(human[i] + 1)))

        if dataset_choice == "cross":
            self.A_public_cross = scio.loadmat('./Dataset/public_data/2023/public_A_cross.mat')['B_train'][0][0][1]
            self.A_public_cross = np.sum([self.A_public_cross, np.eye(64)], axis=0)
        elif dataset_choice == "within":
            self.A_public_within = scio.loadmat('./Dataset/public_data/2023/public_A_within.mat')['A_train'][0][0][1]
            self.A_public_within = np.sum([self.A_public_within, np.eye(64)], axis=0)

        for i in range(len(human)):
            data_EEG = Human_list[i]
            meta_EEG = data_EEG['train']

            # channel_title = meta_EEG['title'][0][human[i]].__array__()
            eegdata = meta_EEG['eegdata'][0][0]
            label = meta_EEG['label'][0][0]
            # 每个人56条
            for j in range(len(eegdata)):
                data_temper = []
                split_num = int((len(eegdata[j][0]) - clip_length) / step + 1)
                for b in range(split_num):
                    data_temper.append(eegdata[j][0][step * b:clip_length + step * b])
                    if len(data_temper) == 1:
                        EEG_train.append(copy.deepcopy(data_temper))
                        # A_train.append(eegdata[j][0])
                        Label_train.append(label[j][0])
                        data_temper.pop(0)

                    if logger_setup is not None:
                        logger_setup.info(
                            "Human {0} Motion_data {1} Seq {2} added".format(human[i] + 1, j + 1, b + 1))

        # self.A_EEG_train = torch.tensor(
        #     np.array(A_train).transpose([0, 2, 1]), dtype=torch.float32)
        # A_train=[]
        self.EEG_train = torch.tensor(
            np.array(EEG_train).transpose([0, 1, 3, 2]), dtype=torch.float32)
        del EEG_train
        self.Y_EEG_train = (torch.tensor(np.array([Label_train])))[0] - 1
        del Label_train

        # meta_EEG = data_EEG.get('test')

        EEG_train = []
        # A_train = []
        Label_train = []

        for i in range(len(human)):
            # channel_title = meta_EEG['title'][0][human[i]].__array__()
            data_EEG = Human_list[i]
            meta_EEG = data_EEG['test']

            # channel_title = meta_EEG['title'][0][human[i]].__array__()
            eegdata = meta_EEG['eegdata'][0][0]
            label = meta_EEG['label'][0][0]
            for j in range(len(eegdata)):
                data_temper = []
                split_num = int((len(eegdata[j][0]) - clip_length) / step + 1)
                for b in range(split_num):
                    data_temper.append(eegdata[j][0][step * b:clip_length + step * b])
                    if len(data_temper) == 1:
                        EEG_train.append(copy.deepcopy(data_temper))
                        # A_train.append(eegdata[j][0])
                        Label_train.append(label[j][0])
                        data_temper.pop(0)

                    if logger_setup is not None:
                        logger_setup.info(
                            "Human {0} Motion_data {1} Seq {2} added".format(human[i] + 1, j + 1, b + 1))

        # self.A_EEG_test = np.array(A_train).transpose([0, 2, 1])
        # del A_train
        self.EEG_test = torch.tensor(
            np.array(EEG_train).transpose([0, 1, 3, 2]), dtype=torch.float32)
        del EEG_train
        self.Y_EEG_test = (torch.tensor(np.array([Label_train])))[0] - 1
        del Label_train

        if dataset_choice == "cross":
            # merge
            # self.A_EEG = torch.cat([self.A_EEG_train, self.A_EEG_test], dim=0)
            # del self.A_EEG_train, self.A_EEG_test
            self.EEG = torch.cat([self.EEG_train, self.EEG_test], dim=0)
            del self.EEG_train, self.EEG_test
            self.EEG = self.EEG / torch.mean(torch.abs(self.EEG))
            self.Y_EEG = torch.cat([self.Y_EEG_train, self.Y_EEG_test], dim=0)
            del self.Y_EEG_train, self.Y_EEG_test
            self.A = torch.tensor(self.A_public_cross)
        elif dataset_choice == "within":
            self.A = torch.tensor(self.A_public_within)
            if train_or_test == "train":
                # self.A_EEG = self.A_EEG_train.cuda()
                self.EEG = self.EEG_train
                # self.EEG = self.EEG / torch.mean(torch.abs(self.EEG))
                self.Y_EEG = self.Y_EEG_train
            elif train_or_test == "test":
                # self.A_EEG = self.A_EEG_test.cuda()
                self.EEG = self.EEG_test
                # self.EEG = self.EEG / torch.mean(torch.abs(self.EEG))
                self.Y_EEG = self.Y_EEG_test
            else:
                print("please check param train_or_test")
        else:
            print("please check param dataset_choice")
        # self.A = torch.eye(64)

        # if dataset_choice == 'raw_fb':
        #     tmp = None
        #     for i in range(self.A_EEG.shape[0]):
        #         tmp_a_eeg = self.A_EEG[i].reshape(5, 22, 750)
        #         for j in range(5):
        #             if tmp is not None:
        #                 tmp = torch.cat([tmp, torch.corrcoef(tmp_a_eeg[j]).unsqueeze(0)], dim=0)
        #             else:
        #                 tmp = torch.corrcoef(tmp_a_eeg[j]).unsqueeze(0)
        #     self.A = (torch.abs(torch.mean(tmp, dim=0)) > torch.mean(torch.abs(tmp))).long()
        #     for i in range(self.A.shape[0]):
        #         self.A[i, i] = 0

    def __len__(self):
        return len(self.EEG)

    def __getitem__(self, item):
        if self.with_remix:
            length = len(self.EEG)
            rand_idx = torch.randint(0, length, [1])
            rand_ratio = torch.rand([1]) * 0.4
            rand_idx_noise = torch.randint(0, length, [1])
            rand_ratio_noise = torch.rand([1]) * 0.05

            EEG_A = self.A
            EEG_label = self.Y_EEG[item]
            while self.Y_EEG[rand_idx] != EEG_label:
                rand_idx = torch.randint(0, length, [1])

            EEG_merged = \
                (self.EEG[item] + self.EEG[rand_idx] * rand_ratio + self.EEG[rand_idx_noise] * rand_ratio_noise)[0]
            # EEG_merged_A = \
            #     ((self.A_EEG[item] * (1 - rand_ratio) + self.A_EEG[rand_idx] * rand_ratio) * (1 - rand_ratio_noise) +
            #      self.A_EEG[rand_idx_noise] * rand_ratio_noise)[0]

        else:
            EEG_merged = self.EEG[item]
            # EEG_merged_A = self.A_EEG[item]
            EEG_label = self.Y_EEG[item]
            EEG_A = self.A

        # calc edge_index
        # edge_index = torch.corrcoef(EEG_merged_A)
        # edge_index = (edge_index > torch.mean(edge_index)).long()
        # edge_index = torch.nonzero(edge_index)

        return EEG_merged, EEG_label, EEG_A

