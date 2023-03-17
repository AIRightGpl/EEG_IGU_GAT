import scipy.io as scio
import torch
from typing import Tuple, List
##'/home/qinyz/Brain_EEGCN/datasets/low-limb-motorimagery-dataset/self_EEG_RAW.mat'
Path = '/home/uestc/Brain_EEGCN/datasets/low-limb-motorimagery-dataset/self_EEG_RAW.mat'
data_dump = scio.loadmat(Path)


def form_onesub_set(subject_num: int, size: int = None, step: int = None, sequence: bool = False) -> \
        Tuple[List[torch.Tensor], List[int]]:
    train_dat = []
    train_lab = []
    test_dat = []
    test_lab = []
    for i in range(len(data_dump['s_train']['label'][0][subject_num].squeeze())):
        parsed = torch.from_numpy(data_dump['s_train']['eegdata'][0][subject_num][i][0].T).unfold(1, size, step).\
            permute(1, 0, 2)
        if sequence:
            train_dat.append(parsed)
            train_lab.append(int(data_dump['s_train']['label'][0][subject_num].squeeze()[i]))
        else:
            train_dat.extend(parsed)
            train_lab.extend([int(data_dump['s_train']['label'][0][subject_num].squeeze()[i])] * len(parsed))

    for i in range(len(data_dump['s_test']['label'][0][subject_num].squeeze())):
        parsed = torch.from_numpy(data_dump['s_test']['eegdata'][0][subject_num][i][0].T).unfold(1, size, step).\
            permute(1, 0, 2)
        if sequence:
            test_dat.append(parsed)
            test_lab.append(data_dump['s_test']['label'][0][subject_num].squeeze()[i])
        else:
            test_dat.extend(parsed)
            test_lab.extend([data_dump['s_test']['label'][0][subject_num].squeeze()[i]] * len(parsed))

    return train_dat, train_lab, test_dat, test_lab


def form_multsub_set(tra_sub_lst: List[int], tes_sub_lst: List[int], size: int = None, step: int = None, seq: bool =
False) -> Tuple[List[torch.Tensor], List[int]]:
    from copy import deepcopy
    train_dat = []
    train_lab = []
    test_dat = []
    test_lab = []
    for tr_sub in tra_sub_lst:
        tr_dat, tr_lab, te_dat, te_lab = form_onesub_set(tr_sub, size=size, step=step, sequence=seq)
        train_dat.extend(deepcopy(tr_dat))
        train_dat.extend(deepcopy(te_dat))
        train_lab.extend(deepcopy(tr_lab))
        train_lab.extend(deepcopy(te_lab))
    for te_sub in tes_sub_lst:
        tra_dat, tra_lab, tes_dat, tes_lab = form_onesub_set(te_sub, size=size, step=step, sequence=seq)
        test_dat.extend(deepcopy(tra_dat))
        test_dat.extend(deepcopy(tes_dat))
        test_lab.extend(deepcopy(tra_lab))
        test_lab.extend(deepcopy(tes_lab))

    return train_dat, train_lab, test_dat, test_lab




if __name__ == '__main__':
    # tr_d, tr_l, te_d, te_l = form_onesub_set(0, size=400, step=50)
    # print('one sub done')
    tr_d, tr_l, te_d, te_l = form_multsub_set([0, 1, 2, 3, 4], [5, 6, 7], size=400, step=50)
    print('multi sub done')