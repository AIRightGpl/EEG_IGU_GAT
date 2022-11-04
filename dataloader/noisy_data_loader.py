import scipy.io as scio
import torch
from typing import Tuple, List

Path1 = '/home/qinyz/Brain_EEGCN/datasets/raw_format_109_public/'
Path2 = '/home/qinyz/Brain_EEGCN//datasets/random_noisy_channel/'



def form_onesub_set(subject_num: int, size: int = None, step: int = None, noise=0.1) -> Tuple[List[torch.Tensor], List[int]]:
    raw_data_dump = scio.loadmat(Path1 + '{}.mat'.format(subject_num))
    noi_data_dump = scio.loadmat(Path2 + str(noise) + '/' + '{}.mat'.format(subject_num))
    train_dat = []
    train_lab = []
    test_dat = []
    test_lab = []
    for i in range(len(raw_data_dump['train'][0]['label'][0].squeeze())):
        parsed = torch.from_numpy(raw_data_dump['train'][0]['eegdata'][0][i][0].T).unfold(1, size, step).permute(1, 0, 2)
        train_dat.extend(parsed)
        train_lab.extend([raw_data_dump['train'][0]['label'][0].squeeze()[i]] * len(parsed))
    for i in range(len(noi_data_dump['test'][0]['label'][0].squeeze())):
        parsed = torch.from_numpy(noi_data_dump['test'][0]['eegdata'][0][i][0].T).unfold(1, size, step).permute(1, 0, 2)
        test_dat.extend(parsed)
        test_lab.extend([noi_data_dump['test'][0]['label'][0].squeeze()[i]] * len(parsed))

    return train_dat, train_lab, test_dat, test_lab


def form_multsub_set(tra_sub_lst: List[int], tes_sub_lst: List[int], size: int = None, step: int = None) -> \
        Tuple[List[torch.Tensor], List[int]]:
    from copy import deepcopy
    train_dat = []
    train_lab = []
    test_dat = []
    test_lab = []
    for tr_sub in tra_sub_lst:
        tr_dat, tr_lab, te_dat, te_lab = form_onesub_set(tr_sub, size=size, step=step)
        train_dat.extend(tr_dat)
        train_dat.extend(te_dat)
        train_lab.extend(tr_lab)
        train_lab.extend(te_lab)
    for te_sub in tes_sub_lst:
        tr_dat, tr_lab, te_dat, te_lab = form_onesub_set(te_sub, size=size, step=step)
        test_dat.extend(tr_dat)
        test_dat.extend(te_dat)
        test_lab.extend(tr_lab)
        test_lab.extend(te_lab)

    return train_dat, train_lab, test_dat, test_lab




if __name__ == '__main__':
    tr_d, tr_l, te_d, te_l = form_onesub_set(1, size=400, step=50)  ## start with 1
    print('one sub done')
    # tr_d, tr_l, te_d, te_l = form_multsub_set([1, 2, 3, 4], [5, 6, 7], size=400, step=50)
    # print('multi sub done')