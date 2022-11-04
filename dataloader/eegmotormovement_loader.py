import numpy
import torch
from typing import List, Union, Tuple
from modules.dataset_conversion_module import SqlConnector


# def Totensor(obje: numpy.ndarray) -> torch.Tensor:


def Create_TrainTest(dataset: List[torch.Tensor], labeset: List[int], testsize: float = 0.3, randstat: int = 0) -> Tuple[List[torch.Tensor], List[int],
                                                                               List[torch.Tensor], List[int]]:
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(dataset, labeset, test_size=testsize, random_state=randstat)

    return x_train, y_train, x_test, y_test


def Create_Kfold(dataset: List[torch.Tensor], labeset: List[int], n_fold: int = 5, randstat: int = 0):
    from sklearn.model_selection import KFold
    from copy import deepcopy
    kf = KFold(n_fold=n_fold, shuffle=True, random_state=randstat)
    kf_traindata = []
    kf_testdata = []
    kf_trainlab = []
    kf_testlab = []
    for i, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        kf_traindata.append(deepcopy(dataset[train_idx]))
        kf_trainlab.append(deepcopy(labeset[train_idx]))
        kf_testdata.append(deepcopy(dataset[test_idx]))
        kf_testlab.append(deepcopy(labeset[test_idx]))

    return kf_traindata, kf_trainlab, kf_testdata, kf_testlab


def Construct_Dataset_withinSubj(subject_num: int, size: int = None, step: int = None) -> Tuple[List[torch.Tensor], List[int]]:
    import pickle
    task_2 = [3, 7, 11]
    task_4 = [5, 9, 13]
    lab_map1 = {2: 1, 3: 2}          ## for this datasets we map task2: 1:'left fist MI', 2:'right fist MI
    lab_map2 = {2: 3, 3: 4}          ## map task4: 3:'both fist MI', 4:'both feet MI'
    assert type(size) == type(step), '\'size\' and \'step\' should be the same type!'
    dataset = SqlConnector('eeg_motor_MI', 'subject_id', 'trails_id', 'eeg_data')
    data_lst = []
    labl_lst = []
    single_subject = dataset.attract('eeg_data', 'subject_id', str(subject_num))
    for i in range(len(single_subject)):
        if i in task_2:
            trail_dat = pickle.loads(single_subject[i][0])['data']
            trail_lab = pickle.loads(single_subject[i][0])['label']
            rem_idx = [i for i, lab in enumerate(trail_lab) if lab == 1]
            rem_idx.reverse()
            for rem in rem_idx:
                del trail_dat[rem]
                del trail_lab[rem]
            if 0< size < 655 and 0< step< 655:
                data_lst.extend([torch.from_numpy(obje[:, :640]).unfold(1, size, step).permute(1, 0, 2) for obje in trail_dat])
                labl_lst.extend(list(map(lambda x: lab_map1[x], trail_lab)))
            else:
                data_lst.extend([torch.from_numpy(obje[:, :640]) for obje in trail_dat])
                labl_lst.extend(list(map(lambda x: lab_map1[x], trail_lab)))
            # print('use map1')

        elif i in task_4:
            trail_dat = pickle.loads(single_subject[i][0])['data']
            trail_lab = pickle.loads(single_subject[i][0])['label']
            rem_idx = [i for i, lab in enumerate(trail_lab) if lab == 1]
            rem_idx.reverse()
            for rem in rem_idx:
                del trail_dat[rem]
                del trail_lab[rem]
            if 0 < size < 655 and 0 < step < 655:
                data_lst.extend([torch.from_numpy(obje[:, :640]).unfold(1, size, step).permute(1, 0, 2) for obje in trail_dat])
                labl_lst.extend(list(map(lambda x: lab_map2[x], trail_lab)))
            else:
                data_lst.extend([torch.from_numpy(obje[:, :640]) for obje in trail_dat])
                labl_lst.extend(list(map(lambda x: lab_map2[x], trail_lab)))
            # print('use map2')

        if 0 < size < 640 and 0 < step < 640:
            out_dat_lst = []
            out_lab_lst = []
            for ind in range(len(labl_lst)):
                out_dat_lst.extend(list(data_lst[ind]))
                out_lab_lst.extend([labl_lst[ind]] * len(data_lst[ind].tolist()))
        else:
            out_dat_lst = data_lst
            out_lab_lst = labl_lst

    return out_dat_lst, out_lab_lst


def Construct_Dataset_crosssub(train_subj: Union[int, List[int]], test_subj: Union[int, List[int]],
                               size: int = None, step: int = None) -> Tuple[List[torch.Tensor], List[int],
                                                                            List[torch.Tensor], List[int]]:
    from copy import deepcopy
    train_data = []
    train_labe = []
    test_data = []
    test_labe = []
    for tra_sub in train_subj:
        datalst, labelst = Construct_Dataset_withinSubj(tra_sub, size=size, step=step)
        train_data.extend(deepcopy(datalst))
        train_labe.extend(deepcopy(labelst))
    for tes_sub in test_subj:
        datalst, labelst = Construct_Dataset_withinSubj(tes_sub, size=size, step=step)
        test_data.extend(deepcopy(datalst))
        test_labe.extend(deepcopy(labelst))

    return train_data, train_labe, test_data, test_labe