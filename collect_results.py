import os
import re
import numpy as np
import pandas as pd
from toolbox_lib.Recursive_travel import walkthroRecur

PATH = './log/self_rgf0412_lr-2wd-3/2023_04_13'


def init_glob_dict():
    global saveddict
    saveddict = {}
    return


def filt_result(path):
    arg_result = re.match('.*/test/csvResults.*', path)
    return arg_result


def save_dict(filename, path, filt_func):
    if filt_func(path):
        regex = '(?<=subject)\d+'
        str_select = re.findall(regex, path)
        saveddict[int(str_select[0])] = path
    return


def find_test_result(path):
    init_glob_dict()
    walkthroRecur(path, save_dict, filt_func=filt_result)
    return saveddict


def collect_data(path, description=None):
    data = pd.read_csv(path)
    # print('data acquire')
    return data['test_acc'].max()


if __name__ == '__main__':
    filted_filename_dict = find_test_result(PATH)
    results = []
    # for key, value in enumerate(filted_filename_dict):   ## for public 1-110   for self 10
    for i in range(10):    #1, 110
        # max_acc = collect_data(filted_filename_dict[value])
        max_acc = collect_data(filted_filename_dict[i])
        results.append(max_acc)
    writes = np.array(results).T

    np.savetxt('./sel_sinrg_0413_lr-2wd-3.csv', writes, delimiter=',')

    print('done')
