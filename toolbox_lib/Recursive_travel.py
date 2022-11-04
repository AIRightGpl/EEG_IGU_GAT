import os
from stat import *


def walkthroRecur(curpath, callback, filt_func=None):
    for f in os.listdir(curpath):
        pathnam = curpath+'/'+f
        try:
            mode = os.stat(pathnam, follow_symlinks=False).st_mode
        except:
            continue
        if S_ISDIR(mode):
            walkthroRecur(pathnam, callback, filt_func=filt_func)
        else:
            try:
                callback(f, pathnam, filt_func)
            except:
                callback(pathnam)
    return


def initiate_glob():
    global savedpath
    savedpath = []
    return


def init_glob_dict():
    global saveddict
    saveddict = {}
    return


def savepath(pathname):
    savedpath.append(pathname)


def saveasdict(filename, pathname):
    saveddict[filename] = pathname


# def save_with_filter(filename, pathname, filter_function):
#     if filter_function(pathname):
#         saveddict[filename] = pathname


def walktreefile(path):
    initiate_glob()
    walkthroRecur(path, savepath)
    return savedpath


def walktreefiledict(path):
    init_glob_dict()
    walkthroRecur(path, saveasdict)
    return saveddict


if __name__ == '__main__':
    # rotpat = 'D:/BCI/dataset/BCICIV_2a_gdf'
    # initiate_glob()
    # walkthroRecur(rotpat,savepath)
    # print('finished!')
    tar_path = '../datasets/eeg-motor-movementimagery-dataset-1.0.0'
    walktreefiledict(tar_path)
    print('stay')