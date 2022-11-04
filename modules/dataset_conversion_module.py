import mne
import re
import pickle
import pymysql
import numpy as np
from copy import deepcopy


def attract_spec_filepath(rootpath, extern='.edf'):
    from toolbox_lib.Recursive_travel import walkthroRecur
    savedpath = []

    def file_extern_filter(filepath, fileextern=extern):
        if filepath.endswith(fileextern):
            savedpath.append(filepath)
        else:
            pass
        return
    walkthroRecur(rootpath, file_extern_filter)
    return savedpath


class SqlConnector:

    def __init__(self, sqltable_name, *attr):
        try:
            self.dbconnect = pymysql.connect(host='localhost',
                                 user='root',
                                 password='msql24693&',
                                 database='eeg_datasets')
        except:
            print('error, database not found')
            exit()
        self.cursor = self.dbconnect.cursor()
        self._sqlinsert_que = """ INSERT INTO {} {} VALUES (%s, %s, %s)""".format(sqltable_name, attr).replace("'", "")
        self._sqlattrac_que = """ SELECT %s FROM {} WHERE %s = %s""".format(sqltable_name).replace("'", "")
        self._sqlsttrspec_q = """ SELECT %s FROM {} WHERE %s = %s AND %s = %s""".format(sqltable_name).replace("'", "")

    def insert(self, *attr):
        self.cursor.execute(self._sqlinsert_que, attr)
        self.dbconnect.commit()

    def specattr(self, *attr):
        self.cursor.execute(self._sqlsttrspec_q % attr)
        data = self.cursor.fetchall()
        return data

    def attract(self, *attr):
        self.cursor.execute(self._sqlattrac_que % attr)
        data = self.cursor.fetchall()
        return data


class EegMotorMovementImageryDataset:

    def __init__(self, root):
        self.edf_files_lst = attract_spec_filepath(root, extern='.edf')
        self.event_files_lst = [f + '.event' for f in self.edf_files_lst]
        self.subjid_lst = []
        self.traiid_lst = []


        for path in self.edf_files_lst:
            subj = re.search("(?<=/S)\d*(?=R)", path)
            trai = re.search("(?<=R)\d*(?=\.edf)", path)
            self.subjid_lst.append(int(subj.group()))
            self.traiid_lst.append(int(trai.group()))

    def load_raw_dat(self):
        from mne.io import read_raw_edf
        writer = SqlConnector('`eeg_motor_MI`', '`subject_id`', '`trails_id`', '`eeg_data`')
        for i in range(len(self.edf_files_lst)):
            simple = read_raw_edf(self.edf_files_lst[i])
            events_from_annot, event_dict = mne.events_from_annotations(simple)
            data_dict = {'data': [], 'label': []}
            for event in events_from_annot:
                data, _ = simple[:, event[0]:event[0]+655]
                label = event[2]
                data_dict['data'].append(deepcopy(data))
                data_dict['label'].append(deepcopy(label))
            writer.insert(self.subjid_lst[i], self.traiid_lst[i], pickle.dumps(data_dict))

        return


if __name__ == '__main__':
    import pickle
    ## build connection to mysql on root@localhost use database 'eeg_datasets'
    # db = pymysql.connect(host='localhost',
    #                      user='root',
    #                      password='msql24693&',
    #                      database='eeg_datasets')
    # ## instantiate an instance 'cursor' with method cursor()
    # cursor = db.cursor()
    # create_tab = """CREATE TABLE IF NOT EXISTS `eeg_motor_MI` (
    #                 `subject_id` INT UNSIGNED,
    #                 `trails_id` INT UNSIGNED,
    #                 `eeg_data` MEDIUMBLOB,
    #                 PRIMARY KEY (`subject_id`,`trails_id`)    )"""
    # try:
    #     cursor.execute(create_tab)
    #     db.commit()
    # except:
    #     db.rollback()
    # print('done')

    # writer = SqlConnector('`eeg_motor_MI`', '`subject_id`', '`trails_id`')
    # cursor.execute(writer.sqlinsert_que, (1, 2))
    # db.commit()
    # print('hault')

    # datate = SqlConnector('eeg_motor_MI', 'subject_id', 'trails_id', 'eeg_data')
    # subject_num = 2
    # i = 3
    # datasu = datate.specattr('eeg_data', 'subject_id', str(subject_num), 'trails_id', str(i))
    # data = pickle.loads(datasu[0][0])
    # print('stop')

    root = '../datasets/eeg-motor-movementimagery-dataset-1.0.0/datas'
    dataset1 = EegMotorMovementImageryDataset(root)
    dataset1.load_raw_dat()
    print('data to sql database')