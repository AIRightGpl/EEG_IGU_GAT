import os
import csv


class logger:
    def __init__(self, logpath='./log', txtHistoEnable=True, csvResulEnable=True):
        if not os.path.exists(logpath):
            os.mkdir(logpath)
        self.logRoot = logpath
        if txtHistoEnable:
            os.makedirs(os.path.join(logpath, 'txtHistory'), exist_ok=True)
            self.txtHistopath = os.path.join(logpath, 'txtHistory')
        if csvResulEnable:
            os.makedirs(os.path.join(logpath, 'csvResults'), exist_ok=True)
            self.csvResulpath = os.path.join(logpath, 'csvResults')

    def begin(self, name):
        self.logwriter = open(os.path.join(self.txtHistopath, name+'.txt'), 'a')
        self.Resulfpath = open(os.path.join(self.csvResulpath, name+'.csv'), 'a')
        self.Reswriter = csv.writer(self.Resulfpath)
        # self.Reswriter.writerow(["test", "time", "epochs", "train_loss", "train_acc", "test_loss", "test_acc"])
        self.Resulfpath.flush()
        print("log initiate done ~ ready to write rows")
        print("use writer-like logwriter.write() and Reswriter.writerow()")

    def writetxt(self, logrow):
        self.logwriter.write(logrow)
        self.logwriter.flush()
        # print('Write in txt: {}'.format(logrow))

    def singlewrite(self, name, contain):
        with open(os.path.join(self.txtHistopath, name+'.txt'), 'w') as f:
            f.write(contain)

    def writecsv(self, csvrow):
        self.Reswriter.writerow(csvrow)
        self.Resulfpath.flush()
        # print('Write in csv: {}'.format(csvrow))

    def close(self):
        self.logwriter.close()
        self.Resulfpath.close()


if __name__ == '__main__':
    test1 = logger(logpath='./testlog')
    import time
    import random
    ast1 = time.asctime(time.localtime())  # 参数为时间元组
    ast2 = time.ctime(time.time())  # 参数为浮点数时间戳
    name = str(ast1)+' model'
    test1.begin(name=name)
    for i in range(10):
        test1.writetxt("num:{} - epoch:{:#x} - time:{} - loss:{} \n".format(i,i,time.time(),random.random()))
        test1.writecsv([i,i,time.time(),random.random()])
    test1.close()
