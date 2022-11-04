import numpy as np
def main(results):
    file_path = "/home/felix/ssd/iccsip_skc_net/experiments/logs_ijcai_within/1674524889.5196972/one_human.log"
    input_file = open(file_path)
    print("--------------------------------start ours----------------------------------")
    results0 = []
    while 1:
        input_line = input_file.readline()
        if not input_line:
            break
        if input_line.find("best") != -1:
            print(input_line.split("acc: ")[-1].split(" with ")[0])
            results0.append(input_line.split("acc: ")[-1].split(" with ")[0])
    input_file.close()
    results.append(np.array(results0))
    print("--------------------------------end ours----------------------------------")

    file_path = "/home/felix/ssd/iccsip_skc_net/experiments/logs_ijcai_eegnet/1674525113.8517106/multi_human.log"
    input_file = open(file_path)
    print("--------------------------------start eegnet----------------------------------")
    results1 = []
    while 1:
        input_line = input_file.readline()
        if not input_line:
            break
        if input_line.find("best") != -1:
            print(input_line.split("acc: ")[-1].split(" with ")[0])
            results1.append(input_line.split("acc: ")[-1].split(" with ")[0])
    input_file.close()
    results.append(np.array(results1))
    print("--------------------------------end eegnet----------------------------------")

    file_path = "/home/felix/ssd/iccsip_skc_net/experiments/logs_ijcai_mcsnet/1674525276.9093356/multi_human.log"
    input_file = open(file_path)
    print("--------------------------------start mcsnet----------------------------------")
    results2 = []
    while 1:
        input_line = input_file.readline()
        if not input_line:
            break
        if input_line.find("best") != -1:
            print(input_line.split("acc: ")[-1].split(" with ")[0])
            results2.append(input_line.split("acc: ")[-1].split(" with ")[0])
    input_file.close()
    results.append(np.array(results2))
    print("--------------------------------end mcsnet----------------------------------")
    return np.array(results)

if __name__ == '__main__':
    results = []
    res = main(results)
    print('finish')
    np.savetxt('./multi_results.csv', res.T.astype('float64'), delimiter=',')
