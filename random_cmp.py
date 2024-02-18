import pandas as pd
import sys
import load_data
from train_batch import check
import random
def get_acc(circuit,n_clusters):
    method = 'k1'
    n_clusters = n_clusters
    # print('circuit: ')
    # print(circuit)
    # print('encoding method : ')
    # print(method)
    test_clusters_data = load_data.load_by_cluster(circuit, load_data.get_circuit_dict_site_input(circuit), method,
                                                   n_clusters, 'test')
    test_set = []
    for tmp in test_clusters_data:
        if tmp == [0]:
            continue
        test_set.extend(tmp)
    ground_true = []
    for candidates in test_set:
        res = []
        for candidate in candidates:
            if candidate[1] == 'True':
                res.append(candidate[0])
        ground_true.append(res)
    origin_order = []
    for chip in test_set:
        tmp = []
        for tup in chip:
            tmp.append(tup[0])
        origin_order.append(tmp)
    seeds = [i for i in range(num)]

    acc2s = []

    top2 = circuit2top_nb2[circuit]
    for my_seed in seeds:
        random.seed(my_seed)
        sort_optimized = []
        for chip in origin_order:
            chip_copy = chip.copy()
            random.shuffle(chip_copy)
            sort_optimized.append(chip_copy)
        acc2 = check(sort_optimized, ground_true, top2)

        acc2s.append(acc2)
    print(circuit)
   
    print("rnd acc:" ,sum(acc2s) / len(acc2s))
    return sum(acc2s) / len(acc2s)



randomtimes = sys.argv[1]



df = pd.read_csv('experiment/1d.csv')
df = df[df['method'] == 'hac']


keys = df['circuit'].tolist()
values = df['remaining_nb2'].tolist()

circuit2top_nb2 = dict(zip(keys, values))
# circuit2top_nb2['fp2int'] = 60.12/100
num = int(randomtimes)
df_random = pd.DataFrame(columns=['circuit', 'top_nb2', 'acc_random'])
for circuit in circuit2top_nb2.keys():
    print(circuit)
    print("vic acc:",df.loc[df['circuit'] == circuit, 'count_nb2'].iloc[0])
    acc = get_acc(circuit, 1)
    value = (circuit, circuit2top_nb2[circuit], acc)
    df_random.loc[df_random.shape[0]] = dict(zip(df_random.columns, value))
path = './experiment'
df_random.to_csv(f'{path}/random_{num}.csv')



# circuit = 'b17'

