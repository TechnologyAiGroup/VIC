# -*- coding: utf-8 -*-
import os.path
import numpy as np
from sklearn.cluster import SpectralClustering, KMeans, AgglomerativeClustering
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin_min
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import pandas as pd
import ast
# import _ucrdtw
import time
import sys
from collections import Counter
import json
# import cuFFT
from fastdtw import dtw
import concurrent.futures

def save_json(circuit, method, n_clusters):
    print(circuit, method, n_clusters)
    if not os.path.exists(clusterInfo_dir):
        os.makedirs(clusterInfo_dir)
    with open(f"{clusterInfo_dir}/{circuit}_{method}_{n_clusters}.json", "w") as json_file:
        json.dump(clusterInfo, json_file)



def cluster_dtw(data_df):
    train = data_df[0]
    test = data_df[1]
    # train_data = []
    # test_data = []
    train_data = [np.array([int(num) for num in ast.literal_eval(i)]) for i in train['fails']]
    test_data = [np.array([int(num) for num in ast.literal_eval(i)]) for i in test['fails']]

    # for i in train['fails']:
    #     tmp = ast.literal_eval(i)
    #     train_data.append([int(num) for num in tmp])

    # for i in test['fails']:
    #     tmp = ast.literal_eval(i)
    #     test_data.append([int(num) for num in tmp])

    print("len of train_data :", len(train_data))
    print("len of test_data :", len(test_data))
    # distance_matrix_train = np.zeros((len(train_data), len(train_data)))
    # distance_matrix_test = np.zeros((len(test_data), len(test_data)))

    # 定义一个函数，用于计算距离矩阵的一部分
    def compute_distance(i, j):
        x, y = train_data[i], train_data[j]
        distance, _ = dtw(x, y)
        return i, j, distance

    n = len(train_data)
    distance_matrix_train = np.zeros((n, n))

    start_time = time.time()

    # 使用线程池并行计算距离矩阵的每一项
    with concurrent.futures.ThreadPoolExecutor(max_workers=None) as executor:  # 使用所有可用线程
        futures = [executor.submit(compute_distance, i, j) for i in range(n) for j in range(i + 1, n)]

    # 初始化最大值和最小值为无穷大和无穷小
    # max_value = float('-inf')
    # min_value = float('inf')
    # 收集计算结果并填充距离矩阵
    for future in concurrent.futures.as_completed(futures):
        i, j, distance = future.result()
            # 更新最大值和最小值
        # if distance > max_value:
        #     max_value = distance
        # if distance < min_value:
        #     min_value = distance
            # print(distance)
        distance_matrix_train[i][j] = distance

    print(distance_matrix_train)
    # 对称化距离矩阵
    distance_matrix_train = distance_matrix_train + distance_matrix_train.T
    max_value = np.max(distance_matrix_train)
    np.fill_diagonal(distance_matrix_train, max_value)
    min_value = np.min(distance_matrix_train)
    np.fill_diagonal(distance_matrix_train, 0)
    print(min_value, max_value)

    # 归一化的距离矩阵
    # normalized_matrix = (distance_matrix_train - min_value) / (max_value - min_value)
    
    
    # print(normalized_matrix)
 
    similarity_matrix = 1 / (1 + distance_matrix_train)
    end_time = time.time()
    execution_time1 = end_time - start_time
    # lines = [f'计算相似度矩阵时间：{execution_time1} 秒\n']

    start_time = time.time()
    # spectral_clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=seed)
    # labels_train = spectral_clustering.fit_predict(similarity_matrix)
    agglomerative_clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=np.mean(distance_matrix_train), affinity='precomputed', linkage='average')
    labels_train = agglomerative_clustering.fit_predict(distance_matrix_train)
    end_time = time.time()
    execution_time = end_time - start_time
    # lines.append(f'聚类时间：{execution_time} 秒')
    print("dtw 聚类时间：", execution_time+execution_time1, "秒(训练集)")
    print(f"训练集聚类数量： {len(set(labels_train))}")
    clusterInfo['clusterNumberOnTrainSet'] = len(set(labels_train))
    clusterInfo['clusterTimeOnTrainingSet'] = execution_time
    # with open(f'logs/{circuit}_{method}_{n_clusters}.out', 'w') as f:
    #     f.writelines(lines)
    labels_test = []
    start_time = time.time()
    for i in range(len(test_data)):
        min_dis = 10000000001.00000
        min_idx = 0
        for j in range(len(train_data)):
            distance,_ = dtw(test_data[i], train_data[j])
            if distance < min_dis and distance != 0:
                min_dis = distance
                min_idx = j
        labels_test.append(labels_train[min_idx])
        
    # for i in range(len(test_data)):
    #     distances = np.array([dtw(test_data[i], train_seq)[0] for train_seq in train_data])

    #     # 将距离为零的索引位置设为最大值，以排除与自身的比较
    #     distances[i] = np.max(distances)

    #     min_idx = np.argmin(distances)
    #     labels_test.append(labels_train[min_idx])


    end_time = time.time()
    execution_time = end_time - start_time
    print("dtw 聚类时间：", execution_time, "秒(测试集)")
    numberOfCluster = len(set(labels_test))
    print(f"测试集聚类数量： {numberOfCluster}")
    clusterInfo['clusterNumberOnTestSet'] = numberOfCluster
    clusterInfo['clusterTimeOnTestSet'] = execution_time
    
    save_json(circuit, method, numberOfCluster)
    return [labels_train, labels_test]


# add on 20231208
def cluster_hac(data_df):
    train, test = data_df
    train_data = [[int(item) for item in sublist] for sublist in [ast.literal_eval(i) for i in train['fails']]]
    test_data = [[int(item) for item in sublist] for sublist in [ast.literal_eval(i) for i in test['fails']]]
    Ntp = getNtp('circuit2test.csv')

    print(f"len of train_data : {len(train_data)}")
    print(f"len of test_data : {len(test_data)}")
    
    start_time_get_distance = time.time()
    mlb = MultiLabelBinarizer(classes=[i for i in range(Ntp)])
    train_data_encoded = mlb.fit_transform(train_data)
    distance_matrix_train = pairwise_distances(train_data_encoded, metric='jaccard')

    print(f"hac 计算相似度矩阵时间：{time.time() - start_time_get_distance} 秒")

    start_time_get_clusters = time.time()
    agglomerative_clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=n_clusters/100, affinity='precomputed', linkage='average')
    labels_train = agglomerative_clustering.fit_predict(distance_matrix_train)
    clusterInfo['clusterTimeOnTrainingSet'] = time.time() - start_time_get_clusters
    print(f"hac 聚类时间：{time.time() - start_time_get_clusters} 秒(训练集)")
    print(f"训练集聚类数量： {len(set(labels_train))}")
    clusterInfo['clusterNumberOnTrainSet'] = len(set(labels_train))

    # 训练集训练，测试集找聚类中心或边界 
    test_data_encoded = mlb.fit_transform(test_data)
    train_data_encoded = np.array(train_data_encoded)
    test_data_encoded = np.array(test_data_encoded)

    labels_test = [None] * len(test_data_encoded)
    start_time_test = time.time()
    # 使用 pairwise_distances_argmin_min 函数计算最近邻的索引
    indices, _ = pairwise_distances_argmin_min(test_data_encoded, train_data_encoded, metric='jaccard')

    # 使用找到的索引从训练标签中获取测试标签
    labels_test = [labels_train[idx] for idx in indices]
    # print(len(set(labels_test)))
    clusterInfo['clusterTimeOnTestSet'] = time.time() - start_time_test
    print(f"hac 聚类时间：{time.time() - start_time_test} 秒(测试集)")
    # hac里更新聚类值
    numberOfCluster = len(set(labels_test))
    print(f"测试集聚类数量： {numberOfCluster}")
    clusterInfo['clusterNumberOnTestSet'] = numberOfCluster
    
    save_json(circuit, method, numberOfCluster)
    return [labels_train, labels_test]



def cluster_f1(data_df):
    train = data_df[0]
    test = data_df[1]
    Ntp = getNtp('circuit2test.csv')

    train_data = []
    test_data = []
    start_time = time.time()
    for i in train['fails']:
        tmp = ast.literal_eval(i)
        idxs = [int(num) for num in tmp]
        fingerprint = np.zeros(Ntp)
        fingerprint[idxs] = 1
        train_data.append(fingerprint)

    for i in test['fails']:
        tmp = ast.literal_eval(i)
        idxs = [int(num) for num in tmp]
        fingerprint = np.zeros(Ntp)
        fingerprint[idxs] = 1
        test_data.append(fingerprint)
    execution_time = time.time() - start_time
    print("len of train_data :", len(train_data))
    print("len of test_data :", len(test_data))
    print("编码时间：", execution_time, "秒")
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    major = min(32, len(train_data[0]))
    pca = PCA(n_components=major, random_state=seed)
    pca_data = pca.fit_transform(train_data)
    start_time = time.time()
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
    train = kmeans.fit(pca_data)
    time1 = time.time()
    clusterInfo['clusterTimeOnTrainingSet'] = time1-start_time
    print("K-means编码1聚类时间：", time1-start_time, "秒(训练集)" )
    train_label = train.labels_
    start_time = time.time()
    test_label = kmeans.predict(pca.transform(test_data))
    time2 = time.time()
    clusterInfo['clusterTimeOnTestSet'] = time2 - start_time
    print("K-means编码1聚类时间：", time2 - start_time, "秒(测试集)")
    
    save_json(circuit, method, n_clusters)
    return [train_label, test_label]



def cluster_f2(data_df):
    train = data_df[0]
    test = data_df[1]
    Ntp = getNtp('circuit2test.csv')

    train_data = []
    test_data = []
    start_time = time.time()
    for i in train['fails']:
        tmp = ast.literal_eval(i)
        idxs = [int(num) for num in tmp]
        fingerprint = np.zeros(Ntp)
        dic = Counter(idxs)
        for key, value in dic.items():
            fingerprint[key] = value
        train_data.append(fingerprint)

    for i in test['fails']:
        tmp = ast.literal_eval(i)
        idxs = [int(num) for num in tmp]
        fingerprint = np.zeros(Ntp)
        fingerprint[idxs] = 1
        test_data.append(fingerprint)

    execution_time = time.time() - start_time
    print("len of train_data :", len(train_data))
    print("len of test_data :", len(test_data))
    print("编码时间：", execution_time, "秒")
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    major = min(8, len(train_data[0]))
    pca = PCA(n_components=major, random_state=seed)
    pca_data = pca.fit_transform(train_data)
    start_time = time.time()
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
    train = kmeans.fit(pca_data)
    time1 = time.time()
    clusterInfo['clusterTimeOnTrainingSet'] = time1-start_time
    print("K-means编码2聚类时间：", time1-start_time, "秒(训练集)" )
    train_label = train.labels_
    start_time = time.time()
    # test_label = spectral.fit_predict(pca.transform(test_data))
    test_label = kmeans.predict(pca.transform(test_data))
    time2 = time.time()
    clusterInfo['clusterTimeOnTestSet'] = time2 - start_time
    print("K-means编码2聚类时间：", time2 - start_time, "秒(测试集)")
    
    save_json(circuit, method, n_clusters)
    return [train_label, test_label]


def cluster_k1(data_df):
    train_label = [0]*len(data_df[0])
    test_label = [0]*len(data_df[1])
    save_json(circuit, method, 1)
    return [train_label, test_label]


def getNtp(path):
    #获得test pattern的维度N
    try:
        c2t = pd.read_csv(path)
        for index, row in c2t.iterrows():
            line = row['circuit']
            tp = row['test_num']
            if line == circuit:
                return int(tp)
    except Exception:
        print("文件错误")
        raise Exception

    print("文件内容错误")
    raise Exception


def exec_cluster(callback_function, data_df):
    return callback_function(data_df)


if __name__ == '__main__':
    circuit = sys.argv[1]
    method = sys.argv[2]
    # circuit = 's1488'
    # method = 'k1'
    '''
    if method!='hac', n_clusters is #clusters
    if method=='hac', n_clusters/100 is the threshold
    '''
    n_clusters = int(sys.argv[3])
    # n_clusters =10
    if method=='k1':
        n_clusters=1
    print(f'circuit: {circuit}, method: {method}, k: {n_clusters}')
    
    seed = 42
    
    
    
    
    # 202401111路径修改
    # filepath = f'./data/raw_data/{circuit}'
    filepath = f'./DiagData/{circuit}'
    # if isTopK > 0:
    #     filepath = filepath + f'_top{isTopK}'
    
    data_df = pd.read_csv(filepath + '.csv')
    train_data_df, test_data_df = train_test_split(data_df, train_size=0.9, test_size=0.1, shuffle=True,
                                                   random_state=seed)

    if method == 'dtw':
        cb = cluster_dtw
    elif method == 'hac':
        cb = cluster_hac
    elif method == 'f1':
        cb = cluster_f1
    elif method == 'f2':
        cb = cluster_f2
    elif method == 'k1':
        cb = cluster_k1
    else:
        exit()
    
    
    # add on 9.11
    # parameters 'circuit' 'method' 'n_clusters' determines a row.
    clusterInfo = dict()
    # 假设 experiment_folder 是存储 JSON 文件的文件夹路径
    clusterInfo_dir = './experiment/clusterInfo_'
    
    [train_label, test_label] = exec_cluster(callback_function=cb, data_df=[train_data_df, test_data_df])

    assert train_data_df.shape[0] == len(train_label)
    # train_data_df[:, 'cluster_label'] = train_label
    train_data_df = train_data_df.assign(cluster_label=train_label)
    assert test_data_df.shape[0] == len(test_label)
    test_data_df = test_data_df.assign(cluster_label=test_label)

    
    # 路径变量
    base_dir = './cluster_data_'
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')


    if method == "hac" or method=="dtw":
        # 仅对于 hac 方法，通过 JSON 文件名来确定 n_clusters
        json_file_pattern = f'{circuit}_{method}_'
        json_files = [f for f in os.listdir(clusterInfo_dir) if f.startswith(json_file_pattern) and f.endswith('.json')]

        if json_files:
            # 假定文件命名符合规则，并且只有一个文件匹配
            json_filename = json_files[0]
            # 分割文件名以获取 n_clusters 部分
            print(json_filename)
            n_clusters = json_filename.replace(json_file_pattern, '').replace('.json', '')
        else:
            # 找不到文件，则抛出错误或者设置默认值
            raise ValueError(f"No JSON file found for circuit {circuit} with method {method}")
    
    res_file_name = f'{circuit}_{method}_{n_clusters}.csv'



    # 确保训练和测试的目录存在
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # 保存新的带有簇标签的数据集
    train_data_df.to_csv(os.path.join(train_dir, res_file_name))
    test_data_df.to_csv(os.path.join(test_dir, res_file_name))

    print('\n')
