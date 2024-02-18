import sys
import os
import numpy as np
import pandas as pd
import time
import load_data_multi-observations as load_data
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import roc_auc_score
from itertools import accumulate
import json
from pomegranate import HiddenMarkovModel,State,MultivariateGaussianDistribution
np.random.seed(0) 
# Calculate transition probability
def t2_given_t1(t2, t1, train_bag):
    tags = [pair[1] for pair in train_bag]
    count_t1 = len([t for t in tags if t == t1])
    count_t2_t1 = 0
    for index in range(len(tags) - 1):
        if tags[index] == t1 and tags[index + 1] == t2:
            count_t2_t1 += 1
    return count_t2_t1, count_t1


def train(trainset, testset):
    print(len(trainset))
    train_data_list = []
    train_labels = []
    train_tagged_words = []
    states_names = ('True', 'Neighbor', 'Neighbor2', 'Neighbor3', 'Noise', 'Separator')
    
    for chip in trainset:
        for candidate in chip:
            train_data_list.append(candidate[2])
            train_labels.append([candidate[1]])
            train_tagged_words.append((candidate[0],candidate[1]))
            
    train_data = np.vstack(train_data_list)
    

    
    
    transition_matrix = np.zeros((len(states_names), len(states_names)), dtype='float32')
    for i, t1 in enumerate(list(states_names)):
        for j, t2 in enumerate(list(states_names)):
            frequency = t2_given_t1(t2, t1, train_tagged_words)
            if  frequency[1]!=0:
                # print(t2, t1, train_tagged_words)
                transition_matrix[i, j] = frequency[0] / frequency[1]
            else:
                transition_matrix[i, j] = 1e-4

    transition_df = pd.DataFrame(transition_matrix, columns=list(states_names), index=list(states_names))
    # print(transition_df)
    
    data_dict_by_states = {
        'True': [],
        'Neighbor': [],
        'Neighbor2': [],
        'Neighbor3': [],
        'Noise': [],
        'Separator': []
    }
    for chip_data in testset:
        for candidatatuple in chip_data:
            data_dict_by_states[candidatatuple[1]].append(candidatatuple[2])

    
    means = [np.random.rand(8) for _ in range(len(states_names))]
    covariances = [np.eye(8) for _ in range(len(states_names))]
    states = [State(MultivariateGaussianDistribution(mean, cov), name=name) for name, (mean, cov) in zip(states_names, zip(means, covariances))]

    model = HiddenMarkovModel()
    model.add_states(*states)
    


    for s1 in states:
        for s2 in states:
            model.add_transition(s1, s2, 1.0 / len(states))
  
    # for i, s1 in enumerate(states):
    #     for j, s2 in enumerate(states):
            
    #         prob = transition_df.iloc[i, j]
    #         model.add_transition(s1, s2, prob)


    separator_probs = transition_df.loc['Separator'].values


    for state, prob in zip(states, separator_probs):
        model.add_transition(model.start, state, prob)

    model.bake()
  
    # print(train_data.shape)
    # print(train_labels.__len__())

    start_time = time.time()
    model.fit(train_data, labels=train_labels, algorithm='labeled', max_iterations=500)

    end_time = time.time()

    training_time = end_time - start_time
    t_time = training_time
    
    original_order = []
    pred_set = []
    inference_time = 0
    for chip in testset:
        tmp = []
        original_l = []
        for candiate in chip:
            tmp.append(candiate[2])
            original_l.append(candiate[0])
        per_chip = np.vstack(tmp)
        original_order.append(original_l)
        start_time = time.time()
        predictions = model.predict(per_chip)
        end_time = time.time()
        inference_time += end_time - start_time

        predicted_states = [states_names[index] for index in predictions]
        # print(predicted_states)
        pred_set.append(list(zip(original_l, predicted_states)))

    # print(model)
    sort_optimized = []
    states = []
    for candidates in pred_set:
        pred_dict = {'Noise': [], 'True': [], 'Neighbor': [], 'Neighbor2': [], 'Neighbor3': [], 'Separator': []}
        for candidate in candidates:
            pred_dict[candidate[1]].append(candidate[0])
        res = pred_dict['True'] + pred_dict['Neighbor'] + pred_dict['Neighbor2'] + pred_dict['Neighbor3'] + pred_dict[
            'Noise'] + pred_dict['Separator']
        state = ['t'] * len(pred_dict['True']) + ['nb'] * len(pred_dict['Neighbor']) + ['nb2'] * len(
            pred_dict['Neighbor2']) + ['nb3'] * len(pred_dict['Neighbor3']) + ['n'] * len(
            pred_dict['Noise']) + ['s'] * len(pred_dict['Separator'])

        states.append(state)
        sort_optimized.append(res)
        # ground-true
    ground_true = []
    type_list = []
    for candidates in testset:
        labels = []
        # res_type = []
        for candidate in candidates:
            if candidate[1] == 'True':
                labels.append(candidate[0])
            # res_type.append(candiate[3])
        ground_true.append(labels)
        type_list.append(f'{candidates[0][3]}_{candidates[0][4]}')
    return sort_optimized, original_order, ground_true, inference_time, states, t_time, type_list


if __name__ == '__main__':
    circuit = sys.argv[1]
    method = sys.argv[2]
    n_clusters = sys.argv[3]
    
    # circuit = 'fp2int'
    # method = 'hac'
    # n_clusters = 25

    print(f"dealing {circuit}_{method}_{n_clusters}")
    
    # Cluster
    n_dim = 7
    train_clusters_data = load_data.load_by_cluster(circuit, load_data.get_circuit_dict_site_input(circuit), method,
                                                    n_clusters, 'train',n_dim=n_dim)
    test_clusters_data = load_data.load_by_cluster(circuit, load_data.get_circuit_dict_site_input(circuit), method,
                                                   n_clusters, 'test',n_dim=n_dim)

    # train
    sorted_data = []
    ground_trues = []
    original_orders = []
    scores_in_cluster = []
    scores_raw_in_cluster = []
    times = []
    t_times = []
    states_li = []
    fault_list = []
    for i in range(len(train_clusters_data)):
        trainset = train_clusters_data[i]
        testset = test_clusters_data[i]
        if testset == [0]:
            continue
        sort_optimized, original_order, ground_true, run_time, states, t_time, type_list = train(trainset, testset)

        sorted_data.extend(sort_optimized)
        original_orders.extend(original_order)
        ground_trues.extend(ground_true)
        fault_list.extend(type_list)
        times.append(run_time)  # test time
        states_li.extend(states)
        t_times.append(t_time)

    res_dict = {'origin': original_orders, 'sorted': sorted_data, 'labels': ground_trues, 'states': states_li, 'fault type': fault_list,
                'times': sum(times), "t_time": sum(t_times)}
    
    
    res_folder = f"experiment/res_{n_dim}d"
    if not os.path.exists(res_folder):
        os.makedirs(res_folder)
    
    with open(f"{res_folder}/{circuit}_{method}_{n_clusters}.json", "w") as f:
        json.dump(res_dict, f, indent=4)
    
    print(f"JSON {circuit} is saved to {res_folder}")
    