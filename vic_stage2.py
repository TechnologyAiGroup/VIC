import sys
import os
import numpy as np
import pandas as pd
import time
import load_data
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from itertools import accumulate
import json
# from sklearnex import patch_sklearn
# patch_sklearn()


def train(trainset, testset):
    # Create labeled candidate states for the training set and test set
    train_set, test_set = trainset, testset
    train_tagged_words = [tup for sent in train_set for tup in sent]

   
    tags = {tag for _, tag,_ in train_tagged_words}

    # train time
    start_time = time.time()
    tags_matrix = np.zeros((len(tags), len(tags)), dtype='float32')
    for i, t1 in enumerate(list(tags)):
        for j, t2 in enumerate(list(tags)):
            tags_matrix[i, j] = t2_given_t1(t2, t1, train_tagged_words)[0] / t2_given_t1(t2, t1, train_tagged_words)[1]


    tags_df = pd.DataFrame(tags_matrix, columns=list(tags), index=list(tags))

    candidates_lens = [len(sent) for sent in test_set]
    candidates_indexs = list(accumulate(candidates_lens))

    test_untagged_words = [tup[0] for sent in test_set for tup in sent]
    end_time = time.time()
    t_time = end_time - start_time

    # test time
    start = time.time()
    tagged_seq = Viterbi(test_untagged_words, train_bag=train_tagged_words, tags_df=tags_df)
    end = time.time()
    difference = end - start

    pred_set = [tagged_seq[:candidates_indexs[0]]]
    for i in range(1, len(candidates_indexs)):
        pred_set.append(tagged_seq[candidates_indexs[i - 1]:candidates_indexs[i]])
    original_order = []
    for chip in pred_set:
        tmp = []
        for tup in chip[:-1]:
            tmp.append(tup[0])
        original_order.append(tmp)

    # Optimize candidate sorting
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
    for candidates in test_set:
        res = []
        for candidate in candidates:
            if candidate[1] == 'True':
                res.append(candidate[0])
        ground_true.append(res)
        type_list.append(f'{candidates[0][2]}')

    return sort_optimized, original_order, ground_true, difference, states, t_time,type_list


def check(sort_optimized, ground_true, quantile):
    # count_complete = 0
    # for i in range(len(sort_optimized)):
    #     retain_candidates = sort_optimized[i]
    #     fault_set = [fault for fault in ground_true[i]]
    #     for candidate in retain_candidates:
    #         if candidate in fault_set:
    #             count_complete += 1
    #             break
            
    # print(count_complete)
    # print(len(sort_optimized))
    
    count = 0
    # print(quantile)
    for i in range(len(sort_optimized)):
        length = int(quantile * len(sort_optimized[i])-0.5)
        if length <=  0:
            continue
        retain_candidates = sort_optimized[i][:length]
        fault_set = set(fault for fault in ground_true[i])
        for candidate in retain_candidates:
            if candidate in fault_set:
                count += 1
                break
    acc_for_chip = count / len(sort_optimized)
    return acc_for_chip


# Calculate emission probability
def word_given_tag(word, tag, train_bag):
    tag_list = [pair for pair in train_bag if pair[1] == tag]
    count_tag = len(tag_list)  
    w_given_tag_list = [pair[0] for pair in tag_list if pair[0] == word]

    count_w_given_tag = len(w_given_tag_list)

    return count_w_given_tag, count_tag


# Calculate transition probability
def t2_given_t1(t2, t1, train_bag):
    tags = [pair[1] for pair in train_bag]
    count_t1 = len([t for t in tags if t == t1])
    count_t2_t1 = 0
    for index in range(len(tags) - 1):
        if tags[index] == t1 and tags[index + 1] == t2:
            count_t2_t1 += 1
    return count_t2_t1, count_t1


def Viterbi_old(words, train_bag, tags_df):
    state = []
    # Get the set of all different tags in the training set
    T = list(set([pair[1] for pair in train_bag]))
    T = sorted(T)
    for key, word in enumerate(words):
       # Initialize the probability column list for the given observation
        p = []
        for tag in T:
            if key == 0:
              
                transition_p = tags_df.loc['Separator', tag]
            else:
                
                transition_p = tags_df.loc[state[-1], tag]

               # Calculate emission and state probabilities
            emission_p = word_given_tag(words[key], tag, train_bag)[0] / word_given_tag(words[key], tag, train_bag)[1]
            state_probability = emission_p * transition_p
            p.append(state_probability)

            # Get the state with the maximum probability
        pmax = max(p)
        state_max = T[p.index(pmax)]
        state.append(state_max)
    return list(zip(words, state))


def Viterbi(words, train_bag, tags_df,smoothing_param=0.5):
    state = []
    # Get the set of all different tags in the training set
    T = list(set([pair[1] for pair in train_bag]))
    T = sorted(T)
    for key, word in enumerate(words):
       # Initialize the probability column list for the given observation
        p = np.zeros(len(T))
        for i, tag in enumerate(T):
            if key == 0:
                # Add Laplace smoothing to transition probability
                transition_p = (tags_df.loc['Separator', tag] + smoothing_param) / \
                               (1 + len(T) * smoothing_param)
              
            else:
                transition_p = (tags_df.loc[state[-1], tag] + smoothing_param) / \
                               (1 + len(T) * smoothing_param)

            # Calculate emission and state probabilities
            temp =  word_given_tag(words[key], tag, train_bag)
            emission_p = temp[0] / temp[1]
            state_probability = emission_p * transition_p
            p[i] = state_probability

            # Get the state with the maximum probability
        # Get the state with the maximum probability using NumPy functions
        state_max = T[np.argmax(p)]
        state.append(state_max)
    return list(zip(words, state))


if __name__ == '__main__':
    circuit = sys.argv[1]
    method = sys.argv[2]
    n_clusters = sys.argv[3]
    
    # circuit = 's1488'
    # method = 'dtw'
    # n_clusters = '2'

    print(f"dealing {circuit}_{method}_{n_clusters}")
    
    # Cluster
    train_clusters_data = load_data.load_by_cluster(circuit, load_data.get_circuit_dict_site_input(circuit), method,
                                                    n_clusters, 'train')
    test_clusters_data = load_data.load_by_cluster(circuit, load_data.get_circuit_dict_site_input(circuit), method,
                                                   n_clusters, 'test')

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
        sort_optimized, original_order, ground_true, run_time, states, t_time,type_list = train(trainset, testset)

        sorted_data.extend(sort_optimized)
        original_orders.extend(original_order)
        ground_trues.extend(ground_true)
        fault_list.extend(type_list)
        times.append(run_time)  # test time
        states_li.extend(states)
        t_times.append(t_time)

    res_dict = {'origin': original_orders, 'sorted': sorted_data, 'labels': ground_trues, 'states': states_li, 'fault type': fault_list,
                'times': sum(times), "t_time": sum(t_times)}
    res_folder = "experiment/batch_1d_"
    if not os.path.exists(res_folder):
        os.makedirs(res_folder)
    
    with open(f"{res_folder}/{circuit}_{method}_{n_clusters}.json", "w") as f:
        json.dump(res_dict, f, indent=4)
    
    print(f"JSON {circuit} is saved to {res_folder}")
    