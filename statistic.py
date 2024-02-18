import os
import json
import csv
from decimal import Decimal, getcontext
import re
from collections import Counter

def getFaultInfo(chip_name, chip_type, chip_id):
    if (chip_name, chip_type) not in cache_dict:
        filename = f'./DiagData/{chip_name}/{chip_type}/{chip_name}.faults'
        with open(filename, 'r') as f:
            cache_dict[(chip_name, chip_type)] = f.readlines()
    result = cache_dict[(chip_name, chip_type)][chip_id-1]
    # # 构建命令
    # command = f"sed -n '{chip_id}p' {filename}"

    # # 运行命令并获取输出
    # result = subprocess.check_output(command, shell=True, text=True)
    '''
    if chip_type in ['ssl', 'and', 'or', 'fe', 'dom']:
        numberOfFaults = 1
        listOfFaults = []
        huakuohaoli = re.search(r'\{(.*?)\}', result).group(1)
        kuohulist = huakuohaoli.split(" + ")
        for kuohutuple in kuohulist:
            matches = re.findall(r'\(([^,]+)', kuohutuple)
            if set(matches) not in listOfFaults:
                listOfFaults.append(set(matches))
            
    elif chip_type=='msl':
        numberOfFaults = 2
        listOfFaults = []
        huakuohaoli = re.search(r'\{(.*?)\}', result).group(1)
        kuohulist = huakuohaoli.split(" + ")
        for kuohutuple in kuohulist:
            matches = re.findall(r'\(([^,]+)', kuohutuple)
            if set(matches) not in listOfFaults:
                listOfFaults.append(set(matches))
    else:
        match = re.match(r'msl(\d+)$', chip_type)
        numberOfFaults = int(match.group(1))
        listOfFaults = []
        huakuohaoli = re.search(r'\{(.*?)\}', result).group(1)
        kuohulist = huakuohaoli.split(" + ")
        for kuohutuple in kuohulist:
            matches = re.findall(r'\(([^,]+)', kuohutuple)
            if set(matches) not in listOfFaults:
                listOfFaults.append(set(matches))
    '''
    listOfFaults = []
    huakuohaoli = re.search(r'\{(.*?)\}', result).group(1)
    kuohulist = huakuohaoli.split(" + ")
    for kuohutuple in kuohulist:
        matches = re.findall(r'\(([^,]+)', kuohutuple)
        if set(matches) not in listOfFaults:
            listOfFaults.append(set(matches))
    # numberOfFaults = len(listOfFaults)
    return listOfFaults


def remove_dot_corresponding_data(chip_sorted, chip_state):
    # 创建一个新的列表，存放处理后的 chip_sorted 和 chip_state
    new_chip_sorted = []
    new_chip_state = []

    # 遍历 chip_sorted 和 chip_state 的对应位置
    for sorted_value, state_value in zip(chip_sorted, chip_state):
        # 如果 chip_sorted 中的值不是 '.'，则添加到新的列表中
        if sorted_value != '.':
            new_chip_sorted.append(sorted_value)
            new_chip_state.append(state_value)

    return new_chip_sorted, new_chip_state


def check(j, state):
    # 检查第j个（下标从0开始）candidates的标签挡位
    count = Counter(state)
    # print(count)
    # assert len(count)==6 # t,nb,nb2,nb3,n,s
    s = state[j]
    ni = 0
    if s == 't':
        ni = count['t']
    elif s == 'nb':
        ni = count['nb'] + count['t']
    elif s == 'nb2':
        ni = count['nb2'] + count['nb'] + count['t']
    elif s == 'nb3':
        ni = count['nb3'] + count['nb2'] + count['nb'] + count['t']
    elif s == 'n':
        ni = count['n'] + count['nb3'] + count['nb2'] + count['nb'] + count['t']
    elif s == 's':
        ni = count['s'] + count['n'] + count['nb3'] + count['nb2'] + count['nb'] + count['t']
    else:
        raise Exception
    return ni, len(state), count


def getRemaining(chip_sorted, chip_labels, chip_states):
    c_sorted = chip_sorted
    c_labels = chip_labels
    top1, top2, top3, top4 = 0, 0, 0, 0
    for j in range(len(c_sorted)):
        if c_sorted[j] in c_labels:
            ni, c_m, count = check(j, chip_states)
            break
        else:
            continue
    # acc and top
    k = j + 1
    if 0 < k <= count['t']:
        top1 = count['t'] / c_m
    if 0 < k <= count['t'] + count['nb']:
        top2 = (count['t'] + count['nb']) / c_m
    if 0 < k <= count['t'] + count['nb'] + count['nb2']:
        top3 = (count['t'] + count['nb'] + count['nb2']) / c_m
    if 0 < k <= count['t'] + count['nb'] + count['nb2'] + count['nb3']:
        top4 = (count['t'] + count['nb'] + count['nb2'] + count['nb3']) / c_m
    
    return top1, top2, top3, top4


def getAllCountAndResolution(_origin, _sorted, _labels, _states, _types, filename, ordered_states=['t', 'nb', 'nb2', 'nb3']):
    # 总芯片数量
    n = len(_origin)
    # 分档位
    counts = {state: 0 for state in ordered_states}
    resolutions = {state: 0 for state in ordered_states}
    diagnosability = {state: 0 for state in ordered_states}
    # accuracy = {state: 0 for state in ordered_states}
    remaining = {state: 0 for state in ordered_states}
    # 工具
    tool_count = 0
    tool_diagnosability = Decimal(0)  # 实际诊断出的正确faults数量/总共插入的faults数量
    tool_resolution_without_deduplication = Decimal(0)
    # tool_accuracy = Decimal(0)

    getcontext().prec = 8

    for i in range(n):
        # origin
        chip_origin = _origin[i]
        chip_sorted = _sorted[i]
        chip_labels = _labels[i]
        chip_states = _states[i]
        # frequency_dict = Counter(chip_states)
        chip_type, chip_id = _types[i].split("_")
        chip_id = int(chip_id)
        chip_faultsList = getFaultInfo(filename.split("_")[0], chip_type, chip_id)  # 真实故障
        # modified
        # chip_origin = chip_origin[:-1]
        
        top1, top2, top3, top4 = getRemaining(chip_sorted=chip_sorted, chip_labels=chip_labels, chip_states=chip_states)
        remaining['t']+=top1
        remaining['nb']+=top2
        remaining['nb2']+=top3
        remaining['nb3']+=top4
        
        chip_sorted, chip_states = remove_dot_corresponding_data(chip_sorted=chip_sorted, chip_state=chip_states)
        
        # VIC
        for priority_idx, priority in enumerate(ordered_states):
            idx = len(chip_states) - 1
            res_idx = -1
            # input_priority = ordered_states.index(priority)
            for j in range(idx, -1, -1):
                if chip_states[j] not in ordered_states:
                    continue
                cur_priority = ordered_states.index(chip_states[j])
                if cur_priority <= priority_idx:
                    res_idx = j
                    break

            if res_idx!=-1:
                set_sorted = set(chip_sorted[:res_idx + 1])
                # set_labels = set(chip_labels[:res_idx + 1])
                set_labels = set(chip_labels)
            else:
                continue

            if not set_sorted.isdisjoint(set_labels):
                counts[priority] += 1
                
            cur_resolution = len(chip_sorted[:res_idx + 1])/len(chip_faultsList)
            cur_diagnosability = 0
            # cur_accuracy = 0
            # for subset in chip_faultsList:
            #     for element in chip_sorted[:res_idx+1]:
            #         if element in subset or any(element in sub_element for sub_element in subset):
            #             cur_accuracy += 1
            #             continue
            
            
            for subset in chip_faultsList:
                for element in chip_sorted[:res_idx+1]:
                    if element in subset or any(element in sub_element for sub_element in subset):
                        cur_diagnosability += 1
                        break
            
            
            resolutions[priority] += cur_resolution
            diagnosability[priority] += cur_diagnosability/len(chip_faultsList)
            # accuracy[priority] += cur_accuracy/len(chip_faultsList)
            # if priority=='t':
            #     remaining[priority] += frequency_dict['t']/len(chip_states)
            # elif priority=='nb':
            #     remaining[priority] += (frequency_dict['t']+frequency_dict['nb'])/len(chip_states)
            # elif priority=='nb2':
            #     remaining[priority] += (frequency_dict['t']+frequency_dict['nb']+frequency_dict['nb2'])/len(chip_states)
            # elif priority=='nb3':
            #     remaining[priority] += (frequency_dict['t']+frequency_dict['nb']+frequency_dict['nb2']+frequency_dict['nb3'])/len(chip_states)
            # else:
            #     raise Exception
           
        # Tool
        tool_resolution_without_deduplication += Decimal(len(chip_origin))/Decimal(len(chip_faultsList))
        # print(chip_faultsList)
        # print(chip_origin)
        # 初始化计数器
        r = 0
        for subset in chip_faultsList:
            for element in chip_origin:
                if element in subset or any(element in sub_element for sub_element in subset):
                    r += 1
                    break
        tool_diagnosability += Decimal(r/len(chip_faultsList))
        # tool_accuracy += Decimal(r/len(chip_faultsList))

        if not set(chip_origin).isdisjoint(set(chip_labels)):
            tool_count += 1
        else:
            print("??")
    
    dn = Decimal(n)
    counts = {f'count_{state}': Decimal(count) / dn for state, count in counts.items()}
    resolutions = {f'resolution_{state}': Decimal(resolution) / dn for state, resolution in resolutions.items()}
    diagnosability = {f'diagnosability_{state}': Decimal(diagnosability) / dn for state, diagnosability in diagnosability.items()}
    remaining = {f'remaining_{state}': Decimal(remaining) / dn for state, remaining in remaining.items()}
    tool_count = Decimal(tool_count)/dn
    tool_resolution_without_deduplication = Decimal(tool_resolution_without_deduplication)/dn
    tool_diagnosability = Decimal(tool_diagnosability/dn)

    return counts, resolutions, diagnosability, remaining, tool_count, tool_resolution_without_deduplication, tool_diagnosability



def process_json_files(root_experiment, root_clusterInfo, output_csv_path):
    chipnumber = {'s13207': 562}

    with open(output_csv_path, 'w', newline='') as csv_file:
        column_circuitInfo = ['circuit', 'method', 'cluster_test', 'cluster_train']
        column_experiment = [f'{key}_{state}' for state in ['t', 'nb', 'nb2', 'nb3'] for key in ['resolution', 'count', 'diagnosability', 'remaining']]
        # column_time = ['ClusterTime_train', 'ClusterTime_test', 'TrainTime_train', 'TrainTime_test']
        column_time = ['ClusterTime_train', 'TrainTime_train', 'PredictTime']
        # column_tool = ['tool_count', 'tool_resolution', 'tool_resolution_deduplication', 'tool_resolution_without_deduplication']
        column_tool = ['tool_count', 'tool_resolution', 'tool_diagnosability']
        fieldnames = column_circuitInfo + column_tool+column_experiment + column_time
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        files = os.listdir(root_experiment)
        files.sort()
        for filename in files:
            if filename.endswith('.json'):
                json_res = os.path.join(root_experiment, filename)
                json_cluster = os.path.join(root_clusterInfo, filename)
                
                # if f'{chip}_' not in json_cluster:
                #     continue
                
                # if not('div' in json_cluster or 'b12' in json_cluster):
                #     continue

                
                if os.path.exists(json_cluster):
                    # 如果存在同名文件
                    with open(json_res, 'r') as file1:
                        data = json.load(file1)

                    _origin = data.get('origin', None)
                    _sorted = data.get('sorted', None)
                    _labels = data.get('labels', None)
                    _states = data.get('states', None)
                    _types = data.get('fault type', None)
                    _testTime = data.get('times', None)
                    _trainTime = data.get('t_time', None)


                    counts, resolutions, diagnosability, remaining, tool_count, tool_resolution_without_deduplication, tool_diagnosability = getAllCountAndResolution(_origin, _sorted, _labels, _states, _types, filename=filename)
                    circuit, method, cluster_test = filename[:-5].split('_')

                    with open(json_cluster, 'r') as file2:
                        data = json.load(file2)

                    _cluster_trainTime = data.get('clusterTimeOnTrainingSet', None)
                    _cluster_testTime = data.get('clusterTimeOnTestSet', None)

                    # 检查 method 是否为 'dtw' 或 'hac'
                    if method == 'dtw' or method == 'hac':
                        cluster_train = data.get('clusterNumberOnTrainSet', None)
                    else:
                        cluster_train = cluster_test

                    
                    row_data = {
                        'circuit': circuit,
                        'method': method,
                        'cluster_train': cluster_train,
                        'cluster_test': cluster_test,
                        'ClusterTime_train': _cluster_trainTime,
                        # 'ClusterTime_test': _cluster_testTime,
                        'TrainTime_train': _trainTime,
                        # 'TrainTime_test': _testTime,
                        'tool_count': tool_count, 
                        'PredictTime': (_cluster_testTime+_testTime)/chipnumber[circuit] * 1000 if _cluster_testTime and _testTime else None,        # 单位是毫秒
                        # 'tool_resolution': None, 
                        # 'tool_resolution_deduplication': None,
                        'tool_resolution': tool_resolution_without_deduplication,
                        'tool_diagnosability': tool_diagnosability                       
                    }

                    row_data.update(counts)
                    row_data.update(resolutions)
                    row_data.update(diagnosability)
                    row_data.update(remaining)

                    writer.writerow(row_data)

# chip = '_table5_3d'
# if __name__ == "__main__":
#     # root_experiment = './experiment/batch_1d_'
#     # root_clusterInfo = './experiment/clusterInfo_'
#     # root_experiment = './experiment/batch_1d'
#     # root_clusterInfo = './experiment/clusterInfo'
#     root_experiment = './experiment/res_3d'
#     root_clusterInfo = './experiment/clusterInfo_'
#     output_csv_path = f'./experiment/{chip}_20240126.csv'
#     cache_dict = {}
chip = 'bd'
if __name__ == "__main__":
    root_experiment = './experiment/res_k1'
    root_clusterInfo = './experiment/clusterInfo_k=1'
    # root_experiment = './experiment/batch_1d'
    # root_clusterInfo = './experiment/clusterInfo'
    # root_experiment = './experiment/res_3d'
    # root_clusterInfo = './experiment/clusterInfo_'
    output_csv_path = f'./experiment/k1.csv'
    cache_dict = {}

    process_json_files(root_experiment, root_clusterInfo, output_csv_path)