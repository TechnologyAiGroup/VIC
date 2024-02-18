import json
import pandas as pd
import ast
import os


class IOGate:
    inputs = []
    outputs = []

    def __init__(self):
        self.inputs = []  # 它的输入
        self.outputs = []  # 它的输出（作为了哪些节点的输入）

    def push_output(self, cur_output):
        self.outputs.append(cur_output)

    def push_inputs(self, cur_inputs):
        self.inputs.extend(cur_inputs)

    def get_inputs(self):
        return self.inputs

    def get_outputs(self):
        return self.outputs

    def get_neighbors(self):
        return self.inputs + self.outputs


def get_circuit_dict(circuit):
    # 根据bench文件提取邻域
    circuit_dict = {}

    # 打开bench文件并读取所有行
    with open(f'./circuit/{circuit}.bench') as f:
        lines = f.readlines()

    # 遍历文件行，处理每个门的定义
    for line in lines:
        # 忽略注释行和空行
        if line.startswith('#') or not line.strip() or '=' not in line or 'DFF' in line:
            continue
        # 分离门的名称和输入输出
        # if 'I326' in line:
        #     print(line)
        output_g, input_str = line.split('=')
        output_g = output_g.strip()
        input_str = input_str.split('(')[1].rstrip(')\n')
        if ',' in input_str:
            input_gates = input_str.split(',')
        else:
            input_gates = [input_str]
        io_gate = IOGate()
        io_gate.push_inputs(input_gates)
        circuit_dict[output_g] = io_gate
        for gate in input_gates:
            if gate in circuit_dict:
                circuit_dict[gate].push_output(output_g)
            else:
                io_gate = IOGate()
                io_gate.push_output(output_g)
                circuit_dict[gate] = io_gate
    return circuit_dict


class IOGate:
    inputs = []
    outputs = []
    # add on 20230515
    side_inputs = []

    def __init__(self):
        self.inputs = []  # 它的输入
        self.outputs = []  # 它的输出（作为了哪些节点的输入）
        self.side_inputs = []

    def push_output(self, cur_output):
        self.outputs.append(cur_output)

    def push_inputs(self, cur_inputs):
        self.inputs.extend(cur_inputs)

    def push_side_inputs(self, cur_side_inputs):
        self.side_inputs.extend(cur_side_inputs)

    def get_inputs(self):
        return self.inputs

    def get_outputs(self):
        return self.outputs

    def get_side_inputs(self):
        return self.side_inputs

    def get_neighbors(self):
        return self.inputs + self.outputs + self.side_inputs


def get_circuit_dict_site_input(circuit):
    # 根据bench文件提取邻域
    circuit_dict = {}

    # 打开bench文件并读取所有行
    with open(f'./circuit/{circuit}.bench') as f:
        lines = f.readlines()

    # 遍历文件行，处理每个门的定义
    for line in lines:
        # 忽略注释行和空行
        if line.startswith('#') or not line.strip() or '=' not in line or 'DFF' in line:
            continue
        # 分离门的名称和输入输出
        # if 'I326' in line:
        #     print(line)
        output_g, input_str = line.split('=')
        output_g = output_g.strip()
        input_str = input_str.split('(')[1].rstrip(')\n')
        if ',' in input_str:
            input_gates = input_str.split(', ')
        else:
            input_gates = [input_str]

        io_gate = IOGate()  # 当前gate
        # add side inputs
        if len(input_gates) > 1:
            for g in input_gates:
                if g not in circuit_dict:
                    side_gate = IOGate()  # 邻域gate
                else:
                    side_gate = circuit_dict[g]
                side_inputs_gates = list(input_gates)
                side_inputs_gates.remove(g)
                assert len(side_inputs_gates) == len(input_gates) - 1
                side_gate.push_side_inputs(side_inputs_gates)
                circuit_dict[g] = side_gate

        io_gate.push_inputs(input_gates)
        circuit_dict[output_g] = io_gate
        for gate in input_gates:
            if gate in circuit_dict:
                circuit_dict[gate].push_output(output_g)
            else:
                io_gate = IOGate()
                io_gate.push_output(output_g)
                circuit_dict[gate] = io_gate
    return circuit_dict


def load(circuit, circuit_dict, method):
    path = f'./cluster_data/{circuit}_{method}.csv'
    data_df = pd.read_csv(path)
    labeled_data = []
    for i in range(data_df.shape[0]):
        faults = ast.literal_eval(data_df['fault'][i])
        neighbors = []
        for fault in faults:
            if fault in circuit_dict:
                neighbors.extend(circuit_dict[fault].get_neighbors())
        candidates = ast.literal_eval(data_df['candidates'][i])
        states = ('True', 'Neighbor', 'Noise', 'Separator')
        res = []
        for candidate in candidates:
            if candidate in faults:
                res.append((candidate, states[0]))
            elif candidate in neighbors:
                res.append((candidate, states[1]))
            else:
                res.append((candidate, states[2]))
        res.append(('.', states[3]))
        labeled_data.append(res)
    return labeled_data


def load_by_cluster(circuit, circuit_dict, method, n_clusters, trainOrTest):
    if method!='k1':
        path = f'./cluster_data_/{trainOrTest}/{circuit}_{method}_{n_clusters}.csv'
    else:
        path = f'./cluster_data_k=1/{trainOrTest}/{circuit}_{method}_{n_clusters}.csv'
    # print(path)
    # path = f'./data_cmp/{trainOrTest}/{circuit}_{method}_{n_clusters}.csv'
    if method == 'hac' or method == 'dtw':
        # 当方法为 hac 时，通过文件名查找 n_clusters
        json_file_pattern = f'{circuit}_{method}_'
        json_files = [f for f in os.listdir('./experiment/clusterInfo_/') if
                      f.startswith(json_file_pattern) and f.endswith('.json')]

        # 确保找到了至少一个JSON文件
        if json_files:
            json_filename = json_files[0]  # 取列表中的第一个文件
            with open(f'./experiment/clusterInfo_/{json_filename}', 'r') as json_file:
                data = json.load(json_file)
                # 从JSON文件内容获取 n_clusters
                n_clusters_code = data['clusterNumberOnTrainSet']   # 写代码用
                # n_clusters_test = data['clusterNumberOnTestSet']  # 文件命名用
    else:
        n_clusters_code = n_clusters
    
    data_df = pd.read_csv(path)
    groups = data_df.groupby('cluster_label')
    groups_data = [[0] for _ in range(int(n_clusters_code))]
    for _, group_df in groups:
        labeled_data = []
        cluster = group_df.iloc[0, -1]
        # assert 0 <= cluster < 16
        for i, _ in group_df.iterrows():
            faults = ast.literal_eval(group_df['fault'][i])
            neighbors = []

            for fault in faults:
                if fault in circuit_dict:
                    neighbors.extend(circuit_dict[fault].get_neighbors())

            neighbors_2 = []
            for neighbor in neighbors:
                neighbors_2.extend(circuit_dict[neighbor].get_neighbors())
            neighbors_3 = []
            for neighbor in neighbors_2:
                neighbors_3.extend(circuit_dict[neighbor].get_neighbors())

            candidates = ast.literal_eval(group_df['candidates'][i])
            fault_type = group_df['fault_type'][i]  
            chip_id = group_df['chip_id'][i]  
            chip_info = f'{fault_type}_{chip_id}'
            states = ('True', 'Neighbor', 'Neighbor2', 'Neighbor3', 'Noise', 'Separator')
            res = []
            for candidate in candidates:
                if candidate in faults:
                    # 20230626
                    if trainOrTest == 'train':
                        res += [(candidate, states[0],chip_info)] * 9
                    res.append((candidate, states[0],chip_info))
                elif candidate in neighbors:
                    res.append((candidate, states[1],chip_info))
                elif candidate in neighbors_2:
                    res.append((candidate, states[2],chip_info))
                elif candidate in neighbors_3:
                    res.append((candidate, states[3],chip_info))
                else:
                    res.append((candidate, states[4],chip_info))
            res.append(('.', states[5],chip_info))

            labeled_data.append(res)
        groups_data[cluster] = labeled_data
    return groups_data


if __name__ == '__main__':
    # 输出结果字典
    # print(circuit_dict)

    circuit = 's35932'
    method = 'f2'
    n_clusters = 16
    train_clusters_data = load_by_cluster(circuit, get_circuit_dict_site_input(circuit), method,
                                                    n_clusters, 'train')
    test_clusters_data = load_by_cluster(circuit, get_circuit_dict_site_input(circuit), method,
                                                   n_clusters, 'test')
