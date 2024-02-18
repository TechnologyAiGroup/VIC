import os
import re 
import pandas as pd
import sys
# circuit = 'b12' 
# circuit = sys.argv[1]
def is_daig_success(fault_gates,candidates):
    for candiate in candidates:
       if candiate in fault_gates:
            return True
    return False
def handle(circuit):
    total_chip =0
    diag_success_num = 0
    defect_num = 0
    fault_types = [path for path in os.listdir(circuit) if os.path.isdir(f'{circuit}/{path}')]
    df = pd.DataFrame(columns=['chip_id', 'fault', 'fails','candidates', 'fault_type','match'])
    for fault_type in fault_types:
        diagnosis_report = f"{circuit}/{fault_type}/diagnosis_report"
        print(diagnosis_report)
        if not os.path.exists(diagnosis_report):
            continue
        fail_path = f"{circuit}/{fault_type}/tmax_fail"
        chip_nums = [int(file_name.split('.')[0]) for file_name in os.listdir(diagnosis_report)]
        faults = f'{circuit}/{fault_type}/{circuit}.faults'
        with open(faults,'r') as f:
            lines = f.readlines()
            for chip_id, fault in enumerate(lines, start=1) :
                if chip_id in chip_nums:
                    

                    # diag
                    with open(f'{diagnosis_report}/{chip_id}.diag','r') as f:
                        report = f.read()
                        
                        # Regular expression pattern to match the string
                        pattern = r'match=.+?%|g_.+?\/'

                        # Find all matches
                        matches = re.findall(pattern, report)
                        # Extract the matched strings
                        
                        # candidates =  [match[2:-1] for match in matches ]
                        candidates = []
                        match_score = []
                        cur_score = 0
                        for item in matches:
                            if re.search('match', item):
                                cur_score = float(item.split('=')[1][:-1])
                                continue
                            elif re.search('g_', item):
                                candidates.append(item[2:-1])
                                match_score.append(cur_score)
                                continue
                            else:
                                raise Exception
                        assert len(candidates)==len(match_score) 

                        flag = 0
                        left = 0
                        fault_gates = []
                        for i in range(len(fault)):
                            if '(' == fault[i]:
                                left = i
                                flag = 1
                            if ',' == fault[i] and flag==1:
                                gate = fault[left+1:i]
                                if '->' in gate:
                                    fault_gates.extend(gate.split('->'))
                                else:
                                    fault_gates.append(gate)
                                flag = 0
                        total_chip +=1
                        if is_daig_success(fault_gates,candidates):
                            defect_num += fault.count('+')+1
                            diag_success_num +=1
                             # create a new row with data
                            new_row = {'chip_id': chip_id, 'fault': fault_gates, 'candidates': candidates, 'match': match_score, 'fault_type':fault_type}
                            # fail data

                            fail_data = [] 
                            with open(f'{fail_path}/{chip_id}.fail','r') as f:
                                lines = f.readlines()
                                for line in lines:

                                    match = re.match(r'\d+', line)

                                    if match:
                                        first_number = int(match.group())
                                        # print(first_number)
                                        fail_data.append(str(first_number))
                                    # if 'SO' in line:
                                    #     print(circuit)
                                    #     print(line)
                                    #     break
                                    # if 'tmax' in fail_path:
                                    #     fail_data.append(line.split('\t')[0])
                                    # else:
                                    #     fail_data.append(line.split(' ')[0])
                            
                            new_row['fails'] = fail_data
                          
                            new_df = pd.DataFrame([new_row])
                            df = pd.concat([df, new_df], ignore_index=True)
    

    df.to_csv(f'./{circuit}_.csv', index=False)
    
    return total_chip, diag_success_num, defect_num                        


if __name__ == '__main__':
    df = pd.DataFrame(columns=['电路', '芯片总数', '诊断成功芯片数', 'count',"缺陷数量"])
    circuits = [path for path in os.listdir('.') if os.path.isdir(path)]
    for circuit in circuits:
            total_chip, diag_success_num,defect_num  = handle(circuit)
            new_row = {'电路': circuit, '芯片总数': total_chip, '诊断成功芯片数': diag_success_num, 'count': diag_success_num* 1.0 /total_chip,"缺陷数量": defect_num }
            new_df = pd.DataFrame([new_row])
            df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv('芯片信息_20240118.csv', index=False)
            

