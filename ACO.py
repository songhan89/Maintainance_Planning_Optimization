import argparse
import time
import numpy as np
import json
import random
import copy
import sys

optimal_value = {
    "A_01": 1767.8156110,
    "A_02": 4671.3766110,
    "A_03": 848.1786111,
    "A_04": 2085.8760540,
    "A_05": 635.2217857,
    "A_06": 590.6235989,
    "A_07": 2272.7822740,
    "A_08": 744.2932352,
    "A_09": 1507.2847840,
    "A_10": 2994.8487350,
    "A_11": 495.2557702,
    "A_12": 789.6349276,
    "A_13": 1998.6621620,
    "A_14": 2264.1243210,
    "A_15": 2268.5691500
}

## Retrieve effective risk distribution given starting times solution
def compute_risk_distribution(Interventions: dict, T_max: int, scenario_numbers):
    # Init risk table
    risk = [scenario_numbers[t] * [0] for t in range(T_max)]
    # Compute for each intervention independently
    for intervention in Interventions.values():
        # Retrieve Intervention's usefull infos
        intervention_risk = intervention['risk']
        # start time should be defined (already checked in scheduled constraint checker)
        if not 'start' in intervention:
            continue
        start_time = intervention['start']
        start_time_idx = int(start_time) - 1 # index for list getter
        delta = int(intervention['Delta'][start_time_idx])
        for time in range(start_time_idx, start_time_idx + delta):
            for i, additional_risk in enumerate(intervention_risk[str(time + 1)][str(start_time)]):
                risk[time][i] += additional_risk
    return risk

## Compute mean for each period
def compute_mean_risk(risk, T_max: int, scenario_numbers):
    mean_risk = np.zeros(T_max)
    # compute mean
    for t in range(T_max):
        mean_risk[t] = sum(risk[t]) / scenario_numbers[t]
    return mean_risk

## Compute quantile for each period
def compute_quantile(risk, T_max: int, scenario_numbers, quantile):
    # Init quantile
    q = np.zeros(T_max)
    for t in range(T_max):
        risk[t].sort()
        q[t] = risk[t][int(np.ceil(scenario_numbers[t] * quantile))-1]
    return q

## Compute both objectives: mean risk and quantile
def compute_objective(Instance: dict):
    # Retrieve usefull infos
    T_max = Instance['T']
    scenario_numbers = Instance['Scenarios_number']
    Interventions = Instance['Interventions']
    quantile = Instance['Quantile']
    # Retrieve risk final distribution
    risk = compute_risk_distribution(Interventions, T_max, scenario_numbers)
    # Compute mean risk
    mean_risk = compute_mean_risk(risk, T_max, scenario_numbers)
    # Compute quantile
    quantile = compute_quantile(risk, T_max, scenario_numbers, quantile)
    alpha = Instance['Alpha']
    q = Instance['Quantile']
    obj_1 = np.mean(mean_risk)
    tmp = np.zeros(len(quantile))
    obj_2 = np.mean(np.max(np.vstack((quantile - mean_risk, tmp)), axis=0))
    obj_tot = alpha * obj_1 + (1-alpha)*obj_2
    return obj_tot

def check_resources(Instance: dict, pnt_set = set(),record_itv = False):
    penalty = 0
    resource_violation_dic = dict.fromkeys(Instance['Resources'].keys(),[])
    T_max = Instance['T']
    Resources = Instance['Resources']
    # Bounds are checked with a tolerance value
    tolerance = 1e-5
    # Compute resource usage
    resource_usage = compute_resources(Instance) # dict on resources and time
    # Compare bounds to usage
    if record_itv:
        for resource_name, resource in Resources.items():
            for time in range(T_max):
                # retrieve bounds values
                upper_bound = resource['max'][time]
                lower_bound = resource['min'][time]
                # Consumed value
                worload = resource_usage[resource_name][time]
                # Check max
                if worload > upper_bound + tolerance:
                    penalty += 1
                    resource_violation_dic[resource_name] = resource_violation_dic[resource_name] + [time+1]
                if worload < lower_bound - tolerance:
                    penalty += 1
                    resource_violation_dic[resource_name] = resource_violation_dic[resource_name] + [time+1]
        if penalty == 0:
            return 0,set()
        else:
            for resource in Instance['Resources'].keys():
                if resource_violation_dic[resource] == []:
                    del resource_violation_dic[resource]
            for intervention in Instance['Interventions'].keys():
                start_time = Instance['Interventions'][intervention]['start']
                end_time = start_time + int(Instance['Interventions'][intervention]['Delta'][start_time-1])
                for resource in resource_violation_dic:
                    if resource in Instance['Interventions'][intervention]['workload'].keys():
                        for t in resource_violation_dic[resource]:
                            if start_time <= t < end_time:
                                pnt_set.add(intervention)
            return penalty, pnt_set
    else:
        for resource_name, resource in Resources.items():
            for time in range(T_max):
                # retrieve bounds values
                upper_bound = resource['max'][time]
                lower_bound = resource['min'][time]
                # Consumed value
                worload = resource_usage[resource_name][time]
                # Check max
                if worload > upper_bound + tolerance:
                    penalty += 1
                if worload < lower_bound - tolerance:
                    penalty += 1
        return penalty

def check_exclusions(Instance: dict, pnt_set = set(),record_itv = False):
    # Retrieve Interventions and Exclusions
    penalty = 0
    Interventions = Instance[ 'Interventions']
    Exclusions = Instance['Exclusions']
    # Assert every exclusion holds
    for exclusion in Exclusions.values():
        # Retrieve exclusion infos
        [intervention_1_name, intervention_2_name, season] = exclusion
        # Retrieve concerned interventions...
        intervention_1 = Interventions[intervention_1_name]
        intervention_2 = Interventions[intervention_2_name]
        # ... their respective starting times...
        intervention_1_start_time = intervention_1['start']
        intervention_2_start_time = intervention_2['start']
        # ... and their respective deltas (duration)
        intervention_1_delta = int(intervention_1['Delta'][intervention_1_start_time - 1]) # get index in list
        intervention_2_delta = int(intervention_2['Delta'][intervention_2_start_time - 1]) # get index in list
        # Check overlaps for each time step of the season
    
        for time_str in Instance['Seasons'][season]:
            time = int(time_str)
            if (intervention_1_start_time <= time < intervention_1_start_time + intervention_1_delta) and (intervention_2_start_time <= time < intervention_2_start_time + intervention_2_delta):
                penalty += 1
                pnt_set.add(intervention_1_name)
                pnt_set.add(intervention_2_name)
    if record_itv:
        return penalty, pnt_set
    else:
        return penalty
   

# process function
def compute_resources(Instance: dict):
    # Retrieve usefull infos
    Interventions = Instance['Interventions']
    T_max = Instance['T']
    Resources = Instance['Resources']
    # Init resource usage dictionnary for each resource and time
    resources_usage = {}
    for resource_name in Resources.keys():
        resources_usage[resource_name] = np.zeros(T_max)
    # Compute value for each resource and time step
    for intervention_name, intervention in Interventions.items():
        start_time = intervention['start']
        start_time_idx = start_time - 1 #index of list starts at 0
        intervention_worload = intervention['workload']
        intervention_delta = int(intervention['Delta'][start_time_idx])
        # compute effective worload
        for resource_name, intervention_resource_worload in intervention_worload.items():
            for time in range(start_time_idx, start_time_idx + intervention_delta):
                # null values are not available
                if str(time+1) in intervention_resource_worload and str(start_time) in intervention_resource_worload[str(time+1)]:
                    resources_usage[resource_name][time] += intervention_resource_worload[str(time+1)][str(start_time)]
    return resources_usage

def compute_penalty(Instance: dict):
    return check_exclusions(Instance) + check_resources(Instance)

def generate_start_time_dic(pnt_set, Instance: dict):
    for intervention in pnt_set:
        minimum = 1
        maximum = int(Instance['Interventions'][intervention]['tmax'])
        Instance['Interventions'][intervention]['start'] = random.randint(minimum,maximum)

def search_for_solution(Instance):
    # initialize with all interventions
    lowest_penalty = len(Instance['Interventions'].keys())
    intervention_set = Instance['Interventions'].keys()
    best_start_time = dict.fromkeys(Instance['Interventions'].keys(),0)
    i = 0
    while lowest_penalty > 0: # if penalty is 0, jump out of the loop
        current_penalty = lowest_penalty
        for _ in range(10): # generate 10 interention combinations and pick the best to proceed
            generate_start_time_dic(intervention_set, Instance)
            pnt_set = set()
            p1, pnt_set = check_resources(Instance, pnt_set,True)
            p2, pnt_set = check_exclusions(Instance, pnt_set,True)
            if p1+p2 < lowest_penalty:
                best_Instance = copy.deepcopy(Instance)
                best_penalty_set = pnt_set # storing the penalty set under the best start_time
                lowest_penalty = p1+p2 # storing the best penalty by far

        if lowest_penalty < current_penalty: # change intervention_set if there's new lowest penalty
            intervention_set = best_penalty_set
            # turn this line on if you wanna see what happen in the searching.
            # print('after',i,'interations, the best penalty is', lowest_penalty, ', and number of interventions violated is',len(best_penalty_set))
        # recover the instance to the original start
        Instance = best_Instance.copy()
        i += 1
        if i == 100: # fail in compute solution after 100 interations, reset everything and start over
            lowest_penalty = len(Instance['Interventions'].keys())
            intervention_set = Instance['Interventions'].keys()
            best_start_time = dict.fromkeys(Instance['Interventions'].keys(),0)
            i = 0
    return Instance


def compute_cost(Instance):
    weight_penalty = 1000 # This shall be adjusted
    return compute_objective(Instance) + weight_penalty * compute_penalty(Instance)

def list2dict(arr, Instance):
    for i in range(len(arr)):
        Instance['Interventions'][intervention_pointer[i]]['start'] = arr[i]+1
    return Instance

def update_cost_matrix(Instance):
    Instance = search_for_solution(Instance)
    prob_matrix_cost = Tau_ori.copy()
    for i in range(n_dim):
        st = Instance['Interventions'][intervention_pointer[i]]['start']
        prob_matrix_cost[i,st-1] += 10 # 因为这个答案也不是很好，所以我也只给十个提示
    return Instance, prob_matrix_cost


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, required=True)

    args = parser.parse_args().file
    if args == 'A_04':
        print('cannot run A_04, will stuck!')
        sys.exit()

    timing_start = time.time()
    time_lst = [60, 300, 600, 900]
    t = 0

    with open("A_set//"+args+".json", "r") as content:
        Instance = json.load(content)
    optimal_value = optimal_value[args]

    # 我们用的是list不是dict，所以需要做一个map的dict以便稍后转成dict
    intervention_pointer = {}
    i = 0
    for intervention in Instance['Interventions'].keys():
        intervention_pointer[i] = intervention
        i += 1
        
    n_dim = len(Instance['Interventions'].keys())
    size_pop = 50 # 蚂蚁数量
    max_iter = 100000 # 迭代次数
    alpha = 1 # 信息素的重要程度
    beta = 2 # 适应度的重要程度
    rho = 0.1 # 信息素的挥发程度
    # 信息素矩阵，每次迭代都会更新
    # 因为有开始时间窗，对超过时间窗的开始时间置零

    Tau_ori = np.zeros((n_dim,Instance['T']))
    i = 0
    for intervention in Instance['Interventions'].keys():
        intervention_array = np.ones((1,int(Instance['Interventions'][intervention]['tmax'])))
        intervention_array.resize(1,Instance['T'])
        Tau_ori[i] = intervention_array
        i+=1
    Tau = Tau_ori.copy()

    # 某一代每个蚂蚁的开始时间选择    
    Table = np.zeros((size_pop, n_dim)).astype(np.int)

    # 某一代每个蚂蚁的总ojb
    y = None

    # 记录各代的最佳情况
    generation_best_X, generation_best_Y = [], []

    # 历史原因，为了保持统一
    x_best_history, y_best_history = generation_best_X, generation_best_Y

    best_x, best_y = None, None

    prob_matrix_cost = Tau_ori.copy()

    solution_chance = 1
    report_lst = []
    for i in range(1,max_iter):  # 对每次迭代
    
        prob_matrix = (Tau ** alpha) * (prob_matrix_cost) ** beta  # 转移概率，无须归一化。
        for j in range(size_pop):  # 对每个蚂蚁
            for k in range(n_dim):  # 蚂蚁到达的每个节点
                prob = prob_matrix[k]
                prob = prob / prob.sum()  # 概率归一化
                start_time = np.random.choice(Instance['T'], size=1, p=prob)[0]
                Table[j,k] = start_time

        # 计算距离
        y_penalty = np.array([compute_penalty(list2dict(i,Instance)) for i in Table])
        y = np.array([compute_cost(list2dict(i,Instance)) for i in Table])
        # 顺便记录历史最好情况
        index_best = y.argmin()
        # print(i, y[index_best],y_penalty[index_best])
        x_best, y_best = Table[index_best, :].copy(), y[index_best].copy()
        generation_best_X.append(x_best)
        generation_best_Y.append(y_best)

        if min(y) <= min(generation_best_Y): # 好的才涂，坏的不要涂
            top=3 # 涂也只给top 3 涂
            y_top = np.argsort(y)[:top]
            delta_tau = np.zeros((n_dim,Instance['T']))
            for j in range(top):  # 每个蚂蚁
                for k in range(n_dim):  # 每个节点
                    original_cost = delta_tau[k,Table[y_top[j],k]]
                    current_cost = 100 / y[y_top[j]]
                    if current_cost > original_cost:
                        delta_tau[k, Table[y_top[j],k]] = current_cost # 涂抹的信息素,涂得多一点~
        
        if len(set(y)) <= 2 or (max(y)-min(y)) < min(y)/100: # 肯定卡住了， 给我来个重启
            Tau = Tau_ori.copy()
            Instance, prob_matrix_cost = update_cost_matrix(Instance) 
            
        # 信息素飘散+信息素涂抹。
        Tau = (1 - rho) * Tau + delta_tau

        if i%15 == 0 and min(y_penalty) >= 5 and solution_chance == 1: # 如果我都循环15次了还无法降低penalty
            Instance, prob_matrix_cost = update_cost_matrix(Instance) # 会帮我死死找到一个更好的起点
            soft_Instance = copy.deepcopy(Instance)
            Tau = Tau_ori.copy()
            solution_chance -= 1 # 但我只想要找一次

        if time.time()- timing_start > time_lst[t]-2:
            best_generation = np.array(generation_best_Y).argmin()
            best_x = generation_best_X[best_generation]
            best_y = generation_best_Y[best_generation]
            penalty = compute_penalty(list2dict(best_x,Instance))
            if penalty == 0:
                Instance = list2dict(best_x,Instance)
            else:
                Instance = copy.deepcopy(soft_Instance)
            penalty = compute_penalty(Instance)
            obj = compute_objective(Instance)
            display_report = 'Duration: '+ str(round(time.time() - timing_start,2))+' seconds, Objective Value: '+str(round(obj,2))+ ', Penalty Count: '+ str(penalty)+ ', Optimality Gap: '+str(round((obj/optimal_value - 1)*100,2))+'%.'
            report_lst.append(display_report)
            print(display_report)
            # Storing solutions
            with open(args+"_ACO_"+str(time_lst[t])+"s.txt", "w") as f: 
                for itv in Instance['Interventions'].keys():
                    f.write(" ".join([itv, str(Instance['Interventions'][itv]['start'])]))
                    f.write("\n")
            t+=1        
            if t == len(time_lst):# Outside of time limit
                break

    with open(args+"_ACO_report.txt", "w") as f:
        f.write("Scenario: "+args+"\n")
        for i in range(len(time_lst)):
            f.write(report_lst[i])
            f.write('\n')