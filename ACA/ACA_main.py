import argparse
import time
import numpy as np
import json

from checker import compute_objective
from checker import compute_penalty
from random_search import search_for_solution


def compute_cost(Instance):
    weight_penalty = 1000 # This shall be adjusted
    return compute_objective(Instance) + weight_penalty * compute_penalty(Instance)

def list2dict(arr, Instance):
    for i in range(len(arr)):
        Instance['Interventions'][intervention_pointer[i]]['start'] = arr[i]+1
    return Instance

def update_cost_matrix(Instance):
    for i in range(5):
        best_obj = float('inf')
        best_Instance = Instance
        Instance = search_for_solution(Instance)
        obj = compute_objective(Instance)
        if obj < best_obj:
            best_obj = obj
            best_Instance = Instance
    prob_matrix_cost = Tau_ori.copy()
    for i in range(n_dim):
        st = best_Instance['Interventions'][intervention_pointer[i]]['start']-1
        prob_matrix_cost[i,st] += 10
    return prob_matrix_cost,best_obj

def display_info():
    best_generation = np.array(generation_best_Y).argmin()
    best_x = generation_best_X[best_generation]
    best_y = generation_best_Y[best_generation]
    penalty = compute_penalty(list2dict(best_x,Instance))
    if penalty == 0:
        obj = compute_objective(list2dict(best_x,Instance))
    else:
        penalty = 0
    print('Duration:', round(time.time() - timing_start,2),'seconds, Objective Value: ',round(obj,2), ', Penalty Count:', penalty, ', Optimality Gap: ',round((obj/optimal_value - 1)*100,2),'%.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, required=True)

    args = parser.parse_args().file
    timing_start = time.time()
    time_lst = [60, 300, 600, 900]
    t = 0


    with open("A_set//"+args+".json", "r") as content:
        Instance = json.load(content)
    from checker import optimal_value
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

    for i in range(1,max_iter):  # 对每次迭代
        if i%10 == 0 and time.time()- timing_start > time_lst[t]:
            display_info()
            t+=1
            if t == 4:
                break
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
    #     print(i, y[index_best],y_penalty[index_best])
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
            prob_matrix_cost, obj = update_cost_matrix(Instance) 
            
        # 信息素飘散+信息素涂抹。
        Tau = (1 - rho) * Tau + delta_tau
        if i%20 == 0 and min(y_penalty) >= 5 and solution_chance == 1: # 如果我都循环20次了还无法降低penalty
            prob_matrix_cost, obj = update_cost_matrix(Instance) # 会帮我死死找到一个更好的起点
            Tau = Tau_ori.copy()
            solution_chance -= 1 # 但我只想要找一次
