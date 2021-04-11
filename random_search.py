'''
For A_04 and A_01, it's hard to find a feasible solutions, so unmute line 190 to see the whole searching process
For other small test cases, mute line 190 and 199 cuz it's really annoying to see solutions jumping out.
'''
import random
import json
import numpy as np
import time
with open("A_set//A_04.json", "r") as content:
    Instance = json.load(content)
minute_to_run = 1 # feel free to let it run for longer time

def generate_start_time_dic(pnt_set, Instance: dict):
    for intervention in pnt_set:
        minimum = 1
        maximum = int(Instance['Interventions'][intervention]['tmax'])
        Instance['Interventions'][intervention]['start'] = random.randint(minimum,maximum)
        
def check_resources(Instance: dict, pnt_set):
    penalty = 0
    resource_violation_dic = dict.fromkeys(Instance['Resources'].keys(),[])
    T_max = Instance['T']
    Resources = Instance['Resources']
    # Bounds are checked with a tolerance value
    tolerance = 1e-5
    # Compute resource usage
    resource_usage = compute_resources(Instance) # dict on resources and time
    # Compare bounds to usage
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

def check_exclusions(Instance: dict, pnt_set):
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
    return penalty, pnt_set

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
    q = compute_quantile(risk, T_max, scenario_numbers, quantile)
    return mean_risk, q

def display_basic(Instance: dict, mean_risk, quantile):
    alpha = Instance['Alpha']
    q = Instance['Quantile']
    obj_1 = np.mean(mean_risk)
    tmp = np.zeros(len(quantile))
    obj_2 = np.mean(np.max(np.vstack((quantile - mean_risk, tmp)), axis=0))
    obj_tot = alpha * obj_1 + (1-alpha)*obj_2
    return obj_tot

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
            p1, pnt_set = check_resources(Instance, pnt_set)
            p2, pnt_set = check_exclusions(Instance, pnt_set)
            if p1+p2 < lowest_penalty:
                for itv in Instance['Interventions'].keys(): # storing the best start_time
                    best_start_time[itv] = Instance['Interventions'][itv]['start']
                best_penalty_set = pnt_set # storing the penalty set under the best start_time
                lowest_penalty = p1+p2 # storing the best penalty by far

        if lowest_penalty < current_penalty: # change intervention_set if there's new lowest penalty
            intervention_set = best_penalty_set
            # turn this line on if you wanna see what happen in the searching.
            print('after',i,'interations, the best penalty is', lowest_penalty, ', and number of interventions violated is',len(best_penalty_set))
        # recover the instance to the original start
        for itv in Instance['Interventions'].keys(): # set the Instance back to the be aligned with the current best
            Instance['Interventions'][itv]['start'] = best_start_time[itv]
        i += 1
        if i == 500: # fail in compute solution after 500 interations
            print('Cannot generate feasible solution, the remaining penalty is', lowest_penalty)
            break
    if lowest_penalty == 0:
        print('Solution generated!')
        feasibility = True
    else:
        feasibility = False
    return Instance, feasibility

if __name__ == '__main__':
    start_cal = time.time()
    best_obj = float('inf')
    best_start_time = dict.fromkeys(Instance['Interventions'].keys(),0)
    while (time.time() - start_cal) / 60 < minute_to_run:
        Instance, feasibility = search_for_solution(Instance)
        if feasibility:
            mean_risk, quantile = compute_objective(Instance)
            obj_tot = display_basic(Instance, mean_risk, quantile)
            if obj_tot < best_obj:
                best_obj = obj_tot
                for intervention in Instance['Interventions'].keys():
                    best_start_time[intervention] = Instance['Interventions'][intervention]['start']
    if best_obj == float('inf'): # suggests no solutions are found, return the infeasible solution
        print('No feasible solution found in this search, constraints are violated.')
        for intervention in Instance['Interventions'].keys():
            best_start_time[intervention] = Instance['Interventions'][intervention]['start']
            mean_risk, quantile = compute_objective(Instance)
            best_obj = display_basic(Instance, mean_risk, quantile)

    print('The best objective is',best_obj)
    with open("output.txt", "w") as f: 
        for itv in Instance['Interventions'].keys():
            f.write(" ".join([itv, str(best_start_time[itv])]))
            f.write("\n")