# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 16:39:09 2019
Lasrt update on Wed Feb 12 10:35:00 2020

@author: rte-challenge-roadef-2020-team
"""
import os
import datetime
import sys
import numpy as np
import json
import glob


####################
### Utils ##########
####################

## Global variables
CUR_DIR = os.getcwd()
PAR_DIR = os.path.dirname(CUR_DIR)
RESOURCES_STR = 'Resources'
SEASONS_STR = 'Seasons'
INTERVENTIONS_STR = 'Interventions'
EXCLUSIONS_STR = 'Exclusions'
T_STR = 'T'
SCENARIO_NUMBER = 'Scenarios_number'
RESOURCE_CHARGE_STR = 'workload'
TMAX_STR = 'tmax'
DELTA_STR = 'Delta'
MAX_STR = 'max'
MIN_STR = 'min'
RISK_STR = 'risk'
START_STR = 'start'
QUANTILE_STR = "Quantile"
ALPHA_STR = "Alpha"
COST_P_SCHEDULE = 200000
# COST_P_RESOURCE = 20
# COST_P_EXCLUDE = 20
COST_P_RESOURCE = 5
COST_P_EXCLUDE = 5
NEIGHBORHOOD_SIZE= 50
TOP_K= 5
TIME_WINDOW= 5
PROB_CHANGE = 0.98
#For A06
# COST_P_RESOURCE = 50
# COST_P_EXCLUDE = 50
# TIME_WINDOW= 20
# PROB_CHANGE = 0.8

#For A01
COST_P_RESOURCE = 20
COST_P_EXCLUDE = 20
TIME_WINDOW= 5
PROB_CHANGE = 0.98



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

def local_beam_search(solution_list, solution: dict, optimal_value, k=TOP_K, gap=0.15):

    min_p_cost = 1
    min_obj_cost = 999999
    stop_obj_cost = (1 + gap) * optimal_value

    while min_obj_cost > stop_obj_cost or min_p_cost > 0:
        p_cost_list = []
        obj_cost_list = []
        total_cost_list = []
        for s in solution_list:
            list_to_dict(s, solution)
            p_cost = compute_penalty(solution)
            obj_cost = compute_objective(solution)
            p_cost_list.append(p_cost)
            obj_cost_list.append(obj_cost)
            total_cost_list.append(obj_cost + p_cost)

        sorted_idx_top_k = np.argsort(total_cost_list)[0:TOP_K]
        # if min_obj_cost >= 5000:
        #     sorted_idx_top_k = np.argsort(total_cost_list)[0:TOP_K]
        # else:
        #     sorted_idx_top_k = np.argsort(p_cost_list)[0:TOP_K]
        #     PROB_CHANGE = 0.99
        #     TIME_WINDOW = 3

        min_idx = sorted_idx_top_k[0]
        min_obj_cost = obj_cost_list[min_idx]
        min_p_cost = p_cost_list[min_idx]
        print (obj_cost_list[min_idx], p_cost_list[min_idx]/COST_P_EXCLUDE)
        if min_obj_cost > stop_obj_cost or min_p_cost > 0:
            solution_list = solution_list[sorted_idx_top_k]
            solution_list = generate_neighbourhood(solution_list=solution_list, solution=solution)
        else:
            list_to_dict(solution_list[min_idx], solution)


def list_to_dict(solution_list, solution: dict):

    INTERVENTIONS_NAMES = list(solution[INTERVENTIONS_STR].keys())
    for i in range(len(solution_list)):
        solution[INTERVENTIONS_STR][INTERVENTIONS_NAMES[i]][START_STR] = solution_list[i]

def generate_neighbourhood(solution_list, solution: dict):
    neighbourhood_list = np.resize(solution_list, (NEIGHBORHOOD_SIZE, solution_list.shape[1]))
    neighbourhood_list = np.apply_along_axis(pertubate_startime, axis=1, arr=neighbourhood_list, solution=solution)

    return neighbourhood_list

def initial_neighbourhood(solution: dict, k=TOP_K):

    interventions = solution[INTERVENTIONS_STR]
    arr = [int(interventions[intervention_name][START_STR]) for intervention_name in interventions.keys()]

    return np.resize(arr,(k, len(arr)))

def random_neighbourhood(solution: dict, k=TOP_K):

    interventions = solution[INTERVENTIONS_STR]
    arr = [[np.random.randint(1, int(interventions[intervention_name][TMAX_STR]) + 1) for intervention_name in interventions.keys()] for i in range(k)]
    return np.array(arr)

def pertubate_startime(solution_list, solution: dict):

    INTERVENTIONS_NAMES = list(solution[INTERVENTIONS_STR].keys())

    for i in range(len(solution_list)):
        intervention = solution[INTERVENTIONS_STR][INTERVENTIONS_NAMES[i]]
        start = int(intervention[START_STR])
        tmax = int(intervention[TMAX_STR])
        step = TIME_WINDOW
        new_start = np.random.randint(max(start - step,1), min(start + step, tmax + 1))
        delta = int(intervention[DELTA_STR][new_start - 1])

        if new_start + delta > tmax:
            new_start = max(tmax - delta, 1)

        if np.random.random() > PROB_CHANGE:
            solution_list[i] = new_start

    return solution_list


## Json reader
def read_json(filename: str):
    """Read a json file and return data as a dict object"""

    print('Reading json file ' + filename + '...')
    f = open(filename, 'r')
    Instance = json.load(f)
    f.close()
    print('Done')

    return Instance

def read_solution_from_txt(Instance: dict, solution_filename: str):
    """Read a txt formated Solution file, and store the solution informations in Instance"""

    print('Loading solution from ' + solution_filename + '...')
    # Load interventions
    Interventions = Instance[INTERVENTIONS_STR]
    # Read file line by line, and store starting time value (no checks yet, except format and duplicate)
    solution_file = open(solution_filename, 'r')
    for line in solution_file:
        # Split line to retrive infos: Intervention name and decided starting date
        tmp = line.split(' ')
        intervention_name = tmp[0]
        start_time_str = tmp[1].split('\n')[0]
        # Assert Intervention exists
        if not intervention_name in Interventions:
            print('ERROR: Unexpected Intervention ' + intervention_name + ' in solution file ' + solution_filename + '.')
            continue
        # Assert starting date is an integer
        start_time: int
        try:
            start_time = int(start_time_str)
        except ValueError:
            print('ERROR: Unexpected starting time ' + start_time_str + ' for Intervention ' + intervention_name + '. Expect integer value.')
            continue
        # Assert no duplicate
        if START_STR in Interventions[intervention_name]:
            print('ERROR: Duplicate entry for Intervention ' + intervention_name + '. Only first read value is being considered.')
            continue
        # Store starting time
        Interventions[intervention_name][START_STR] = start_time
    solution_file.close()
    print('Done')

################################
## Results processing ##########
################################

## Compute effective worload
def compute_resources(Instance: dict):
    """Compute effective workload (i.e. resources consumption values) for every resource and every time step"""

    # Retrieve usefull infos
    Interventions = Instance[INTERVENTIONS_STR]
    T_max = Instance[T_STR]
    Resources = Instance[RESOURCES_STR]
    # Init resource usage dictionnary for each resource and time
    resources_usage = {}
    for resource_name in Resources.keys():
        resources_usage[resource_name] = np.zeros(T_max)
    # Compute value for each resource and time step
    for intervention_name, intervention in Interventions.items():
        # start time should be defined (already checked in scheduled constraint checker)
        if not START_STR in intervention:
            continue
        start_time = intervention[START_STR]
        start_time_idx = start_time - 1 #index of list starts at 0
        intervention_worload = intervention[RESOURCE_CHARGE_STR]
        intervention_delta = int(intervention[DELTA_STR][start_time_idx])
        # compute effective worload
        for resource_name, intervention_resource_worload in intervention_worload.items():
            for time in range(start_time_idx, start_time_idx + intervention_delta):
                # null values are not available
                if str(time+1) in intervention_resource_worload and str(start_time) in intervention_resource_worload[str(time+1)]:
                    resources_usage[resource_name][time] += intervention_resource_worload[str(time+1)][str(start_time)]

    return resources_usage


## Retrieve effective risk distribution given starting times solution
def compute_risk_distribution(Interventions: dict, T_max: int, scenario_numbers):
    """Compute risk distributions for all time steps, given the interventions starting times"""

    # Init risk table
    risk = [scenario_numbers[t] * [0] for t in range(T_max)]
    # Compute for each intervention independently
    for intervention in Interventions.values():
        # Retrieve Intervention's usefull infos
        intervention_risk = intervention[RISK_STR]
        # start time should be defined (already checked in scheduled constraint checker)
        if not START_STR in intervention:
            continue
        start_time = intervention[START_STR]
        start_time_idx = int(start_time) - 1 # index for list getter
        delta = int(intervention[DELTA_STR][start_time_idx])
        for time in range(start_time_idx, start_time_idx + delta):
            for i, additional_risk in enumerate(intervention_risk[str(time + 1)][str(start_time)]):
                risk[time][i] += additional_risk

    return risk

## Compute mean for each period
def compute_mean_risk(risk, T_max: int, scenario_numbers):
    """Compute mean risk values over each time period"""

    # Init mean risk
    mean_risk = np.zeros(T_max)
    # compute mean
    for t in range(T_max):
        mean_risk[t] = sum(risk[t]) / scenario_numbers[t]

    return mean_risk

## Compute quantile for each period
def compute_quantile(risk, T_max: int, scenario_numbers, quantile):
    """Compute Quantile values over each time period"""

    # Init quantile
    q = np.zeros(T_max)
    for t in range(T_max):
        risk[t].sort()
        q[t] = risk[t][int(np.ceil(scenario_numbers[t] * quantile))-1]

    return q

## Compute total objective
def compute_objective(Instance: dict):
    """Compute total objectives (mean and expected excess)"""

    # Retrieve usefull infos
    T_max = Instance[T_STR]
    scenario_numbers = Instance[SCENARIO_NUMBER]
    Interventions = Instance[INTERVENTIONS_STR]
    quantile = Instance[QUANTILE_STR]
    # Retrieve risk final distribution
    risk = compute_risk_distribution(Interventions, T_max, scenario_numbers)
    # Compute mean risk
    mean_risk = compute_mean_risk(risk, T_max, scenario_numbers)
    # Compute quantile
    quantile = compute_quantile(risk, T_max, scenario_numbers, quantile)

    # Usefull infos
    alpha = Instance[ALPHA_STR]
    q = Instance[QUANTILE_STR]
    obj_1 = np.mean(mean_risk)
    tmp = np.zeros(len(quantile))
    obj_2 = np.mean(np.max(np.vstack((quantile - mean_risk, tmp)), axis=0))
    obj_tot = alpha * obj_1 + (1-alpha)*obj_2

    return obj_tot



##################################
## Constraints checkers ##########
##################################

## Launch each Constraint checks
def compute_penalty(Instance: dict):
    """Run all constraint checks"""

    total_penalty_cost = 0
    # Schedule constraints
    total_penalty_cost += COST_P_SCHEDULE * check_schedule(Instance)
    # Resources constraints
    total_penalty_cost += COST_P_RESOURCE * check_resources(Instance)
    # Exclusions constraints
    total_penalty_cost += COST_P_EXCLUDE * check_exclusions(Instance)

    return total_penalty_cost

## Schedule constraints: §4.1 in model description
def check_schedule(Instance: dict):
    """Check schedule constraints"""

    # Continuous interventions: §4.1.1
    #   This constraint is implicitly checked by the resource computation:
    #   computation is done under continuity hypothesis, and resource bounds will ensure the feasibility
    # Checks is done on each Intervention
    count = 0
    Interventions = Instance[INTERVENTIONS_STR]
    for intervention_name, intervention in Interventions.items():
        # Interventions are planned once: §4.1.2
        #   assert a starting time has been assigned to the intervention
        if not START_STR in intervention:
            print('ERROR: Schedule constraint 4.1.2: Intervention ' + intervention_name + ' has not been scheduled.')
            continue
        # Starting time validity: no explicit constraint
        start_time = intervention[START_STR]
        horizon_end = Instance[T_STR]
        if not (1 <= start_time <= horizon_end):
            count += 1
        # No work left: §4.1.3
        #   assert intervention is not ongoing after time limit or end of horizon
        time_limit = int(intervention[TMAX_STR])
        if time_limit < start_time:
            count += 1

    return count

## Resources constraints: §4.2 in model description
def check_resources(Instance: dict):
    """Check resources constraints"""

    count = 0
    T_max = Instance[T_STR]
    Resources = Instance[RESOURCES_STR]
    # Bounds are checked with a tolerance value
    tolerance = 1e-5
    # Compute resource usage
    resource_usage = compute_resources(Instance) # dict on resources and time
    # Compare bounds to usage
    for resource_name, resource in Resources.items():
        for time in range(T_max):
            # retrieve bounds values
            upper_bound = resource[MAX_STR][time]
            lower_bound = resource[MIN_STR][time]
            # Consumed value
            worload = resource_usage[resource_name][time]
            # Check max
            if worload > upper_bound + tolerance:
                count += 1
            # Check min
            if worload < lower_bound - tolerance:
                count += 1

    return count

## Exclusions constraints: §4.3 in model description
def check_exclusions(Instance: dict):
    """Check exclusions constraints"""

    count = 0
    # Retrieve Interventions and Exclusions
    Interventions = Instance[INTERVENTIONS_STR]
    Exclusions = Instance[EXCLUSIONS_STR]
    # Assert every exclusion holds
    for exclusion in Exclusions.values():
        # Retrieve exclusion infos
        [intervention_1_name, intervention_2_name, season] = exclusion
        # Retrieve concerned interventions...
        intervention_1 = Interventions[intervention_1_name]
        intervention_2 = Interventions[intervention_2_name]
        # start time should be defined (already checked in scheduled constraint checker)
        if (not START_STR in intervention_1) or (not START_STR in intervention_2):
            continue
        # ... their respective starting times...
        intervention_1_start_time = intervention_1[START_STR]
        intervention_2_start_time = intervention_2[START_STR]
        # ... and their respective deltas (duration)
        intervention_1_delta = int(intervention_1[DELTA_STR][intervention_1_start_time - 1]) # get index in list
        intervention_2_delta = int(intervention_2[DELTA_STR][intervention_2_start_time - 1]) # get index in list
        # Check overlaps for each time step of the season
        for time_str in Instance[SEASONS_STR][season]:
            time = int(time_str)
            if (intervention_1_start_time <= time < intervention_1_start_time + intervention_1_delta) and (intervention_2_start_time <= time < intervention_2_start_time + intervention_2_delta):
                count += 1

    return count


if __name__ == '__main__':

    for f in glob.glob("A_set/A_*.json"):

        filename = os.path.basename(f).split(".")[0]
        print(filename)

        # if os.path.exists(f"output\{filename}_lbs.txt"):
        #     continue

        solution = read_json(f)

        # read_solution_from_txt(solution, "./output/A_06_relaxed.txt")

        # s = initial_neighbourhood(solution)
        start = datetime.datetime.now()

        s = random_neighbourhood(solution)
        list_to_dict(s[0], solution)

        s = generate_neighbourhood(solution_list=s, solution=solution)

        local_beam_search(solution_list=s, solution=solution, optimal_value=optimal_value[filename])

        end = datetime.datetime.now()
        runtime = (end - start).total_seconds()

        try:
            with open(f"output\{filename}_lbs.txt", "w") as f:
                for itv in solution['Interventions'].keys():
                    f.write(" ".join([itv, str(solution[INTERVENTIONS_STR][itv][START_STR])]))
                    f.write("\n")

            with open(f"output\{filename}_lbs_stats.txt", "w") as f:
                f.write(f"Duration: {runtime} seconds, Objective Value: {compute_objective(solution)}")
                f.write("\n")
        except:
            print ("No solution!")
