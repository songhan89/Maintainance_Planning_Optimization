'''
For A_04 and A_01, it's hard to find a feasible solutions, so unmute line 190 to see the whole searching process
For other small test cases, mute line 190 and 199 cuz it's really annoying to see solutions jumping out.
'''
import random
import json
import numpy as np
import time
from checker import check_resources
from checker import check_exclusions
from checker import compute_objective

with open("A_set//A_06.json", "r") as content:
    Instance = json.load(content)
minute_to_run = 1 # feel free to let it run for longer time

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
                for itv in Instance['Interventions'].keys(): # storing the best start_time
                    best_start_time[itv] = Instance['Interventions'][itv]['start']
                best_penalty_set = pnt_set # storing the penalty set under the best start_time
                lowest_penalty = p1+p2 # storing the best penalty by far

        if lowest_penalty < current_penalty: # change intervention_set if there's new lowest penalty
            intervention_set = best_penalty_set
            # turn this line on if you wanna see what happen in the searching.
            # print('after',i,'interations, the best penalty is', lowest_penalty, ', and number of interventions violated is',len(best_penalty_set))
        # recover the instance to the original start
        for itv in Instance['Interventions'].keys(): # set the Instance back to the be aligned with the current best
            Instance['Interventions'][itv]['start'] = best_start_time[itv]
        i += 1
        if i == 200: # fail in compute solution after 200 interations, reset everything and start over
            lowest_penalty = len(Instance['Interventions'].keys())
            intervention_set = Instance['Interventions'].keys()
            best_start_time = dict.fromkeys(Instance['Interventions'].keys(),0)
            i = 0
    return Instance

if __name__ == '__main__':
    start_cal = time.time()
    best_obj = float('inf')
    best_start_time = dict.fromkeys(Instance['Interventions'].keys(),0)
    while (time.time() - start_cal) / 60 < minute_to_run:
        Instance = search_for_solution(Instance)
        obj_tot = compute_objective(Instance)
        if obj_tot < best_obj:
            best_obj = obj_tot
            for intervention in Instance['Interventions'].keys():
                best_start_time[intervention] = Instance['Interventions'][intervention]['start']
    if best_obj == float('inf'): # suggests no solutions are found, return the infeasible solution
        print('No feasible solution found in this search, constraints are violated.')
        for intervention in Instance['Interventions'].keys():
            best_start_time[intervention] = Instance['Interventions'][intervention]['start']
            best_obj = compute_objective(Instance)

    print('The best objective is',best_obj)
    with open("output.txt", "w") as f: 
        for itv in Instance['Interventions'].keys():
            f.write(" ".join([itv, str(best_start_time[itv])]))
            f.write("\n")
