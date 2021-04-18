import numpy as np

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
        if record_itv:
            for time_str in Instance['Seasons'][season]:
                time = int(time_str)
                if (intervention_1_start_time <= time < intervention_1_start_time + intervention_1_delta) and (intervention_2_start_time <= time < intervention_2_start_time + intervention_2_delta):
                    penalty += 1
                    pnt_set.add(intervention_1_name)
                    pnt_set.add(intervention_2_name)
            return penalty, pnt_set
        else:
            for time_str in Instance['Seasons'][season]:
                time = int(time_str)
                if (intervention_1_start_time <= time < intervention_1_start_time + intervention_1_delta) and (intervention_2_start_time <= time < intervention_2_start_time + intervention_2_delta):
                    penalty += 1
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