import json
import glob
import os
from docplex.mp.model import Model

for f in glob.glob("A_set/A_06*.json"):
    filename = os.path.basename(f).split(".")[0]
    print (filename)

    if os.path.exists(f"output\{filename}.txt"):
        continue

    with open(f, "r") as content:
        data = json.load(content)

    model = Model("maintainance")

    # Decision Variables
    start = model.integer_var_dict([itv for itv in data['Interventions'].keys()],0, data['T'],name='start')
    delta = model.integer_var_dict([itv for itv in data['Interventions'].keys()],0, data['T'],name='delta')
    start_time = model.binary_var_dict([(itv,t) for itv in data['Interventions'].keys() for t in range(1, data['T']+1)],name='st')
    ongoing = model.binary_var_dict([(itv,t) for itv in data['Interventions'].keys() for t in range(1, data['T']+1)],name='ongoing')
    resource = model.binary_var_dict([(itv,c,t,st) for itv in data['Interventions'].keys() for c in data['Interventions'][itv]['workload'].keys() for t in data['Interventions'][itv]['workload'][c].keys() for st in data['Interventions'][itv]['workload'][c][t].keys()], name='resource' )

    # Objective Function
    risk_t = 0
    excess = 0
    risk_std = 0
    for t in range(1,data['T']+1):
        risk_s_t = [0 for _ in range(data['Scenarios_number'][t-1])]
        q = int(data['Quantile'] * data['Scenarios_number'][t-1])+1
        for sce in range(data['Scenarios_number'][t-1]):
            for itv in data['Interventions'].keys():
                if str(t) in data['Interventions'][itv]['risk'].keys():
                    for st in data['Interventions'][itv]['risk'][str(t)].keys():
                        risk_s_t[sce] += data['Interventions'][itv]['risk'][str(t)][st][sce] * ongoing[itv,t] * start_time[itv,int(st)]
        risk_mean = sum(risk_s_t) / data['Scenarios_number'][t-1]

        # for sce in range(data['Scenarios_number'][t-1]):
        #     risk_std += (risk_s_t[sce] - risk_mean)**2 / data['Scenarios_number'][t-1]
        risk_t += risk_mean




    # testing bubble sort, not working.
    #     for passnum in range(len(risk_s_t)-1, 0, -1):
    #         for i in range(passnum):
    #             if risk_s_t[i] - risk_s_t[i+1] >= 0:
    #                 risk_s_t[i+1], risk_s_t[i] = risk_s_t[i], risk_s_t[i+1]
    #     qn = risk_s_t[q] - risk_mean
    #     if qn >= 0:
    #         excess += qn

    obj_1 = risk_t / data['T']
    # obj_2 = risk_std / data['T']
    # # obj_2 = excess / data['T']
    # model.minimize(data['Alpha'] * obj_1 + (1 - data['Alpha']) * obj_2)
    model.minimize(obj_1 )

    # Constraints

    # Constraints between decision variables
    # Resource & start_time
    for itv in data['Interventions'].keys():
        for c in data['Interventions'][itv]['workload'].keys():
            for t in data['Interventions'][itv]['workload'][c].keys():
                for st in data['Interventions'][itv]['workload'][c][t].keys():
                    model.add(model.if_then(start_time[itv,int(st)] + ongoing[itv,int(t)] == 2, resource[itv,c,t,st] == 1))
                    model.add(model.if_then(start_time[itv,int(st)] + ongoing[itv,int(t)] != 2, resource[itv,c,t,st] == 0))
    for itv in data['Interventions'].keys():
        for t in range(data['T']):
            # start_time and start
            model.add((start[itv] == t+1) == start_time[itv,t+1])
            if t >= int(data['Interventions'][itv]['tmax']): # intevention can only be executed before tmax
                model.add(start_time[itv,t] == 0)
            # project must end before the period
            model.add(model.if_then(start_time[itv,t+1] == 1, start[itv] + data['Interventions'][itv]['Delta'][t] <= data['T']))
            model.add(model.logical_and(t+1 >= start[itv], t <= start[itv] + delta[itv] -2) == ongoing[itv,t+1])

    # All interventions have to be executed.
    for itv in data['Interventions'].keys():
        model.add(model.sum(start_time[itv,t] for t in range(1, data['T']+1)) == 1)
        model.add(delta[itv] == model.sum(data['Interventions'][itv]['Delta'][t] * start_time[itv,t+1] for t in range(data['T'])))

    # The needed resources cannot exceed the resources capacity but have to be at least equal to the minimum workload
    for t in range(1, data['T']+1):
        for c in data['Resources'].keys():
            resource_c_t = 0
            for itv in data['Interventions'].keys():
                if c in data['Interventions'][itv]['workload'].keys():
                    if str(t) in data['Interventions'][itv]['workload'][c].keys():
                        for st in data['Interventions'][itv]['workload'][c][str(t)].keys():
                            resource_c_t += resource[itv,c,str(t),st] * data['Interventions'][itv]['workload'][c][str(t)][st]
            model.add(resource_c_t <= data['Resources'][c]['max'][t-1])
            model.add(resource_c_t >= data['Resources'][c]['min'][t-1])

    for e, lst in data['Exclusions'].items():
        i_1 = lst[0]
        i_2 = lst[1]
        t = data['Seasons'][lst[2]]
        for period in t:
            model.add(model.if_then(ongoing[i_1,int(period)]== 1, ongoing[i_2,int(period)] == 0))
            model.add(model.if_then(ongoing[i_2,int(period)]== 1, ongoing[i_1,int(period)] == 0))


    model.context.update_cplex_parameters({'mip.strategy.variableselect':3,'mip.tolerances.mipgap': 0.01,'timelimit': 600})
    sol = model.solve(log_output=True)

    try:
        with open(f"output\{filename}.txt", "w") as f:
            for itv in data['Interventions'].keys():
                f.write(" ".join([itv, str(int(sol[start[itv]]))]))
                f.write("\n")
    except:
        print ("No solution!")