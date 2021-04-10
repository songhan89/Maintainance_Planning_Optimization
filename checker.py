optimal_value = {
    "A01": 1767.8156110,
    "A02": 4671.3766110,
    "A03": 848.1786111,
    "A04": 2085.8760540,
    "A05": 635.2217857,
    "A06": 590.6235989,
    "A07": 2272.7822740,
    "A08": 744.2932352,
    "A09": 1507.2847840,
    "A10": 2994.8487350,
    "A11": 495.2557702,
    "A12": 789.6349276,
    "A13": 1998.6621620,
    "A14": 2264.1243210,
    "A15": 2268.5691500
}

import math

def checker(
    instance: str,
    solution: dict,
):
    global optimal_value

    SCHEDULED = True
    if SCHEDULED:
        total_obj, conflict = 0, 0
        # Absolute Percetange Error
        gap = math.fabs(optimal_value[instance] - total_obj) / optimal_value[instance] 
        return(total_obj, conflict, gap)
    else:
        raise ValueError("Not all interventions has been scheduled")