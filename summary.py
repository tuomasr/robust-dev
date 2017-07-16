import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from common_data import scenarios, hours, units, unit_to_node, unit_to_generation_type, real_node_names
from helpers import get_start_node, get_end_node

# 0: coal
# 1: gas
# 2: ccgt
# 3: oil
# 4: biomass
# 5: oil shale
# 6: nuclear
# 7: hydro
# 8: wind
# 9: pv
# different chp types (e=extraction, b=back pressure):
# 10: coal-e
# 11: gas-b
# 12: gas-e
# 13: oil-b
# 14: oil-e
# 15: biomass-e
# 16: waste-e
# 17: peat-e
fuel_types = [
    "coal",
    "gas",
    "oil",
    "biomass",
    "oil shale",
    "nuclear",
    "hydro",
    "wind",
    "solar",
]

order = [6, 8, 7, 2, 5, 0, 3, 1, 4]

generation_type_to_fuel_type = {
    0: 0,
    1: 1,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    9: 8,
    10: 0,
    11: 1,
    12: 1,
    13: 2,
    14: 2,
    15: 3,
    16: 3,
    17: 3,
}

def mysort(X):
    return [X[o] for o in order]

basedir = "result_files2"

results = []

for idx, subdir in enumerate(os.walk(basedir)):
    if idx == 0:
        continue

    subdir = subdir[0]

    if "stoch" not in subdir:
        continue

    if "benders_dc_miqp_dc" not in subdir:
        continue

    input_file = subdir + "/" + "investment.pickle"
    split = subdir.split("_")

    last_idx = -1
    if "stoch" in subdir:
        last_idx = -2

    increase = split[last_idx].split("=")
    if len(increase) != 2:
        continue
    increase = increase[1]
    if float(increase) < 0.03:
        continue
    ub = split[last_idx-1].split("=")[1]

    with open(input_file, "rb") as file_:
        investments = pickle.load(file_)

    g_by_fuel = [0.0] * len(fuel_types)

    #import pdb; pdb.set_trace()

    for k, v in investments['x'].items():
        t, u = k
        if t != 9:
            continue
        t = unit_to_generation_type[u]
        f = generation_type_to_fuel_type[t]
        g_by_fuel[f] += v / 1000

    finv = []

    for k, v in investments['y'].items():
        t, l = k
        if t != 9:
            continue

        if v < 1.0:
            continue

        start = get_start_node(l)
        end = get_end_node(l)
        line_label = real_node_names[start] + "-" + real_node_names[end]


        finv.append(line_label)

    z = zip(fuel_types, g_by_fuel)
    inv = [str(k) + ": " + str(round(v, 2)) for k, v in z if v > 0]
    z = zip(fuel_types, g_by_fuel)
    inv2 = ["{:.2f}".format(v) for k, v in z if v > 0]
    results.append((int(ub), int(float(increase)*100), inv, inv2, finv))


results = sorted(results, key=lambda tup: (tup[0], tup[1]))
for l in results:
    print(l)

for l in results:
    if len(l[3]) < 3:
        ltx = "%d & %d & 1.0 & 1.0 & %s & %s \\\\" % (l[0], l[1], l[3][0], l[3][1])
    else:
        ltx = "%d & %d & 1.0 & 1.0 & %s & %s & %s \\\\" % (l[0], l[1], l[3][0], l[3][1], l[3][2])
    print(ltx)