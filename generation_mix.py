import pickle

import matplotlib.pyplot as plt
import numpy as np

from common_data import scenarios, hours, units, unit_to_generation_type


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

input_files = [
    "result_files/milp_miqp_5_scenarios_no_emission_constr/operation_decisions.pickle",
    "result_files/milp_miqp_5_scenarios/operation_decisions.pickle",
]

titles = ["no emission constraint", "with emission constraint"]
# light gray, red, black, green, dark gray, yellow, dark blue, light blue, orange
colors = [
    "#a6a6a6",
    "#ff0000",
    "#000000",
    "#00ff08",
    "#636363",
    "#fffb00",
    "#00058a",
    "#99dbff",
    "#ffa200",
]


def mysort(X):
    return [X[o] for o in order]


fig, ax = plt.subplots(2, 2, figsize=(12, 9))

for idx, input_file in enumerate(input_files):
    with open(input_file, "rb") as file_:
        operation_decisions = pickle.load(file_)
        g = operation_decisions["g"]
        f = operation_decisions["f"]

    for idx2, (offset, year) in enumerate([(0, 1), (len(hours) - 24, 10)]):
        g_by_fuel = [0.0] * len(fuel_types)

        for o in scenarios:
            for u in units:
                generation_type = unit_to_generation_type[u]
                fuel = generation_type_to_fuel_type[generation_type]

                for t in range(24):
                    g_by_fuel[fuel] += g[o, offset + t, u]

        currrent_labels = [
            l + " %.1f" % (g_by_fuel[i] / sum(g_by_fuel) * 100.0) + "%"
            for i, l in enumerate(fuel_types)
        ]

        current_title = titles[idx] + ", t = %d" % year

        ax[idx2, idx].pie(
            mysort(g_by_fuel),
            labels=mysort(currrent_labels),
            colors=mysort(colors),
            # autopct="%.2f",
        )
        ax[idx2, idx].set_title(current_title)

plt.savefig("generation_mixes.png")
