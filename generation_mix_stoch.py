import pickle

import matplotlib.pyplot as plt
import numpy as np

from common_data import scenarios, hours, units, unit_to_generation_type
plt.rcParams.update({'font.size': 18})

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

order = [6, 8, 7, 3, 5, 0, 1, 4, 2]

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


fig, ax = plt.subplots(2, 2, figsize=(14, 12))

with open("stoch.pickle", "rb") as file_:
    data = pickle.load(file_)

idx = 0

models = list(data.keys())
if "stoch" in models[0]:
    sorted_models = [models[1], models[0]]
else:
    sorted_models = models

for model in sorted_models:
    operation_decisions = data[model]
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
            #l + " %.1f" % (g_by_fuel[i] / 1000.0) + "GW"
            for i, l in enumerate(fuel_types)
        ]

        model_name = "SARO"
        if "stoch" in model:
            model_name = "SP"

        current_title = model_name + ", t = %d" % year

        my_pie, texts = ax[idx2, idx].pie(
            mysort(g_by_fuel),
            labels=mysort(currrent_labels),
            colors=mysort(colors),
            radius=0.7,
            # autopct="%.2f",
            #pctdistance=0.9,
            labeldistance=1.1,
            explode=mysort([0.15, 0.15, 0.15, 0.1, 0.15, 0.1, 0.1, 0.1, 0.1]),
        )

        #import pdb; pdb.set_trace()

        for txt in texts:
            t = txt.get_text()

            if "oil shale" in t:
                pos = txt.get_position()
                pos2 = (pos[0], pos[1]-0.025)
                txt.set_position(pos2)

            if "oil" in t and "oil shale" not in t:
                pos = txt.get_position()
                pos2 = (pos[0], pos[1]+0.05)
                txt.set_position(pos2)

            if "gas" in t or "coal" in t:
                pos = txt.get_position()
                pos2 = (pos[0], pos[1]-0.05)
                txt.set_position(pos2)

            if "wind" in t:
                pos = txt.get_position()
                pos2 = (pos[0], pos[1]-0.05)
                txt.set_position(pos2)

            if "biomass" in t:
                pos = txt.get_position()
                pos2 = (pos[0], pos[1]-0.1)
                txt.set_position(pos2)

        ax[idx2, idx].set_title(current_title)

    idx = idx + 1

plt.savefig("generation_mixes_stoch.png", bbox_inches="tight")

# fuel_types = [
#     "coal",       0
#     "gas",        1
#     "oil",        2
#     "biomass",    3
#     "oil shale",  4
#     "nuclear",    5
#     "hydro",      6
#     "wind",       7
#     "solar",      8
# ]