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

input_files = [
    "result_files2/scenarios_15_benders_dc_miqp_dc_ub=4_increase=0.05/operation_decisions.pickle",
    "result_files2/scenarios_15_benders_dc_miqp_dc_ub=4_increase=0.05_stoch/operation_decisions.pickle",
]

titles = ["with emission constraint", "no emission constraint"]
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


fig, ax = plt.subplots(2, 2, figsize=(15, 12), gridspec_kw={'width_ratios': [1.55, 1]})

for idx, input_file in enumerate(input_files):
    with open(input_file, "rb") as file_:
        operation_decisions = pickle.load(file_)
        g = operation_decisions["g"]
        f = operation_decisions["f"]

        if idx == 0:
            g1 = operation_decisions["g"]

    for idx2, (offset, year) in enumerate([(0, 1), (len(hours) - 24, 10)]):
        g_by_fuel = [0.0] * len(fuel_types)
        if idx == 1:
            g1_by_fuel = [0.0] * len(fuel_types)

        for o in scenarios:
            for u in units:
                generation_type = unit_to_generation_type[u]
                fuel = generation_type_to_fuel_type[generation_type]

                for t in range(24):
                    if idx == 0:
                        g_by_fuel[fuel] += g[o, offset + t, u]
                    if idx == 1:
                        g_by_fuel[fuel] += g[o, offset + t, u] #- g1[o, offset + t, u]
                        g1_by_fuel[fuel] += g1[o, offset + t, u]

        if idx == 0:
            currrent_labels = [
                l + " %.1f" % (g_by_fuel[i] / sum(g_by_fuel) * 100.0) + "%"
                for i, l in enumerate(fuel_types)
            ]
        else:
            #import pdb; pdb.set_trace()

            share_g_by_fuel = [x / sum(g_by_fuel) for x in g_by_fuel]
            share_g1_by_fuel = [x / sum(g1_by_fuel) for x in g1_by_fuel]
            #import pdb; pdb.set_trace()
            g_by_fuel = [100 * (x - y) for x, y in zip(share_g_by_fuel, share_g1_by_fuel)]
            #g_by_fuel = [100 * (a - b) / b for a, b in zip(g_by_fuel, g1_by_fuel)]

        current_title = titles[idx] + ", t = %d" % year

        if idx == 0:
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
                    pos2 = (pos[0]-0.02, pos[1]-0.025)
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

        if idx == 1:
            ind = range(len(g_by_fuel))

            ax[idx2, idx].bar(
                ind,
                g_by_fuel,
                color=colors,
            )


            ax[idx2, idx].set_xticks(ind)

            myy = 0.24 if idx2 == 0 else 0.3


            ax[idx2, idx].spines['bottom'].set_position("zero")
            ax[idx2, idx].spines['right'].set_visible(False)
            ax[idx2, idx].spines['top'].set_visible(False)
            #ax[idx2, idx].yaxis.set_label_coords(-0.16,0.5)
            ax[idx2, idx].set_xticklabels(fuel_types, rotation=45, va="top", y=0)
            ax[idx2, idx].set_ylabel("change in generation share (%)")
            #ax[idx2, idx].tick_params(direction='out', pad=10)
            #ax[idx2, idx].set_xscale(0.5)

            #if idx2 == 0:

            #ax[idx2, idx].yaxis.set_label_coords(-0.16,0.5)

        #     for i,(g,t) in enumerate(zip(h, ax.get_xticklabels())):
        #         if g<0:
        #             t.set_ha('left')
        #             t.set_va('bottom')
        #         else:
        #             t.set_ha('right')
        #             t.set_va('top')
        #         t.set_rotation_mode('anchor')
        #         t.set_rotation(45)
        #         t.set_transform(ax.transData)
        #         t.set_position((i,0))

        ax[idx2, idx].set_title(current_title)

plt.savefig("generation_mixes_diff_stoch2.png", bbox_inches="tight")

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