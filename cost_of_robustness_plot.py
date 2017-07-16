# Run an experiment in investment decisions are fixed but demand is varied.
# This is to quantify the quality of the investment decisions.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

from common_data import scenarios


plt.switch_backend("agg")  # Enable plotting without a monitor.
plt.rcParams.update({'font.size': 14})


def plot():
    models = ["SP", "SARO"]
    emission_costs = [100, 1000, 10000]
    markers = ["o", "x", "^", "v", "s", "P"]
    assert len(markers) >= len(models) * len(emission_costs)

    with open("cost_of_robustness.npy", "rb") as f:
        y = np.load(f)
        #import pdb; pdb.set_trace()
        y = np.nanmean(y, axis=3) / len(scenarios) / 1e6
        y = np.round(y, 2)



    num_costs = len(emission_costs)

    num_models = len(models)
    num_samples = 1

    start_percentage = -5.0
    end_percentage = 15.0
    percentage_increase = 1.0
    num_levels = int(np.round((end_percentage - start_percentage) / percentage_increase)) + 1

    x = np.arange(start_percentage, end_percentage+0.5, percentage_increase) / 100.

    #import pdb; pdb.set_trace()

    # Estimate the cost of different investment decisions.
    for e, emission_cost in enumerate(emission_costs):
        for i, model in enumerate(models):
            plt.plot(x[:-5], y[e, i, :-5], marker=markers[e*num_models+i], label=model + ", %s €/tonne" % int(emission_cost))

    plt.xlabel("Load change (L)")
    plt.ylabel("Expected total cost (million €)")
    plt.legend(loc="upper left")
    plt.savefig("cost_of_robustness_emission_costs.png", bbox_inches="tight")


if __name__ == "__main__":
    plot()
