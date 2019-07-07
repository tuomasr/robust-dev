import pickle

import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend("agg")  # Enable plotting without a monitor.

from common_data import scenarios, real_node_names, years, unit_to_node, incidence, F_max, \
    candidate_line_capacity
from helpers import get_start_node, get_end_node


def stacked_bar(data, series_labels, category_labels, y_label, grid=True):
    """Plots a stacked bar chart with the data and labels provided.

    Modified from: https://stackoverflow.com/a/50205834
    """
    ny = len(data[0])
    ind = list(range(ny))

    axes = []
    cum_size = np.zeros(ny)

    data = np.array(data)

    for i, row_data in enumerate(data):
        axes.append(plt.bar(ind, row_data, bottom=cum_size, width=0.9))
        cum_size += row_data

    if category_labels:
        plt.xticks(ind, category_labels)

    if y_label:
        plt.ylabel(y_label)

    if grid:
        plt.grid()

    for i, axis in enumerate(axes):
        for bar in axis:
            w, h = bar.get_width(), bar.get_height()

            if h > 0:
                plt.text(bar.get_x() + w/2, bar.get_y() + h/2,
                         series_labels[i], ha="center", va="center")

def create_investment_plots(xhat, yhat, master_problem_algorithm, subproblem_algorithm):
    """Create plots for transmission and generation investments."""
    # Pickling is for debugging... ignore.
    with open("xhat.pickle", "w") as f:
        pickle.dump(xhat, f)

    with open("yhat.pickle", "w") as f:
        pickle.dump(yhat, f)

    # Collect data for generation investments.
    series_labels = []
    data = []
    category_labels = years
    investment_years = []

    for key, val in xhat.items():
        # Only show investments >= 10 MW.
        trunc_val = int(np.round(val / 10.0) * 10.0)

        if trunc_val > 0.0:
            year, unit = key
            node = unit_to_node[unit]
            unit_label = real_node_names[node] + "\n" + str(trunc_val) + " MW"

            series_labels.append(unit_label)

            capacity = np.zeros(len(years))
            capacity[year:] += val

            data.append(capacity)
            investment_years.append(year)   # For reordering the data points by investment year.

    # Reorder data so that they appear in the order of the investment year.
    series_labels = [x for _, x in
                     sorted(zip(investment_years, series_labels), key=lambda pair: pair[0])]
    data = [x for _, x in sorted(zip(investment_years, data),
            key=lambda pair: pair[0])]

    if len(data) > 0:
        width, height = max(1.5*len(years), 12), max(1.6*len(data), 12)
        plt.figure(figsize=(width, height))
        stacked_bar(
            data,
            series_labels,
            category_labels,
            y_label="Cumulative new wind power capacity (MW)"
        )
        plt.savefig(
            "generation_investment_%s_%s.png"
            % (master_problem_algorithm, subproblem_algorithm)
        )
    else:
        print("No generation investments. Nothing to plot.")

    # Collect data for transmission investments.
    series_labels = []
    data = []
    category_labels = years
    investment_years = []

    for key, val in yhat.items():
        if val > 0.0:
            year, line = key
            start = get_start_node(line)
            end = get_end_node(line)
            line_label = real_node_names[start] + "-" + real_node_names[end]

            series_labels.append(line_label)

            capacity = np.zeros(len(years))
            capacity[year:] += candidate_line_capacity

            data.append(capacity)
            investment_years.append(year)   # For reordering the data points by investment year.

    # Reorder data so that they appear in the order of the investment year.
    series_labels = [x for _, x in
                     sorted(zip(investment_years, series_labels), key=lambda pair: pair[0])]
    data = [x for _, x in sorted(zip(investment_years, data),
            key=lambda pair: pair[0])]

    if len(data) > 0:
        width, height = max(1.1*len(years), 12), max(1.1*len(data), 10)
        plt.figure(figsize=(width, height))
        stacked_bar(
            data,
            series_labels,
            category_labels,
            y_label="Cumulative new transmission line capacity (MW)"
        )
        plt.savefig(
            "transmission_investment_%s_%s.png"
            % (master_problem_algorithm, subproblem_algorithm)
        )
    else:
        print("No transmission investments. Nothing to plot.")


def create_emission_plots(emissions, emission_prices,
                          master_problem_algorithm, subproblem_algorithm):
    """Create plots for emission prices and for emission reductions."""
    plt.figure()

    markers = ["ro--", "bs--", "kx--", "yd--", "c*--", "m^--"]

    for i, o in enumerate(scenarios):
        plt.plot(years, emissions[o, :], markers[i], label="Oper. cond. %d" % o)

    plt.xlabel("master problem time step")
    plt.ylabel("emissions (tonne)")
    lgd = plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.savefig(
        "emissions_trajectory_%s_%s.png"
        % (master_problem_algorithm, subproblem_algorithm),
        bbox_extra_artists=(lgd,),
        bbox_inches="tight",
    )

    # Emission prices plot.
    plt.figure()
    for i, o in enumerate(scenarios):
        price_list = [emission_prices[o, y] for y in years]
        plt.plot(years, price_list, markers[i], label="Oper. cond. %d" % o)

    plt.xlabel("master problem time step")
    plt.ylabel("emissions price (EUR/tonne)")
    lgd = plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.savefig(
        "emissions_prices_trajectory_%s_%s.png"
        % (master_problem_algorithm, subproblem_algorithm),
        bbox_extra_artists=(lgd,),
        bbox_inches="tight",
    )



def test():
    """For debugging the plotting functions."""
    with open("xhat.pickle", "r") as f:
        xhat = pickle.load(f)

    with open("yhat.pickle", "r") as f:
        yhat = pickle.load(f)

    create_investment_plots(xhat, yhat, "test", "test")


# test()
