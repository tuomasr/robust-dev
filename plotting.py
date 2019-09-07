# Plotting functions for the results.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from common_data import (
    real_node_names,
    years,
    unit_to_node,
    candidate_line_capacity,
    unit_to_generation_type,
    candidate_unit_type_names,
    candidate_unit_types,
)
from helpers import get_start_node, get_end_node

plt.switch_backend("agg")  # Enable plotting without a monitor.


def stacked_bar(
    data,
    series_labels,
    category_labels,
    y_label,
    colors=None,
    legend_labels=None,
    grid=True,
):
    """Plots a stacked bar chart with the data and labels provided.

    Modified from: https://stackoverflow.com/a/50205834
    """
    ny = len(data[0])
    ind = list(range(ny))

    axes = []
    cum_size = np.zeros(ny)

    data = np.array(data)

    legend_data = dict()

    for i, row_data in enumerate(data):
        color = colors[i] if colors is not None else "b"
        bar = plt.bar(ind, row_data, bottom=cum_size, color=color, width=0.9)
        axes.append(bar)
        cum_size += row_data

        if legend_labels:
            label = legend_labels[i]
            legend_data[color] = (bar, label)

    if category_labels:
        plt.xticks(ind, category_labels)

    if y_label:
        plt.ylabel(y_label)

    plt.xlabel("master problem time step")

    if grid:
        plt.grid()

    for i, axis in enumerate(axes):
        for bar in axis:
            w, h = bar.get_width(), bar.get_height()

            if h > 0:
                plt.text(
                    bar.get_x() + w / 2,
                    bar.get_y() + h / 2,
                    series_labels[i],
                    ha="center",
                    va="center",
                )

    if legend_data:
        handles, labels = zip(*legend_data.values())
        plt.legend(handles, labels, loc="upper left")


def create_investment_plots(
    xhat, yhat, master_problem_algorithm, subproblem_algorithm, output_dir
):
    """Create plots for transmission and generation investments."""
    # Pickling is for debugging... ignore.
    algo_choice = (master_problem_algorithm, subproblem_algorithm)
    with open("xhat_%s_%s.pickle" % algo_choice, "wb") as f:
        pickle.dump(xhat, f)

    with open("yhat_%s_%s.pickle" % algo_choice, "wb") as f:
        pickle.dump(yhat, f)

    # Collect data for generation investments.
    series_labels = []
    data = []
    category_labels = years
    investment_years = []
    colors = []
    legend_labels = []

    palette = {
        k: v for k, v in zip(candidate_unit_types, ["r", "c", "m", "g", "b", "y"])
    }
    candidate_unit_name_map = {
        k: v for k, v in zip(candidate_unit_types, candidate_unit_type_names)
    }

    for idx, (key, val) in enumerate(xhat.items()):
        # Only show investments >= 10 MW.
        trunc_val = int(np.round(val / 10.0) * 10.0)

        if trunc_val > 0.0:
            year, unit = key

            node = unit_to_node[unit]
            unit_type = unit_to_generation_type[unit]
            unit_label = real_node_names[node]  # + "\n" + str(trunc_val) + " MW"

            series_labels.append(unit_label)

            capacity = np.zeros(len(years))
            capacity[year:] += val

            data.append(capacity)
            investment_years.append(
                year * 1000 + idx
            )  # For reordering the data points by investment year.
            colors.append(palette[unit_type])
            legend_labels.append(candidate_unit_name_map[unit_type])

    # Reorder data so that they appear in the order of the investment year.
    series_labels = [
        x
        for _, x in sorted(
            zip(investment_years, series_labels), key=lambda pair: pair[0]
        )
    ]
    data = [x for _, x in sorted(zip(investment_years, data), key=lambda pair: pair[0])]
    colors = [
        x for _, x in sorted(zip(investment_years, colors), key=lambda pair: pair[0])
    ]
    legend_labels = [
        x
        for _, x in sorted(
            zip(investment_years, legend_labels), key=lambda pair: pair[0]
        )
    ]

    if len(data) > 0:
        width, height = max(1.5 * len(years), 12), max(1.6 * len(data), 12)
        plt.figure(figsize=(width, height))
        stacked_bar(
            data,
            series_labels,
            category_labels,
            y_label="Cumulative new generation capacity (MW)",
            colors=colors,
            legend_labels=legend_labels,
        )
        filename = "generation_investment_%s_%s.png" % (
            master_problem_algorithm,
            subproblem_algorithm,
        )
        plt.savefig(os.path.join(output_dir, filename))
    else:
        print("No generation investments. Nothing to plot.")

    # Collect data for transmission investments.
    series_labels = []
    data = []
    category_labels = years
    investment_years = []

    for idx, (key, val) in enumerate(yhat.items()):
        if val > 0.0:
            year, line = key
            start = get_start_node(line)
            end = get_end_node(line)
            line_label = real_node_names[start] + "-" + real_node_names[end]

            series_labels.append(line_label)

            capacity = np.zeros(len(years))
            capacity[year:] += candidate_line_capacity

            data.append(capacity)
            investment_years.append(
                year * 1000 + idx
            )  # For reordering the data points by investment year.

    # Reorder data so that they appear in the order of the investment year.
    series_labels = [
        x
        for _, x in sorted(
            zip(investment_years, series_labels), key=lambda pair: pair[0]
        )
    ]
    data = [x for _, x in sorted(zip(investment_years, data), key=lambda pair: pair[0])]
    colors = [
        x for _, x in sorted(zip(investment_years, colors), key=lambda pair: pair[0])
    ]

    if len(data) > 0:
        width, height = max(1.1 * len(years), 12), max(1.1 * len(data), 10)
        plt.figure(figsize=(width, height))
        stacked_bar(
            data,
            series_labels,
            category_labels,
            y_label="Cumulative new transmission line capacity (MW)",
        )
        filename = "transmission_investment_%s_%s.png" % (
            master_problem_algorithm,
            subproblem_algorithm,
        )
        plt.savefig(os.path.join(output_dir, filename))
    else:
        print("No transmission investments. Nothing to plot.")


def create_emission_plots(
    emissions,
    emission_prices,
    master_problem_algorithm,
    subproblem_algorithm,
    output_dir,
):
    """Create plots for emission prices and for emission reductions."""
    # plt.figure()
    # plt.plot(years, emissions)
    # plt.xlabel("master problem time step")
    # plt.ylabel("emissions (tonne)")
    # plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    # plt.savefig(
    #     "emissions_trajectory_%s_%s.png"
    #     % (master_problem_algorithm, subproblem_algorithm)
    # )

    # # Emission prices plot.
    # plt.figure()
    # plt.plot(years, emission_prices)
    # plt.xlabel("master problem time step")
    # plt.ylabel("emissions price (EUR/tonne)")
    # plt.savefig(
    #     "emissions_prices_trajectory_%s_%s.png"
    #     % (master_problem_algorithm, subproblem_algorithm)
    # )

    fig, ax1 = plt.subplots()

    color = "b"
    ax1.set_xlabel("master problem time step")
    ax1.set_ylabel("emissions (tonne)", color=color)
    line1, = ax1.plot(years, emissions, "b-", label="emissions (LHS)")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = "r"
    ax2.set_ylabel(
        "emission price (EUR/tonne)", color=color
    )  # we already handled the x-label with ax1
    line2, = ax2.plot(years, emission_prices, "r--", label="emission price (RHS)")
    ax2.tick_params(axis="y", labelcolor=color)

    fig.legend(
        (line1, line2),
        ("emission (LHS)", "emission price (RHS)"),
        loc="center",
        bbox_to_anchor=(0.5, 0.075),
    )

    fig.tight_layout(
        rect=[0, 0.15, 1, 1]
    )  # otherwise the right y-label is slightly clipped

    filename = "emissions_%s_%s.png" % (master_problem_algorithm, subproblem_algorithm)
    plt.savefig(os.path.join(output_dir, filename))


def test():
    """For debugging the plotting functions."""
    algo_choice = "milp_dc", "miqp_dc"

    with open("xhat_%s_%s.pickle" % algo_choice, "rb") as f:
        xhat = pickle.load(f)

    with open("yhat_%s_%s.pickle" % algo_choice, "rb") as f:
        yhat = pickle.load(f)

    create_investment_plots(xhat, yhat, algo_choice[0], algo_choice[1])


def test2():
    """For debugging the plotting functions."""
    master_problem_algorithm, subproblem_algorithm = "milp_dc", "miqp_dc"

    with open("emission_data.pickle", "rb") as f:
        emission_data = pickle.load(f)

    emissions = emission_data["emissions"]
    emission_prices = emission_data["emission_prices"]

    create_emission_plots(
        emissions, emission_prices, master_problem_algorithm, subproblem_algorithm
    )


# test2()
