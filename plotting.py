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
    hatches=None,
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
        hatch = hatches[i] if hatches is not None else ""
        bar = plt.bar(ind, row_data, bottom=cum_size, color=color, hatch=hatch, width=0.9, edgecolor='black')
        axes.append(bar)
        cum_size += row_data

        if legend_labels:
            label = legend_labels[i]
            legend_data[color] = (bar, label)

    if category_labels:
        plt.xticks(ind, category_labels)

    if y_label:
        plt.ylabel(y_label)

    plt.xlabel("master problem time step $t$")

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
                    weight="bold",
                )

    if legend_data:
        handles, labels = zip(*legend_data.values())
        plt.legend(handles, labels, loc="upper left", prop={"size": 20})


def create_investment_plots(
    xhat, yhat, master_problem_algorithm, subproblem_algorithm, output_dir
):
    """Create plots for transmission and generation investments."""
    plt.rcParams.update({'font.size': 20})
    # Pickling is for debugging... ignore.
    algo_choice = (master_problem_algorithm, subproblem_algorithm)
    with open("xhat_%s_%s.pickle" % algo_choice, "wb") as f:
        pickle.dump(xhat, f)

    with open("yhat_%s_%s.pickle" % algo_choice, "wb") as f:
        pickle.dump(yhat, f)

    # Collect data for generation investments.
    series_labels = []
    nodes = []
    data = []
    category_labels = years
    investment_years = []
    colors = []
    hatches = []
    legend_labels = []

    palette = {
        k: v for k, v in zip(candidate_unit_types, ["r", "lightgrey", "m", "lightgreen", "lightblue", "y"])
    }
    hatch_palette = {
        k: v for k, v in zip(candidate_unit_types, ["o", "", "o", "", "", "o"])
    }
    candidate_unit_name_map = {
        k: v for k, v in zip(candidate_unit_types, candidate_unit_type_names)
    }

    for idx, (key, val) in enumerate(xhat.items()):
        # Only show investments >= 100 MW.
        trunc_val = int(np.round(val / 100.0) * 100.0)

        if trunc_val > 0.0:
            year, unit = key

            node = unit_to_node[unit]
            unit_type = unit_to_generation_type[unit]
            unit_label = real_node_names[node]  # + "\n" + str(trunc_val) + " MW"
            nodes.append(node)

            series_labels.append(unit_label)

            capacity = np.zeros(len(years))
            capacity[year:] += val

            data.append(capacity)
            investment_years.append(
                year * 1000 + idx
            )  # For reordering the data points by investment year.
            colors.append(palette[unit_type])
            hatches.append(hatch_palette[unit_type])
            legend_labels.append(candidate_unit_name_map[unit_type])

    # Reorder data so that they appear in the order of the investment year.
    series_labels = [
        x
        for _, _, x in sorted(
            zip(investment_years, nodes, series_labels), key=lambda pair: (pair[0], pair[1])
        )
    ]
    data = [x for _, _, x in sorted(zip(investment_years, nodes, data), key=lambda pair: (pair[0], pair[1]))]
    colors = [
        x for _, _, x in sorted(zip(investment_years, nodes, colors), key=lambda pair: (pair[0], pair[1]))
    ]
    hatches = [
        x for _, _, x in sorted(zip(investment_years, nodes, hatches), key=lambda pair: (pair[0], pair[1]))
    ]
    legend_labels = [
        x
        for _, _, x in sorted(
            zip(investment_years, nodes, legend_labels), key=lambda pair: (pair[0], pair[1])
        )
    ]

    if len(data) > 0:
        width, height = max(1.5 * len(years), 12), max(2.5 * len(data), 25)
        plt.figure(figsize=(width, height))
        stacked_bar(
            data,
            series_labels,
            category_labels,
            y_label="Cumulative new generation capacity (MW)",
            colors=colors,
            hatches=hatches,
            legend_labels=legend_labels,
        )
        filename = "generation_investment_%s_%s.png" % (
            master_problem_algorithm,
            subproblem_algorithm,
        )
        plt.savefig(os.path.join(output_dir, filename), bbox_inches="tight")
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
    plt.rcParams.update({'font.size': 14})
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
    ax1.set_xlabel("master problem time step $t$")
    ax1.set_ylabel("$\mathrm{CO_2}$ emissions (ktonne)", color=color)
    line1, = ax1.plot(years, emissions / 1000.0, "b-", label="emissions")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = "r"
    ax2.set_ylabel(
        "$\mathrm{CO_2}$ emission price (€/tonne)", color=color
    )  # we already handled the x-label with ax1
    line2, = ax2.plot(years, emission_prices, "r--", label="$\mathrm{CO_2}$ emission price")
    ax2.tick_params(axis="y", labelcolor=color)

    fig.legend(
        (line1, line2),
        ("$\mathrm{CO_2}$ emissions", "$\mathrm{CO_2}$ emission price"),
        loc="center",
        bbox_to_anchor=(0.5, 0.075),
        ncol=2,
    )

    fig.tight_layout(
        rect=[0, 0.15, 1, 1]
    )  # otherwise the right y-label is slightly clipped

    filename = "emissions_%s_%s.png" % (master_problem_algorithm, subproblem_algorithm)
    plt.savefig(os.path.join(output_dir, filename), bbox_inches="tight")


def test():
    """For debugging the plotting functions."""
    basedir = "result_files2/scenarios_15_benders_dc_miqp_dc_ub=4_increase=0.05_no_emissions/"
    with open(basedir + "investment.pickle", "rb") as f:
        inv = pickle.load(f)

    create_investment_plots(inv["xhat"], inv["yhat"], "benders_dc", "miqp_dc", ".")


def test2():
    """For debugging the plotting functions."""
    basedir = "result_files2/scenarios_15_benders_dc_miqp_dc_ub=4_increase=0.05_no_emissions/"

    with open(basedir+"emission_data.pickle", "rb") as f:
        emission_data = pickle.load(f)

    emissions = emission_data["emissions"]
    emission_prices = emission_data["emission_prices"]

    create_emission_plots(
        emissions, emission_prices, "benders_dc", "miqp_dc", "."
    )


def both():
    test()
    #test2()

both()
