# Helper functions.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time

import numpy as np

from common_data import (
    scenarios,
    weights,
    years,
    num_hours_per_year,
    units,
    candidate_units,
    candidate_lines,
    incidence,
    G_max,
    availability_rates,
    ramp_rates,
    wind_unit_idx,
    pv_unit_idx,
    unit_to_generation_type,
    G_emissions,
    maximum_candidate_unit_capacity_by_type,
)


def compute_objective_gap(lb, ub):
    # Compute the relative gap between upper and lower bounds of the algorithm.
    return (ub - lb) / float(ub)


def to_year(h):
    # Map hour to the year it belongs to. Assume each year has the same amount of hours.
    return int(h / num_hours_per_year)


def to_hours(y):
    # Return the hours belonging to a particular year.
    return range(y * num_hours_per_year, (y + 1) * num_hours_per_year)


def get_installed_capacity(o, t, u, x):
    # Get installed generation capacity at any given time.
    if u in candidate_units:
        # Compute the total capacity of the candidate unit during this year.
        year = to_year(t)
        capacity = x[year, u]
    else:
        capacity = G_max[o, t, u]

    return np.maximum(capacity, 0.0)


def get_effective_capacity(o, t, u, x):
    # Get effective available generation capacity at any given time.
    installed_capacity = get_installed_capacity(o, t, u, x)
    # Multiply installed capacity with availability rate.
    # For wind and solar, availability rate is weather-dependent.
    # Otherwise it takes into account outages.
    available = installed_capacity * availability_rates[o, t, u]

    return np.maximum(available, 0.0)


def get_maximum_ramp(o, t, u, x):
    # Get maximum ramp at any given time.
    # Wind and PV can ramp up freely within the installed capacity limits.
    if unit_to_generation_type[u] in (wind_unit_idx, pv_unit_idx):
        ramp = get_installed_capacity(o, t, u, x)
    else:
        # Otherwise, take a fraction of the effective capacity.
        ramp = get_effective_capacity(o, t, u, x) * ramp_rates[o, t, u]

    return np.maximum(ramp, 0.0)


def unit_built(x, h, u):
    # Check if a generation unit is built at hour h.
    year = to_year(h)

    return x[year, u] > 0.0 if u in candidate_units else 1


def line_built(y, h, l):
    # Check if a transmission line is built at hour h.
    year = to_year(h)

    return y[year, l] if l in candidate_lines else 1


def get_start_node(l):
    # Get the node at which a transmission line starts.
    row = incidence[l]
    starts = np.where(row == -1)[0]
    assert len(starts) == 1
    return starts[0]


def get_end_node(l):
    # Get the node to which a transmission line ends.
    row = incidence[l]
    ends = np.where(row == 1)[0]
    assert len(ends) == 1
    return ends[0]


def get_lines_starting(n):
    # Get all transmission lines starting at node n.
    col = incidence[:, n]
    return np.where(col == -1)[0]


def get_lines_ending(n):
    # Get all transmission lines ending at node n.
    col = incidence[:, n]
    return np.where(col == 1)[0]


def get_ramp_hours():
    # Return the hours for which ramp constraints are defined. Exclude the last hour in each year
    # because the constraint is defined as lb <= g_{t+1} - g_t <= ub. By excluding the last hour in
    # each year we also avoid linking subsequent years.
    ramp_hours_in_year = range(num_hours_per_year - 1)

    ramp_hours = []

    for year in years:
        offset = year * num_hours_per_year

        for i in ramp_hours_in_year:
            hour = offset + i
            ramp_hours.append(hour)

    return ramp_hours


def is_year_first_hour(h):
    # Check if the given hour is the first hour of the year.
    remainder = h % num_hours_per_year

    return remainder == 0


def is_year_last_hour(h):
    # Check if the given hour is the last hour of the year.
    remainder = (h + 1) % num_hours_per_year

    return remainder == 0


def concatenate_to_uncertain_variables_array(current_d, new_d):
    # Add a new column to the uncertain variables array.
    new_column = np.expand_dims(new_d, axis=-1)

    current_d = np.concatenate((current_d, new_column), axis=-1)

    return current_d


def read_investment_and_availability_decisions(
    x, xhat, y, yhat, initial, many_solutions
):
    # Read current investments to generation and transmission and whether the units and lines are
    # operational at some time point.
    # At the first CC iteration, return full investment.
    current_xhat = dict()
    current_yhat = dict()

    current_x = dict()
    current_y = dict()

    initial_transmission_investment = 1.0

    for t in years:
        for u in candidate_units:
            unit_type = unit_to_generation_type[u]
            initial_generation_investment = maximum_candidate_unit_capacity_by_type[
                unit_type
            ]

            if initial:
                if t == 0:
                    current_xhat[t, u] = initial_generation_investment
                else:
                    current_xhat[t, u] = 0.0

                current_x[t, u] = initial_generation_investment
            else:
                current_xhat[t, u] = xhat[t, u].x
                current_x[t, u] = x[t, u].x

        for l in candidate_lines:
            if initial:
                if t == 0:
                    current_yhat[t, l] = initial_transmission_investment
                else:
                    current_yhat[t, l] = 0.0

                current_y[t, l] = initial_transmission_investment
            else:
                if many_solutions:
                    current_yhat[t, l] = float(int(yhat[t, l].Xn))
                    current_y[t, l] = float(int(y[t, l].Xn))
                else:
                    current_yhat[t, l] = float(int(yhat[t, l].x))
                    current_y[t, l] = float(int(y[t, l].x))

    return current_xhat, current_yhat, current_x, current_y


def get_emissions(g):
    # Get emissions corresponding to a generation schedule.
    i = g.keys()[0][-1]

    emissions = np.zeros(len(years))

    for y in years:
        emissions[y] = sum(
            weights[o] * g[o, t, u, i].x * G_emissions[o, t, u]
            for o in scenarios
            for t in to_hours(y)
            for u in units
        )

    return emissions


class Timer:
    """Context manager for timing a function."""

    def __init__(self):
        self._collection = list()

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *args):
        self._end = time.time()
        self._elapsed = self._end - self._start
        self._collection.append(round(self._elapsed, 2))

    @property
    def solution_times(self):
        return self._collection


# Based on:
# https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting
# Makes print() to output to both terminal and a file.
class MyLogger(object):
    def __init__(self, master_problem_algorithm, subproblem_algorithm):
        self.terminal = sys.stdout
        self.log = open(
            "%s_%s.log" % (master_problem_algorithm, subproblem_algorithm), "w"
        )

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass
