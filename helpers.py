# Helper functions.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np

from common_data import (
    years,
    num_hours_per_year,
    candidate_units,
    candidate_lines,
    incidence,
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


def get_candidate_generation_capacity(t, u, x):
    # Compute the total capacity of the candidate unit during this year.
    assert u in candidate_units
    year = to_year(t)
    capacity = x[year, u]

    return capacity


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