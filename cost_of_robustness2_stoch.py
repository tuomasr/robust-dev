# Run an experiment in investment decisions are fixed but demand is varied.
# This is to quantify the quality of the investment decisions.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import socket
import sys

hostname = socket.gethostname()
print("HOSTNAME", hostname)

from gurobipy import GRB
import matplotlib.pyplot as plt
import numpy as np

from common_data import hours, years, real_nodes, scenarios, load, units, lines, discount_factor
from helpers import get_investment_cost, MyLogger
from master_problem_dc import CCMasterProblem


plt.switch_backend("agg")  # Enable plotting without a monitor.


class FixedCCMasterProblem(CCMasterProblem):
    # A version of the CC master problem where the investment decisions are fixed.
    # The resulting problem is an LP.

    def __init__(self, x, y, xhat, yhat, emission_penalty):
        CCMasterProblem.__init__(self)
        self._fix_investment_decisions(x, y, xhat, yhat)
        self._relax_emission_constraint(emission_penalty)

    def _fix_investment_decisions(self, x, y, xhat, yhat):
        # Fix the values of the variables x, y, xhat, and yhat.
        for var, val in [
            (self._x, x),
            (self._y, y),
            (self._xhat, xhat),
            (self._yhat, yhat),
        ]:
            for k, v in var.items():
                v.lb = v.ub = val[k]

        self._model.update()

    def _relax_emission_constraint(self, emission_penalty):
        # Relax the emission constraint by adding a slack variable with a penalty.
        m = self._model
        old_obj = m.getObjective()

        slack = m.addVars(years, name="emission_slack", lb=0.0, ub=GRB.INFINITY)
        m.setObjective(
            (
                old_obj
                + sum(
                    emission_penalty * (discount_factor ** (-y)) * slack[y]
                    for y in years
                )
            ),
            GRB.MINIMIZE,
        )
        self._emission_slack = slack

        # By setting this variable to True, the emission constraint will be relaxed.
        self._relaxed_emission_constraint = True

        self._model.update()


def estimate_cost(models, input_files, emission_costs):
    emission_costs = [1000.0]

    markers = ["o", "x", "^", "v", "s", "P"]
    assert len(markers) >= len(models) * len(emission_costs)

    num_costs = len(emission_costs)

    num_models = len(models)
    num_samples = 1

    start_percentage = 8.0
    end_percentage = 8.0
    percentage_increase = 1.0
    num_levels = 1

    results = dict()

    # Estimate the cost of different investment decisions.
    num_scenarios, num_hours, num_real_nodes = (
        len(scenarios),
        len(hours),
        len(real_nodes),
    )
    d = np.zeros((num_scenarios, num_hours, num_real_nodes, 2))

    percentages = []
    costs = np.zeros((num_models, num_levels, num_samples))

    for level in range(num_levels):
        print("level", level)
        percentage = (start_percentage + level * percentage_increase) / 100.0
        percentages.append(percentage)

        for sample in range(num_samples):
            print("sample", sample)
            # Sample random load changes.
            size = (num_scenarios, num_hours, num_real_nodes)
            if percentage >= 0.0:
                random_percentage_change = np.random.uniform(0.0, percentage, size=size)
            else:
                random_percentage_change = np.random.uniform(percentage, 0.0, size=size)

            # Random load for this sample.
            d[:, :, :, 1] = load * (1.0 + random_percentage_change)

            for model, input_file in enumerate(input_files):
                print("model", model)
                with open(input_file, "rb") as f:
                    investment_decisions = pickle.load(f)

                x = investment_decisions["x"]
                y = investment_decisions["y"]
                xhat = investment_decisions["xhat"]
                yhat = investment_decisions["yhat"]

                for eidx, emission_cost in enumerate(emission_costs):
                    print("SOLVING MODEL %d, LEVEL %d, SAMPLE %d, EMISSION COST %f" % (model, level, sample, emission_cost))

                    problem = FixedCCMasterProblem(x, y, xhat, yhat, emission_cost)
                    try:
                        total_cost, g, s, f = problem.solve(current_iteration=1, d=d)
                    except Exception as e:
                        print("FAILED:", str(e))
                        total_cost = np.nan

                    # Save operation decisions to an output file.
                    gg = dict()
                    ff = dict()
                    for o in scenarios:
                        for t in hours:
                            for u in units:
                                gg[o, t, u] = g[o, t, u, 1].x
                            for l in lines:
                                ff[o, t, l] = f[o, t, l, 1].x

                    model_result = {"g": gg, "f": ff}
                    results[input_file] = model_result

    with open("stoch.pickle", "wb") as file_:
        pickle.dump(results, file_)


def main(emission_cost):
    """Run robust optimization."""
    models = ["Stochastic", "Robust"]
    input_files = ["result_files2/scenarios_15_benders_dc_miqp_dc_ub=4_increase=0.05_stoch/investment.pickle", "result_files/scenarios_15_benders_dc_miqp_dc_ub=4_increase=0.05/investment.pickle"]

    sys.stdout = MyLogger(".", "cost_of_robustness", "emission_costs")

    estimate_cost(models, input_files, emission_cost)


if __name__ == "__main__":
    assert len(sys.argv) >= 2
    emission_costs = [float(e) for e in sys.argv[1:]]

    main(emission_costs)
