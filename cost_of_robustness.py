# Run an experiment in investment decisions are fixed but demand is varied.
# This is to quantify the quality of the investment decisions.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle

from gurobipy import GRB
import matplotlib.pyplot as plt
import numpy as np

from common_data import hours, years, real_nodes, scenarios, load, discount_factor
from helpers import get_investment_cost
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


def estimate_cost(models, input_files):
    # Estimate the cost of different investment decisions.
    num_scenarios, num_hours, num_real_nodes = (
        len(scenarios),
        len(hours),
        len(real_nodes),
    )
    d = np.zeros((num_scenarios, num_hours, num_real_nodes, 2))

    emission_penalty = 1e3  # EUR per tonne

    num_models = len(models)
    num_levels = 12
    num_samples = 1

    percentages = []
    costs = np.zeros((num_models, num_levels, num_samples))

    start_percentage = -4.0
    percentage_increase = 1.0

    for level in range(num_levels):
        percentage = (start_percentage + level * percentage_increase) / 100.0
        percentages.append(percentage)

        for sample in range(num_samples):
            # Sample random load changes.
            size = (num_scenarios, num_hours, num_real_nodes)
            if percentage >= 0.0:
                random_percentage_change = np.random.uniform(0.0, percentage, size=size)
            else:
                random_percentage_change = np.random.uniform(percentage, 0.0, size=size)

            # Random load for this sample.
            d[:, :, :, 1] = load * (1.0 + random_percentage_change)

            for model, input_file in enumerate(input_files):
                with open(input_file, "rb") as f:
                    investment_decisions = pickle.load(f)

                x = investment_decisions["x"]
                y = investment_decisions["y"]
                xhat = investment_decisions["xhat"]
                yhat = investment_decisions["yhat"]

                print("SOLVING MODEL %d, LEVEL %d, SAMPLE %d" % (model, level, sample))

                problem = FixedCCMasterProblem(x, y, xhat, yhat, emission_penalty)
                try:
                    total_cost, _, _, _ = problem.solve(current_iteration=1, d=d)
                except Exception as e:
                    print("FAILED:", str(e))
                    total_cost = np.nan

                costs[
                    model, level, sample
                ] = total_cost  # - get_investment_cost(xhat, yhat)

    x = np.array(percentages)
    y = np.mean(costs, axis=2) / num_scenarios / 1e6
    y = np.round(y, 2)

    markers = ["o", ".", "x", "^", "v", ">", "<"]

    for i, model in enumerate(models):
        plt.plot(x, y[i, :], marker=markers[i], label=model)

    plt.xlabel("Maximum load increase")
    plt.ylabel("Expected total cost (MEUR)")
    plt.legend(loc="lower right")
    plt.savefig("cost_of_robustness.png")


def main():
    """Run robust optimization."""
    models = ["SP", "robust 2"]
    input_files = ["sp_investment.pickle", "robust_2_investment.pickle"]

    # models = ["robust 2"]
    # input_files = ["robust_2_investment.pickle"]

    estimate_cost(models, input_files)


if __name__ == "__main__":
    main()
