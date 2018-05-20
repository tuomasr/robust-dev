# Solve robust optimization problem for power markets.
# Master problem and subproblem as well as input data are defined in their respective files.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend("agg")  # Enable plotting without a monitor.

from common_data import (
    scenarios,
    years,
    hours,
    nodes,
    units,
    candidate_units,
    candidate_lines,
    G_ramp_max,
    G_ramp_min,
)
from helpers import (
    compute_objective_gap,
    concatenate_to_uncertain_variables_array,
    Timer,
)
from master_problem import (
    master_problem,
    augment_master_problem,
    get_investment_cost,
    get_investment_and_availability_decisions,
    get_emissions,
)


# Configure the algorithm for solving the robust optimization problem.
MAX_ITERATIONS = 5

# Compile solution times for the master problem and subproblem to these objects.
master_problem_timer = Timer()
subproblem_timer = Timer()

# Threshold for algorithm convergence.
EPSILON = 1e-6  # From Minguez et al. (2016)

# Numerical precision issues can cause LB to become higher than UB in some cases.
# This threshold allows some slack but an error is raised if the threshold is exceeded.
BAD_GAP_THRESHOLD = -1e-6

# Initial upper and lower bounds for the algorithm.
UB = np.inf
LB = -np.inf

# Used for collecting gaps when the algorithm runs.
gaps = []

# A line separating stuff in standard output.
separator = "-" * 50


def print_iteration_counter(iteration):
    # Print the current iteration number.
    print("ITERATION", iteration)
    print(separator)


def print_solution_quality(problem, problem_name):
    # Print information about the solution quality and problem statistics.
    print(separator)
    print("%s quality and stats:" % problem_name)
    problem.printQuality()
    problem.printStats()
    print(separator)


def run_robust_optimization(subproblem_algorithm):
    # Has the algorithm converged?
    converged = False

    # Import an implementation depending on the subproblem algorithm choice.
    if subproblem_algorithm == "benders":
        from subproblem_benders import (
            solve_subproblem,
            get_uncertain_variables,
            get_uncertainty_decisions,
        )
    elif subproblem_algorithm == "milp":
        from subproblem_milp import (
            solve_subproblem,
            get_uncertain_variables,
            get_uncertainty_decisions,
        )
    elif subproblem_algorithm == "miqp":
        from subproblem_miqp import (
            solve_subproblem,
            get_uncertain_variables,
            get_uncertainty_decisions,
        )

    # Initial uncertain variables for the first master problem iteration.
    d = np.zeros((len(hours), len(nodes), 1))

    # The main loop of the algorithm starts here.
    for iteration in range(MAX_ITERATIONS):
        print_iteration_counter(iteration)

        # Augment the master problem for the current iteration.
        if iteration > 0:
            g = augment_master_problem(iteration, d)

            # Solve the master problem. The context manager measures solution time.
        with master_problem_timer as t:
            print("Solving master problem.")
            master_problem.optimize()

        print_solution_quality(master_problem, "Master problem")

        # Update lower bound to the master problem objective value.
        LB = master_problem.objVal

        # Obtain investment and availability decisions from the master problem solution.
        xhat, yhat, x, y = get_investment_and_availability_decisions()

        # Solve the subproblem with the newly updated availability decisions.
        with subproblem_timer as t:
            print("Solving subproblem.")
            subproblem_objval, emission_prices = solve_subproblem(x, y)

        # Update the algorithm upper bound and compute new a gap.
        UB = get_investment_cost(xhat, yhat) + subproblem_objval

        GAP = compute_objective_gap(LB, UB)
        gaps.append(GAP)

        # Read the values of the uncertain variables
        _, uncertain_variable_vals = get_uncertain_variables()

        # Fill the next column in the uncertain variables array.
        d = concatenate_to_uncertain_variables_array(d, uncertain_variable_vals)

        # Exit if the algorithm converged.
        if GAP < EPSILON:
            converged = True
            assert GAP >= BAD_GAP_THRESHOLD, "Upper bound %f, lower bound %f." % (
                UB,
                LB,
            )
            break

    # Report solution.
    print_primal_variables = False

    if print_primal_variables:
        print("Primal variables:")
        print(separator)
        for v in master_problem.getVars():
            print(v.varName, v.x)

    print_uncertain_variables = True

    if print_uncertain_variables:
        print(separator)
        print("Uncertain variables:")
        print(separator)
        print(get_uncertainty_decisions())

    # Report if the algorithm converged.
    print(separator)

    if converged:
        print(
            "Converged at iteration %d! Gap: %s, UB-LB: %s" % (iteration, GAP, UB - LB)
        )
    else:
        print("Did not converge. Gap: %s, UB-LB: %s" % (GAP, UB - LB))

    print(separator)
    print("Objective value:", master_problem.objVal)
    print(
        "Investment cost %s, operation cost %s "
        % (get_investment_cost(xhat, yhat), subproblem_objval)
    )
    print(separator)

    # Report solution times.
    print(separator)
    print(
        "Master problem solution times (seconds):", master_problem_timer.solution_times
    )
    print("Subproblem solution times (seconds):", subproblem_timer.solution_times)
    print(
        "Total solution time (seconds):",
        sum(master_problem_timer.solution_times + subproblem_timer.solution_times),
    )

    print("xhat")
    for key, val in xhat.items():
        if val > 0.0:
            print(key, val)
    print("----")

    print("y")
    for key, val in y.items():
        if val > 0.0:
            print(key, val)
    print("----")

    print("yhat")
    for key, val in yhat.items():
        if val > 0.0:
            print(key, val)
    print("----")

    up_ramp_active = False
    down_ramp_active = False

    for o in scenarios:
        for u in units:
            for t in hours[1:-1]:
                gen_diff = g[o, t, u, iteration].x - g[o, t - 1, u, iteration].x

                if np.isclose(gen_diff, G_ramp_max[o, t, u]):
                    up_ramp_active = True

                if np.isclose(gen_diff, G_ramp_min[o, t, u]):
                    down_ramp_active = True

    print("up_ramp", up_ramp_active)
    print("down_ramp", down_ramp_active)

    # Generate emission plots.
    plot_emissions = True

    if plot_emissions:
        emissions = get_emissions(g)
        plt.figure()

        markers = ["ro--", "bs--", "kx--", "yd--", "c*--", "m^--"]

        for i, o in enumerate(scenarios):
            plt.plot(years, emissions[o, :], markers[i], label="Scenario %d" % o)

        plt.xlabel("master problem time step")
        plt.ylabel("emissions (kg)")
        lgd = plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.savefig(
            "emissions_trajectory_%s.png" % subproblem_algorithm,
            bbox_extra_artists=(lgd,),
            bbox_inches="tight",
        )

        # Emission prices plot.
        plt.figure()
        for i, o in enumerate(scenarios):
            price_list = [emission_prices[o, y] for y in years]
            plt.plot(years, price_list, markers[i], label="Scenario %d" % o)

        plt.xlabel("master problem time step")
        plt.ylabel("emissions price (EUR/kg)")
        lgd = plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.savefig(
            "emissions_prices_trajectory_%s.png" % subproblem_algorithm,
            bbox_extra_artists=(lgd,),
            bbox_inches="tight",
        )


def main():
    # Run robust optimization.
    parser = argparse.ArgumentParser(description="Run robust optimization.")
    parser.add_argument(
        "subproblem_algorithm", type=str, choices=("benders", "milp", "miqp")
    )

    args = parser.parse_args()

    run_robust_optimization(args.subproblem_algorithm)


if __name__ == "__main__":
    main()
