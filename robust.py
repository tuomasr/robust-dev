# Solve robust optimization problem for power markets.
# Master problem and subproblem as well as input data are defined in their respective files.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pickle
import sys

import numpy as np

from common_data import (
    scenarios,
    years,
    hours,
    real_nodes,
    units,
    hydro_units,
    unit_to_node,
    initial_storage,
    storage_change_lb,
    storage_change_ub,
    ac_lines,
    lines,
)
from helpers import (
    compute_objective_gap,
    concatenate_to_uncertain_variables_array,
    Timer,
    get_maximum_ramp,
    MyLogger,
    get_emissions,
    get_investment_cost,
)
from master_problem_dc import CCMasterProblem
from master_problem_benders_dc import CCMasterProblemBenders
from master_problem_benders_dc_dual import CCMasterProblemDualBenders
from plotting import create_investment_plots, create_emission_plots

# Configure the algorithm for solving the robust optimization problem.
MAX_ITERATIONS = 7

# Compile solution times for the master problem and subproblem to these objects.
master_problem_timer = Timer()
subproblem_timer = Timer()

# Threshold for algorithm convergence.
EPSILON = 1e-6  # From Minguez et al. (2016)

# Numerical precision issues can cause LB to become higher than UB in some cases.
# This threshold allows some slack but an error is raised if the threshold is exceeded.
BAD_GAP_THRESHOLD = -1e-3

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


def run_robust_optimization(master_problem_algorithm, subproblem_algorithm, output_dir):
    """Solve the robust optimization problem.

    Use a column-and-constraint (CC) algorithm at the top-level.
    """
    # Create output directory.
    os.makedirs(output_dir)

    converged = False

    if master_problem_algorithm == "benders_dc":
        master_problem_class = CCMasterProblemBenders
    elif master_problem_algorithm == "dual_benders_dc":
        master_problem_class = CCMasterProblemDualBenders
    elif master_problem_algorithm == "milp_dc":
        master_problem_class = CCMasterProblem

    master_problem = master_problem_class()

    # Import an implementation depending on the subproblem algorithm choice.
    if subproblem_algorithm in ("miqp_dc", "milp_dc"):
        subproblem_milp = subproblem_algorithm == "milp_dc"
        from subproblem import (
            solve_subproblem,
            get_uncertain_variables,
            get_uncertainty_decisions,
        )

    # Initial uncertain variables for the first master problem iteration.
    d = np.zeros((len(scenarios), len(hours), len(real_nodes), 1))

    # The main loop of the algorithm starts here.
    for iteration in range(MAX_ITERATIONS):
        print_iteration_counter(iteration)

        # Solve the master problem. The context manager measures solution time.
        with master_problem_timer as t:
            print("Solving master problem.")
            master_problem_objval, g, s, f = master_problem.solve(iteration, d)

        # Update lower bound to the master problem objective value.
        LB = master_problem_objval

        # Initial master problem solution (full investment) is used for the first iteration.
        initial = iteration == 0

        # Obtain investment and availability decisions from the master problem solution.
        xhat, yhat, x, y = master_problem.get_investment_and_availability_decisions(
            initial
        )

        # Solve the subproblem with the newly updated availability decisions.
        with subproblem_timer as t:
            print("Solving subproblem.")
            subproblem_objval, emission_prices = solve_subproblem(x, y, subproblem_milp)

        # Update the algorithm upper bound and compute new a gap.
        UB = get_investment_cost(xhat, yhat) + subproblem_objval

        GAP = compute_objective_gap(LB, UB)
        gaps.append(GAP)

        print("GAP is:", GAP)

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

    # Report if the algorithm converged.
    print(separator)

    if converged:
        print(
            "Converged at iteration %d! Gap: %s, UB-LB: %s" % (iteration, GAP, UB - LB)
        )
    else:
        print("Did not converge. Gap: %s, UB-LB: %s" % (GAP, UB - LB))

    if converged:
        # Save investment decisions to an output file.
        investment_decisions = {"x": x, "xhat": xhat, "y": y, "yhat": yhat}
        with open(os.path.join(output_dir, "investment.pickle"), "wb") as file_:
            pickle.dump(investment_decisions, file_)

        # Save operation decisions to an output file.
        gg = dict()
        ff = dict()
        for o in scenarios:
            for t in hours:
                for u in units:
                    gg[o, t, u] = g[o, t, u, iteration].x

                for l in lines:
                    ff[o, t, l] = f[o, t, l, iteration].x

        operation_decisions = {"g": gg, "f": ff}
        with open(
            os.path.join(output_dir, "operation_decisions.pickle"), "wb"
        ) as file_:
            pickle.dump(operation_decisions, file_)

    # Report costs.
    print(separator)
    print("Objective value:", master_problem_objval)
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

    # Report solution.
    print_primal_variables = False

    if print_primal_variables:
        print("Primal variables:")
        print(separator)
        for v in master_problem.getVars():
            if "flow_%d" % iteration in v.varName:
                key = v.varName.split(",")
                if int(key[1]) == 17 and int(key[2]) in ac_lines:
                    print(v.varName, v.x)

    print_uncertain_variables = True

    if print_uncertain_variables:
        print(separator)
        print("Uncertain variables:")
        print(separator)
        print(get_uncertainty_decisions())

    # Print investments.
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

    # Plot investments.
    plot_investments = True

    if plot_investments:
        create_investment_plots(
            xhat, yhat, master_problem_algorithm, subproblem_algorithm, output_dir
        )

    # Check if ramping constraints were active.
    up_ramp_active = False
    down_ramp_active = False

    for o in scenarios:
        for u in units:
            for t in hours[1:-1]:
                gen_diff = g[o, t, u, iteration].x - g[o, t - 1, u, iteration].x

                max_ramp = get_maximum_ramp(o, t, u, x)

                if np.isclose(gen_diff, max_ramp):
                    up_ramp_active = True

                if np.isclose(gen_diff, -max_ramp):
                    down_ramp_active = True

    print("up_ramp", up_ramp_active)
    print("down_ramp", down_ramp_active)
    print("----")

    # Check if final storage constraints were active.
    storage_lb_active = False
    storage_ub_active = False
    for o in scenarios:
        for u in hydro_units:
            for y in years:
                num_hours_per_year = len(hours) / len(years)
                t1 = (y + 1) * num_hours_per_year - 1

                final_storage = s[o, t1, u, iteration].x
                final_storage_lb = (
                    initial_storage[u][o, y] * storage_change_lb[unit_to_node[u]]
                )
                final_storage_ub = (
                    initial_storage[u][o, y] * storage_change_ub[unit_to_node[u]]
                )

                if np.isclose(final_storage, final_storage_lb):
                    storage_lb_active = True

                if np.isclose(final_storage, final_storage_ub):
                    storage_ub_active = True

    print("final_storage_lb", storage_lb_active)
    print("final_storage_ub", storage_ub_active)
    print("----")

    # Print storage values.
    print_storage = False

    if print_storage:
        print("storage")
        for key, val in s.items():
            print(key, val.x)
        print("----")

    # Generate emission plots.
    plot_emissions = True

    if plot_emissions:
        emissions = get_emissions(g)

        emission_data = {"emissions": emissions, "emission_prices": emission_prices}
        with open(os.path.join(output_dir, "emission_data.pickle"), "wb") as file_:
            pickle.dump(emission_data, file_)

        create_emission_plots(
            emissions,
            emission_prices,
            master_problem_algorithm,
            subproblem_algorithm,
            output_dir,
        )


def main():
    """Run robust optimization."""
    parser = argparse.ArgumentParser(description="Run robust optimization.")
    parser.add_argument(
        "master_problem_algorithm",
        type=str,
        choices=("benders_dc", "dual_benders_dc", "milp_dc"),
    )
    parser.add_argument(
        "subproblem_algorithm", type=str, choices=("miqp_dc", "milp_dc")
    )
    parser.add_argument("output_dir", type=str)

    args = parser.parse_args()

    sys.stdout = MyLogger(args.master_problem_algorithm, args.subproblem_algorithm)

    run_robust_optimization(
        args.master_problem_algorithm, args.subproblem_algorithm, args.output_dir
    )


if __name__ == "__main__":
    main()
