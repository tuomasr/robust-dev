# Subproblem Benders formulation.
# Note: Assumes that candidate lines are AC.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gurobipy import *
import numpy as np

from common_data import (
    hours,
    scenarios,
    real_nodes,
    nodes,
    load,
    units,
    lines,
    existing_units,
    existing_lines,
    candidate_units,
    candidate_lines,
    hydro_units,
    G_max,
    F_max,
    F_min,
    B,
    ref_node,
    G_ramp_max,
    G_ramp_min,
    incidence,
    weights,
    C_g,
    initial_storage,
    inflows,
    unit_to_node,
    emission_targets,
    G_emissions,
    years,
    subproblem_method,
    enable_custom_configuration,
    GRB_PARAMS,
    uncertainty_budget,
    uncertainty_demand_increase,
)
from helpers import (
    to_year,
    unit_built,
    line_built,
    is_year_first_hour,
    is_year_last_hour,
    get_lines_starting,
    get_lines_ending,
    get_candidate_generation_capacity,
    get_ramp_hours,
    compute_objective_gap,
)


# Problem-specific data: Demand at each node in each time hour and uncertainty in it.
nominal_demand = load[:, : len(real_nodes)]
demand_increase = uncertainty_demand_increase * np.ones_like(nominal_demand)

K = 100.0
max_lambda_ = K
min_lambda_ = -K

# Master problem definition.
mp = Model("subproblem_master")

delta = mp.addVars(scenarios, hours, name="delta", lb=-GRB.INFINITY)

w = mp.addVars(real_nodes, name="demand_deviation", vtype=GRB.BINARY)

mp.setObjective(sum(delta[o, t] for o in scenarios for t in hours), GRB.MAXIMIZE)

mp.addConstr(
    sum(w[n] for n in real_nodes) - uncertainty_budget == 0.0,
    name="uncertainty_set_budget",
)

# Solver methods.
mp_method = sp_method = subproblem_method

mp.params.Method = mp_method

if enable_custom_configuration:
    for parameter, value in GRB_PARAMS:
        mp.setParam(parameter, value)


def initialize_master():
    # Initialize the Benders master problem by removing old cuts.
    existing_constraints = [
        c for c in mp.getConstrs() if "delta_constraint_" in c.ConstrName
    ]

    if existing_constraints:
        mp.remove(existing_constraints)
        mp.update()


def augment_master(x, dual_values, iteration):
    # Augment the Benders master problem with a new cut from the slave problem.

    # Unpack the dual values. Be careful with the order.
    sigma, rho_underline, rho_bar, omega_underline, omega_bar = dual_values

    # Enable/disable multicut version of Benders. Single cut seems to be faster.
    multicut = False

    if not multicut:
        mp.addConstr(
            sum(delta[o, t] for o in scenarios for t in hours)
            - sum(
                (
                    max_lambda_
                    * sum(
                        (rho_bar[o, t, n] - omega_bar[o, t, n]) * w[n]
                        for n in real_nodes
                    )
                    - min_lambda_
                    * sum(
                        (rho_underline[o, t, n] - omega_underline[o, t, n]) * w[n]
                        for n in real_nodes
                    )
                    + sum(
                        C_g[o, t, u] * weights[o] * sigma[o, t, u]
                        for u in units
                        if unit_built(x, t, u)
                    )
                    + max_lambda_ * sum(omega_bar[o, t, n] for n in real_nodes)
                    - min_lambda_ * sum(omega_underline[o, t, n] for n in real_nodes)
                )
                for o in scenarios
                for t in hours
            )
            <= 0.0,
            name="delta_constraint_single_cut_%d" % iteration,
        )
    else:
        mp.addConstrs(
            (
                delta[o, t]
                - (
                    max_lambda_
                    * sum(
                        (rho_bar[o, t, n] - omega_bar[o, t, n]) * w[n]
                        for n in real_nodes
                    )
                    - min_lambda_
                    * sum(
                        (rho_underline[o, t, n] - omega_underline[o, t, n]) * w[n]
                        for n in real_nodes
                    )
                    + sum(
                        C_g[o, t, u] * weights[o] * sigma[o, t, u]
                        for u in units
                        if unit_built(x, t, u)
                    )
                    + max_lambda_ * sum(omega_bar[o, t, n] for n in real_nodes)
                    - min_lambda_ * sum(omega_underline[o, t, n] for n in real_nodes)
                )
                <= 0.0
                for o in scenarios
                for t in hours
            ),
            name="delta_constraint_multi_cut_%d" % iteration,
        )


def get_uncertain_demand_decisions(initial):
    # Construct an initial solution for the first iteration, otherwise read the optimal solution.
    if initial:
        uncertain = {n: 0.0 for n in w.keys()}
    else:
        if mp.SolCount == 1:
            uncertain = {n: np.round(w[n].x) for n in w.keys()}
        else:
            uncertain = {n: np.round(w[n].Xn) for n in w.keys()}

    return uncertain


def get_solution_hash():
    # Generate a hash of the solution.
    if mp.SolCount == 1:
        return "-".join(
            [str(idx) for idx, n in enumerate(w.keys()) if np.isclose(w[n].x, 1.0)]
        )
    else:
        return "-".join(
            [str(idx) for idx, n in enumerate(w.keys()) if np.isclose(w[n].Xn, 1.0)]
        )


def create_slave(x, y, ww):
    # Benders slave problem definition
    sp = Model("subproblem_slave")

    sp.Params.Method = sp_method  # Dual simplex.

    if enable_custom_configuration:
        for parameter, value in GRB_PARAMS:
            sp.setParam(parameter, value)

    ub1 = GRB.INFINITY
    lb2 = -GRB.INFINITY
    ub2 = GRB.INFINITY

    # Variables for linearizing bilinear terms lambda_[o, t, n] * w[n]
    z = sp.addVars(
        scenarios,
        hours,
        nodes,
        name="linearization_z",
        lb=-GRB.INFINITY,
        ub=GRB.INFINITY,
    )
    lambda_tilde = sp.addVars(
        scenarios,
        hours,
        nodes,
        name="linearization_lambda_tilde",
        lb=-GRB.INFINITY,
        ub=GRB.INFINITY,
    )

    # Maximum and minimum generation dual variables.
    beta_underline = sp.addVars(
        scenarios, hours, units, name="dual_minimum_generation", lb=0.0, ub=ub1
    )
    beta_candidate_bar = sp.addVars(
        scenarios,
        hours,
        candidate_units,
        name="dual_maximum_candidate_generation",
        lb=0.0,
        ub=ub1,
    )
    beta_bar = sp.addVars(
        scenarios, hours, units, name="dual_maximum_generation", lb=0.0, ub=ub1
    )

    ramp_hours = get_ramp_hours()

    # Storage dual variables.
    year_first_hours = [t for t in hours if is_year_first_hour(t)]

    beta_storage_underline = sp.addVars(
        scenarios, hours, hydro_units, name="dual_minimum_storage", lb=0.0, ub=ub1
    )

    phi_initial_storage = sp.addVars(
        scenarios,
        year_first_hours,
        hydro_units,
        name="dual_initial_storage",
        lb=lb2,
        ub=ub2,
    )

    phi_storage = sp.addVars(
        scenarios, ramp_hours, hydro_units, name="dual_storage", lb=lb2, ub=ub2
    )

    # Transmission flow dual variables.
    phi = sp.addVars(scenarios, hours, lines, name="dual_flow", lb=lb2, ub=ub2)

    # Maximum and minimum transmission flow dual variables.
    mu_underline = sp.addVars(
        scenarios, hours, lines, name="dual_minimum_flow", lb=0.0, ub=ub1
    )
    mu_bar = sp.addVars(
        scenarios, hours, lines, name="dual_maximum_flow", lb=0.0, ub=ub1
    )

    # Dual variables for voltage angle bounds.
    mu_angle_bar = sp.addVars(
        scenarios, hours, nodes, name="dual_maximum_angle", lb=0.0, ub=ub1
    )
    mu_angle_underline = sp.addVars(
        scenarios, hours, nodes, name="dual_minimum_angle", lb=0.0, ub=ub1
    )

    # Dual variable for the reference node voltage angle.
    eta = sp.addVars(scenarios, hours, name="dual_reference_node_angle", lb=lb2, ub=ub2)

    # Maximum up- and down ramp dual variables.
    beta_ramp_bar = sp.addVars(
        scenarios, ramp_hours, units, name="dual_maximum_ramp_upwards", lb=0.0, ub=ub1
    )
    beta_ramp_underline = sp.addVars(
        scenarios, ramp_hours, units, name="dual_maximum_ramp_downwards", lb=0.0, ub=ub1
    )

    # Maximum emissions dual variables.
    beta_emissions = sp.addVars(
        scenarios, years, name="dual_maximum_emissions", lb=0.0, ub=ub1
    )

    obj = sum(
        sum(
            z[o, t, n] * demand_increase[t, n]
            + (z[o, t, n] + lambda_tilde[o, t, n]) * nominal_demand[t, n]
            for n in real_nodes
        )
        - sum(beta_bar[o, t, u] * G_max[o, t, u] for u in units if unit_built(x, t, u))
        - sum(
            beta_candidate_bar[o, t, u] * get_candidate_generation_capacity(t, u, x)
            for u in candidate_units
        )
        + sum(
            initial_storage[u][o, to_year(t)] * phi_initial_storage[o, t, u]
            if t in year_first_hours
            else 0.0
            for u in hydro_units
        )
        + sum(
            inflows[u][o, t] * phi_storage[o, t, u] if t in ramp_hours else 0.0
            for u in hydro_units
        )
        - sum(
            (mu_bar[o, t, l] * F_max[o, t, l] - mu_underline[o, t, l] * F_min[o, t, l])
            for l in lines
            if line_built(y, t, l)
        )
        - sum(
            np.pi * (mu_angle_bar[o, t, n] + mu_angle_underline[o, t, n]) for n in nodes
        )
        - sum(
            beta_ramp_bar[o, t, u] * G_ramp_max[o, t, u] if t in ramp_hours else 0.0
            for u in units
        )
        + sum(
            beta_ramp_underline[o, t, u] * G_ramp_min[o, t, u]
            if t in ramp_hours
            else 0.0
            for u in units
        )
        for o in scenarios
        for t in hours
    )

    obj -= sum(
        beta_emissions[o, y] * emission_targets[y] for o in scenarios for y in years
    )

    sp.setObjective(obj, GRB.MAXIMIZE)

    # Dual constraints.
    sigma_constrs = sp.addConstrs(
        (
            z[o, t, unit_to_node[u]]
            + lambda_tilde[o, t, unit_to_node[u]]
            - (beta_candidate_bar[o, t, u] if u in candidate_units else 0.0)
            - beta_bar[o, t, u]
            + beta_underline[o, t, u]
            + (phi_storage[o, t, u] if t in ramp_hours and u in hydro_units else 0.0)
            - (beta_ramp_bar[o, t - 1, u] if not is_year_first_hour(t) else 0.0)
            + (beta_ramp_bar[o, t, u] if not is_year_last_hour(t) else 0.0)
            + (beta_ramp_underline[o, t - 1, u] if not is_year_first_hour(t) else 0.0)
            - (beta_ramp_underline[o, t, u] if not is_year_last_hour(t) else 0.0)
            - (beta_emissions[o, to_year(t)] * G_emissions[o, t, u])
            - C_g[o, t, u] * weights[o]
            == 0.0
            for o in scenarios
            for t in hours
            for u in units
            if unit_built(x, t, u)
        ),
        name="generation_dual_constraint",
    )

    sp.addConstrs(
        (
            (phi_initial_storage[o, t, u] if is_year_first_hour(t) else 0.0)
            + beta_storage_underline[o, t, u]
            - (phi_storage[o, t, u] if not is_year_last_hour(t) else 0.0)
            + (phi_storage[o, t - 1, u] if not is_year_first_hour(t) else 0.0)
            == 0.0
            for o in scenarios
            for t in hours
            for u in hydro_units
        ),
        name="storage_dual_constraint",
    )

    sp.addConstrs(
        (
            sum(incidence[l, n] * (z[o, t, n] + lambda_tilde[o, t, n]) for n in nodes)
            + phi[o, t, l]
            - mu_bar[o, t, l]
            + mu_underline[o, t, l]
            == 0.0
            for o in scenarios
            for t in hours
            for l in lines
            if line_built(y, t, l)
        ),
        name="flow_dual_constraint",
    )

    sp.addConstrs(
        (
            -sum(
                line_built(y, t, l) * B[l] * phi[o, t, l] for l in get_lines_starting(n)
            )
            + sum(
                line_built(y, t, l) * B[l] * phi[o, t, l] for l in get_lines_ending(n)
            )
            - mu_angle_bar[o, t, n]
            + mu_angle_underline[o, t, n]
            + (eta[o, t] if n == ref_node else 0.0)
            == 0.0
            for o in scenarios
            for t in hours
            for n in nodes
        ),
        name="voltage_angle_dual_constraint",
    )

    rho_underline_constrs = sp.addConstrs(
        (
            ww[n] * min_lambda_ - z[o, t, n] <= 0.0
            for o in scenarios
            for t in hours
            for n in real_nodes
        ),
        name="linearization_z_lb",
    )

    rho_bar_constrs = sp.addConstrs(
        (
            z[o, t, n] - ww[n] * max_lambda_ <= 0.0
            for o in scenarios
            for t in hours
            for n in real_nodes
        ),
        name="linearization_z_ub",
    )

    omega_underline_constrs = sp.addConstrs(
        (
            (1.0 - ww[n]) * min_lambda_ - lambda_tilde[o, t, n] <= 0.0
            for o in scenarios
            for t in hours
            for n in real_nodes
        ),
        name="lambda_tilde_lb",
    )

    omega_bar_constrs = sp.addConstrs(
        (
            lambda_tilde[o, t, n] - (1.0 - ww[n]) * max_lambda_ <= 0.0
            for o in scenarios
            for t in hours
            for n in real_nodes
        ),
        name="lambda_tilde_ub",
    )

    all_constrs = (
        sigma_constrs,
        rho_underline_constrs,
        rho_bar_constrs,
        omega_underline_constrs,
        omega_bar_constrs,
    )

    updatable_constrs = (
        rho_underline_constrs,
        rho_bar_constrs,
        omega_underline_constrs,
        omega_bar_constrs,
    )

    return sp, all_constrs, updatable_constrs, beta_emissions


def update_slave(updatable_constrs, ww):
    # Update constraints involving the value of w. Be careful with the order when unpacking.
    rho_underline_constrs, rho_bar_constrs, omega_underline_constrs, omega_bar_constrs = (
        updatable_constrs
    )

    for o in scenarios:
        for t in hours:
            for n in real_nodes:
                rho_underline_constrs[o, t, n].RHS = -ww[n] * min_lambda_
                rho_bar_constrs[o, t, n].RHS = ww[n] * max_lambda_

                omega_underline_constrs[o, t, n].RHS = -(1.0 - ww[n]) * min_lambda_
                omega_bar_constrs[o, t, n].RHS = (1.0 - ww[n]) * max_lambda_


def get_dual_variables(sp, all_constrs):
    # Get dual variable values of all (relevant) constraints of the slave problem.
    def _get_dual_values(constrs):
        dual_values = dict()

        for key, val in constrs.iteritems():
            dual_values[key] = val.getAttr("Pi")

        return dual_values

    return [_get_dual_values(constrs) for constrs in all_constrs]


def get_uncertain_variables():
    # Get the names and values of uncertain variables of the subproblem.
    names = np.array([w[n].varName for n in real_nodes])
    values = np.array(
        [w[n].Xn * demand_increase[:, n] + nominal_demand[:, n] for n in real_nodes]
    )
    values = np.transpose(values)

    return names, values


def get_uncertainty_decisions():
    # Get a vector of the uncertainty decisions w.
    values = np.array([w[n].Xn for n in real_nodes])
    return values


def solve_subproblem(x, y):
    # Solve the subproblem using Benders.
    max_iterations = 9999
    threshold = 1e-6
    bad_threshold = -1e-3
    separator = "-" * 50

    lb = -np.inf
    ub = np.inf

    # Initialize Benders.
    initialize_master()
    ww = get_uncertain_demand_decisions(initial=True)
    sp, all_constrs, updatable_constrs, beta_emissions = create_slave(x, y, ww)
    sp.optimize()
    dual_values = get_dual_variables(sp, all_constrs)
    augment_master(x, dual_values, iteration=0)

    # The Benders master problem is likely to generate the same solution twice so don't explore
    # the same solution more than once.
    solution_hashes = set()

    for iteration in range(1, max_iterations):
        print(separator)
        print("Starting Benders iteration:", iteration)

        mp.Params.SolutionNumber = 0

        print("Solving Benders master problem.")
        mp.update()
        mp.optimize()
        print(separator)

        if mp.Status != GRB.OPTIMAL:
            raise RuntimeError("Benders master problem not optimal.")

        ub = mp.objVal

        # Explore all master problem solutions.
        for k in range(mp.SolCount):
            mp.Params.SolutionNumber = k

            solution_hash = get_solution_hash()

            if k > 0 and solution_hash in solution_hashes:
                continue

            ww = get_uncertain_demand_decisions(initial=False)

            update_slave(updatable_constrs, ww)

            print(separator)
            print("Solving Benders slave problem.")
            sp.update()
            sp.optimize()
            print(separator)

            if sp.Status != GRB.OPTIMAL:
                raise RuntimeError("Benders slave problem not optimal.")

            # Only consider exit in case of an optimal solution.
            if k == 0:
                lb = sp.objVal

                gap = compute_objective_gap(lb, ub)

                if gap < threshold:
                    if not gap >= bad_threshold:
                        raise RuntimeError("lb (%f) > ub (%f) in Benders." % (lb, ub))

                    print("Took %d Benders iterations." % (iteration + 1))

                    # Collect emission prices.
                    prices = dict()
                    for s in scenarios:
                        for y in years:
                            print(
                                "beta_emissions[%d, %d]:" % (s, y),
                                beta_emissions[s, y].x,
                            )
                            prices[s, y] = beta_emissions[s, y].x

                    return (lb + ub) / 2, prices

            dual_values = get_dual_variables(sp, all_constrs)
            augment_master(x, dual_values, iteration)

            solution_hashes.add(solution_hash)

    raise RuntimeError("Max iterations hit in Benders. LB: %f, UB: %f" % (lb, ub))