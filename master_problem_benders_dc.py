# Master problem Benders formulation with the assumption that all candidate lines are DC.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gurobipy import *
import numpy as np

from common_data import (
    years,
    hours,
    num_hours_per_year,
    annualizer,
    scenarios,
    real_nodes,
    nodes,
    units,
    lines,
    existing_units,
    existing_lines,
    candidate_units,
    candidate_lines,
    ac_lines,
    ac_nodes,
    hydro_units,
    unit_to_node,
    G_max,
    maximum_candidate_unit_capacity_by_type,
    unit_to_generation_type,
    F_max,
    F_min,
    B,
    ref_node,
    G_emissions,
    C_g,
    initial_storage,
    inflows,
    storage_change_lb,
    storage_change_ub,
    incidence,
    weights,
    node_to_unit,
    emission_targets,
    C_x,
    C_y,
    discount_factor,
    master_method,
    enable_custom_configuration,
    GRB_PARAMS,
)
from helpers import (
    to_year,
    to_hours,
    line_built,
    get_ramp_hours,
    get_start_node,
    get_end_node,
    get_effective_capacity,
    get_maximum_ramp,
    compute_objective_gap,
    is_year_first_hour,
    is_year_last_hour,
)


def get_investment_cost(xhat, yhat):
    # Compute total investment cost for fixed generation and transmission investment decisions.
    return (
        sum(
            sum(C_x[t, u] * xhat[t, u] for u in candidate_units)
            + sum(C_y[t, l] * yhat[t, l] for l in candidate_lines)
            for t in years
        )
        / annualizer
    )


def get_transmission_investment_cost(yhat):
    # Compute total investment cost for fixed transmission investment decisions.
    return (
        sum(sum(C_y[t, l] * yhat[t, l] for l in candidate_lines) for t in years)
        / annualizer
    )


def get_initial_transmission_investments():
    # Get initial transmission line investments.
    initial_yhat = dict()
    initial_y = dict()

    for t in years:
        for l in candidate_lines:
            if t == 0:
                initial_yhat[t, l] = 1.0
            else:
                initial_yhat[t, l] = 0.0

            initial_y[t, l] = 1.0

    return initial_yhat, initial_y


def add_primal_variables(sp, iteration):
    # Generation variables for existing and candidate units. Upper bounds are set as constraints.
    g = sp.addVars(
        scenarios,
        hours,
        units,
        [iteration],
        name="generation_%d" % iteration,
        lb=0.0,
        ub=GRB.INFINITY,
    )

    # Storage variables.
    s = sp.addVars(
        scenarios,
        hours,
        hydro_units,
        [iteration],
        name="storage_%d" % iteration,
        lb=0.0,
        ub=GRB.INFINITY,
    )

    # Flow variables for existing and candidate lines. Upper and lower bound are set as constraints.
    f = sp.addVars(
        scenarios,
        hours,
        lines,
        [iteration],
        name="flow_%d" % iteration,
        lb=-GRB.INFINITY,
        ub=GRB.INFINITY,
    )

    delta = sp.addVars(
        scenarios,
        hours,
        ac_nodes,
        [iteration],
        name="voltage_angle_%d" % iteration,
        lb=-GRB.INFINITY,
        ub=GRB.INFINITY,
    )

    return g, s, f, delta


# Create the initial model without operation constraints.
mp = Model("master_problem")
mp.Params.Method = master_method

if enable_custom_configuration:
    for parameter, value in GRB_PARAMS:
        mp.setParam(parameter, value)

# Variables representing investment to generation units and transmission lines.
yhat = mp.addVars(years, candidate_lines, vtype=GRB.BINARY, name="line_investment")

# Variables indicating whether candidate generation units and transmission lines can be operated.
y = mp.addVars(years, candidate_lines, vtype=GRB.BINARY, name="line_available")


# Constraints defining that candidate units and transmission lines can be operated if investment
# has been made.
# mp.addConstrs(
#     (sum(yhat[t, l] for t in years) <= 1 for l in candidate_lines),
#     name="line_invest_once",
# )

mp.addConstrs(
    (
        y[t, l] - sum(yhat[tt, l] for tt in range(t + 1)) == 0.0
        for t in years
        for l in candidate_lines
    ),
    name="line_operational1",
)

# mp.addConstrs(
#     (
#         y[t, l] >= yhat[t, l]
#         for t in years
#         for l in candidate_lines
#     ),
#     name="line_operational2",
# )

delta = mp.addVar(name="delta", lb=0.0, ub=GRB.INFINITY)

mp.setObjective(get_transmission_investment_cost(yhat) + delta, GRB.MINIMIZE)

# Slave problem definition.
sp = Model("slave_problem")
sp.Params.Method = master_method

if enable_custom_configuration:
    for parameter, value in GRB_PARAMS:
        sp.setParam(parameter, value)

xhat = sp.addVars(
    years, candidate_units, lb=0.0, ub=GRB.INFINITY, name="unit_investment"
)
x = sp.addVars(years, candidate_units, lb=0.0, ub=GRB.INFINITY, name="unit_available")

sp.addConstrs(
    (
        x[t, u] - sum(xhat[tt, u] for tt in range(t + 1)) == 0.0
        for t in years
        for u in candidate_units
    ),
    name="unit_operational1",
)

sp.addConstrs(
    (
        sum(xhat[t, u] for t in years)
        <= maximum_candidate_unit_capacity_by_type[unit_to_generation_type[u]]
        for u in candidate_units
    ),
    name="maximum_unit_investment1",
)

# sp.addConstrs(
#     (
#         xhat[t, u] <= maximum_candidate_unit_capacity_by_type[unit_to_generation_type[u]]
#         for t in years
#         for u in candidate_units
#     ),
#     name="maximum_unit_investment2",
# )

# sp.addConstrs(
#     (
#         x[t, u] <= maximum_candidate_unit_capacity_by_type[unit_to_generation_type[u]]
#         for t in years
#         for u in candidate_units
#     ),
#     name="maximum_unit_investment3",
# )

# Variable representing the subproblem objective value.
theta = sp.addVar(name="theta", lb=0.0, ub=GRB.INFINITY)


def initialize_master():
    # Initialize the Benders master problem by removing old cuts.
    existing_constraints = [
        c for c in mp.getConstrs() if "delta_constraint_" in c.ConstrName
    ]

    if existing_constraints:
        mp.remove(existing_constraints)
        mp.update()


def augment_master(
    dual_values, iteration, benders_iteration, solution_number, d, unbounded=False
):
    # Augment the Benders master problem with a new cut from the slave problem.

    # Unpack the dual values. Be careful with the order.
    sigma, beta_bar, mu_underline, mu_bar, beta_ramp_underline, beta_ramp_bar, beta_emissions, rho_bar, rho_underline, phi_initial_storage, phi_storage, phi_storage_change_lb, phi_storage_change_ub = (
        dual_values
    )

    year_first_hours = [t for t in hours if is_year_first_hour(t)]
    year_last_hours = [t for t in hours if is_year_last_hour(t)]
    ramp_hours = get_ramp_hours()

    mp.update()

    if not unbounded:
        mp.addConstr(
            delta
            - sum(
                (
                    +sum(
                        get_effective_capacity(o, t, u, x) * beta_bar[o, t, u, v]
                        for u in existing_units
                    )
                    - sum(
                        F_min[o, t, l] * line_built(y, t, l) * mu_underline[o, t, l, v]
                        for l in lines
                    )
                    + sum(
                        F_max[o, t, l] * line_built(y, t, l) * mu_bar[o, t, l, v]
                        for l in lines
                    )
                    - sum(
                        (
                            (-get_maximum_ramp(o, t, u, x))
                            * beta_ramp_underline[o, t, u, v]
                            if t in ramp_hours
                            else 0.0
                        )
                        for u in existing_units
                    )
                    + sum(
                        (
                            get_maximum_ramp(o, t, u, x) * beta_ramp_bar[o, t, u, v]
                            if t in ramp_hours
                            else 0.0
                        )
                        for u in existing_units
                    )
                    + sum(d[o, t, n, v] * sigma[o, t, n, v] for n in real_nodes)
                    + sum(np.pi * rho_underline[o, t, n, v] for n in ac_nodes)
                    + sum(np.pi * rho_bar[o, t, n, v] for n in ac_nodes)
                    + sum(
                        (
                            initial_storage[u][o, to_year(t)]
                            * phi_initial_storage[o, t, u, v]
                            if t in year_first_hours
                            else 0.0
                        )
                        for u in hydro_units
                    )
                    + sum(
                        (
                            inflows[u][o, t] * phi_storage[o, t, u, v]
                            if t in ramp_hours
                            else 0.0
                        )
                        for u in hydro_units
                    )
                    + sum(
                        phi_storage_change_lb[o, t, u, v]
                        * (
                            -initial_storage[u][o, to_year(t)]
                            * storage_change_lb[unit_to_node[u]]
                        )
                        if t in year_last_hours
                        else 0.0
                        for u in hydro_units
                    )
                    + sum(
                        phi_storage_change_ub[o, t, u, v]
                        * initial_storage[u][o, to_year(t)]
                        * storage_change_ub[unit_to_node[u]]
                        if t in year_last_hours
                        else 0.0
                        for u in hydro_units
                    )
                )
                for o in scenarios
                for t in hours
                for v in range(1, iteration + 1)
            )
            - sum(
                emission_targets[y] * beta_emissions[y, v]
                for y in years
                for v in range(1, iteration + 1)
            )
            >= 0.0,
            name="delta_constraint_single_cut_%d_%d"
            % (benders_iteration, solution_number),
        )
    else:
        mp.addConstr(
            sum(
                (
                    +sum(
                        get_effective_capacity(o, t, u, x) * beta_bar[o, t, u, v]
                        for u in existing_units
                    )
                    - sum(
                        F_min[o, t, l] * line_built(y, t, l) * mu_underline[o, t, l, v]
                        for l in lines
                    )
                    + sum(
                        F_max[o, t, l] * line_built(y, t, l) * mu_bar[o, t, l, v]
                        for l in lines
                    )
                    - sum(
                        (
                            (-get_maximum_ramp(o, t, u, x))
                            * beta_ramp_underline[o, t, u, v]
                            if t in ramp_hours
                            else 0.0
                        )
                        for u in existing_units
                    )
                    + sum(
                        (
                            get_maximum_ramp(o, t, u, x) * beta_ramp_bar[o, t, u, v]
                            if t in ramp_hours
                            else 0.0
                        )
                        for u in existing_units
                    )
                    + sum(d[o, t, n, v] * sigma[o, t, n, v] for n in real_nodes)
                    + sum(np.pi * rho_underline[o, t, n, v] for n in ac_nodes)
                    + sum(np.pi * rho_bar[o, t, n, v] for n in ac_nodes)
                    + sum(
                        (
                            initial_storage[u][o, to_year(t)]
                            * phi_initial_storage[o, t, u, v]
                            if t in year_first_hours
                            else 0.0
                        )
                        for u in hydro_units
                    )
                    + sum(
                        (
                            inflows[u][o, t] * phi_storage[o, t, u, v]
                            if t in ramp_hours
                            else 0.0
                        )
                        for u in hydro_units
                    )
                    + sum(
                        phi_storage_change_lb[o, t, u, v]
                        * (
                            -initial_storage[u][o, to_year(t)]
                            * storage_change_lb[unit_to_node[u]]
                        )
                        if t in year_last_hours
                        else 0.0
                        for u in hydro_units
                    )
                    + sum(
                        phi_storage_change_ub[o, t, u, v]
                        * initial_storage[u][o, to_year(t)]
                        * storage_change_ub[unit_to_node[u]]
                        if t in year_last_hours
                        else 0.0
                        for u in hydro_units
                    )
                )
                for o in scenarios
                for t in hours
                for v in range(1, iteration + 1)
            )
            - sum(
                emission_targets[y] * beta_emissions[y, v]
                for y in years
                for v in range(1, iteration + 1)
            )
            >= 0.0,
            name="delta_constraint_single_cut_infeasibility_%d_%d"
            % (benders_iteration, solution_number),
        )


def augment_slave(current_iteration, d, yhat, y):
    sp.setObjective(get_investment_cost(xhat, yhat) + theta, GRB.MINIMIZE)

    # Augment the slave problem for the current iteration.
    v = current_iteration

    # Create additional primal variables indexed with the current iteration.
    g, s, f, delta = add_primal_variables(sp, v)

    ramp_hours = get_ramp_hours()  # Hours for which the ramp constraints are defined.

    # Minimum value for the subproblem objective function.
    sp.addConstr(
        theta
        - sum(
            sum(
                discount_factor ** (-to_year(t))
                * sum(C_g[o, t, u] * g[o, t, u, v] for u in units)
                for t in hours
            )
            * weights[o]
            for o in scenarios
        )
        >= 0.0,
        name="minimum_subproblem_objective_%d" % current_iteration,
    )

    # Balance equation. Note that d[o, t, n, v] is input data from the subproblem.
    sp.addConstrs(
        (
            sum(g[o, t, u, v] for u in node_to_unit[n])
            + sum(incidence[l, n] * f[o, t, l, v] for l in lines)
            - (d[o, t, n, v] if n in real_nodes else 0.0)
            == 0.0
            for o in scenarios
            for t in hours
            for n in nodes
        ),
        name="balance_%d" % current_iteration,
    )

    # Generation constraint for the units.
    sp.addConstrs(
        (
            g[o, t, u, v] - get_effective_capacity(o, t, u, x) <= 0.0
            for o in scenarios
            for t in hours
            for u in units
        ),
        name="maximum_generation_%d" % current_iteration,
    )

    # Storage constraints.
    # Initial storage level.
    year_first_hours = [t for t in hours if is_year_first_hour(t)]

    sp.addConstrs(
        (
            s[o, t, u, v] - initial_storage[u][o, to_year(t)] == 0.0
            for o in scenarios
            for t in year_first_hours
            for u in hydro_units
        ),
        name="initial_storage_%d" % current_iteration,
    )

    # Storage evolution.
    sp.addConstrs(
        (
            s[o, t + 1, u, v] - s[o, t, u, v] + g[o, t, u, v] - inflows[u][o, t] == 0.0
            for o in scenarios
            for t in ramp_hours
            for u in hydro_units
        ),
        name="storage_%d" % current_iteration,
    )

    # Final storage.
    year_last_hours = [t for t in hours if is_year_last_hour(t)]
    sp.addConstrs(
        (
            initial_storage[u][o, to_year(t)] * storage_change_lb[unit_to_node[u]]
            - s[o, t, u, v]
            <= 0.0
            for o in scenarios
            for t in year_last_hours
            for u in hydro_units
        ),
        name="final_storage_lb_%d" % current_iteration,
    )

    sp.addConstrs(
        (
            s[o, t, u, v]
            - initial_storage[u][o, to_year(t)] * storage_change_ub[unit_to_node[u]]
            <= 0.0
            for o in scenarios
            for t in year_last_hours
            for u in hydro_units
        ),
        name="final_storage_ub_%d" % current_iteration,
    )

    # Flow constraints for the existing lines.
    # Note: all candidate lines are assumed to be DC.
    sp.addConstrs(
        (
            f[o, t, l, v]
            - B[l]
            * (delta[o, t, get_start_node(l), v] - delta[o, t, get_end_node(l), v])
            == 0.0
            for o in scenarios
            for t in hours
            for l in existing_lines
            if l in ac_lines
        ),
        name="power_flow_existing_%d" % current_iteration,
    )

    # Maximum flows for all lines.
    sp.addConstrs(
        (
            f[o, t, l, v] - line_built(y, t, l) * F_max[o, t, l] <= 0.0
            for o in scenarios
            for t in hours
            for l in lines
        ),
        name="maximum_flow_%d" % current_iteration,
    )

    sp.addConstrs(
        (
            F_min[o, t, l] * line_built(y, t, l) - f[o, t, l, v] <= 0.0
            for o in scenarios
            for t in hours
            for l in lines
        ),
        name="minimum_flow_%d" % current_iteration,
    )

    sp.addConstrs(
        (delta[o, t, ref_node, v] == 0.0 for o in scenarios for t in hours),
        name="reference_node_%d" % current_iteration,
    )

    # Ramping constraints for generation units.
    # Maximum ramp downwards.
    sp.addConstrs(
        (
            -get_maximum_ramp(o, t, u, x) - g[o, t + 1, u, v] + g[o, t, u, v] <= 0.0
            for o in scenarios
            for t in ramp_hours
            for u in units
        ),
        name="max_down_ramp_%d" % current_iteration,
    )

    # Maximum ramp upwards.
    sp.addConstrs(
        (
            g[o, t + 1, u, v] - g[o, t, u, v] - get_maximum_ramp(o, t, u, x) <= 0.0
            for o in scenarios
            for t in ramp_hours
            for u in units
        ),
        name="max_up_ramp_%d" % current_iteration,
    )

    # Emission constraint.
    sp.addConstrs(
        (
            sum(
                weights[o] * g[o, t, u, v] * G_emissions[o, t, u]
                for o in scenarios
                for t in to_hours(y)
                for u in units
            )
            - emission_targets[y]
            <= 0.0
            for y in years
        ),
        name="maximum_emissions_%d" % current_iteration,
    )

    # Voltage angle constraints.
    sp.addConstrs(
        (
            delta[o, t, n, v] <= np.pi
            for o in scenarios
            for t in hours
            for n in ac_nodes
        ),
        name="voltage_angle_ub_%d" % current_iteration,
    )

    sp.addConstrs(
        (
            -delta[o, t, n, v] <= np.pi
            for o in scenarios
            for t in hours
            for n in ac_nodes
        ),
        name="voltage_angle_lb_%d" % current_iteration,
    )

    return g, s


def obtain_constraints(current_iteration):
    # Obtain relevant constraints in a nice data structure.
    sigma_constrs = dict()
    beta_bar_constrs = dict()
    mu_underline_constrs = dict()
    mu_bar_constrs = dict()
    beta_ramp_underline_constrs = dict()
    beta_ramp_bar_constrs = dict()
    beta_emissions_constrs = dict()
    rho_underline_constrs = dict()
    rho_bar_constrs = dict()
    phi_initial_storage_constrs = dict()
    phi_storage_constrs = dict()
    phi_storage_change_lb_constrs = dict()
    phi_storage_change_ub_constrs = dict()

    year_first_hours = [t for t in hours if is_year_first_hour(t)]
    year_last_hours = [t for t in hours if is_year_last_hour(t)]
    ramp_hours = get_ramp_hours()

    for v in range(1, current_iteration + 1):
        for o in scenarios:
            for t in hours:
                for n in nodes:
                    name = "balance_%d[%d,%d,%d]" % (v, o, t, n)
                    sigma_constrs[o, t, n, v] = sp.getConstrByName(name)

                    if n in ac_nodes:
                        lb_name = "voltage_angle_lb_%d[%d,%d,%d]" % (v, o, t, n)
                        ub_name = "voltage_angle_ub_%d[%d,%d,%d]" % (v, o, t, n)

                        rho_underline_constrs[o, t, n, v] = sp.getConstrByName(lb_name)
                        rho_bar_constrs[o, t, n, v] = sp.getConstrByName(ub_name)

                for u in existing_units:
                    name = "maximum_generation_%d[%d,%d,%d]" % (v, o, t, u)
                    beta_bar_constrs[o, t, u, v] = sp.getConstrByName(name)

                    if t in ramp_hours:
                        down_name = "max_down_ramp_%d[%d,%d,%d]" % (v, o, t, u)
                        up_name = "max_up_ramp_%d[%d,%d,%d]" % (v, o, t, u)
                        beta_ramp_underline_constrs[o, t, u, v] = sp.getConstrByName(
                            down_name
                        )
                        beta_ramp_bar_constrs[o, t, u, v] = sp.getConstrByName(up_name)

                        if u in hydro_units:
                            storage_name = "storage_%d[%d,%d,%d]" % (v, o, t, u)
                            phi_storage_constrs[o, t, u, v] = sp.getConstrByName(
                                storage_name
                            )

                    if t in year_first_hours:
                        if u in hydro_units:
                            initial_storage_name = "initial_storage_%d[%d,%d,%d]" % (
                                v,
                                o,
                                t,
                                u,
                            )
                            phi_initial_storage_constrs[
                                o, t, u, v
                            ] = sp.getConstrByName(initial_storage_name)

                    if t in year_last_hours:
                        if u in hydro_units:
                            storage_change_lb_name = "final_storage_lb_%d[%d,%d,%d]" % (
                                v,
                                o,
                                t,
                                u,
                            )
                            phi_storage_change_lb_constrs[
                                o, t, u, v
                            ] = sp.getConstrByName(storage_change_lb_name)

                            storage_change_ub_name = "final_storage_ub_%d[%d,%d,%d]" % (
                                v,
                                o,
                                t,
                                u,
                            )
                            phi_storage_change_ub_constrs[
                                o, t, u, v
                            ] = sp.getConstrByName(storage_change_ub_name)

                for l in lines:
                    min_name = "minimum_flow_%d[%d,%d,%d]" % (v, o, t, l)
                    max_name = "maximum_flow_%d[%d,%d,%d]" % (v, o, t, l)
                    mu_underline_constrs[o, t, l, v] = sp.getConstrByName(min_name)
                    mu_bar_constrs[o, t, l, v] = sp.getConstrByName(max_name)

            for y in years:
                name = "maximum_emissions_%d[%d]" % (v, y)
                beta_emissions_constrs[y, v] = sp.getConstrByName(name)

    all_constrs = (
        sigma_constrs,
        beta_bar_constrs,
        mu_underline_constrs,
        mu_bar_constrs,
        beta_ramp_underline_constrs,
        beta_ramp_bar_constrs,
        beta_emissions_constrs,
        rho_underline_constrs,
        rho_bar_constrs,
        phi_initial_storage_constrs,
        phi_storage_constrs,
        phi_storage_change_lb_constrs,
        phi_storage_change_ub_constrs,
    )

    updatable_constrs = (mu_underline_constrs, mu_bar_constrs)

    return all_constrs, updatable_constrs


def update_slave(updatable_constrs, current_iteration, x, yhat, y):
    # Update constraints involving the value of x and y. Be careful with the order when unpacking.
    sp.setObjective(get_investment_cost(xhat, yhat) + theta, GRB.MINIMIZE)

    mu_underline_constrs, mu_bar_constrs = updatable_constrs

    for o in scenarios:
        for t in hours:
            for v in range(1, current_iteration + 1):
                for l in candidate_lines:
                    mu_underline_constrs[o, t, l, v].RHS = line_built(y, t, l) * (
                        -F_min[o, t, l]
                    )
                    mu_bar_constrs[o, t, l, v].RHS = (
                        line_built(y, t, l) * F_max[o, t, l]
                    )


def get_investment_and_availability_decisions(initial=False, many_solutions=False):
    # Read current investments to generation and transmission and whether the units and lines are
    # operational at some time point.
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


def get_dual_variables(all_constrs):
    # Get dual variable values of all (relevant) constraints of the slave problem.
    def _get_dual_values(constrs):
        dual_values = dict()

        for key, val in constrs.items():
            dual_values[key] = val.getAttr("Pi")

        return dual_values

    return [_get_dual_values(constrs) for constrs in all_constrs]


def get_unbounded_ray(all_constrs):
    # Get unbounded rays of all (relevant) constraints of the slave problem.
    def _get_dual_values(constrs):
        dual_values = dict()

        for key, val in constrs.items():
            dual_values[key] = val.getAttr("FarkasDual")

        return dual_values

    return [_get_dual_values(constrs) for constrs in all_constrs]


def get_solution_hash(yhat, y):
    # Generate a hash of the solution.
    solution_hash = ""

    for t in years:
        for l in candidate_lines:
            solution_hash += str(int(yhat[t, l])) + str(int(y[t, l]))

    return solution_hash


def solve_master_problem(current_iteration, d):
    # Solve the master problem using Benders.

    # Dummy return values on the first iteration.
    if current_iteration == 0:
        return -np.inf, None, None

    max_iterations = 9999
    threshold = 1e-8
    bad_threshold = -1e-3
    separator = "-" * 50

    lb = -np.inf
    ub = np.inf

    sp.Params.InfUnbdInfo = 1

    duplicates = 0

    # Initialize Benders.
    initialize_master()
    yhat, y = get_initial_transmission_investments()
    g, s = augment_slave(current_iteration, d, yhat, y)
    sp.update()
    sp.optimize()
    all_constrs, updatable_constrs = obtain_constraints(current_iteration)
    dual_values = get_dual_variables(all_constrs)
    augment_master(dual_values, current_iteration, 0, 0, d)

    solution_hashes = set()

    for iteration in range(1, max_iterations):
        print(separator)
        print("Starting Benders iteration:", iteration)

        print("Solving Benders master problem.")
        mp.update()
        mp.optimize()
        print(separator)

        if mp.Status != GRB.OPTIMAL:
            raise RuntimeError("Benders master problem not optimal.")

        lb = mp.objVal

        # Explore all master problem solutions.
        many_solutions = mp.SolCount > 1
        sol_indices = list(range(mp.SolCount))

        for k in sol_indices:
            mp.Params.SolutionNumber = k

            if mp.SolCount == 0:
                break

            xhat, yhat, x, y = get_investment_and_availability_decisions(
                initial=False, many_solutions=many_solutions
            )

            solution_hash = get_solution_hash(yhat, y)

            if k > 0 and solution_hash in solution_hashes:
                continue
            elif k == 0 and solution_hash in solution_hashes:
                duplicates += 1
                print("Current duplicates: %d. :(" % duplicates)

            solution_hashes.add(solution_hash)

            update_slave(updatable_constrs, current_iteration, x, yhat, y)

            print(separator)
            print("Solving Benders slave problem.")
            sp.update()
            sp.optimize()
            print(separator)

            if sp.Status != GRB.OPTIMAL:
                unbounded_ray = get_unbounded_ray(all_constrs)
                augment_master(
                    unbounded_ray, current_iteration, iteration, k, d, unbounded=True
                )
                continue

            ub = sp.objVal

            gap = compute_objective_gap(lb, ub)

            if k == 0 and gap < threshold:
                if not gap >= bad_threshold:
                    raise RuntimeError("lb (%f) > ub (%f) in Benders." % (lb, ub))

                print("Took %d Benders iterations." % (iteration + 1))

                return lb, g, s

            dual_values = get_dual_variables(all_constrs)
            augment_master(dual_values, current_iteration, iteration, k, d)

    raise RuntimeError("Max iterations hit in Benders. LB: %f, UB: %f" % (lb, ub))


master_problem = sp
