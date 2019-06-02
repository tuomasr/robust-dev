# Master problem Benders formulation with the assumption that candidate lines are AC.

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
    hydro_units,
    G_max,
    F_max,
    F_min,
    B,
    ref_node,
    G_ramp_max,
    G_ramp_min,
    G_emissions,
    C_g,
    initial_storage,
    inflows,
    incidence,
    weights,
    node_to_unit,
    emission_targets,
    C_x,
    C_y,
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
    get_candidate_generation_capacity,
    compute_objective_gap,
    is_year_first_hour,
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
        nodes,
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


def get_initial_transmission_investments():
    initial_yhat = dict()
    initial_y = dict()

    initial_transmission_investment = 0.0

    for y in years:
        for l in candidate_lines:
            if y == 0:
                initial_yhat[y, l] = initial_transmission_investment
            else:
                initial_yhat[y, l] = 0.0

            initial_y[y, l] = initial_transmission_investment

    return initial_yhat, initial_y


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

delta = mp.addVars(scenarios, years, name="delta", lb=0.0)

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


def augment_master(dual_values, iteration, benders_iteration, kk, d):
    # Augment the Benders master problem with a new cut from the slave problem.

    # Unpack the dual values. Be careful with the order.
    sigma, beta_bar, mu_underline, mu_bar, beta_ramp_underline, beta_ramp_bar, beta_emissions, rho_bar, rho_underline, phi_initial_storage, phi_storage, rho_flow_underline, rho_flow_bar, omega_underline, omega_bar = (
        dual_values
    )

    # Enable/disable multicut version of Benders. Single cut seems to be faster.
    multicut = False

    year_first_hours = [t for t in hours if is_year_first_hour(t)]
    ramp_hours = get_ramp_hours()

    mp.update()

    if not multicut:
        mp.addConstr(
            delta
            - sum(
                (
                    +sum(G_max[o, t, u] * beta_bar[o, t, u, v] for u in units)
                    - sum(
                        2
                        * np.pi
                        * (-1.0 if l in candidate_lines else 1.0)
                        * line_built(y, t, l)
                        * (rho_flow_underline[o, t, l, v] + rho_flow_bar[o, t, l, v])
                        for l in candidate_lines
                    )
                    + sum(
                        2
                        * np.pi
                        * (-1.0 if l in candidate_lines else 1.0)
                        * line_built(y, t, l)
                        * (omega_underline[o, t, l, v] + omega_bar[o, t, l, v])
                        for l in candidate_lines
                    )
                    + sum(
                        2
                        * np.pi
                        * (omega_underline[o, t, l, v] + omega_bar[o, t, l, v])
                        for l in candidate_lines
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
                            G_ramp_min[o, t, u] * beta_ramp_underline[o, t, u, v]
                            if t in ramp_hours
                            else 0.0
                        )
                        for u in units
                    )
                    + sum(
                        (
                            G_ramp_max[o, t, u] * beta_ramp_bar[o, t, u, v]
                            if t in ramp_hours
                            else 0.0
                        )
                        for u in units
                    )
                    + sum(d[t, n, v] * sigma[o, t, n, v] for n in real_nodes)
                    + sum(np.pi * rho_underline[o, t, n, v] for n in nodes)
                    + sum(np.pi * rho_bar[o, t, n, v] for n in nodes)
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
                )
                for o in scenarios
                for t in hours
                for v in range(1, iteration + 1)
            )
            - sum(
                emission_targets[y] * beta_emissions[o, y, v]
                for o in scenarios
                for y in years
                for v in range(1, iteration + 1)
            )
            >= 0.0,
            name="delta_constraint_single_cut_%d_%d_%d" % (iteration, benders_iteration, kk)
        )
    else:
        raise NotImplementedError()


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
            sum(sum(C_g[o, t, u] * g[o, t, u, v] for u in units) for t in hours)
            * weights[o]
            for o in scenarios
        )
        >= 0.0,
        name="minimum_subproblem_objective_%d" % current_iteration,
    )

    # Balance equation. Note that d[n, v] is input data from the subproblem.
    sp.addConstrs(
        (
            sum(g[o, t, u, v] for u in node_to_unit[n])
            + sum(incidence[l, n] * f[o, t, l, v] for l in lines)
            - (d[t, n, v] if n in real_nodes else 0.0)
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
            g[o, t, u, v] - get_candidate_generation_capacity(t, u, x) <= 0.0
            for o in scenarios
            for t in hours
            for u in candidate_units
        ),
        name="maximum_candidate_generation_%d" % current_iteration,
    )

    sp.addConstrs(
        (
            g[o, t, u, v] - G_max[o, t, u] <= 0.0
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

    # Flow constraints for the lines.
    # Existing lines.
    sp.addConstrs(
        (
            f[o, t, l, v]
            - B[l]
            * (delta[o, t, get_start_node(l), v] - delta[o, t, get_end_node(l), v])
            == 0.0
            for o in scenarios
            for t in hours
            for l in existing_lines
        ),
        name="power_flow_existing_%d" % current_iteration,
    )

    # Candidate lines. Linearize y * (delta1 - delta2) = y * q = z.
    z = sp.addVars(
        scenarios,
        hours,
        candidate_lines,
        [v],
        lb=-GRB.INFINITY,
        ub=GRB.INFINITY,
        name="linearization_z_%d" % current_iteration,
    )
    q = sp.addVars(
        scenarios,
        hours,
        candidate_lines,
        [v],
        lb=-GRB.INFINITY,
        ub=GRB.INFINITY,
        name="linearization_q_%d" % current_iteration,
    )
    q_tilde = sp.addVars(
        scenarios,
        hours,
        candidate_lines,
        [v],
        lb=-GRB.INFINITY,
        ub=GRB.INFINITY,
        name="linearization_q_tilde_%d" % current_iteration,
    )

    sp.addConstrs(
        (
            q[o, t, l, v]
            == delta[o, t, get_start_node(l), v] - delta[o, t, get_end_node(l), v]
            for o in scenarios
            for t in hours
            for l in candidate_lines
        ),
        name="linearization_q_definition_%d" % current_iteration,
    )

    sp.addConstrs(
        (
            z[o, t, l, v] == q[o, t, l, v] - q_tilde[o, t, l, v]
            for o in scenarios
            for t in hours
            for l in candidate_lines
        ),
        name="linearization_z_definition_%d" % current_iteration,
    )

    sp.addConstrs(
        (
            line_built(y, t, l) * (-2 * np.pi) - z[o, t, l, v] <= 0.0
            for o in scenarios
            for t in hours
            for l in candidate_lines
        ),
        name="linearization_z_lb_%d" % current_iteration,
    )

    sp.addConstrs(
        (
            z[o, t, l, v] - line_built(y, t, l) * 2 * np.pi <= 0.0
            for o in scenarios
            for t in hours
            for l in candidate_lines
        ),
        name="linearization_z_ub_%d" % current_iteration,
    )

    sp.addConstrs(
        (
            (1.0 - line_built(y, t, l)) * (-2 * np.pi) - q_tilde[o, t, l, v] <= 0.0
            for o in scenarios
            for t in hours
            for l in candidate_lines
        ),
        name="linearization_q_tilde_lb_%d" % current_iteration,
    )

    sp.addConstrs(
        (
            q_tilde[o, t, l, v] - (1.0 - line_built(y, t, l)) * 2 * np.pi <= 0.0
            for o in scenarios
            for t in hours
            for l in candidate_lines
        ),
        name="linearization_q_tilde_ub_%d" % current_iteration,
    )

    # Candidate line flow.
    sp.addConstrs(
        (
            f[o, t, l, v] - B[l] * z[o, t, l, v] == 0.0
            for o in scenarios
            for t in hours
            for l in candidate_lines
        ),
        name="power_flow_candidate_%d" % current_iteration,
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
            G_ramp_min[o, t, u] - g[o, t + 1, u, v] + g[o, t, u, v] <= 0.0
            for o in scenarios
            for t in ramp_hours
            for u in units
        ),
        name="max_down_ramp_%d" % current_iteration,
    )

    # Maximum ramp upwards.
    sp.addConstrs(
        (
            g[o, t + 1, u, v] - g[o, t, u, v] - G_ramp_max[o, t, u] <= 0.0
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
                g[o, t, u, v] * G_emissions[o, t, u] for u in units for t in to_hours(y)
            )
            - emission_targets[y]
            <= 0.0
            for o in scenarios
            for y in years
        ),
        name="maximum_emissions_%d" % current_iteration,
    )

    # Voltage angle constraints.
    sp.addConstrs(
        (delta[o, t, n, v] <= np.pi for o in scenarios for t in hours for n in nodes),
        name="voltage_angle_ub_%d" % current_iteration,
    )

    sp.addConstrs(
        (-delta[o, t, n, v] <= np.pi for o in scenarios for t in hours for n in nodes),
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
    rho_flow_underline_constrs = dict()
    rho_flow_bar_constrs = dict()
    omega_underline_constrs = dict()
    omega_bar_constrs = dict()

    year_first_hours = [t for t in hours if is_year_first_hour(t)]
    ramp_hours = get_ramp_hours()

    for v in range(1, current_iteration + 1):
        for o in scenarios:
            for t in hours:
                for n in nodes:
                    name = "balance_%d[%d,%d,%d]" % (v, o, t, n)
                    sigma_constrs[o, t, n, v] = sp.getConstrByName(name)

                    lb_name = "voltage_angle_lb_%d[%d,%d,%d]" % (v, o, t, n)
                    ub_name = "voltage_angle_ub_%d[%d,%d,%d]" % (v, o, t, n)

                    rho_underline_constrs[o, t, n, v] = sp.getConstrByName(lb_name)
                    rho_bar_constrs[o, t, n, v] = sp.getConstrByName(ub_name)

                for u in units:
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

                for l in lines:
                    min_name = "minimum_flow_%d[%d,%d,%d]" % (v, o, t, l)
                    max_name = "maximum_flow_%d[%d,%d,%d]" % (v, o, t, l)
                    mu_underline_constrs[o, t, l, v] = sp.getConstrByName(min_name)
                    mu_bar_constrs[o, t, l, v] = sp.getConstrByName(max_name)

                    if l in candidate_lines:
                        z_lb_name = "linearization_z_lb_%d[%d,%d,%d]" % (v, o, t, l)
                        z_ub_name = "linearization_z_ub_%d[%d,%d,%d]" % (v, o, t, l)
                        q_tilde_lb_name = "linearization_q_tilde_lb_%d[%d,%d,%d]" % (
                            v,
                            o,
                            t,
                            l,
                        )
                        q_tilde_ub_name = "linearization_q_tilde_ub_%d[%d,%d,%d]" % (
                            v,
                            o,
                            t,
                            l,
                        )

                        rho_flow_underline_constrs[o, t, l, v] = sp.getConstrByName(
                            z_lb_name
                        )
                        rho_flow_bar_constrs[o, t, l, v] = sp.getConstrByName(z_ub_name)
                        omega_underline_constrs[o, t, l, v] = sp.getConstrByName(
                            q_tilde_lb_name
                        )
                        omega_bar_constrs[o, t, l, v] = sp.getConstrByName(
                            q_tilde_ub_name
                        )

            for y in years:
                name = "maximum_emissions_%d[%d,%d]" % (v, o, y)
                beta_emissions_constrs[o, y, v] = sp.getConstrByName(name)

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
        rho_flow_underline_constrs,
        rho_flow_bar_constrs,
        omega_underline_constrs,
        omega_bar_constrs,
    )

    updatable_constrs = (
        mu_underline_constrs,
        mu_bar_constrs,
        rho_flow_underline_constrs,
        rho_flow_bar_constrs,
        omega_underline_constrs,
        omega_bar_constrs,
    )

    return all_constrs, updatable_constrs


def update_slave(updatable_constrs, current_iteration, yhat, y):
    # Update constraints involving the value of x and y. Be careful with the order when unpacking.
    sp.setObjective(get_investment_cost(xhat, yhat) + theta, GRB.MINIMIZE)

    mu_underline_constrs, mu_bar_constrs, rho_flow_underline_constrs, rho_flow_bar_constrs, omega_underline_constrs, omega_bar_constrs = (
        updatable_constrs
    )

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

                    rho_flow_underline_constrs[o, t, l, v].RHS = (
                        line_built(y, t, l) * 2 * np.pi
                    )
                    rho_flow_bar_constrs[o, t, l, v].RHS = (
                        line_built(y, t, l) * 2 * np.pi
                    )

                    omega_underline_constrs[o, t, l, v].RHS = (
                        (1.0 - line_built(y, t, l)) * 2 * np.pi
                    )
                    omega_bar_constrs[o, t, l, v].RHS = (
                        (1.0 - line_built(y, t, l)) * 2 * np.pi
                    )


def get_investment_and_availability_decisions(initial=False, many_solutions=False):
    # Read current investments to generation and transmission and whether the units and lines are
    # operational at some time point.
    current_xhat = dict()
    current_yhat = dict()

    current_x = dict()
    current_y = dict()

    initial_generation_investment = 500.0
    initial_transmission_investment = 1.0

    for t in years:
        for u in candidate_units:
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

    emissions = np.zeros((len(scenarios), len(years)))

    for o in scenarios:
        for y in years:
            emissions[o, y] = sum(
                g[o, t, u, i].x * G_emissions[o, t, u]
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
    threshold = 1e-6
    bad_threshold = -1e-3
    separator = "-" * 50

    lb = -np.inf
    ub = np.inf

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

        yhats = []
        ys = []

        for k in sol_indices:
            mp.Params.SolutionNumber = k

            mp.update()

            if mp.SolCount == 0:
                break

            _, yhat, _, y = get_investment_and_availability_decisions(
                initial=False, many_solutions=many_solutions
            )

            solution_hash = get_solution_hash(yhat, y)

            if k > 0 and solution_hash in solution_hashes:
                continue

            solution_hashes.add(solution_hash)

            yhats.append(yhat)
            ys.append(y)

        for kk, (yhat, y) in enumerate(zip(yhats, ys)):
            update_slave(updatable_constrs, current_iteration, yhat, y)

            print(separator)
            print("Solving Benders slave problem.")
            sp.update()
            sp.optimize()
            print(separator)

            if sp.Status != GRB.OPTIMAL:
                raise RuntimeError("Benders slave problem not optimal.")

            ub = sp.objVal

            gap = compute_objective_gap(lb, ub)

            if kk == 0 and gap < threshold:
                if not gap >= bad_threshold:
                    raise RuntimeError("lb (%f) > ub (%f) in Benders." % (lb, ub))

                print("Took %d Benders iterations." % (iteration + 1))

                return (ub + lb) / 2, g, s

            dual_values = get_dual_variables(all_constrs)
            augment_master(dual_values, current_iteration, iteration, kk, d)

    raise RuntimeError("Max iterations hit in Benders. LB: %f, UB: %f" % (lb, ub))
