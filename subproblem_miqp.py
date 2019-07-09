# Subproblem MIQP formulation.
# Note: This assumes that all candidates lines are DC.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gurobipy import *
import numpy as np

from common_data import (
    hours,
    years,
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
    ac_lines,
    ac_nodes,
    hydro_units,
    G_max,
    F_max,
    F_min,
    B,
    ref_node,
    incidence,
    weights,
    C_g,
    initial_storage,
    inflows,
    storage_change_lb,
    storage_change_ub,
    unit_to_node,
    emission_targets,
    G_emissions,
    discount_factor,
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
    get_effective_capacity,
    get_maximum_ramp,
    get_ramp_hours,
)

num_hours = len(hours)

# Problem-specific data: Demand at each node in each time hour and uncertainty in it.
nominal_demand = load[:, : len(real_nodes)]
demand_increase = uncertainty_demand_increase * np.ones_like(nominal_demand)

# Create the model for the subproblem.
m = Model("subproblem")
m.Params.Method = subproblem_method

if enable_custom_configuration:
    for parameter, value in GRB_PARAMS:
        m.setParam(parameter, value)

K = 1000.0

# Demand is fixed for the dummy nodes that do not contain any load.
# For real nodes that contain load, add hour- and nodewise uncertain demand variables and
# nodewise binary variables for deviating from the nominal demand values.
d = m.addVars(
    hours,
    real_nodes,
    name="uncertain_demand",
    lb=0.0,
    ub=nominal_demand + demand_increase,
)
# Deviations are possible only for real nodes that contain load.
w = m.addVars(real_nodes, name="demand_deviation", vtype=GRB.BINARY)

# Maximum and minimum generation dual variables.
ub1 = GRB.INFINITY
lb2 = -GRB.INFINITY
ub2 = GRB.INFINITY

beta_bar = m.addVars(
    scenarios, hours, units, name="dual_maximum_generation", lb=0.0, ub=ub1
)
beta_underline = m.addVars(
    scenarios, hours, units, name="dual_minimum_generation", lb=0.0, ub=ub1
)

ramp_hours = get_ramp_hours()

# Storage dual variables.
year_first_hours = [t for t in hours if is_year_first_hour(t)]
year_last_hours = [t for t in hours if is_year_last_hour(t)]

beta_storage_underline = m.addVars(
    scenarios, hours, hydro_units, name="dual_minimum_storage", lb=0.0, ub=ub1
)

phi_initial_storage = m.addVars(
    scenarios,
    year_first_hours,
    hydro_units,
    name="dual_initial_storage",
    lb=lb2,
    ub=ub2,
)

phi_storage = m.addVars(
    scenarios, ramp_hours, hydro_units, name="dual_storage", lb=lb2, ub=ub2
)

phi_storage_change_lb = m.addVars(
    scenarios,
    year_last_hours,
    hydro_units,
    name="dual_storage_change_lb",
    lb=0.0,
    ub=ub1,
)

phi_storage_change_ub = m.addVars(
    scenarios,
    year_last_hours,
    hydro_units,
    name="dual_storage_change_ub",
    lb=0.0,
    ub=ub1,
)

# Maximum up- and down ramp dual variables.
beta_ramp_bar = m.addVars(
    scenarios, ramp_hours, units, name="dual_maximum_ramp_upwards", lb=0.0, ub=ub1
)
beta_ramp_underline = m.addVars(
    scenarios, ramp_hours, units, name="dual_maximum_ramp_downwards", lb=0.0, ub=ub1
)

# Maximum emissions dual variables.
beta_emissions = m.addVars(
    scenarios, years, name="dual_maximum_emissions", lb=0.0, ub=ub1
)

# Transmission flow dual variables.
phi = m.addVars(scenarios, hours, ac_lines, name="dual_flow", lb=lb2, ub=ub2)

# Maximum and minimum transmission flow dual variables.
mu_bar = m.addVars(scenarios, hours, lines, name="dual_maximum_flow", lb=0.0, ub=ub1)
mu_underline = m.addVars(
    scenarios, hours, lines, name="dual_minimum_flow", lb=0.0, ub=ub1
)

# Dual variables for voltage angle bounds.
mu_angle_bar = m.addVars(
    scenarios, hours, ac_nodes, name="dual_maximum_angle", lb=0.0, ub=ub1
)
mu_angle_underline = m.addVars(
    scenarios, hours, ac_nodes, name="dual_minimum_angle", lb=0.0, ub=ub1
)

# Dual variable for the reference node voltage angle.
eta = m.addVars(scenarios, hours, name="dual_reference_node_angle", lb=lb2, ub=ub2)

# Balance equation dual (i.e. price).
lambda_ = m.addVars(scenarios, hours, nodes, name="dual_balance", lb=-K, ub=K)


def get_objective(x, y):
    # Define subproblem objective function for fixed x and y (unit and line investments).
    obj = sum(
        sum(
            w[n] * demand_increase[t, n] * lambda_[o, t, n]
            + lambda_[o, t, n] * nominal_demand[t, n]
            for n in real_nodes
        )
        - sum(beta_bar[o, t, u] * get_effective_capacity(o, t, u, x) for u in units if unit_built(x, t, u))
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
            phi_storage_change_lb[o, t, u] * (-initial_storage[u][o, to_year(t)] * storage_change_lb[unit_to_node[u]])
            if t in year_last_hours
            else 0.0
            for u in hydro_units
        )
        - sum(
            phi_storage_change_ub[o, t, u] * initial_storage[u][o, to_year(t)] * storage_change_ub[unit_to_node[u]]
            if t in year_last_hours
            else 0.0
            for u in hydro_units
        )
        - sum(
            (mu_bar[o, t, l] * F_max[o, t, l] - mu_underline[o, t, l] * F_min[o, t, l])
            for l in lines
            if line_built(y, t, l)
        )
        - sum(
            np.pi * (mu_angle_bar[o, t, n] + mu_angle_underline[o, t, n]) for n in ac_nodes
        )
        - sum(
            beta_ramp_bar[o, t, u] * get_maximum_ramp(o, t, u, x) if t in ramp_hours else 0.0
            for u in units
            if unit_built(x, t, u)
        )
        + sum(
            beta_ramp_underline[o, t, u] * (-get_maximum_ramp(o, t, u, x))
            if t in ramp_hours
            else 0.0
            for u in units
            if unit_built(x, t, u)
        )
        for o in scenarios
        for t in hours
    )

    obj -= sum(
        beta_emissions[o, y] * emission_targets[y] for o in scenarios for y in years
    )

    return obj


def set_subproblem_objective(x, y):
    # Set objective function for the subproblem for fixed x and y (unit and line investments).
    obj = get_objective(x, y)

    m.setObjective(obj, GRB.MAXIMIZE)


# Constraints defining the uncertainty set.
m.addConstr(
    sum(w[n] for n in real_nodes) - uncertainty_budget == 0.0,
    name="uncertainty_set_budget",
)

# Storage dual constraints.
m.addConstrs(
    (
        (phi_initial_storage[o, t, u] if is_year_first_hour(t) else 0.0)
        + beta_storage_underline[o, t, u]
        - (phi_storage[o, t, u] if not is_year_last_hour(t) else 0.0)
        + (phi_storage[o, t - 1, u] if not is_year_first_hour(t) else 0.0)
        + (phi_storage_change_lb[o, t, u] if is_year_last_hour(t) else 0.0)
        - (phi_storage_change_ub[o, t, u] if is_year_last_hour(t) else 0.0)
        == 0.0
        for o in scenarios
        for t in hours
        for u in hydro_units
    ),
    name="storage_dual_constraint",
)


def set_dependent_constraints(x, y):
    constraint_names = [
        "voltage_angle_dual_constraint",
        "flow_dual_constraint",
        "generation_dual_constraint",
    ]

    for constraint_name in constraint_names:
        existing_constraints = [
            c for c in m.getConstrs() if constraint_name in c.ConstrName
        ]

        if existing_constraints:
            m.remove(existing_constraints)
            m.update()

    m.addConstrs(
        (
            -sum(
                (B[l] * phi[o, t, l] if l in existing_lines and l in ac_lines else 0.0)
                for l in get_lines_starting(n)
            )
            + sum(
                (B[l] * phi[o, t, l] if l in existing_lines and l in ac_lines else 0.0)
                for l in get_lines_ending(n)
            )
            - mu_angle_bar[o, t, n]
            + mu_angle_underline[o, t, n]
            + (eta[o, t] if n == ref_node else 0.0)
            == 0.0
            for o in scenarios
            for t in hours
            for n in ac_nodes
        ),
        name="voltage_angle_dual_constraint",
    )

    m.addConstrs(
        (
            (
                sum(incidence[l, n] * lambda_[o, t, n] for n in nodes)
                + (phi[o, t, l] if l in existing_lines and l in ac_lines else 0.0)
                - mu_bar[o, t, l]
                + mu_underline[o, t, l]
            )
            == 0.0
            for o in scenarios
            for t in hours
            for l in lines
            if line_built(y, t, l)
        ),
        name="flow_dual_constraint",
    )

    m.addConstrs(
        (
            lambda_[o, t, unit_to_node[u]]
            - beta_bar[o, t, u]
            + beta_underline[o, t, u]
            + (phi_storage[o, t, u] if t in ramp_hours and u in hydro_units else 0.0)
            - (beta_ramp_bar[o, t - 1, u] if not is_year_first_hour(t) else 0.0)
            + (beta_ramp_bar[o, t, u] if not is_year_last_hour(t) else 0.0)
            + (beta_ramp_underline[o, t - 1, u] if not is_year_first_hour(t) else 0.0)
            - (beta_ramp_underline[o, t, u] if not is_year_last_hour(t) else 0.0)
            - (beta_emissions[o, to_year(t)] * G_emissions[o, t, u])
            - discount_factor ** (-to_year(t)) * C_g[o, t, u] * weights[o]
            == 0.0
            for o in scenarios
            for t in hours
            for u in units
            if unit_built(x, t, u)
        ),
        name="generation_dual_constraint",
    )


def solve_subproblem(x, y):
    set_dependent_constraints(x, y)
    set_subproblem_objective(x, y)
    m.optimize()

    prices = dict()
    for s in scenarios:
        for y in years:
            print("beta_emissions[%d, %d]:" % (s, y), beta_emissions[s, y].x)
            prices[s, y] = beta_emissions[s, y].x

    return m.objVal, prices


def get_uncertain_variables():
    # Get the names and values of uncertain variables of the subproblem.
    names = np.array([w[n].varName for n in real_nodes])
    values = np.array(
        [
            float(int(w[n].x)) * demand_increase[:, n] + nominal_demand[:, n]
            for n in real_nodes
        ]
    )
    values = np.transpose(values)

    return names, values


def get_uncertainty_decisions():
    # Get a vector of the uncertainty decisions w.
    values = np.array([float(int(w[n].x)) for n in real_nodes])
    return values
