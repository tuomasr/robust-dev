# Subproblem MILP/MIQP formulation.
# Note: This assumes that all candidates lines are DC.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gurobipy import GRB, Model
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
    existing_lines,
    ac_lines,
    ac_nodes,
    hydro_units,
    F_max,
    F_min,
    B,
    ref_node,
    incidence,
    weights,
    C_g,
    inflows,
    unit_to_node,
    emission_targets,
    G_emissions,
    discount_factor,
    subproblem_method,
    enable_custom_configuration_subproblem,
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
nominal_demand = load
demand_increase = uncertainty_demand_increase

# Create the model for the subproblem.
m = Model("subproblem")
m.Params.Method = subproblem_method

if enable_custom_configuration_subproblem:
    for parameter, value in GRB_PARAMS:
        m.setParam(parameter, value)

K = 9999.9

# Demand is fixed for the dummy nodes that do not contain any load.
# For real nodes that contain load, add hour- and nodewise uncertain demand variables and
# nodewise binary variables for deviating from the nominal demand values.
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

# Maximum up- and down ramp dual variables.
beta_ramp_bar = m.addVars(
    scenarios, ramp_hours, units, name="dual_maximum_ramp_upwards", lb=0.0, ub=ub1
)
beta_ramp_underline = m.addVars(
    scenarios, ramp_hours, units, name="dual_maximum_ramp_downwards", lb=0.0, ub=ub1
)

# Maximum emissions dual variables.
beta_emissions = m.addVars(years, name="dual_maximum_emissions", lb=0.0, ub=ub1)

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

linearization_variables_created = False
z, lambda_tilde = None, None


def get_or_create_linearization_variables():
    # Variables for linearizing bilinear terms lambda_[o, t, n] * w[n].
    global linearization_variables_created, z, lambda_tilde
    if not linearization_variables_created:
        z = m.addVars(
            scenarios,
            hours,
            real_nodes,
            name="linearization_z",
            lb=-GRB.INFINITY,
            ub=GRB.INFINITY,
        )
        lambda_tilde = m.addVars(
            scenarios,
            hours,
            real_nodes,
            name="linearization_lambda_tilde",
            lb=-GRB.INFINITY,
            ub=GRB.INFINITY,
        )
        linearization_variables_created = True

    return z, lambda_tilde


def get_objective(x, y, milp):
    # Define subproblem objective function for fixed x and y (unit and line investments).
    if milp:
        z, _ = get_or_create_linearization_variables()
        term1 = sum(
            (
                z[o, t, n] * demand_increase[o, t, n] + lambda_[o, t, n] * nominal_demand[o, t, n]
            )
            for o in scenarios
            for t in hours
            for n in real_nodes
        )
    else:
        term1 = sum(
            (
                w[n] * demand_increase[o, t, n] * lambda_[o, t, n]
                + lambda_[o, t, n] * nominal_demand[o, t, n]
            )
            for o in scenarios
            for t in hours
            for n in real_nodes
        )

    obj = term1 + sum(
        - sum(
            beta_bar[o, t, u] * get_effective_capacity(o, t, u, x)
            for u in units
            if unit_built(x, t, u)
        )
        - sum(
            (mu_bar[o, t, l] * F_max[o, t, l] - mu_underline[o, t, l] * F_min[o, t, l])
            for l in lines
            if line_built(y, t, l)
        )
        - sum(
            np.pi * (mu_angle_bar[o, t, n] + mu_angle_underline[o, t, n])
            for n in ac_nodes
        )
        - sum(
            beta_ramp_bar[o, t, u] * get_maximum_ramp(o, t, u, x)
            if t in ramp_hours
            else 0.0
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

    obj -= sum(beta_emissions[y] * emission_targets[y] for y in years)

    return obj


def set_subproblem_objective(x, y, milp):
    # Set objective function for the subproblem for fixed x and y (unit and line investments).
    obj = get_objective(x, y, milp)

    m.setObjective(obj, GRB.MAXIMIZE)


# Constraints defining the uncertainty set.
m.addConstr(
    sum(w[n] for n in real_nodes) - uncertainty_budget == 0.0,
    name="uncertainty_set_budget",
)

# Flow constraints for existing AC lines.
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


def set_dependent_constraints(x, y, milp):
    constraint_names = [
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
            - (beta_ramp_bar[o, t - 1, u] if not is_year_first_hour(t) else 0.0)
            + (beta_ramp_bar[o, t, u] if not is_year_last_hour(t) else 0.0)
            + (beta_ramp_underline[o, t - 1, u] if not is_year_first_hour(t) else 0.0)
            - (beta_ramp_underline[o, t, u] if not is_year_last_hour(t) else 0.0)
            - (beta_emissions[to_year(t)] * weights[o] * G_emissions[o, t, u])
            - discount_factor ** (-to_year(t)) * C_g[o, t, u] * weights[o]
            == 0.0
            for o in scenarios
            for t in hours
            for u in units
            if unit_built(x, t, u)
        ),
        name="generation_dual_constraint",
    )

    if milp:
        z, lambda_tilde = get_or_create_linearization_variables()

        # Constraints for linearizing lambda_[n, o] * d[n].
        m.addConstrs(
            (
                z[o, t, n] - lambda_[o, t, n] + lambda_tilde[o, t, n] == 0.0
                for o in scenarios
                for t in hours
                for n in real_nodes
            ),
            name="linearization_z_definition",
        )

        m.addConstrs(
            (
                w[n] * (-K) - z[o, t, n] <= 0.0
                for o in scenarios
                for t in hours
                for n in real_nodes
            ),
            name="linearization_z_lb",
        )

        m.addConstrs(
            (
                z[o, t, n] - w[n] * K <= 0.0
                for o in scenarios
                for t in hours
                for n in real_nodes
            ),
            name="linearization_z_ub",
        )

        m.addConstrs(
            (
                (1.0 - w[n]) * (-K) - lambda_tilde[o, t, n] <= 0.0
                for o in scenarios
                for t in hours
                for n in real_nodes
            ),
            name="lambda_tilde_lb",
        )

        m.addConstrs(
            (
                lambda_tilde[o, t, n] - (1.0 - w[n]) * K <= 0.0
                for o in scenarios
                for t in hours
                for n in real_nodes
            ),
            name="lambda_tilde_ub",
        )


def solve_subproblem(x, y, milp):
    set_dependent_constraints(x, y, milp)
    set_subproblem_objective(x, y, milp)
    m.optimize()

    prices = []
    for y in years:
        print("beta_emissions[%d]:" % y, beta_emissions[y].x)
        prices.append(beta_emissions[y].x)

    return m.objVal, prices


def get_uncertain_variables():
    # Get the names and values of uncertain variables of the subproblem.
    names = np.array([w[n].varName for n in real_nodes])

    values = np.zeros_like(nominal_demand)
    for n in real_nodes:
        values[:, :, n] = (
            float(int(w[n].x)) * demand_increase[:, :, n] + nominal_demand[:, :, n]
        )

    return names, values


def get_uncertainty_decisions():
    # Get a vector of the uncertainty decisions w.
    values = np.array([float(int(w[n].x)) for n in real_nodes])
    return values
