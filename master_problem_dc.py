# Master problem formulation.
# Note: this assumes that all candidate lines are DC.

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
    G_max,
    maximum_candidate_unit_capacity,
    F_max,
    F_min,
    B,
    ref_node,
    candidate_rates,
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
    get_candidate_generation_capacity,
    is_year_first_hour,
)


EMISSION_CONSTRAINTS_NAME = "maximum_emissions"


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


def add_primal_variables(iteration):
    # Generation variables for existing and candidate units. Upper bounds are set as constraints.
    g = m.addVars(
        scenarios,
        hours,
        units,
        [iteration],
        name="generation_%d" % iteration,
        lb=0.0,
        ub=GRB.INFINITY,
    )

    # Storage variables.
    s = m.addVars(
        scenarios,
        hours,
        hydro_units,
        [iteration],
        name="storage_%d" % iteration,
        lb=0.0,
        ub=GRB.INFINITY,
    )

    # Flow variables for existing and candidate lines. Upper and lower bound are set as constraints.
    f = m.addVars(
        scenarios,
        hours,
        lines,
        [iteration],
        name="flow_%d" % iteration,
        lb=-GRB.INFINITY,
        ub=GRB.INFINITY,
    )

    delta = m.addVars(
        scenarios,
        hours,
        ac_nodes,   # Nodes that are part of the AC circuit.
        [iteration],
        name="voltage_angle_%d" % iteration,
        lb=-np.pi,
        ub=np.pi,
    )

    return g, s, f, delta


# Create the initial model without operation constraints.
m = Model("master_problem")
m.Params.Method = master_method

if enable_custom_configuration:
    for parameter, value in GRB_PARAMS:
        m.setParam(parameter, value)

# Variables representing investment to generation units and transmission lines.
xhat = m.addVars(
    years, candidate_units, lb=0.0, ub=GRB.INFINITY, name="unit_investment"
)
yhat = m.addVars(years, candidate_lines, vtype=GRB.BINARY, name="line_investment")

# Variables indicating whether candidate generation units and transmission lines can be operated.
x = m.addVars(years, candidate_units, lb=0.0, ub=GRB.INFINITY, name="unit_available")
y = m.addVars(years, candidate_lines, vtype=GRB.BINARY, name="line_available")

# Constraints defining that candidate units and transmission lines can be operated if investment
# has been made.
# m.addConstrs(
#     (sum(yhat[t, l] for t in years) <= 1 for l in candidate_lines),
#     name="line_invest_once",
# )

m.addConstrs(
    (
        x[t, u] - sum(xhat[tt, u] for tt in range(t + 1)) == 0.0
        for t in years
        for u in candidate_units
    ),
    name="unit_operational1",
)

m.addConstrs(
    (
        y[t, l] - sum(yhat[tt, l] for tt in range(t + 1)) == 0.0
        for t in years
        for l in candidate_lines
    ),
    name="line_operational1",
)

# Variable representing the subproblem objective value.
theta = m.addVar(name="theta", lb=-999999999.999, ub=GRB.INFINITY)

# Set master problem objective function. The optimal solution is no investment initially.
m.setObjective(get_investment_cost(xhat, yhat) + theta, GRB.MINIMIZE)


def augment_master_problem(current_iteration, d):
    # Augment the master problem for the current iteration.
    v = current_iteration

    # Create additional primal variables indexed with the current iteration.
    g, s, f, delta = add_primal_variables(v)

    ramp_hours = get_ramp_hours()  # Hours for which the ramp constraints are defined.

    # Minimum value for the subproblem objective function.
    m.addConstr(
        theta
        - sum(
            sum(discount_factor ** (-to_year(t)) * sum(C_g[o, t, u] * g[o, t, u, v] for u in units) for t in hours)
            * weights[o]
            for o in scenarios
        )
        >= 0.0,
        name="minimum_subproblem_objective_%d" % current_iteration,
    )

    # Balance equation. Note that d[n, v] is input data from the subproblem.
    m.addConstrs(
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
    m.addConstrs(
        (
            g[o, t, u, v] - candidate_rates[u][t] * get_candidate_generation_capacity(t, u, x) <= 0.0
            for o in scenarios
            for t in hours
            for u in candidate_units
        ),
        name="maximum_candidate_generation_%d" % current_iteration,
    )

    m.addConstrs(
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

    m.addConstrs(
        (
            s[o, t, u, v] - initial_storage[u][o, to_year(t)] == 0.0
            for o in scenarios
            for t in year_first_hours
            for u in hydro_units
        ),
        name="initial_storage_%d" % current_iteration,
    )

    # Storage evolution.
    m.addConstrs(
        (
            s[o, t + 1, u, v] - s[o, t, u, v] + g[o, t, u, v] - inflows[u][o, t] == 0.0
            for o in scenarios
            for t in ramp_hours
            for u in hydro_units
        ),
        name="storage_%d" % current_iteration,
    )

    # Flow constraints for the lines.
    m.addConstrs(
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
    m.addConstrs(
        (
            f[o, t, l, v] - F_max[o, t, l] * line_built(y, t, l) <= 0.0
            for o in scenarios
            for t in hours
            for l in lines
        ),
        name="maximum_flow_%d" % current_iteration,
    )

    m.addConstrs(
        (
            F_min[o, t, l] * line_built(y, t, l) - f[o, t, l, v] <= 0.0
            for o in scenarios
            for t in hours
            for l in lines
        ),
        name="minimum_flow_%d" % current_iteration,
    )

    m.addConstrs(
        (delta[o, t, ref_node, v] == 0.0 for o in scenarios for t in hours),
        name="reference_node_%d" % current_iteration,
    )

    # Maximum ramp downwards.
    m.addConstrs(
        (
            G_ramp_min[o, t, u] - g[o, t + 1, u, v] + g[o, t, u, v] <= 0.0
            for o in scenarios
            for t in ramp_hours
            for u in units
        ),
        name="max_down_ramp_%d" % current_iteration,
    )

    # Maximum ramp upwards.
    m.addConstrs(
        (
            g[o, t + 1, u, v] - g[o, t, u, v] - G_ramp_max[o, t, u] <= 0.0
            for o in scenarios
            for t in ramp_hours
            for u in units
        ),
        name="max_up_ramp_%d" % current_iteration,
    )

    # Emission constraint.
    m.addConstrs(
        (
            sum(
                g[o, t, u, v] * G_emissions[o, t, u] for u in units for t in to_hours(y)
            )
            - emission_targets[y]
            <= 0.0
            for o in scenarios
            for y in years
        ),
        name=EMISSION_CONSTRAINTS_NAME + "_%d" % current_iteration,
    )

    return g, s


def get_investment_and_availability_decisions(initial=False):
    # Read current investments to generation and transmission and whether the units and lines are
    # operational at some time point.
    current_xhat = dict()
    current_yhat = dict()

    current_x = dict()
    current_y = dict()

    initial_generation_investment = maximum_candidate_unit_capacity
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


# Assign the master problem to a variable that can be imported elsewhere.
master_problem = m


def solve_master_problem(current_iteration, d):
    # Dummy return values on the first iteration.
    if current_iteration == 0:
        return -np.inf, None, None

    g, s = augment_master_problem(current_iteration, d)
    m.update()
    m.optimize()
    return m.objVal, g, s
