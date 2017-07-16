# CC master problem MILP formulation.
# Note: this assumes that all candidate lines are DC.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gurobipy import GRB, Model
import numpy as np

from common_data import (
    years,
    hours,
    annualizer,
    scenarios,
    real_nodes,
    nodes,
    units,
    lines,
    existing_lines,
    candidate_units,
    candidate_lines,
    ac_lines,
    ac_nodes,
    hydro_units,
    unit_to_node,
    unit_to_generation_type,
    maximum_candidate_unit_capacity_by_type,
    F_max,
    F_min,
    B,
    ref_node,
    G_emissions,
    C_g,
    inflows,
    incidence,
    weights,
    node_to_unit,
    emission_targets,
    C_x,
    C_y,
    discount_factor,
    master_method,
    enable_custom_configuration_master_problem,
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
    is_year_first_hour,
    is_year_last_hour,
    get_investment_cost,
    read_investment_and_availability_decisions,
)


class CCMasterProblemNoStorage:
    # A class representing the master problem of the CC algorithm.
    # All candidate lines are assumed to be DC in this formulation.

    def __init__(self):
        # Initialize a model.
        m = self._init_model("master_problem")
        self._model = m

        # Add binary (transmission) and continuous (generation) investment variables to the model.
        y, yhat = self._add_binary_investment_variables(m)
        self._y, self._yhat = y, yhat

        x, xhat = self._add_continuous_investment_variables(m)
        self._x, self._xhat = x, xhat

        # Set the model objective value.
        theta = self._set_objective(xhat, yhat)
        self._theta = theta

        # Emission constraint is strict.
        self._relaxed_emission_constraint = False

    def _apply_model_parameters(self, m):
        m.Params.Method = master_method

        if enable_custom_configuration_master_problem:
            for parameter, value in GRB_PARAMS:
                m.setParam(parameter, value)

    def _init_model(self, name):
        # Create the initial model without objective and constraints.
        m = Model(name)
        self._apply_model_parameters(m)
        return m

    def _add_binary_investment_variables(self, model):
        # Add binary investment variables to the given model.
        m = model

        # Variables representing investment to transmission lines.
        yhat = m.addVars(years, candidate_lines, vtype=GRB.BINARY, name="line_investment")

        # Variables indicating whether candidate transmission lines can be operated.
        y = m.addVars(years, candidate_lines, vtype=GRB.BINARY, name="line_available")

        # Constraints defining that candidate transmission lines can be operated if investment
        # has been made.
        m.addConstrs(
            (
                y[t, l] - sum(yhat[tt, l] for tt in range(t + 1)) == 0.0
                for t in years
                for l in candidate_lines
            ),
            name="line_operational1",
        )

        # m.addConstrs(
        #     (sum(yhat[t, l] for t in years) <= 1 for l in candidate_lines),
        #     name="line_invest_once",
        # )

        # mp.addConstrs(
        #     (
        #         y[t, l] >= yhat[t, l]
        #         for t in years
        #         for l in candidate_lines
        #     ),
        #     name="line_operational2",
        # )

        return y, yhat

    def _add_continuous_investment_variables(self, model):
        # Add continuous investment variables to the given model.
        m = model

        # Variables representing investment to generation units.
        xhat = m.addVars(
            years, candidate_units, lb=0.0, ub=GRB.INFINITY, name="unit_investment"
        )

        # Variables indicating whether candidate generation units can be operated.
        x = m.addVars(
            years, candidate_units, lb=0.0, ub=GRB.INFINITY, name="unit_available",
        )

        # Constraints defining that candidate units can be operated if investment
        # has been made.
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
                sum(xhat[t, u] for t in years)
                <= maximum_candidate_unit_capacity_by_type[unit_to_generation_type[u]]
                for u in candidate_units
            ),
            name="maximum_unit_investment",
        )

        # m.addConstrs(
        #     (
        #         xhat[t, u] <= maximum_candidate_unit_capacity_by_type[unit_to_generation_type[u]]
        #         for t in years
        #         for u in candidate_units
        #     ),
        #     name="maximum_unit_investment2",
        # )

        # m.addConstrs(
        #     (
        #         x[t, u] <= maximum_candidate_unit_capacity_by_type[unit_to_generation_type[u]]
        #         for t in years
        #         for u in candidate_units
        #     ),
        #     name="maximum_unit_investment3",
        # )

        return x, xhat

    def _set_objective(self, xhat, yhat):
        # Set the model objective.
        # Variable representing the subproblem objective value.
        m = self._model

        theta = m.addVar(name="theta", lb=0.0, ub=GRB.INFINITY)

        # Set master problem objective function. The optimal solution is no investment initially.
        m.setObjective(get_investment_cost(xhat, yhat) + theta, GRB.MINIMIZE)

        return theta

    def _add_primal_variables(self, model, iteration):
        # Add a new set of primal variables for the current CC iteration.
        m = model

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

        # Flow variables for existing and candidate lines.
        # Upper and lower bound are set as constraints.
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
            ac_nodes,  # Nodes that are part of the AC circuit.
            [iteration],
            name="voltage_angle_%d" % iteration,
            lb=-np.pi,
            ub=np.pi,
        )

        return g, f, delta

    def _augment_master_problem(self, current_iteration, d):
        # Augment the master problem for the current CC iteration.
        m = self._model
        x, y, xhat, yhat, theta = self._x, self._y, self._xhat, self._yhat, self._theta

        v = current_iteration

        # Create additional primal variables indexed with the current iteration.
        g, f, delta = self._add_primal_variables(m, v)

        ramp_hours = get_ramp_hours()  # Hours for which the ramp constraints are defined.

        # Minimum value for the subproblem objective function.
        m.addConstr(
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
        m.addConstrs(
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
        m.addConstrs(
            (
                g[o, t, u, v] - get_effective_capacity(o, t, u, x) <= 0.0
                for o in scenarios
                for t in hours
                for u in units
            ),
            name="maximum_generation_%d" % current_iteration,
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
                -get_maximum_ramp(o, t, u, x) - g[o, t + 1, u, v] + g[o, t, u, v] <= 0.0
                for o in scenarios
                for t in ramp_hours
                for u in units
            ),
            name="max_down_ramp_%d" % current_iteration,
        )

        # Maximum ramp upwards.
        m.addConstrs(
            (
                g[o, t + 1, u, v] - g[o, t, u, v] - get_maximum_ramp(o, t, u, x) <= 0.0
                for o in scenarios
                for t in ramp_hours
                for u in units
            ),
            name="max_up_ramp_%d" % current_iteration,
        )

        # Emission constraint.
        self._set_emission_constraint(m, g, v, current_iteration)

        return g, f

    def _set_emission_constraint(self, m, g, v, current_iteration):
        constr_name = "maximum_emissions_%d"

        if not self._relaxed_emission_constraint:
            constr = m.addConstrs(
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
                name=constr_name % current_iteration,
            )
        else:
            constr = m.addConstrs(
                (
                    sum(
                        weights[o] * g[o, t, u, v] * G_emissions[o, t, u]
                        for o in scenarios
                        for t in to_hours(y)
                        for u in units
                    )
                    - self._emission_slack[y]
                    - emission_targets[y]
                    <= 0.0
                    for y in years
                ),
                name=constr_name % current_iteration,
            )

    def get_investment_and_availability_decisions(self, initial=False, many_solutions=False):
        # Read current investments to generation and transmission and whether the units and lines are
        # operational at some time point.
        # At the first CC iteration, return full investment.
        return read_investment_and_availability_decisions(
            self._x, self._xhat, self._y, self._yhat, initial, many_solutions
        )

    def solve(self, current_iteration, d):
        # Return dummy values on the first iteration.
        # The initial solution is full investment to ensure that the subproblem is feasible.
        if current_iteration == 0:
            return -np.inf, None, None, None

        m = self._model

        g, f = self._augment_master_problem(current_iteration, d)
        m.update()
        m.optimize()
        return m.objVal, g, None, f
