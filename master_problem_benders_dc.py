# CC master problem Benders formulation with the assumption that all candidate lines are DC.

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
    existing_units,
    existing_lines,
    candidate_units,
    candidate_lines,
    ac_lines,
    ac_nodes,
    hydro_units,
    unit_to_node,
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
    get_investment_cost,
    get_transmission_investment_cost,
    get_initial_transmission_investments,
    read_investment_and_availability_decisions,
)
from master_problem_dc import CCMasterProblem


class CCMasterProblemBenders(CCMasterProblem):
    # A class representing the Benders decomposition of the master problem of the CC algorithm.
    # All candidate lines are assumed to be DC in this formulation.
    BENDERS_MASTER_CUT_NAME = "benders_master_cut_constraint"


    def __init__(self):
        # Initialize a model.
        # Create a model for the Benders master problem.
        mp = self._init_model("master_problem")
        self._mp = mp

        # Add binary investment variables (transmission investment) to the Benders master problem.
        y, yhat = self._add_binary_investment_variables(mp)
        self._y, self._yhat = y, yhat

        # Set Benders master problem objective.
        delta = self._set_master_problem_objective(yhat)
        self._delta = delta

        # Create models for the Benders slave problem.
        sp = self._init_model("slave_problem")
        # sp.Params.InfUnbdInfo = 1   # For obtaining infeasibility information.
        self._sp = sp

        # Add continuous investment variables (generation units) to the Benders slave problem.
        x, xhat = self._add_continuous_investment_variables(self._sp)
        self._x, self._xhat = x, xhat

        # Bender slave problem objective value is updated at every Benders iteration.
        # Variable representing the CC subproblem objective value.
        theta = sp.addVar(name="theta", lb=0.0, ub=GRB.INFINITY)
        self._theta = theta

        # Emission constraint is strict.
        self._relaxed_emission_constraint = False

    def _set_master_problem_objective(self, yhat):
        # Set objective function for the Benders master problem.
        mp = self._mp
        delta = mp.addVar(name="delta", lb=0.0, ub=GRB.INFINITY)
        mp.setObjective(get_transmission_investment_cost(yhat) + delta, GRB.MINIMIZE)
        mp.update()

        return delta

    def _initialize_benders_master(self):
        # Initialize the Benders master problem by removing old cuts.
        mp = self._mp
        existing_constraints = [
            c for c in mp.getConstrs() if self.BENDERS_MASTER_CUT_NAME in c.ConstrName
        ]

        if existing_constraints:
            mp.remove(existing_constraints)
            mp.update()

    def _augment_benders_master(
        self, x, dual_values, cc_iteration, benders_iteration, solution_number, d, unbounded=False
    ):
        self._check_are_values(x, x)
        # Augment the Benders master problem with a new cut from the slave problem.
        mp = self._mp
        y, yhat = self._y, self._yhat
        delta = self._delta

        # Unpack the dual values. Be careful with the order.
        sigma, beta_bar, mu_underline, mu_bar, beta_ramp_underline = dual_values[:5]
        beta_ramp_bar, beta_emissions, rho_bar, rho_underline, phi_initial_storage = dual_values[
            5:10
        ]
        phi_storage, phi_storage_change_lb, phi_storage_change_ub, beta_max_investment = dual_values[10:]

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
                    for v in range(1, cc_iteration + 1)
                )
                - sum(
                    emission_targets[y] * beta_emissions[y, v]
                    for y in years
                    for v in range(1, cc_iteration + 1)
                )
                - sum(
                    maximum_candidate_unit_capacity_by_type[unit_to_generation_type[u]] * beta_max_investment[u]
                    for u in candidate_units
                )
                >= 0.0,
                name="%s_single_cut_%d_%d"
                % (self.BENDERS_MASTER_CUT_NAME, benders_iteration, solution_number),
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
                    for v in range(1, cc_iteration + 1)
                )
                - sum(
                    emission_targets[y] * beta_emissions[y, v]
                    for y in years
                    for v in range(1, cc_iteration + 1)
                )
                - sum(
                    maximum_candidate_unit_capacity_by_type[unit_to_generation_type[u]] * beta_max_investment[u]
                    for u in candidate_units
                )
                >= 0.0,
                name="%s_single_cut_infeasibility_%d_%d"
                % (self.BENDERS_MASTER_CUT_NAME, benders_iteration, solution_number),
            )

        mp.update()

    def _check_are_values(self, some, somehat):
        assert all([isinstance(v, float) for v in some.values()])
        assert all([isinstance(v, float) for v in somehat.values()])

    def _augment_benders_slave(self, current_iteration, d, yhat, y):
        # Augment the Benders slave with results from a new CC and Benders iteration.
        # yhat and y are expected to be values here, not Gurobi values.
        self._check_are_values(y, yhat)
        sp = self._sp
        theta = self._theta
        x, xhat = self._x, self._xhat

        sp.setObjective(get_investment_cost(xhat, yhat) + theta, GRB.MINIMIZE)

        v = current_iteration

        # Create additional primal variables indexed with the current iteration.
        g, s, f, delta = self._add_primal_variables(sp, v)

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

        # Balance equation. Note that d[o, t, n, v] is input data from the CC subproblem.
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
        sp.update()

        return g, s, f


    def _obtain_constraints(self, current_iteration):
        # Obtain relevant constraints in a nice data structure.
        sp = self._sp

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
        max_investment_constrs = dict()

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

        for u in candidate_units:
            name = "maximum_unit_investment[%d]" % u
            max_investment_constrs[u] = sp.getConstrByName(name)

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
            max_investment_constrs
        )

        updatable_constrs = (mu_underline_constrs, mu_bar_constrs)

        return all_constrs, updatable_constrs


    def _update_benders_slave(self, updatable_constrs, current_iteration, yhat, y):
        # Update constraints involving the value of x and y. Be careful with the order when unpacking.
        sp = self._sp
        x, xhat = self._x, self._xhat
        theta = self._theta
        self._check_are_values(y, yhat)

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

        sp.update()

    def _get_dual_variables(self, all_constrs):
        # Get dual variable values of all (relevant) constraints of the slave problem.
        def _get_dual_values(constrs):
            dual_values = dict()

            for key, val in constrs.items():
                dual_values[key] = val.getAttr("Pi")

            return dual_values

        return [_get_dual_values(constrs) for constrs in all_constrs]


    def _get_unbounded_ray(self, all_constrs):
        # Get unbounded rays of all (relevant) constraints of the slave problem.
        def _get_dual_values(constrs):
            dual_values = dict()

            for key, val in constrs.items():
                dual_values[key] = val.getAttr("FarkasDual")

            return dual_values

        return [_get_dual_values(constrs) for constrs in all_constrs]


    def _get_solution_hash(self, yhat, y):
        # Generate a hash of the solution.
        solution_hash = ""

        for t in years:
            for l in candidate_lines:
                solution_hash += str(int(yhat[t, l])) + str(int(y[t, l]))

        return solution_hash

    def _get_initial_x(self):
        # Get initial values of x to bootstrap Benders.
        initial_x = dict()

        for t in years:
            for u in candidate_units:
                current_x[t, u] = self._x[t, u].x

    def _fix_x(self, xhat_values, x_values):
        x, xhat = self._x, self._xhat

        for k in x_values.keys():
            x[k].lb = x[k].ub = x_values[k]
            xhat[k].lb = xhat[k].ub = xhat_values[k]

    def _relax_x(self):
        x, xhat = self._x, self._xhat

        for k in x.keys():
            x[k].lb = 0.0
            xhat[k].lb = 0.0
            x[k].ub = GRB.INFINITY
            xhat[k].ub = GRB.INFINITY

    def solve(self, current_iteration, d):
        # Solve the CC master problem using Benders.
        mp, sp = self._mp, self._sp

        # Dummy return values on the first iteration.
        # Full investment is used as the initial CC master problem solution to ensure CC subproblem
        # feasibility.
        if current_iteration == 0:
            return -np.inf, None, None, None

        max_iterations = 9999
        threshold = 1e-6
        bad_threshold = -1e-3
        separator = "-" * 50

        lb = -np.inf
        ub = np.inf

        duplicates = 0  # For monitoring if Benders visits the same solution multiple times.

        # Initialize Benders by running one iteration with full transmission investments.
        print("Getting initial Benders master problem solution")
        self._initialize_benders_master()
        xhat, yhat, x, y = self.get_investment_and_availability_decisions(
            initial=True, many_solutions=False
        )

        print("Getting initial Benders slave problem solution")
        g, s, f = self._augment_benders_slave(current_iteration, d, yhat, y)
        self._fix_x(xhat, x)
        sp.update()
        sp.optimize()
        self._relax_x()

        all_constrs, updatable_constrs = self._obtain_constraints(current_iteration)
        dual_values = self._get_dual_variables(all_constrs)
        x = {k: v.x for k, v in self._x.items()}
        self._augment_benders_master(x, dual_values, current_iteration, 0, 0, d)

        solution_hashes = set()

        for iteration in range(1, max_iterations):
            print(separator)
            print("Starting Benders iteration:", iteration)

            print("Solving Benders master problem.")
            mp.update()
            #mp.write("benders_master_%d_%d.mps" % (current_iteration, iteration))
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

                _, yhat, _, y = self.get_investment_and_availability_decisions(
                    initial=False, many_solutions=many_solutions
                )

                solution_hash = self._get_solution_hash(yhat, y)

                if k > 0 and solution_hash in solution_hashes:
                    continue
                elif k == 0 and solution_hash in solution_hashes:
                    # Count the number of times Benders visits the same solution. If this number is
                    # high, then there is an opportunity to start caching slave problem solutions.
                    duplicates += 1
                    print("Current duplicates: %d. :(" % duplicates)

                solution_hashes.add(solution_hash)

                self._update_benders_slave(updatable_constrs, current_iteration, yhat, y)

                print(separator)
                print("Solving Benders slave problem.")
                sp.update()
                #sp.write("benders_slave_%d_%d.mps" % (current_iteration, iteration))
                sp.optimize()
                print(separator)

                if sp.Status != GRB.OPTIMAL:
                    unbounded_ray = self._get_unbounded_ray(all_constrs)
                    self._augment_benders_master(
                        x, unbounded_ray, current_iteration, iteration, k, d, unbounded=True
                    )
                    continue

                ub = sp.objVal

                gap = compute_objective_gap(lb, ub)

                if k == 0 and gap < threshold:
                    if not gap >= bad_threshold:
                        raise RuntimeError("lb (%f) > ub (%f) in Benders." % (lb, ub))

                    print("Took %d Benders iterations." % (iteration + 1))

                    return lb, g, s, f

                dual_values = self._get_dual_variables(all_constrs)
                _, _, x, _ = self.get_investment_and_availability_decisions(
                    initial=False, many_solutions=many_solutions
                )
                self._augment_benders_master(x, dual_values, current_iteration, iteration, k, d)

        raise RuntimeError("Max iterations hit in Benders. LB: %f, UB: %f" % (lb, ub))
