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
    get_lines_starting,
    get_lines_ending,
    get_ramp_rate,
    get_availability_rate,
)
from master_problem_dc import CCMasterProblem


class CCMasterProblemDualBenders(CCMasterProblem):
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

        # Create model for the Benders slave problem.
        sp = self._init_model("slave_problem")
        self._sp = sp

        # Create fixed variables and constraints for the Benders slave problem.
        self._add_continuous_investment_variables(sp)

        # Emission constraint is strict.
        self._relaxed_emission_constraint = False

    def _add_continuous_investment_variables(self, sp):
        # Add fixed dual variables and constraints corresponding to continuous investments.
        ub1 = GRB.INFINITY
        lb2 = -GRB.INFINITY
        ub2 = GRB.INFINITY

        beta_x_underline = sp.addVars(
            years,
            candidate_units,
            name="dual_min_x",
            lb=0.0,
            ub=ub1,
        )

        beta_xhat = sp.addVars(
            years,
            candidate_units,
            name="dual_x_xhat",
            lb=lb2,
            ub=ub2,
        )

        self._beta_x_underline = beta_x_underline
        self._beta_xhat = beta_xhat

        beta_xhat_total_bar = sp.addVars(
            candidate_units,
            name="dual_max_total_xhat",
            lb=0.0,
            ub=ub1,
        )

        self._beta_xhat_total_bar = beta_xhat_total_bar

        beta_xhat_underline = sp.addVars(
            years,
            candidate_units,
            name="dual_min_xhat",
            lb=0.0,
            ub=ub1,
        )

        # w.r.t. xhat.
        sp.addConstrs(
            (
                beta_xhat_underline[t, u]
                - beta_xhat_total_bar[u]
                - sum(beta_xhat[tt, u] for tt in range(t, len(years)))
                - C_x[t, u] / annualizer
                <= 0.0
                for t in years
                for u in candidate_units
            ),
            name="dual_xhat_constr"
        )

        sp.update()

    def get_investment_and_availability_decisions(self, initial=False, many_solutions=False):
        # Read current investments to generation and transmission and whether the units and lines are
        # operational at some time point.
        # At the first CC iteration, return full investment.
        if initial:
            x = xhat = None
        return read_investment_and_availability_decisions(
            x, xhat, self._y, self._yhat, initial, many_solutions
        )

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
        self,
        dual_variables,
        cc_iteration,
        benders_iteration,
        solution_number,
        d,
        unbounded=False,
    ):
        # Augment the Benders master problem with a new cut from the slave problem.
        mp = self._mp
        y, yhat = self._y, self._yhat
        delta = self._delta

        # Unpack the dual values. Be careful with the order.
        lambda_, beta_bar, mu_underline, mu_bar, beta_ramp_underline = dual_variables[:5]
        beta_ramp_bar, beta_emissions, rho_bar, rho_underline, phi_initial_storage = dual_variables[
            5:10
        ]
        phi_storage, phi_storage_change_lb, phi_storage_change_ub = dual_variables[10:]

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
                            get_effective_capacity(o, t, u, None) * beta_bar[o, t, u, v]
                            for u in existing_units
                        )
                        - sum(
                            F_min[o, t, l]
                            * line_built(y, t, l)
                            * mu_underline[o, t, l, v]
                            for l in lines
                        )
                        + sum(
                            F_max[o, t, l] * line_built(y, t, l) * mu_bar[o, t, l, v]
                            for l in lines
                        )
                        - sum(
                            (
                                -get_maximum_ramp(o, t, u, None)
                                * beta_ramp_underline[o, t, u, v]
                                if t in ramp_hours
                                else 0.0
                            )
                            for u in existing_units
                        )
                        + sum(
                            (
                                get_maximum_ramp(o, t, u, None) * beta_ramp_bar[o, t, u, v]
                                if t in ramp_hours
                                else 0.0
                            )
                            for u in existing_units
                        )
                        + sum(d[o, t, n, v] * lambda_[o, t, n, v] for n in real_nodes)
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
                            F_min[o, t, l]
                            * line_built(y, t, l)
                            * mu_underline[o, t, l, v]
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

        v = current_iteration

        # Maximum and minimum generation dual variables.
        ub1 = GRB.INFINITY
        lb2 = -GRB.INFINITY
        ub2 = GRB.INFINITY

        beta_x_underline = self._beta_x_underline
        beta_xhat = self._beta_xhat

        beta_theta = sp.addVars(
            [current_iteration],
            name="dual_theta_%d" % current_iteration,
            lb=0.0,
            ub=ub1,
        )

        beta_bar = sp.addVars(
            scenarios,
            hours,
            units,
            [current_iteration],
            name="dual_maximum_generation_%d" % current_iteration,
            lb=0.0,
            ub=ub1,
        )
        beta_underline = sp.addVars(
            scenarios,
            hours,
            units,
            [current_iteration],
            name="dual_minimum_generation_%d" % current_iteration,
            lb=0.0,
            ub=ub1,
        )

        ramp_hours = get_ramp_hours()

        # Storage dual variables.
        year_first_hours = [t for t in hours if is_year_first_hour(t)]
        year_last_hours = [t for t in hours if is_year_last_hour(t)]

        beta_storage_underline = sp.addVars(
            scenarios,
            hours,
            hydro_units,
            [current_iteration],
            name="dual_minimum_storage_%d" % current_iteration,
            lb=0.0,
            ub=ub1,
        )

        phi_initial_storage = sp.addVars(
            scenarios,
            year_first_hours,
            hydro_units,
            [current_iteration],
            name="dual_initial_storage_%d" % current_iteration,
            lb=lb2,
            ub=ub2,
        )

        phi_storage = sp.addVars(
            scenarios,
            ramp_hours,
            hydro_units,
            [current_iteration],
            name="dual_storage_%d" % current_iteration,
            lb=lb2,
            ub=ub2,
        )

        phi_storage_change_lb = sp.addVars(
            scenarios,
            year_last_hours,
            hydro_units,
            [current_iteration],
            name="dual_storage_change_lb_%d" % current_iteration,
            lb=0.0,
            ub=ub1,
        )

        phi_storage_change_ub = sp.addVars(
            scenarios,
            year_last_hours,
            hydro_units,
            [current_iteration],
            name="dual_storage_change_ub_%d" % current_iteration,
            lb=0.0,
            ub=ub1,
        )

        # Maximum up- and down ramp dual variables.
        beta_ramp_bar = sp.addVars(
            scenarios,
            ramp_hours,
            units,
            [current_iteration],
            name="dual_maximum_ramp_upwards_%d" % current_iteration,
            lb=0.0,
            ub=ub1,
        )
        beta_ramp_underline = sp.addVars(
            scenarios,
            ramp_hours,
            units,
            [current_iteration],
            name="dual_maximum_ramp_downwards_%d" % current_iteration,
            lb=0.0,
            ub=ub1,
        )

        # Maximum emissions dual variables.
        beta_emissions = sp.addVars(
            years,
            [current_iteration],
            name="dual_maximum_emissions_%d" % current_iteration,
            lb=0.0,
            ub=ub1,
        )

        # Transmission flow dual variables.
        phi = sp.addVars(
            scenarios,
            hours,
            ac_lines,
            [current_iteration],
            name="dual_flow_%d" % current_iteration,
            lb=lb2,
            ub=ub2,
        )

        # Maximum and minimum transmission flow dual variables.
        mu_bar = sp.addVars(
            scenarios,
            hours,
            lines,
            [current_iteration],
            name="dual_maximum_flow_%d" % current_iteration,
            lb=0.0,
            ub=ub1,
        )
        mu_underline = sp.addVars(
            scenarios,
            hours,
            lines,
            [current_iteration],
            name="dual_minimum_flow_%d" % current_iteration,
            lb=0.0,
            ub=ub1,
        )

        # Dual variables for voltage angle bounds.
        mu_angle_bar = sp.addVars(
            scenarios,
            hours,
            ac_nodes,
            [current_iteration],
            name="dual_maximum_angle_%d" % current_iteration,
            lb=0.0,
            ub=ub1,
        )
        mu_angle_underline = sp.addVars(
            scenarios,
            hours,
            ac_nodes,
            [current_iteration],
            name="dual_minimum_angle_%d" % current_iteration,
            lb=0.0,
            ub=ub1,
        )

        # Dual variable for the reference node voltage angle.
        eta = sp.addVars(
            scenarios,
            hours,
            [current_iteration],
            name="dual_reference_node_angle_%d" % current_iteration,
            lb=lb2,
            ub=ub2,
        )

        # Balance equation dual (i.e. price).
        lambda_ = sp.addVars(
            scenarios,
            hours,
            nodes,
            [current_iteration],
            name="dual_balance_%d" % current_iteration,
            lb=lb2,
            ub=ub2,
        )

        # w.r.t. s.
        sp.addConstrs(
            (
                (phi_initial_storage[o, t, u, v] if is_year_first_hour(t) else 0.0)
                + beta_storage_underline[o, t, u, v]
                - (phi_storage[o, t, u, v] if not is_year_last_hour(t) else 0.0)
                + (phi_storage[o, t - 1, u, v] if not is_year_first_hour(t) else 0.0)
                + (phi_storage_change_lb[o, t, u, v] if is_year_last_hour(t) else 0.0)
                - (phi_storage_change_ub[o, t, u, v] if is_year_last_hour(t) else 0.0)
                <= 0.0
                for o in scenarios
                for t in hours
                for u in hydro_units
            ),
            name="storage_dual_constraint_%d" % current_iteration,
        )

        # w.r.t. delta.
        sp.addConstrs(
            (
                -sum(
                    (
                        B[l] * phi[o, t, l, v]
                        if l in existing_lines and l in ac_lines
                        else 0.0
                    )
                    for l in get_lines_starting(n)
                )
                + sum(
                    (
                        B[l] * phi[o, t, l, v]
                        if l in existing_lines and l in ac_lines
                        else 0.0
                    )
                    for l in get_lines_ending(n)
                )
                - mu_angle_bar[o, t, n, v]
                + mu_angle_underline[o, t, n, v]
                + (eta[o, t, v] if n == ref_node else 0.0)
                <= 0.0
                for o in scenarios
                for t in hours
                for n in ac_nodes
            ),
            name="voltage_angle_dual_constraint_%d" % current_iteration,
        )

        # w.r.t. f.
        sp.addConstrs(
            (
                (
                    sum(incidence[l, n] * lambda_[o, t, n, v] for n in nodes)
                    + (
                        phi[o, t, l, v]
                        if l in existing_lines and l in ac_lines
                        else 0.0
                    )
                    - mu_bar[o, t, l, v]
                    + mu_underline[o, t, l, v]
                )
                <= 0.0
                for o in scenarios
                for t in hours
                for l in lines
                if line_built(y, t, l)
            ),
            name="flow_dual_constraint_%d" % current_iteration,
        )

        # w.r.t. g.
        sp.addConstrs(
            (
                lambda_[o, t, unit_to_node[u], v]
                - beta_bar[o, t, u, v]
                + beta_underline[o, t, u, v]
                + (
                    phi_storage[o, t, u, v]
                    if t in ramp_hours and u in hydro_units
                    else 0.0
                )
                - (beta_ramp_bar[o, t - 1, u, v]
                    if not is_year_first_hour(t)
                    else 0.0
                )
                + (beta_ramp_bar[o, t, u, v]
                    if not is_year_last_hour(t)
                    else 0.0
                )
                + (beta_ramp_underline[o, t - 1, u, v]
                    if not is_year_first_hour(t)
                    else 0.0
                )
                - (beta_ramp_underline[o, t, u, v]
                    if not is_year_last_hour(t)
                    else 0.0
                )
                - (beta_emissions[to_year(t), v] * weights[o] * G_emissions[o, t, u])
                - discount_factor ** (-to_year(t))
                * C_g[o, t, u]
                * weights[o]
                * beta_theta[v]
                <= 0.0
                for o in scenarios
                for t in hours
                for u in units
            ),
            name="generation_dual_constraint_%d" % current_iteration,
        )

        # w.r.t. x.
        sp.addConstrs(
            beta_x_underline[y, u]
            + sum(
                get_availability_rate(o, t, u) * beta_bar[o, t, u, v]
                for o in scenarios
                for t in to_hours(y)
            )
            + sum(
                (get_ramp_rate(o, t, u) * (beta_ramp_bar[o, t, u, v] - beta_ramp_underline[o, t, u, v])) if t in ramp_hours else 0.0
                for o in scenarios
                for t in to_hours(y)
            )
            + beta_xhat[y, u] <= 0.0
            for y in years
            for u in candidate_units
        )

        # Theta constraint.
        constr_name = "beta_theta_constr"
        try:
            existing = sp.getConstrByName(constr_name)
            sp.remove(existing)
            sp.update()
        except:
            pass

        sp.addConstr(
            sum(beta_theta[v] for v in range(1, current_iteration)) - 1.0 <= 0.0,
            name=constr_name,
        )

        obj = sum(
            sum(d[o, t, n, v] * lambda_[o, t, n, v] for n in real_nodes)
            - sum(
                beta_bar[o, t, u, v] * get_effective_capacity(o, t, u, None)
                for u in existing_units
            )
            + sum(
                initial_storage[u][o, to_year(t)] * phi_initial_storage[o, t, u, v]
                if t in year_first_hours
                else 0.0
                for u in hydro_units
            )
            + sum(
                inflows[u][o, t] * phi_storage[o, t, u, v] if t in ramp_hours else 0.0
                for u in hydro_units
            )
            - sum(
                phi_storage_change_lb[o, t, u, v]
                * (
                    -initial_storage[u][o, to_year(t)]
                    * storage_change_lb[unit_to_node[u]]
                )
                if t in year_last_hours
                else 0.0
                for u in hydro_units
            )
            - sum(
                phi_storage_change_ub[o, t, u, v]
                * initial_storage[u][o, to_year(t)]
                * storage_change_ub[unit_to_node[u]]
                if t in year_last_hours
                else 0.0
                for u in hydro_units
            )
            - sum(
                (
                    mu_bar[o, t, l, v] * F_max[o, t, l]
                    - mu_underline[o, t, l, v] * F_min[o, t, l]
                )
                for l in lines
                if line_built(y, t, l)
            )
            - sum(
                np.pi * (mu_angle_bar[o, t, n, v] + mu_angle_underline[o, t, n, v])
                for n in ac_nodes
            )
            - sum(
                beta_ramp_bar[o, t, u, v] * get_maximum_ramp(o, t, u, None)
                if t in ramp_hours
                else 0.0
                for u in existing_units
            )
            + sum(
                beta_ramp_underline[o, t, u, v] * (-get_maximum_ramp(o, t, u, None))
                if t in ramp_hours
                else 0.0
                for u in existing_units
            )
            for o in scenarios
            for t in hours
        )

        obj -= sum(beta_emissions[y, v] * emission_targets[y] for y in years)

        beta_xhat_total_bar = self._beta_xhat_total_bar
        obj -= sum(
            maximum_candidate_unit_capacity_by_type[unit_to_generation_type[u]] * beta_xhat_total_bar[u]
            for u in candidate_units
        )

        sp.update()

        sp.setObjective(obj, GRB.MAXIMIZE)

        sp.update()

        dual_variables = [
            lambda_,
            beta_bar,
            phi_initial_storage,
            phi_storage,
            phi_storage_change_lb,
            phi_storage_change_ub,
            mu_bar,
            mu_underline,
            mu_angle_bar,
            mu_angle_underline,
            beta_ramp_bar,
            beta_ramp_underline,
            beta_emissions,
        ]

        return dual_variables

    def _get_dual_variables(self, current_iteration):
        # Obtain relevant dual variables in a nice data structure.
        sp = self._sp

        lambda_ = dict()
        beta_bar = dict()
        mu_underline = dict()
        mu_bar = dict()
        beta_ramp_underline = dict()
        beta_ramp_bar = dict()
        beta_emissions = dict()
        rho_underline = dict()
        rho_bar = dict()
        phi_initial_storage = dict()
        phi_storage = dict()
        phi_storage_change_lb = dict()
        phi_storage_change_ub = dict()

        year_first_hours = [t for t in hours if is_year_first_hour(t)]
        year_last_hours = [t for t in hours if is_year_last_hour(t)]
        ramp_hours = get_ramp_hours()

        for v in range(1, current_iteration + 1):
            for o in scenarios:
                for t in hours:
                    for n in nodes:
                        name = "dual_balance_%d[%d,%d,%d]" % (v, o, t, n)
                        lambda_[o, t, n, v] = sp.getVarByName(name)

                        if n in ac_nodes:
                            lb_name = "dual_minimum_angle_%d[%d,%d,%d]" % (v, o, t, n)
                            ub_name = "dual_maximum_angle_%d[%d,%d,%d]" % (v, o, t, n)

                            rho_underline[o, t, n, v] = sp.getVarByName(
                                lb_name
                            )
                            rho_bar[o, t, n, v] = sp.getVarByName(ub_name)

                    for u in existing_units:
                        name = "dual_maximum_generation_%d[%d,%d,%d]" % (v, o, t, u)
                        beta_bar[o, t, u, v] = sp.getVarByName(name)

                        if t in ramp_hours:
                            down_name = "dual_maximum_ramp_downwards_%d[%d,%d,%d]" % (v, o, t, u)
                            up_name = "dual_maximum_ramp_upwards_%d[%d,%d,%d]" % (v, o, t, u)
                            beta_ramp_underline[
                                o, t, u, v
                            ] = sp.getVarByName(down_name)
                            beta_ramp_bar[o, t, u, v] = sp.getVarByName(
                                up_name
                            )

                            if u in hydro_units:
                                storage_name = "dual_storage_%d[%d,%d,%d]" % (v, o, t, u)
                                phi_storage[o, t, u, v] = sp.getVarByName(
                                    storage_name
                                )

                        if t in year_first_hours:
                            if u in hydro_units:
                                initial_storage_name = (
                                    "dual_initial_storage_%d[%d,%d,%d]" % (v, o, t, u)
                                )
                                phi_initial_storage[
                                    o, t, u, v
                                ] = sp.getVarByName(initial_storage_name)

                        if t in year_last_hours:
                            if u in hydro_units:
                                storage_change_lb_name = (
                                    "dual_storage_change_lb_%d[%d,%d,%d]" % (v, o, t, u)
                                )
                                phi_storage_change_lb[
                                    o, t, u, v
                                ] = sp.getVarByName(storage_change_lb_name)

                                storage_change_ub_name = (
                                    "dual_storage_change_ub_%d[%d,%d,%d]" % (v, o, t, u)
                                )
                                phi_storage_change_ub[
                                    o, t, u, v
                                ] = sp.getVarByName(storage_change_ub_name)

                    for l in lines:
                        min_name = "dual_minimum_flow_%d[%d,%d,%d]" % (v, o, t, l)
                        max_name = "dual_maximum_flow_%d[%d,%d,%d]" % (v, o, t, l)
                        mu_underline[o, t, l, v] = sp.getVarByName(min_name)
                        mu_bar[o, t, l, v] = sp.getVarByName(max_name)

                for y in years:
                    name = "dual_maximum_emissions_%d[%d]" % (v, y)
                    beta_emissions[y, v] = sp.getVarByName(name)

        dual_variables = (
            lambda_,
            beta_bar,
            mu_underline,
            mu_bar,
            beta_ramp_underline,
            beta_ramp_bar,
            beta_emissions,
            rho_underline,
            rho_bar,
            phi_initial_storage,
            phi_storage,
            phi_storage_change_lb,
            phi_storage_change_ub,
        )

        return dual_variables

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

    def solve(self, current_iteration, d):
        # Solve the CC master problem using Benders.
        mp, sp = self._mp, self._sp

        # Dummy return values on the first iteration.
        # Full investment is used as the initial CC master problem solution to ensure CC subproblem
        # feasibility.
        if current_iteration == 0:
            return -np.inf, None, None

        max_iterations = 9999
        threshold = 1e-4
        bad_threshold = -1e-3
        separator = "-" * 50

        lb = -np.inf
        ub = np.inf

        duplicates = (
            0
        )  # For monitoring if Benders visits the same solution multiple times.

        # Initialize Benders by running one iteration with full transmission investments.
        print("Getting initial Benders master problem solution")
        self._initialize_benders_master()
        yhat, y = get_initial_transmission_investments()

        print("Getting initial Benders slave problem solution")
        self._augment_benders_slave(current_iteration, d, yhat, y)
        sp.update()
        sp.optimize()

        dual_variables = self._get_dual_variables()
        self._augment_benders_master(dual_variables, current_iteration, 0, 0, d)

        solution_hashes = set()

        for iteration in range(1, max_iterations):
            print(separator)
            print("Starting Benders iteration:", iteration)

            print("Solving Benders master problem.")
            mp.update()
            mp.write("benders_master_%d_%d.mps" % (current_iteration, iteration))
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

                self._update_benders_slave(
                    updatable_constrs, current_iteration, yhat, y
                )

                print(separator)
                print("Solving Benders slave problem.")
                sp.update()
                sp.write("benders_slave_%d_%d.mps" % (current_iteration, iteration))
                sp.optimize()
                print(separator)

                if sp.Status != GRB.OPTIMAL:
                    unbounded_ray = self._get_unbounded_ray(all_constrs)
                    self._augment_benders_master(
                        x,
                        unbounded_ray,
                        current_iteration,
                        iteration,
                        k,
                        d,
                        unbounded=True,
                    )
                    continue

                ub = sp.objVal

                gap = compute_objective_gap(lb, ub)

                if k == 0 and gap < threshold:
                    if not gap >= bad_threshold:
                        raise RuntimeError("lb (%f) > ub (%f) in Benders." % (lb, ub))

                    print("Took %d Benders iterations." % (iteration + 1))

                    return lb, g, s

                dual_values = self._get_dual_variables(all_constrs)
                _, _, x, _ = self.get_investment_and_availability_decisions(
                    initial=False, many_solutions=many_solutions
                )
                self._augment_benders_master(
                    x, dual_values, current_iteration, iteration, k, d
                )

        raise RuntimeError("Max iterations hit in Benders. LB: %f, UB: %f" % (lb, ub))
