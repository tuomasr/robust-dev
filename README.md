# Robust optimization for power markets

An upper-level agent makes generation and transmission line investment decisions, while the market is cleared in the lower-level.
The lower-level problem is a robust optimization problem in which some parameters are stochastic.

# Requirements

Tested with Python 2.7 and 3.6 and Gurobi 8.1.

# Usage

In clustering.py, select N_CLUSTERS (operating conditions). Generate operating
conditions by running

```
python clustering.py
```

In common_data.py, set num_scenarios to match the selected number of operating conditions. Call robust.py with the selected master problem and subproblem algorithm:

```
python robust.py <benders_dc|milp_dc> <miqp_dc|milp_dc> <output_dir>
```

Generate plots by running

```
python generation_mix.py
```

and

```
python cost_of_robustness.py
```
