# Robust optimization for power markets

An upper-level agent makes generation and transmission line investment decisions, while the market is cleared in the lower-level.
The lower-level problem is a robust optimization problem in which some parameters are stochastic.

# Requirements

Tested with Python 3* and Gurobi 9*.

# Usage

In clustering.py, select N_CLUSTERS (operating conditions). Generate operating
conditions by running

```
python clustering.py
```

In common_data.py, set num_scenarios to match the selected number of operating conditions.
Set other flags to modify the model behavior. Call robust.py with the selected master problem and subproblem algorithm:

```
python robust.py <benders_dc|milp_dc> <miqp_dc|milp_dc> <output_dir>
```

Using the output files of `robust.py` as input, generate plots and tables in the manuscript by running

```
python cost_of_robustness2.py <list of costs, e.g. 100 1000 10000>
```

and `plotting.py`, `generation_mix*.py`, `summary.py`, etc.
