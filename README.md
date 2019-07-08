# Robust optimization for power markets

An upper-level agent makes generation and transmission line investment decisions, while the market is cleared in the lower-level.
The lower-level problem is a robust optimization problem in which some parameters are stochastic.

# Requirements

Tested with Python 2.7 and 3.6 and Gurobi 8.1.

# Usage

Call robust.py with the selected master problem algorithm:

```
python robust.py <benders_dc|milp_dc> miqp_dc
```
