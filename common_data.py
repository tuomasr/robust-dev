# Define data used by both the master problem and subproblem.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

np.random.seed(13)

# Configure Gurobi.

# Solver method for master problem and subproblem.
# -1 = automatic
# 4 = deterministic concurrent (runs out of memory for bigger instances).
master_method = 4
subproblem_method = 4

# Custom configuration for both problems.
enable_custom_configuration = False

GRB_PARAMS = [
    ("MIPGap", 0),
    ("FeasibilityTol", 1e-9),
    ("IntFeasTol", 1e-9),
    ("MarkowitzTol", 1e-4),
    ("OptimalityTol", 1e-9),
    # ('MIPFocus', 2),
    # ('Presolve', 0),
    # ('Cuts', 0),
    # ('Aggregate', 0)]
]

# Data taken from http://orbit.dtu.dk/files/120568114/An_Updated_Version_of_the_IEEE_RTS_24Bus_System_for_Electricty_Market_an....pdf

# Uncertainty parameters.
uncertainty_demand_increase = 100.0
uncertainty_budget = 2.0

# Scenarios.
num_scenarios = 3
scenarios = list(range(num_scenarios))

# Years and hours.
num_years = 4
num_hours_per_year = 5
num_hours = num_years * num_hours_per_year

annualizer = float(num_years) * 365.0 / float(num_hours)

years = list(range(num_years))
hours = list(range(num_hours))

# Nodes.
num_nodes = 24
nodes = list(range(num_nodes))

# Load.
hourly_load = np.array(
    [
        1775.835,
        1669.815,
        1590.3,
        1563.795,
        1563.795,
        1590.3,
        1961.37,
        2279.43,
        2517.975,
        2544.48,
        2544.48,
        2517.975,
        2517.975,
        2517.975,
        2464.965,
        2464.965,
        2623.995,
        2650.5,
        2650.5,
        2544.48,
        2411.955,
        2199.915,
        1934.865,
        1669.815,
    ]
)

hourly_load = hourly_load[:num_hours_per_year]

load_shares = (
    np.array(
        [
            3.8,
            3.4,
            6.3,
            2.6,
            2.5,
            4.8,
            4.4,
            6.0,
            6.1,
            6.8,
            0.0,
            0.0,
            9.3,
            6.8,
            11.1,
            3.5,
            0.0,
            11.7,
            6.4,
            4.5,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    / 100.0
)

assert len(hourly_load) == num_hours_per_year
assert len(load_shares) == num_nodes
assert np.isclose(sum(load_shares), 1.0)

# num_hours x num_nodes array
load = np.dot(np.array([hourly_load]).T, np.array([load_shares]))

# num_hours x num_nodes array
load = np.tile(load, (num_years, 1))

# Maximum emissions by year.
adversarial_hourly_load = hourly_load.copy()
adversarial_hourly_load[0] += 100
adversarial_hourly_load[1] += 100
start_emissions = sum(adversarial_hourly_load)
emission_targets = np.linspace(start_emissions, 0.0, num_years)

assert len(emission_targets) == num_years

# Units.
existing_units = list(range(12))

# Generator to node mapping.
existing_unit_to_node = {
    0: 0,
    1: 1,
    2: 6,
    3: 12,
    4: 14,
    5: 14,
    6: 15,
    7: 17,
    8: 20,
    9: 21,
    10: 22,
    11: 22,
}

# Generation types:
# 0: conventional
# 1: wind
unit_to_generation_type = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
    7: 0,
    8: 0,
    9: 0,
    10: 0,
    11: 0,
}

# Generation costs.
generation_type_to_cost = {0: 20.0, 1: 1.0}

C_g = np.array([], dtype=np.float32)
for u in existing_units:
    cost = generation_type_to_cost[unit_to_generation_type[u]]
    C_g = np.append(C_g, [cost * np.random.uniform(1.0, 1.5)])

# Generation limits.
G_max = np.array(
    [[152, 152, 350, 591, 60, 155, 155, 400, 400, 300, 310, 350]], dtype=np.float32
)

# Greenhouse gas emissions.
generation_type_to_emissions = {0: 1.0, 1: 0.0}

G_emissions = np.array([], dtype=np.float32)
for u in existing_units:
    emissions = generation_type_to_emissions[unit_to_generation_type[u]]
    G_emissions = np.append(G_emissions, [emissions])

# Generate a candidate unit for each node.
candidate_units = []
candidate_unit_to_node = dict()
unit_idx = len(existing_units)

generate_candidate_units = True

if generate_candidate_units:
    for node_idx, node in enumerate(nodes):
        # Set mapping from node to unit id.
        candidate_units.append(unit_idx)
        candidate_unit_to_node[unit_idx] = node_idx

        # Set generation parameters.
        generation_type = 1  # Wind.
        cost = generation_type_to_cost[generation_type]
        emissions = generation_type_to_emissions[generation_type]

        C_g = np.append(C_g, [cost * np.random.uniform(1.0, 1.5)])
        G_max = np.concatenate((G_max, [[200.0]]), axis=1)
        G_emissions = np.append(G_emissions, [emissions])
        unit_idx += 1
else:
    candidate_units = []
    candidate_unit_to_node = dict()

# Update the total mapping from units to nodes.
unit_to_node = existing_unit_to_node.copy()
unit_to_node.update(candidate_unit_to_node)

units = existing_units + candidate_units

# Construct inverse mapping from units to nodes.
node_to_unit = {node: [] for node in nodes}

for unit, node in unit_to_node.items():
    node_to_unit[node].append(unit)

# Total number of units.
num_units = len(units)

# Augment the arrays to have correct dimensions.
G_max = np.tile(G_max, (num_scenarios, num_hours, 1))
G_max += np.random.uniform(-10.0, 10.0, (num_scenarios, num_hours, num_units))

C_g = np.tile(C_g, (num_scenarios, num_hours, 1))

G_emissions = np.tile(G_emissions, (num_scenarios, num_hours, 1))

# Ramping limits from time step to another (both up- and down-ramp).
G_ramp_max = np.array([[50.0]])
G_ramp_max = np.tile(G_ramp_max, (num_scenarios, num_hours, num_units))
G_ramp_min = -G_ramp_max

assert (
    G_max.shape[-1]
    == C_g.shape[-1]
    == G_emissions.shape[-1]
    == G_ramp_max.shape[-1]
    == num_units
)

# Build lines x nodes incidence matrix for existing lines.
# List pairs of nodes that are connected. Note: these are in 1-based indexing.
lines = [
    (1, 2),
    (1, 3),
    (1, 5),
    (2, 4),
    (2, 6),
    (3, 9),
    (3, 24),
    (4, 9),
    (5, 10),
    (6, 10),
    (7, 8),
    (8, 9),
    (8, 10),
    (9, 11),
    (9, 12),
    (10, 11),
    (10, 12),
    (11, 13),
    (11, 14),
    (12, 13),
    (12, 23),
    (13, 23),
    (14, 16),
    (15, 16),
    (15, 21),
    (15, 24),
    (16, 17),
    (16, 19),
    (17, 18),
    (17, 22),
    (18, 21),
    (19, 20),
    (20, 23),
    (21, 22),
]

candidate_line_connections = [
    (1, 4),
    (2, 5),
    (3, 10),
    (4, 10),
    (5, 11),
    (6, 11),
    (7, 9),
    (8, 10),
    (9, 13),
    (10, 13),
    (11, 15),
    (12, 14),
    (13, 24),
    (14, 17),
    (15, 17),
    (16, 18),
    (17, 19),
    (18, 22),
    (19, 21),
    (20, 24),
    (21, 23),
    (22, 24),
    (23, 24),
    (24, 9),
]

num_existing_lines = len(lines)
existing_lines = list(range(len(lines)))

incidence = np.zeros((num_existing_lines, num_nodes))

for line_idx, line in enumerate(lines):
    start, end = sorted(line)

    # Correct for the 1-based indexing in node numbering.
    incidence[line_idx, start - 1] = -1
    incidence[line_idx, end - 1] = 1

F_max = np.array(
    [
        [
            175,
            175,
            350,
            175,
            175,
            175,
            400,
            175,
            350,
            175,
            350,
            175,
            175,
            400,
            400,
            400,
            400,
            500,
            500,
            500,
            500,
            500,
            500,
            500,
            1000,
            500,
            500,
            500,
            500,
            500,
            1000,
            1000,
            1000,
            500,
        ]
    ],
    dtype=np.float32,
)

assert num_existing_lines == len(incidence) == len(F_max[0])

# Generate candidate lines between each pair of nodes.
line_idx = len(existing_lines)
candidate_lines = []
use_candidate_lines = True

if use_candidate_lines:
    for line in candidate_line_connections:
        start, end = sorted(line)

        # Add a new row to the lines x nodes incidence matrix.
        row = np.zeros((1, num_nodes))
        row[0, start - 1] = -1.0
        row[0, end - 1] = 1.0

        incidence = np.concatenate((incidence, row), axis=0)

        # Line parameters.
        F_max = np.concatenate((F_max, [[200.0]]), axis=1)

        candidate_lines.append(line_idx)
        line_idx += 1
else:
    candidate_lines = []

# Update the total transmission lines.
lines = existing_lines + candidate_lines
num_lines = len(lines)

# Augment the transmission capacity array to have expected dimensions.
F_max = np.tile(F_max, (num_scenarios, num_hours, 1))
F_max += np.random.uniform(-10.0, 10.0, (num_scenarios, num_hours, num_lines))

F_min = -F_max  # Same transmission capacity to both directions.

# Susceptance.
B = np.ones(num_lines) * 1e3

# Reference node.
ref_node = 0

assert len(incidence) == F_max.shape[-1] == num_lines

# Equal scenario weights.
weights = np.ones(num_scenarios) / float(num_scenarios)
