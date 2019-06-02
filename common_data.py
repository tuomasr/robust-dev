# Define data used by both the master problem and subproblem.

# TODO: discount operation costs, availability rates for all units, maximum storage constraint, existing lines AC/DC

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

np.random.seed(13)

# Configure Gurobi.

# Solver method for master problem and subproblem.
# -1 = automatic
# 4 = deterministic concurrent (runs out of memory for bigger instances).
master_method = -1
subproblem_method = -1

# Custom configuration for both problems.
enable_custom_configuration = False

GRB_PARAMS = [
    # ("MIPGap", 0),
    ("FeasibilityTol", 1e-9),
    # ("IntFeasTol", 1e-9),
    # ("MarkowitzTol", 1e-4),
    # ("OptimalityTol", 1e-9),
    # ('MIPFocus', 2),
    # ('Presolve', 0),
    # ('Cuts', 0),
    # ('Aggregate', 0)]
]

# Operation data taken from
# https://ieeexplore-ieee-org.libproxy.aalto.fi/stamp/stamp.jsp?tp=&arnumber=8309280

# Uncertainty parameters.
uncertainty_demand_increase = 100.0
uncertainty_budget = 2.0

# Scenarios.
num_scenarios = 3
scenarios = list(range(num_scenarios))

# Years and hours.
num_years = 6
num_hours_per_year = 9
num_hours = num_years * num_hours_per_year

annualizer = float(num_years) * 365.0 / float(num_hours)

years = list(range(num_years))
hours = list(range(num_hours))

# Sample starting days for each year.
start_indices = np.random.randint(0, 364, size=num_years)*24
end_indices = start_indices + num_hours_per_year

# Nodes.
# There are 8 "real" nodes and 6 dummy nodes with no generation or load.
# The dummy nodes are used to represent the transmission network.
# The "real" nodes are:
# 0: dk1
# 1: dk2
# 2: ee
# 3: fi
# 4: lt
# 5: lv
# 6: no
# 7: se
# 8-13: dummy nodes

num_real_nodes = 8
real_nodes = list(range(num_real_nodes))

num_nodes = 14
nodes = list(range(num_nodes))

# Load.
load_real_nodes = np.genfromtxt("load.csv", delimiter=";", skip_header=1)
load = np.zeros((num_hours, num_real_nodes))

# Read load for the sampled days.
for i, (start, end) in enumerate(zip(start_indices, end_indices)):
    load_slice = load_real_nodes[start:end]
    load[i*num_hours_per_year:(i+1)*num_hours_per_year, :] = load_slice

# Units.
generation_capacities = np.genfromtxt(
    "generation_capacity.csv", delimiter=";", skip_header=1
)

# Wind and PV production varies with weather conditions. The installed capacity is scaled
# with a "rate" in [0, 1].
all_wind_rates = np.genfromtxt(
    "wind_rates.csv", delimiter=";", skip_header=1,
)
all_pv_rates = np.genfromtxt(
    "pv_rates.csv", delimiter=";", skip_header=1,
)

# Read wind and PV rates for the sampled days.
wind_unit_idx = 8
pv_unit_idx = 9

for i, (start, end) in enumerate(zip(start_indices, end_indices)):
    wind_slice = all_wind_rates[start:end, 1:]  # Skip first (hour) column.
    pv_slice = all_pv_rates[start:end, 1:]

    if i == 0:
        wind_rates = wind_slice
        pv_rates = pv_slice
    else:
        wind_rates = np.concatenate((wind_rates, wind_slice))
        pv_rates = np.concatenate((pv_rates, pv_slice))

# Compile information about existing units by looping the table of
# generation capacities (real nodes x generation types).
unit_idx = 0
G_max = []
existing_unit_to_node = dict()
unit_to_generation_type = dict()

# Loop real nodes.
for i in range(generation_capacities.shape[0]):
    # Loop generation types.
    for j in range(generation_capacities.shape[1]):
        if generation_capacities[i, j] > 0:
            existing_unit_to_node[unit_idx] = i
            unit_to_generation_type[unit_idx] = j
            # Convert GWs to MWs.
            G_max.append(generation_capacities[i, j] * 1000.0)

            unit_idx += 1

existing_units = list(range(unit_idx))

# Convert G_max into an array that is easier to reshape.
G_max = np.array([np.array(G_max)], dtype=np.float32)

# Generation types:
# 0: coal
# 1: gas
# 2: ccgt
# 3: oil
# 4: biomass
# 5: oil shale
# 6: nuclear
# 7: hydro
# 8: wind
# 9: pv
# different chp types (e=extraction, b=back pressure):
# 10: coal-e
# 11: gas-b
# 12: gas-e
# 13: oil-b
# 14: oil-e
# 15: biomass-e
# 16: waste-e
# 17: peat-e

# Hydro units are treated specially because they act as a storage.
hydro_units = [u for u, t in unit_to_generation_type.items() if t == 7]

# Generation costs.
generation_type_to_cost = {
    0: 29.0,
    1: 85.0,
    2: 47.0,
    3: 78.0,
    4: 62.0,
    5: 33.0,
    6: 9.0,
    7: 5.0,
    8: 0.0,
    9: 0.0,
    10: 15.0,
    11: 45.0,
    12: 46.0,
    13: 37.0,
    14: 38.0,
    15: 28.0,
    16: 25.0,
    17: 22.0,
}

C_g = np.array([], dtype=np.float32)
for u in existing_units:
    cost = generation_type_to_cost[unit_to_generation_type[u]]
    C_g = np.append(C_g, [cost * np.random.uniform(1.0, 1.0)])

# Greenhouse gas emissions.
# The first figure is emissions factor in kg/kWh, which is the same as tonne/MWh.
# The second one (if applicable and known) is efficiency of the generation type.
# 0: coal
# 1: gas
# 2: ccgt
# 3: oil
# 4: biomass
# 5: oil shale
# 6: nuclear
# 7: hydro
# 8: wind
# 9: pv
# different chp types (e=extraction, b=back pressure):
# 10: coal-e
# 11: gas-b
# 12: gas-e
# 13: oil-b
# 14: oil-e
# 15: biomass-e
# 16: waste-e
# 17: peat-e
generation_type_to_emissions = {
    0: 0.34 / 0.41,
    1: 0.2 / 0.4,
    2: 0.2 / 0.54,
    3: 0.28 / 0.39,
    4: 0.0 / 0.35,
    5: 0.39 / 0.37,
    6: 0.0 / 0.33,
    7: 0.0 / 0.85,
    8: 0.0,
    9: 0.0,
    10: 0.34 / 0.81,
    11: 0.2 / 0.85,
    12: 0.2 / 0.85,
    13: 0.28 / 0.85,
    14: 0.28 / 0.85,
    15: 0.0 / 0.81,
    16: 0.33 / 0.61,
    17: 0.38 / 0.81,
}

G_emissions = np.array([], dtype=np.float32)
for u in existing_units:
    emissions = generation_type_to_emissions[unit_to_generation_type[u]]
    G_emissions = np.append(G_emissions, [emissions])

# Maximum emissions by year.
start_emissions = 50000.0
final_emissions = 5000.0
emission_targets = np.linspace(start_emissions, final_emissions, num_years)

assert len(emission_targets) == num_years

# Generate a candidate wind power unit for each real node.
candidate_units = []
candidate_unit_to_node = dict()
unit_idx = len(existing_units)

generate_candidate_units = True

if generate_candidate_units:
    for node_idx, node in enumerate(real_nodes):
        # Set mapping from node to unit id.
        candidate_units.append(unit_idx)
        candidate_unit_to_node[unit_idx] = node_idx

        # Set generation parameters.
        generation_type = 8  # Wind.
        unit_to_generation_type[unit_idx] = generation_type
        cost = generation_type_to_cost[generation_type]
        emissions = generation_type_to_emissions[generation_type]

        C_g = np.append(C_g, [cost * np.random.uniform(1.0, 1.0)])
        G_max = np.concatenate((G_max, [[10000.0]]), axis=1)
        G_emissions = np.append(G_emissions, [emissions])
        unit_idx += 1

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

# Apply wind and PV rates.
wind_units = [u for u, t in unit_to_generation_type.items() if t == wind_unit_idx]
pv_units = [u for u, t in unit_to_generation_type.items() if t == pv_unit_idx]

for u in wind_units:
    G_max[:, :, u] = np.max(G_max[:, :, u] * wind_rates[:, unit_to_node[u]], 0)

for u in pv_units:
    G_max[:, :, u] = np.max(G_max[:, :, u] * pv_rates[:, unit_to_node[u]], 0)

# Rates for the candidate units.
candidate_rates = dict()

for u in candidate_units:
    candidate_rates[u] = wind_rates[:, unit_to_node[u]]

C_g = np.tile(C_g, (num_scenarios, num_hours, 1))

G_emissions = np.tile(G_emissions, (num_scenarios, num_hours, 1))

# Ramping limits from time step to another (both up- and down-ramp) as a percentage of total
# generation capacity.
# 0: coal
# 1: gas
# 2: ccgt
# 3: oil
# 4: biomass
# 5: oil shale
# 6: nuclear
# 7: hydro
# 8: wind
# 9: pv
# different chp types (e=extraction, b=back pressure):
# 10: coal-e
# 11: gas-b
# 12: gas-e
# 13: oil-b
# 14: oil-e
# 15: biomass-e
# 16: waste-e
# 17: peat-e
generation_type_to_ramp_rate = {
    0: 0.2,
    1: 0.5,
    2: 0.5,
    3: 0.7,
    4: 0.2,
    5: 0.4,
    6: 0.1,
    7: 0.3,
    8: 1.0,
    9: 1.0,
    10: 0.2,
    11: 0.3,
    12: 0.3,
    13: 0.7,
    14: 0.7,
    15: 0.2,
    16: 0.2,
    17: 0.2,
}

G_ramp_max = np.zeros_like(G_max)

for u in units:
    t = unit_to_generation_type[u]
    G_ramp_max[:, :, u] = G_max[:, :, u] * generation_type_to_ramp_rate[t]

G_ramp_min = -G_ramp_max

assert (
    G_max.shape[-1]
    == C_g.shape[-1]
    == G_emissions.shape[-1]
    == G_ramp_max.shape[-1]
    == num_units
)

# Set initial hydro reservoir to the weeks which the sampled days belong to.
# Set maximum hydro reservoir as a new constraint.
# FI: http://wwwi2.ymparisto.fi/i2/95/fie7814.txt
# SE: https://www.energiforetagen.se/globalassets/energiforetagen/statistik/el/vecko--och-manadsstatistik/vecka_01-52_2014_ver_a.pdf?v=5Lem8eD7Yda4I_l3j6tHMZx4Gus
# NO: https://energifaktanorge.no/en/norsk-energiforsyning/kraftproduksjon/
# LV, LT: Entso-e

weekly_inflow = np.genfromtxt("inflow.csv", delimiter=";", skip_header=1)
weekly_reservoir = np.genfromtxt("reservoir.csv", delimiter=";", skip_header=1)

initial_storage = {u: np.zeros((num_scenarios, num_years)) for u in hydro_units}
inflows = {u: np.zeros((num_scenarios, num_hours)) for u in hydro_units}

# Map FI, NO, SE to columns in the inflow and reservoir CSV files.
nodemap = {3: 0, 4: 1, 5: 2, 6: 1, 7: 2}

for y, idx in enumerate(start_indices):
    week = int(idx / (7 * 24))     # Fix. 01/01/2014 is Wednesday.

    for u in hydro_units:
        n = nodemap[unit_to_node[u]]
        initial_storage[u][:, y] = weekly_reservoir[week, n]
        # Convert weekly inflow to hourly.
        inflows[u][:, y*num_hours_per_year:(y+1)+num_hours_per_year] = weekly_inflow[week, n] / (24 * 7)

# TODO
# Maximum storage levels.
max_storage_by_node = {n: 0.0 for n in real_nodes}
# Max
max_storage_by_node[3] = 3950
max_storage_by_node[6] = 69430
max_storage_by_node[7] = 25140
max_storage = dict()

# Build lines x nodes incidence matrix for existing lines.
# List pairs of nodes that are connected. Note: these are in 1-based indexing.
# node ids:
# dk1: 1
# dk2: 2
# ee: 3
# fi: 4
# lt: 5
# lv: 6
# no: 7
# se: 8
lines = [
    (7, 9),
    (7, 10),
    (7, 11),
    (7, 12),
    (7, 1),
    (9, 8),
    (10, 8),
    (11, 8),
    (12, 8),
    (8, 1),
    (1, 2),
    (2, 8),
    (8, 13),
    (8, 14),
    (13, 4),
    (14, 4),
    (4, 3),
    (3, 6),
    (6, 5),
]

# These lines are DC. Others are AC.
dc_lines = [4, 9, 10, 13, 15, 16]

num_existing_lines = len(lines)
assert num_existing_lines, "Incorrect amount of existing lines"
existing_lines = list(range(len(lines)))

incidence = np.zeros((num_existing_lines, num_nodes))

for line_idx, line in enumerate(lines):
    start, end = sorted(line)

    # Correct for the 1-based indexing in node numbering (i.e. make it 0-based).
    incidence[line_idx, start - 1] = -1
    incidence[line_idx, end - 1] = 1

F_max = np.array(
    [
        [
            650,
            150,
            600,
            2145,
            950,
            650,
            150,
            600,
            2145,
            680,
            590,
            1700,
            1480,
            1200,
            1480,
            1200,
            860,
            750,
            1234,
        ]
    ],
    dtype=np.float32,
)

F_min = np.array(
    [
        [
            -450,
            -250,
            -1000,
            -2095,
            -1000,
            -450,
            -250,
            -1000,
            -2095,
            -740,
            -600,
            -1300,
            -1120,
            -1200,
            -1120,
            -1200,
            -1016,
            -779,
            -684,
        ]
    ],
    dtype=np.float32,
)

assert num_existing_lines == len(incidence) == len(F_max[0]) == len(F_min[0])

# Generate candidate lines between each pair of (real and dummy) nodes.
line_idx = len(existing_lines)
candidate_lines = []
generate_candidate_lines = True

if generate_candidate_lines:
    for i in range(num_real_nodes):
        for j in range(i + 1, num_real_nodes):
            row = np.zeros((1, num_nodes))
            row[0, i] = -1.0
            row[0, j] = 1.0

            incidence = np.concatenate((incidence, row), axis=0)
            F_max = np.concatenate((F_max, [[2000.0]]), axis=1)
            F_min = np.concatenate((F_min, [[-2000.0]]), axis=1)

            candidate_lines.append(line_idx)
            line_idx += 1
else:
    candidate_lines = []

# Update the total transmission lines.
lines = existing_lines + candidate_lines
num_lines = len(lines)

# Augment the transmission capacity arrays to have expected dimensions.
F_max = np.tile(F_max, (num_scenarios, num_hours, 1))
F_max += np.random.uniform(-10.0, 10.0, (num_scenarios, num_hours, num_lines))

F_min = np.tile(F_min, (num_scenarios, num_hours, 1))
F_min += np.random.uniform(-10.0, 10.0, (num_scenarios, num_hours, num_lines))

# Susceptance.
B = np.ones(num_lines) * 1e3

# Reference node.
ref_node = 0

assert len(incidence) == F_max.shape[-1] == F_min.shape[-1] == num_lines

# Equal scenario weights.
weights = np.ones(num_scenarios) / float(num_scenarios)

# Investment parameters
discount_factor = 1.10

C_x = {
    (year, unit): 1e6 * discount_factor ** (-year)
    for year in years
    for unit in candidate_units
}
C_y = {
    (year, line): 1e8 * discount_factor ** (-year)
    for year in years
    for line in candidate_lines
}
