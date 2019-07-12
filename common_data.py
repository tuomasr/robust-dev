# Define data used by both the master problem and subproblem.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

np.random.seed(13)

# Configure Gurobi.

# Solver method for master problem and subproblem.
# -1 = automatic
# 2 = Barrier + Crossover
# 3 = concurrent
# 4 = deterministic concurrent (runs out of memory for bigger instances).
master_method = 3
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
uncertainty_demand_increase_factor = 100.0
uncertainty_budget = 2.0

# Scenarios.
num_scenarios = 3
scenarios = list(range(num_scenarios))

# Years and hours.
num_years = 10
num_hours_per_year = 24
num_hours = num_years * num_hours_per_year

annualizer = 365.0 * 24.0 / float(num_hours_per_year)

years = list(range(num_years))
hours = list(range(num_hours))

# Load representative days
representative_days = np.load("representative_days.npy")

assert num_scenarios == len(representative_days)

start_indices = representative_days*24
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
real_node_names = ["DK1", "DK2", "EE", "FI", "LT", "LV", "NO", "SE"]

num_real_nodes = 8
real_nodes = list(range(num_real_nodes))

num_nodes = 14
nodes = list(range(num_nodes))

# Load.
load_real_nodes = np.genfromtxt("load.csv", delimiter=";", skip_header=1)
load = np.zeros((num_scenarios, num_hours, num_real_nodes))

# Read load for the sampled days.
load_growth = 1.01  # Yearly growth.

uncertainty_demand_increase = np.ones_like(load) * uncertainty_demand_increase_factor

for y in years:
    for o, (start, end) in enumerate(zip(start_indices, end_indices)):
        load_growth_factor = load_growth ** y
        load_slice = load_real_nodes[start:end]

        h1 = y*num_hours_per_year
        h2 = (y+1)*num_hours_per_year

        uncertainty_demand_increase[o, h1:h2, :] *= load_growth_factor
        load[o, h1:h2, :] = load_slice * load_growth_factor

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

wind_rates = np.zeros_like(load)
pv_rates = np.zeros_like(load)

for y in years:
    for o, (start, end) in enumerate(zip(start_indices, end_indices)):
        wind_slice = all_wind_rates[start:end, 1:]  # Skip first (hour) column.
        pv_slice = all_pv_rates[start:end, 1:]

        h1 = y*num_hours_per_year
        h2 = (y+1)*num_hours_per_year

        wind_rates[o, h1:h2, :] = wind_slice
        pv_rates[o, h1:h2, :] = pv_slice

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
# See also: https://www.ipcc-nggip.iges.or.jp/public/2006gl/pdf/2_Volume2/V2_2_Ch2_Stationary_Combustion.pdf
generation_type_to_emissions = {
    0: 0.34 / 0.41,
    1: 0.2 / 0.4,
    2: 0.2 / 0.54,
    3: 0.28 / 0.39,
    4: 0.0 / 0.35,  # Biomass not in ETS, debated. First value is estimated to be 0.2 - 0.3.
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
# See:
# https://www.eea.europa.eu/data-and-maps/data/data-viewers/greenhouse-gases-viewer
# https://emis.vito.be/sites/emis.vito.be/files/articles/3331/2016/CO2EmissionsfromFuelCombustion_Highlights_2016.pdf
start_emissions = 90000.0
final_emissions = 35000.0   # approx. 90 000 * (1-0.022)^9.
emission_targets = np.linspace(start_emissions, final_emissions, num_years)

assert len(emission_targets) == num_years

# Generate a candidate wind power unit for each real node.
candidate_units = []
candidate_unit_types = [1, 2, 3, 4, 8, 9]  # gas, CCGT, oil, biomass, wind, solar.
candidate_unit_type_names = ["Gas", "CCGT", "Oil", "Biomass", "Wind", "Solar"]
candidate_unit_to_node = dict()
unit_idx = len(existing_units)

generate_candidate_units = True
maximum_candidate_unit_capacity_by_type = {
    1: 10000.0,
    2: 10000.0,
    3: 10000.0,
    4: 1000.0,  # Biomass potential.
    8: 10000.0,
    9: 10000.0,
}

if generate_candidate_units:
    for node_idx, node in enumerate(real_nodes):
        for candidate_type in candidate_unit_types:
            # Set mapping from node to unit id.
            candidate_units.append(unit_idx)
            candidate_unit_to_node[unit_idx] = node_idx

            # Set generation parameters.
            unit_to_generation_type[unit_idx] = candidate_type
            cost = generation_type_to_cost[candidate_type]
            emissions = generation_type_to_emissions[candidate_type]

            C_g = np.append(C_g, [cost * np.random.uniform(1.0, 1.0)])
            max_capacity = maximum_candidate_unit_capacity_by_type[candidate_type]
            G_max = np.concatenate((G_max, [[max_capacity]]), axis=1)
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
G_max += np.random.uniform(-50.0, 0.0, (num_scenarios, num_hours, num_units))
G_max[:, :, :] = np.maximum(G_max[:, :, :], 0.0)
G_max = np.round(G_max, 0)  # Capacities are not fractional usually.

# Gather availability rates for all generation types.
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
conventional_rates = {
    0: 0.9,
    1: 0.95,
    2: 0.8,
    3: 0.86,
    4: 0.95,
    5: 0.9,
    6: 0.9,
    # TODO: Better value for this.
    7: 0.9,
    # Wind and PV are handled separately.
    10: 0.9 / 2.74,
    11: 0.95 / 2.29,
    12: 0.95 / 1.85,
    13: 0.95 / 2.29,
    14: 0.95 / 1.85,
    15: 0.95 / 3.35,
    16: 0.9 / 4.64,
    17: 0.95 / 3.35
}

availability_rates = np.zeros((len(scenarios), len(hours), len(units)))
wind_unit_idx = 8
pv_unit_idx = 9

for u in units:
    t = unit_to_generation_type[u]

    if t == wind_unit_idx:
        availability_rates[:, :, u] = wind_rates[:, :, unit_to_node[u]]
    elif t == pv_unit_idx:
        availability_rates[:, :, u] = pv_rates[:, :, unit_to_node[u]]
    else:
        availability_rates[:, :, u] = conventional_rates[t]

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
ramp_rates = np.zeros((len(scenarios), len(hours), len(units)))
for u in units:
    t = unit_to_generation_type[u]
    ramp_rates[:, :, u] = generation_type_to_ramp_rate[t]

assert (
    G_max.shape[-1]
    == C_g.shape[-1]
    == G_emissions.shape[-1]
    == num_units
)

# Set initial hydro reservoir to the weeks which the sampled days belong to.
# FI: http://wwwi2.ymparisto.fi/i2/95/fie7814.txt
# SE: https://www.energiforetagen.se/globalassets/energiforetagen/statistik/el/vecko--och-manadsstatistik/vecka_01-52_2014_ver_a.pdf?v=5Lem8eD7Yda4I_l3j6tHMZx4Gus
# NO: https://energifaktanorge.no/en/norsk-energiforsyning/kraftproduksjon/
# LV, LT: Entso-e

weekly_inflow = np.genfromtxt("inflow.csv", delimiter=";", skip_header=1)
weekly_reservoir = np.genfromtxt("reservoir.csv", delimiter=";", skip_header=1)

hydro_unit_idx = 7
hydro_units = [u for u, t in unit_to_generation_type.items() if t == hydro_unit_idx]

initial_storage = {u: np.zeros((num_scenarios, num_years)) for u in hydro_units}
inflows = {u: np.zeros((num_scenarios, num_hours)) for u in hydro_units}

# Map FI, LT, LV, NO, SE to columns in the inflow and reservoir CSV files.
nodemap = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4}

# Scale initial reservoir down to avoid numerical issues as the initial reservoir figures are
# much larger than other parameter values. Take the scaling into account in final storage
# lower and upper bound constraints. The scaling values should be such that initial reservoir
# values still remain higher than maximum hydro power generation.
scalers = {
    0: 1.0,
    1: 1.0,
    2: 1.0,
    3: 1.0,
    4: 1.0,
    5: 1.0,
    6: 1.0,
    7: 1.0,
}

for y in years:
    for o, idx in enumerate(start_indices):
        week = int(idx / (7 * 24))     # Fix. 01/01/2014 is Wednesday.

        for u in hydro_units:
            n = nodemap[unit_to_node[u]]

            h1 = y*num_hours_per_year
            h2 = (y+1)*num_hours_per_year

            initial_storage[u][o, y] = weekly_reservoir[week, n] / scalers[n]
            # Convert weekly inflow to hourly.
            inflows[u][o, h1:h2] = weekly_inflow[week, n] / 168

storage_change_ub = {
    0: 0.0,
    1: 0.0,
    2: 0.0,
    3: 0.145, #0.05802,
    4: 0.233, #0.14784,
    5: 0.825, #0.16817,
    6: 0.143, #0.05178,
    7: 0.204, #0.04949,
}
storage_change_ub = {n: ((1.0 + v / 168 * num_hours_per_year) * scalers[n] - (scalers[n] - 1.0)) for n, v in storage_change_ub.items()}

storage_change_lb = {
    0: 0.0,
    1: 0.0,
    2: 0.0,
    3: -0.044, #-0.0330,
    4: -0.198, #-0.1342,
    5: -0.449, #-0.1523,
    6: -0.055, #-0.0415,
    7: -0.07, #-0.0633,
}
storage_change_lb = {n: ((1.0 + v / 168 * num_hours_per_year) * scalers[n] - (scalers[n] - 1.0)) for n, v in storage_change_lb.items()}

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

# Lists for AC and DC lines.
dc_lines = set([4, 9, 10, 13, 15, 16])
ac_lines = set(range(len(lines))) - dc_lines

num_existing_lines = len(lines)
assert num_existing_lines, "Incorrect amount of existing lines"
existing_lines = list(range(len(lines)))

incidence = np.zeros((num_existing_lines, num_nodes))

ac_nodes = set()    # The set of nodes which are part of the AC circuit.
dc_nodes = set()

for line_idx, line in enumerate(lines):
    start, end = sorted(line)
    # Correct for the 1-based indexing in node numbering (i.e. make it 0-based).
    start, end = start - 1, end - 1

    incidence[line_idx, start] = -1
    incidence[line_idx, end] = 1

    if line_idx in ac_lines:
        ac_nodes.add(start)
        ac_nodes.add(end)
    else:
        dc_nodes.add(start)
        dc_nodes.add(end)

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

# Generate candidate lines between each pair of real nodes.
line_idx = len(existing_lines)
candidate_lines = []
candidate_line_capacity = 1000.0
generate_candidate_lines = True

num_build_options = 1

# 0: dk1
# 1: dk2
# 2: ee
# 3: fi
# 4: lt
# 5: lv
# 6: no
# 7: se
neighbors = {
    0: [1, 6, 7],
    1: [1, 6, 7],
    2: [3, 5],
    3: [7, 2, 6],
    4: [5],
    5: [2, 4],
    6: [0, 1, 3, 7],
    7: [0, 1, 3, 6],
}

if generate_candidate_lines:
    for i in range(num_real_nodes):
        for j in range(i + 1, num_real_nodes):
            # A line can be only built if the two real nodes are neighbors, i.e., their
            # distance is not too high.
            if j in neighbors[i]:
                # num_build_options lines can be built between each pair of nodes.
                for _ in range(num_build_options):
                    row = np.zeros((1, num_nodes))
                    row[0, i] = -1.0
                    row[0, j] = 1.0

                    incidence = np.concatenate((incidence, row), axis=0)
                    F_max = np.concatenate((F_max, [[candidate_line_capacity]]), axis=1)
                    F_min = np.concatenate((F_min, [[-candidate_line_capacity]]), axis=1)

                    candidate_lines.append(line_idx)

                    # Assume all candidate lines are DC.
                    dc_lines.add(line_idx)
                    dc_nodes.add(i)
                    dc_nodes.add(j)

                    line_idx += 1
else:
    candidate_lines = []

# Update the total transmission lines.
lines = existing_lines + candidate_lines
num_lines = len(lines)

# Augment the transmission capacity arrays to have expected dimensions.
F_max = np.tile(F_max, (num_scenarios, num_hours, 1))
F_max += np.random.uniform(-50.0, 0.0, (num_scenarios, num_hours, num_lines))
F_max[:, :, :] = np.maximum(F_max[:, :, :], 0.0)
F_max = np.round(F_max, 0)  # Capacities are not usually fractional.

F_min = np.tile(F_min, (num_scenarios, num_hours, 1))
F_min += np.random.uniform(0.0, 50.0, (num_scenarios, num_hours, num_lines))
F_min[:, :, :] = np.minimum(F_min[:, :, :], 0.0)
F_min = np.round(F_min, 0)

# Susceptance.
B = np.ones(num_lines) * 1e5

# Reference node.
ref_node = 7

assert len(incidence) == F_max.shape[-1] == F_min.shape[-1] == num_lines

# Scenarios weights.
weights = np.load("scenario_weights.npy")

# Investment parameters.
discount_factor = 1.03

# Cost per megawatt.
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
# See also: https://www.eia.gov/outlooks/aeo/assumptions/pdf/table_8.2.pdf
cost_per_type = {
    1: 0.8,
    2: 1.0,
    3: 0.8,
    4: 3.9,
    8: 1.6,
    9: 1.8,
}   # In millions.

C_x = {
    (year, unit): 1e6 * cost_per_type[unit_to_generation_type[unit]] * discount_factor ** (-year)
    for year in years
    for unit in candidate_units
}
# Cost per project.
C_y = {
    (year, line): 1.0e9 * discount_factor ** (-year)
    for year in years
    for line in candidate_lines
}
