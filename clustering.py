# Define data used by both the master problem and subproblem.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize

np.random.seed(13)


# Load data used for clustering
load = np.genfromtxt("load.csv", delimiter=";", skip_header=1)

wind = np.genfromtxt("wind_rates.csv", delimiter=";", skip_header=1)[:, 1:]

load = normalize(load, axis=0)
wind = normalize(wind, axis=0)

def reshape(arr, num_hours_to_average=1):
    """Reshape array for clustering."""
    # Average every num_hours_to_average elements.
    cols = arr.shape[1]
    arr2 = np.transpose(arr)    # 8x8760
    arr2 = np.reshape(arr2, (-1, num_hours_to_average)) # (8*8760/2) x 2
    arr2 = np.mean(arr2, axis=1) # (8*8760/2) x 1
    arr2 = np.reshape(arr2, (cols, -1)) # 8 x (8760/2)
    arr2 = np.transpose(arr2)

    # By default, 8760x8 array is reshaped into 365x(24*8)
    num_hours = 24 / num_hours_to_average
    nrows = arr2.shape[0]
    nrows2 = int(nrows / num_hours)
    assert nrows2 == nrows / num_hours == 365
    arr3 = np.reshape(arr2, (nrows2, -1))

    return arr3

num_hours_to_average = 2

load = reshape(load, num_hours_to_average)
wind = reshape(wind, num_hours_to_average)

data = np.concatenate((load, wind), axis=1)

print(load.shape)
print(wind.shape)
print(data.shape)

n_clusters = 3

clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")

labels = clustering.fit_predict(data)
weights = [list(labels).count(i) / len(labels) for i in range(n_clusters)]
print(weights)

centers = []
for i in range(n_clusters):
    center_data = data[labels == i]
    centers.append(np.mean(center_data, axis=0))

def find_nearest(arr, val):
    """Find nearest data point to a value."""
    diffs = np.abs(arr - val)
    total_diffs = np.sum(diffs, axis=1)
    idx = np.argmin(total_diffs)
    return idx

days = [find_nearest(data, centers[i]) for i in range(n_clusters)]
weekdays = [(datetime.datetime(2014, 1, 1) + datetime.timedelta(day)).isoweekday() for day in days]
approx_months = [int(day / 30) + 1 for day in days]

print("Representative days are the indices:", days)
print("The respective weekdays are:", weekdays)
print("They are approximately in these months:", approx_months)

weights = [w for d, w in sorted(zip(days, weights))]
days = sorted(days)

np.save("representative_days.npy", np.array(days))
np.save("scenario_weights.npy", np.array(weights))