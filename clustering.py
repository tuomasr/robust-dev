# Define representative days used as operating conditions in the robust optimization problem.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize

np.random.seed(13)

# Number of clusters (=scenarios/operating conditions).
N_CLUSTERS = 15

# Allows averaging consecutive hours to reduce the dimension of data points to cluster.
NUM_HOURS_TO_AVERAGE = 2

# 1: 0.0476
# 2: 0.0471
# 3: 0.049
# 4: 0.057
# 8: 0.045


def estimate_representative_days():
    """Run the algorithm to generate representative days."""
    # Load data used for clustering
    load = np.genfromtxt("load.csv", delimiter=";", skip_header=1)

    wind = np.genfromtxt("wind_rates.csv", delimiter=";", skip_header=1)[:, 1:]

    load = normalize(load, axis=0)
    wind = normalize(wind, axis=0)

    def reshape(arr, num_hours_to_average=1):
        """Reshape array for clustering."""
        # Average every num_hours_to_average elements.
        cols = arr.shape[1]
        arr2 = np.transpose(arr)  # 8x8760
        arr2 = np.reshape(arr2, (-1, num_hours_to_average))  # (8*8760/2) x 2
        arr2 = np.mean(arr2, axis=1)  # (8*8760/2) x 1
        arr2 = np.reshape(arr2, (cols, -1))  # 8 x (8760/2)
        arr2 = np.transpose(arr2)

        # By default, 8760x8 array is reshaped into 365x(24*8)
        num_hours = 24 / num_hours_to_average
        nrows = arr2.shape[0]
        nrows2 = int(nrows / num_hours)
        assert nrows2 == nrows / num_hours == 365
        arr3 = np.reshape(arr2, (nrows2, -1))

        return arr3

    load = reshape(load, NUM_HOURS_TO_AVERAGE)
    wind = reshape(wind, NUM_HOURS_TO_AVERAGE)

    data = np.concatenate((load, wind), axis=1)

    print(load.shape)
    print(wind.shape)
    print(data.shape)

    clustering = AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage="ward")

    labels = clustering.fit_predict(data)
    weights = [list(labels).count(i) / len(labels) for i in range(N_CLUSTERS)]
    print(weights)

    centers = []
    for i in range(N_CLUSTERS):
        center_data = data[labels == i]
        centers.append(np.mean(center_data, axis=0))

    def find_nearest(arr, val):
        """Find nearest data point to a value."""
        diffs = np.abs(arr - val)
        total_diffs = np.sum(diffs, axis=1)
        idx = np.argmin(total_diffs)
        return idx

    days = [find_nearest(data, centers[i]) for i in range(N_CLUSTERS)]

    # Sort weights and days by day number.
    weights = [w for d, w in sorted(zip(days, weights))]
    days = [d for d, w in sorted(zip(days, weights))]

    weekdays = [
        (datetime.datetime(2014, 1, 1) + datetime.timedelta(int(day))).isoweekday()
        for day in days
    ]
    months = [
        (datetime.datetime(2014, 1, 1) + datetime.timedelta(int(day))).month
        for day in days
    ]

    print("Representative days are the indices:", days)
    print("The respective ISO weekdays are:", weekdays)
    print("They are in these months:", months)

    weight_diff = np.mean(
        np.abs(np.array([w - 1 / float(N_CLUSTERS) for w in weights]))
    )

    print("Diff to ideal scenario weight:", weight_diff)

    np.save("representative_days.npy", np.array(days))
    np.save("scenario_weights.npy", np.array(weights))


def main():
    """Run representative days algorithm."""
    estimate_representative_days()


if __name__ == "__main__":
    main()
