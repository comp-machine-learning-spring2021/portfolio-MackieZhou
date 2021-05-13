# hw2.py
# Mackie Zhou

import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from scipy.spatial import distance


def recompute_centers(labels, data_pd, k):
    """
    labels: labels of each point, numpy array
    data_pd: dataset, a pandas DataFrame
    k: the number of centers, an integer
    """
    centers = np.empty([0, 2])

    # loop over all clusters
    for i in range(k):
        cluster = []  # index of all points in this cluster

        # loop over all observations
        for j in range(len(labels)):
            if labels[j] == i:
                cluster.append(j)

        subset = data_pd.iloc[cluster, :]
        center = [subset.iloc[:, 0].mean(),
                  subset.iloc[:, 1].mean()]
        centers = np.vstack((centers, center))

    return centers


def my_kmeans(data, k, random):
    """
    data: the dataset as a numpy array
    k: the number of clusters as an integer
    random: random_state as an integer

    output includes
    1. centers: the cluster centers and
    2. labels: cluster labels for the data points
    """
    # create a pandas DataFrame of data
    data_pd = pd.DataFrame(data)

    # randomly pick centers for the first iteration
    centers = data_pd.sample(k, random_state=random)

    # stopping method: interate for a maximum of 300 times
    i = 0
    max_iter = 300
    while i <= max_iter:
        i += 1

        # reassign clusters
        dists = distance.cdist(data, centers, 'euclidean')
        labels = np.argmin(dists, axis=1)
        # recompute centers
        centers = recompute_centers(labels, data_pd, k)

    return (centers, labels)


def calculate_within_cluster_sse(fit, data):
    """
    fit: the fit get from sklearn
    data: a numpy array
    """
    SSE = 0

    # Loop over all clusters
    for c in range(len(fit.cluster_centers_)):
        # Extract the cluster's center and associated points:
        center = [fit.cluster_centers_[c]]
        points = data[np.where(fit.labels_ == c)]
        # Compute the following for each cluster:
        cluster_spread = distance.cdist(points, center, 'euclidean')
        cluster_total = np.sum(cluster_spread)
        # Add this cluster's within sum of squares to within_cluster_sumsqs
        SSE += cluster_total

    return SSE


def looping_kmeans(data, kList):
    """
    data: a numpy array
    kList: a list of all k values
    """

    # 1. normalize the data
    shape = data.shape
    data_norm = np.empty([shape[0], 0])

    for i in range(shape[1]):
        var = data[:, i]
        mx = np.max(var)
        mn = np.min(var)

        var_norm = (var - mn)/(mx - mn)
        var_norm = np.around(var_norm, decimals=2)
        data_norm = np.hstack((data_norm, np.array([var_norm]).transpose()))

    # 2. calculate within-cluster SSE for each K
    SSE_list = []
    for i in kList:
        km = KMeans(n_clusters=i, init="k-means++",
                    random_state=1, max_iter=200)
        fit = km.fit(data_norm)
        SSE = calculate_within_cluster_sse(fit, data_norm)
        SSE_list.append(SSE)

    return SSE_list
