# revision of hw2.py
# Mackie Zhou
# May 2021

import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from scipy.spatial import distance
from sklearn import preprocessing


def recompute_centers(labels, data_pd, k):
    """
    labels: labels of each point, numpy array
    data_pd: dataset, a pandas DataFrame
    k: the number of centers, an integer
    """
    n_features = data_pd.shape[1]
    centers = np.empty([0, n_features])

    # loop over all clusters
    for i in range(k):
        cluster = []  # index of all points in this cluster

        # loop over all observations
        for j in range(len(labels)):
            if labels[j] == i:
                cluster.append(j)

        subset = data_pd.iloc[cluster, :]
        center = []
        for i in range(n_features):
            center.append(subset.iloc[:, i].mean())
        centers = np.vstack((centers, center))

    return centers


def my_kmeans(data, k):
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
    centers = data_pd.sample(k, random_state=2021)

    # stopping method: interate for a maximum of 300 times
    i = 0
    max_iter = 300
    labels = np.zeros(data_pd.shape[0])
    while i <= max_iter:
        i += 1

        # reassign clusters
        dists = distance.cdist(data, centers, 'euclidean')

        # stopping method: grouping assignments don't change
        new_labels = np.argmin(dists, axis=1)
        if (new_labels == labels).all():
            break
        else:
            labels = new_labels

        # recompute centers
        centers = recompute_centers(labels, data_pd, k)

    return (centers, labels)


def calculate_within_cluster_sse(centers, labels, data):
    """
    centers: list, centers got from a kmeans model
    labels: list, labels of all points got from a kmeans model
    data: np.ndarray, data

    return: float, SSE
    """
    SSE = 0

    # Loop over all clusters
    for c in range(len(centers)):
        # Extract the cluster's center and associated points:
        center = centers[c]
        points = data[np.where(np.array(labels) == c)]
        # Compute the following for each cluster:
        # print("points\n", points)
        # print("\ncenter\n", [center])
        cluster_spread = distance.cdist(points, [center], 'euclidean')
        cluster_total = np.sum(cluster_spread)
        # Add this cluster's within sum of squares to within_cluster_sumsqs
        SSE += cluster_total

    return SSE


def looping_kmeans_sklearn(data, kList):
    """
    data: np.ndarray
    kList: list, a list of all k values

    return: a list of SSE
    """

    # 1. normalize the data
    # shape = data.shape
    data_norm = preprocessing.normalize(data)

    # 2. calculate within-cluster SSE for each K
    SSE_list = []
    for k in kList:
        km = KMeans(n_clusters=k, init="k-means++",
                    random_state=2021, max_iter=300)
        fit = km.fit(data_norm)
        centers = fit.cluster_centers_
        labels = fit.labels_
        SSE = calculate_within_cluster_sse(centers, labels, data_norm)
        SSE_list.append(SSE)

    return SSE_list


def looping_kmeans_my(data, kList):
    """
    data: np.ndarray
    kList: list, a list of all k values

    return: a list of SSE
    """

    # 1. normalize the data
    # shape = data.shape
    data_norm = preprocessing.normalize(data)

    # 2. calculate within-cluster SSE for each K
    SSE_list = []
    for k in kList:
        centers, labels = my_kmeans(data_norm, k)

        # print("centers\n", centers)
        # print("\nlabels\n", labels)
        SSE = calculate_within_cluster_sse(centers, labels, data_norm)
        SSE_list.append(SSE)

    return SSE_list
