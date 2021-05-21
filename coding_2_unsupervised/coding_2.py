import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as hc


# I consulted this notebook in writing the replace_null_mean function
# https://www.kaggle.com/sabanasimbutt/clustering-visualization-of-clusters-using-pca#Data-Preprocessing


def replace_null_mean(data, features):
    """replace null values in features with mean

    Args:
        data (pandas.DataFrame): the input dataset
        features (list): the list of features that need to be processed

    Returns:
        pandas.DataFrame
    """
    for feature in features:
        mean = data[feature].mean()
        data.loc[(data[feature].isnull() == True), feature] = mean

    return data


# below is the function that plots hierarchical clustering dendrogram
# from scikit learn:
# https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html#sphx-glr-auto-examples-cluster-plot-agglomerative-dendrogram-py


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    hc.dendrogram(linkage_matrix, **kwargs)
