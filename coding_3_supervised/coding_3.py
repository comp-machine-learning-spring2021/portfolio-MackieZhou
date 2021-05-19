# revision of hw5
# Mackie Zhou
# May 2021

import numpy as np
import random
from sklearn.svm import SVC


def split_val_data(all_data, percent):
    """ split the dataset into validation and train/test sets, "percent" percents of all_data go into the validation set

    Args:
        all_data (numpy.ndarray): the dataset you want to split
        percent (int): the percentage of data to be put into the validation set

    Returns:
        (numpy.ndarray, numpy.ndarray): (validation dataset, train/test dataset)
    """

    n_all = all_data.shape[0]
    n_val = int(n_all * percent / 100)

    # validation dataset
    val_inds = random.sample(list(range(n_all)), n_val)
    val_data = all_data[val_inds, :]

    # train/test dataset
    data_inds = list(set(range(n_all)).difference(set(val_inds)))
    data = all_data[data_inds, :]

    return val_data, data


def ten_fold(features, target, kernel, degree=3):
    """ten-fold cross-validation with SVM

    Args:
        features (numpy.ndarray): all features that will be fed into SVM
        target (np.ndarray): the actual classes of each datapoint
        kernel (string): the kernel used in SVM
        degree (int): degree of the polynomial kernel function 

    Returns:
        float: cross-validated error
    """

    n = len(target)  # number of data points
    group_n = n//10  # numbrt of data points in each group

    # implement ten_fold cross-val
    i = 0
    test_errors = []
    while i < 9:
        # test data
        test_inds = list(range(n))[group_n*i:group_n*(i+1)]
        test_data = features[test_inds, :]
        test_target = target[test_inds]

        # train data
        train_inds = list(set(range(n)).difference(set(test_inds)))
        train_data = features[train_inds, :]
        train_target = target[train_inds]

        # tune the model
        if kernel == "poly":
            SVM = SVC(kernel='poly', degree=degree, gamma='auto', C=1000)
        else:
            SVM = SVC(kernel=kernel, C=1000)
        SVM.fit(train_data, train_target)

        # calculate test error
        test_preds = SVM.predict(test_data)
        test_error = np.mean((test_preds - test_target)**2)
        test_errors.append(test_error)

        # update i at the END of the loop!!!
        i += 1

    # test data
    test_inds = list(range(n))[group_n*i:]
    test_data = features[test_inds, :]
    test_target = target[test_inds]

    # train data
    train_inds = list(set(range(n)).difference(set(test_inds)))
    train_data = features[train_inds, :]
    train_target = target[train_inds]

    # tune the model
    if kernel == "poly":
        SVM = SVC(kernel='poly', degree=degree, gamma='auto', C=1000)
    else:
        SVM = SVC(kernel=kernel, C=1000)
    SVM.fit(train_data, train_target)

    # calculate test error
    test_preds = SVM.predict(test_data)
    test_error = np.mean((test_preds - test_target)**2)
    test_errors.append(test_error)

    return np.average(test_errors)
