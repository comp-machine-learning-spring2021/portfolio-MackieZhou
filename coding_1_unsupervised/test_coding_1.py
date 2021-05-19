# revision of test_hw2.py
# Mackie Zhou
# May 2021

import pandas as pd
import numpy as np
import coding_1

students = pd.read_csv("coding_1_unsupervised\students_info.csv")
justthree_np = students[["sleep", "coffee", "gpa"]].to_numpy()


def test_my_kmeans_type():
    assert isinstance(coding_1.my_kmeans(justthree_np, 6), tuple)


def test_my_kmeans_shape():
    expected = 2
    assert len(coding_1.my_kmeans(justthree_np, 6)) == expected


def test_my_kmeans_center_shape():
    expected = (6, 3)
    centers_shape = coding_1.my_kmeans(justthree_np, 6)[0].shape
    assert centers_shape == expected


def test_my_kmeans_labels():
    expected = 5
    label_max = np.max(coding_1.my_kmeans(justthree_np, 6)[1])
    assert label_max == expected


def test_different_cols():
    expected = False
    centers = coding_1.my_kmeans(justthree_np, 6)[0]
    comp_cols = sum(centers[:, 0] == centers[:, 1]) == 6
    assert comp_cols == expected


def test_looping_kmeans_sklearn_type():
    assert isinstance(coding_1.looping_kmeans_sklearn(justthree_np,
                                                      list(range(2, 6))), list)


def test_looping_kmeans_sklearn_size():
    expected = 4
    assert len(coding_1.looping_kmeans_sklearn(justthree_np,
                                               list(range(2, 6)))) == expected


def test_looping_kmeans_sklearn_goodness():
    out = coding_1.looping_kmeans_sklearn(justthree_np, list(range(2, 6)))
    assert (out[1:] <= out[:-1])


def test_looping_kmeans_my_type():
    assert isinstance(coding_1.looping_kmeans_my(justthree_np,
                                                 list(range(2, 6))), list)


def test_looping_kmeans_my_size():
    expected = 4
    assert len(coding_1.looping_kmeans_my(justthree_np,
                                          list(range(2, 6)))) == expected


def test_looping_kmeans_my_goodness():
    out = coding_1.looping_kmeans_my(justthree_np, list(range(2, 6)))
    assert (out[1:] <= out[:-1])
