# Mackie Zhou
# May 2021


import numpy as np
import coding_3

data = np.genfromtxt("coding_3_supervised\hw6data.csv", delimiter=',')
features = data[:, :2]
targets = data[:, 2]


def test_split_val_data_type():
    assert isinstance(coding_3.split_val_data(data, 10), tuple)


def test_split_val_data_shape():
    assert len(coding_3.split_val_data(data, 10)) == 2


def test_random_forest_cross_val_type():
    out = coding_3.random_forest_cross_val(10, features, targets, 10, 3, 2)
    assert isinstance(out, float)
