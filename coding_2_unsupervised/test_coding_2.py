import numpy as np
import pandas as pd
import coding_2


data = pd.read_csv('coding_2_unsupervised\CC_GENERAL.csv', sep=',')


def test_replace_null_mean_shape():
    features = ['MINIMUM_PAYMENTS', 'CREDIT_LIMIT']
    out = coding_2.replace_null_mean(data, features)
    assert out.shape == data.shape


def test_replace_null_mean_type():
    features = ['MINIMUM_PAYMENTS', 'CREDIT_LIMIT']
    out = coding_2.replace_null_mean(data, features)
    assert isinstance(out, pd.DataFrame)
