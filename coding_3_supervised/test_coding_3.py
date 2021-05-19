# revision of hw5
# Mackie Zhou
# May 2021

import pytest
import pandas as pd
import numpy as np
import coding_3

hw_data = np.genfromtxt("hw6data.csv", delimiter=',')


def test_tenfold_type():
    CV = coding_3.ten_fold(hw_data, 'rbf', 0)
    assert isinstance(CV, float)
