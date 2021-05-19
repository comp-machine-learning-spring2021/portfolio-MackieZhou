# Mackie Zhou
# May 2021

import pytest
import pandas as pd
import numpy as np
import coding_3

data = np.genfromtxt("hw6data.csv", delimiter=',')


def split_val_data_shape():
    assert coding_3.split_val_data(data, 10).shape == (1, 2)


def split_val_data_type():
    assert isinstance(coding_3.split_val_data(data, 10), tuple)
