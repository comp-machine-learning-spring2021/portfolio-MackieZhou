import pytest
import pandas as pd
import numpy as np
import hw2

students = pd.read_csv("students_info.csv")
justtwo_np = students[["sleep","coffee"]].to_numpy()

# Tests for question 1.
# Note - These do not test what is contained within your clusters

def test_my_kmeans_type():
	assert isinstance(hw2.my_kmeans(justtwo_np, 6, 2019), tuple)

def test_my_kmeans_shape():
	expected = 2
	assert len(hw2.my_kmeans(justtwo_np, 6, 2019)) == expected

def test_my_kmeans_center_num():
	expected = (6,2)
	centers_shape = hw2.my_kmeans(justtwo_np, 6, 2019)[0].shape
	assert centers_shape == expected

def test_my_kmeans_labels():
	expected = 5
	label_max = np.max(hw2.my_kmeans(justtwo_np, 6, 2019)[1])
	assert label_max == expected

def test_different_cols():
	expected = False
	centers=hw2.my_kmeans(justtwo_np, 6, 2019)[0]
	comp_cols = sum(centers[:,0] == centers[:,1]) == 6
	assert comp_cols == expected

# Tests for question 2:

def test_looping_kmeans_type():
	assert isinstance(hw2.looping_kmeans(justtwo_np,
		list(range(1,15))), list)

def test_looping_kmeans_size():
	expected = 14
	assert len(hw2.looping_kmeans(justtwo_np,
		list(range(1,15)))) == expected

def test_looping_kmeans_goodness():
	out = hw2.looping_kmeans(justtwo_np,list(range(1,15)))
	assert (out[1:] <= out[:-1])
