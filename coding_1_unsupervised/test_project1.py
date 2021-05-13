import pytest
import pandas as pd
import numpy as np
from project1 import top_20_rate_count, SVD, recommend


# preparing data for tests
rating = pd.read_csv('rating.csv', sep=',')
rating['rating'] = rating['rating'].apply(lambda x: np.nan if x == -1 else x)
anime = pd.read_csv('anime.csv', sep=',')
anime = anime.rename(columns={'name': 'anime_name'})

rate_count_20 = top_20_rate_count(rating, "anime_id", "rating")
rate_count_20 = pd.merge(rate_count_20, anime, on='anime_id')
rate_count_20 = rate_count_20[['anime_id', 'count', 'anime_name']]

rating_only_20 = rating[rating["anime_id"].isin(rate_count_20['anime_id'])]
pivot = rating_only_20.pivot_table(index=['user_id'],
                                   columns=['anime_id'],
                                   values='rating')
pivot.fillna(0, inplace=True)
reduced, push_back = SVD(pivot, dim=10, full_matrices=False)
rec_sort = np.argsort(push_back, axis=1).set_index(pivot.index).iloc[:, ::-1]


# for SVD tests
students = pd.read_csv('students_info.csv', sep=',')
justnum = students[["coffee", "sleep", "gym", "gpa"]]


#####################################################
# 1. 'surface' level tests: shape/size and type


def test_top_20_size():
    expected = 20
    assert len(top_20_rate_count(rating, "anime_id", "rating")) == expected


def test_top_20_type():
    assert isinstance(top_20_rate_count(rating, "anime_id", "rating"),
                      pd.DataFrame)


def test_SVD_reduced_shape():
    dim = 2
    reduced, push_back = SVD(justnum, dim, False)
    expected = (len(justnum.index), dim)
    assert reduced.shape == expected


def test_SVD_push_back_shape():
    dim = 2
    reduced, push_back = SVD(justnum, dim, False)
    expected = (len(justnum.index), len(justnum.columns))
    assert push_back.shape == expected


def test_SVD_reduced_type():
    reduced, push_back = SVD(justnum, 2, False)
    assert isinstance(reduced, pd.DataFrame)


def test_SVD_push_back_type():
    reduced, push_back = SVD(justnum, 2, False)
    assert isinstance(push_back, pd.DataFrame)


def test_recommend_shape():
    expected = (1000, 3)  # each of the 1000 users get at most 3 recs
    assert recommend(pivot, rec_sort, rate_count_20).shape == expected


def test_recommend_type():
    assert isinstance(recommend(pivot, rec_sort, rate_count_20), pd.DataFrame)


#####################################################
# 2. check that rated items are not recommended to the user

def test_not_rated():
    rec = recommend(pivot, rec_sort, rate_count_20)
    for user_id in rec.index:
        rec_list = rec.loc[user_id]
        user = pivot.loc[user_id]
        for j in rec_list:
            if j != 0:
                assert user.loc[j] == 0
