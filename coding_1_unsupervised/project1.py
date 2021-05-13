import numpy as np
import pandas as pd


def top_100_rate_count(df, groupby, count_var):
    """
    df - a pandas DataFrame
    groupby - string - count after grouped by
    count_var - string - a variable that will be counted
    Count the number of time a var is rated.
    Select the top 20 values based on var count."""

    count = df[df[count_var] != -1]
    count = count.groupby(by=[groupby])[count_var].count()
    count = count.reset_index().rename(columns={count_var: 'count'})
    count = count.sort_values(by='count', ascending=False)
    count_100 = count.iloc[:100]

    return count_100


# def SVD(df, dim, full_matrices):
#     """
#     apply numpy.linalg.svd() to the df.
#     reduce to dim dimensions.
#     push back to the original dimension.
#     return the reduced and pushed back matrices as pandas.DataFrame"""

#     # SVD implementation
#     U, S, V = np.linalg.svd(df, full_matrices=full_matrices)
#     # reduce to the dimension of dim
#     reduced = U[:, :dim] * S[:dim]
#     # push back to the original dimension
#     push_back = np.dot(U[:, :dim]*S[:dim], V[:dim, :])

#     return (pd.DataFrame(reduced), pd.DataFrame(push_back))


# def recommend(rating, rec_order, anime_id_name):
#     """
#     rating - pandas.DataFrame - the pivot dataframe with original ratings
#     rec_order - pandas.DataFrame
#     anime_id_name - pandas.DataFrame - help find the name of the anime

#     give THREE recommendations to each user.
#     recommendations must be anime that has NOT been rated by the user"""

#     # to save my computer ans also same some time,
#     # I will only give recommendations to 1000 users
#     all_users = rating.index[:1000]
#     columns = ['rec1', 'rec2', 'rec3']
#     rec_matrix = pd.DataFrame(index=all_users, columns=columns)

#     for user_id in all_users:
#         rec_list = []

#         user_rec_order = rec_order.loc[user_id]
#         user_rating = rating.loc[user_id]
#         anime_order = rating.columns
#         max_anime = len(rating.columns)

#         # create rec_list
#         i = 0
#         rec_num = 0
#         while (i < max_anime) and (rec_num < 3):
#             anime_rec = user_rec_order.iloc[i]
#             anime_id = anime_order[anime_rec]  # get the id of that anime

#             if user_rating.loc[anime_id] == 0:
#                 rec_num += 1
#                 rec_list.append(anime_id)  # if not rated, recommend

#             i += 1

#         if len(rec_list) < 3:
#             for i in range(len(rec_list), 3):
#                 rec_list.append(0)

#         rec_matrix.loc[user_id] = rec_list

#     return rec_matrix


# ######################################################
# # Prepareing Data
# # 1. reading datasets to pandas DataFrame
# # the first dataset - rating.csv
# rating = pd.read_csv('rating.csv', sep=',')
# rating['rating'] = rating['rating'].apply(lambda x: np.nan if x == -1 else x)
# # the second dataset - anime.csv
# anime = pd.read_csv('anime.csv', sep=',').rename(
#     columns={'name': 'anime_name'})

# # 2. focus on a subset: top 20 anime based on rating counts
# # select the top 20 anime based on rating count
# rating_count_20 = top_20_rate_count(rating, "anime_id", "rating")
# # add names of the 20 anime to the dataset
# rating_count_20 = pd.merge(rating_count_20, anime, on='anime_id')[
#     ['anime_id', 'count', 'anime_name']]
# rating_count_20.head()

# # From now on, we will only focus on the top 20 animes that are most commonly
# # rated by users. We are only interested in users who have rated at least one
# # of the 20 most commonly rated anime. And we'll leave other users out.
# rating20 = rating[rating["anime_id"].isin(rating_count_20['anime_id'])]

# # 3. transform to a pivot_table: one user per row
# # create a pivot table
# rating_pivot = rating20.pivot_table(index=['user_id'],
#                                     columns=['anime_id'],
#                                     values='rating')
# # put 0 if a user has not rated an anime
# rating_pivot.fillna(0, inplace=True)

# ######################################################
# # Implementing svd
# reduced, push_back = SVD(rating_pivot, dim=10, full_matrices=False)
# # sort the recommendation ratings ascendingly for each user
# rec_sort = np.argsort(push_back, axis=1).set_index(
#     rating_pivot.index).iloc[:, ::-1]

# ######################################################
# # Recommendation
# rec = recommend(rating_pivot, rec_sort, rating_count_20)
# print(rec.head(20))

# Code Consultation:
# https://www.kaggle.com/dipayanpal/anime-recommendation#First-few-lines
# I consulted this kaggle notebook in selecting the top 10 anime by rating counts
