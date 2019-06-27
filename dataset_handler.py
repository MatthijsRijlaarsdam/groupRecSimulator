import pandas as pd
import numpy as np
from surprise import Reader, Dataset
import random


def merge_datasets_df(trainset, new_trainset):
    #print("merging trainsets")

    # for index, rating in new_trainset.iterrows():
    #     assert len(trainset[
    #                    trainset['userId'] == rating['userId']]) == 0, "User already in database"

    return get_data_from_df(pd.concat([trainset, new_trainset], sort=False))


def get_data_from_df(df):
    reader = Reader(rating_scale=(1, 5))

    # The columns must correspond to user id, item id and ratings (in that order).
    data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)

    return data


def remove_users_from_trainset(df, n_users):
    users = []
    for i in range(n_users):
        user, df = get_random_user(df)
        users.append(user)
    return df, users


def get_random_user(df):
    user_id = random.choice(list(df.userId.unique()))
    return user_id, df.loc[df.userId != user_id]
