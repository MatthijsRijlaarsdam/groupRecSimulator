"""
This module illustrates how to retrieve the top-10 items with highest rating
prediction. We first train an SVD algorithm on the MovieLens dataset, and then
predict all the ratings for the pairs (user, item) that are not in the training
set. We then retrieve the top-10 prediction for each user.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import random


import user_adder
import pandas as pd
import math
import dataset_handler as data_handler


class Recommender:

    def __init__(self, algo, trainset):
        self.original_trainset = trainset
        self.original_dataset = data_handler.get_data_from_df(self.original_trainset).build_full_trainset()
        self.algo = algo
        print("Training Recommender")
        self.algo.fit(self.original_dataset)
        self.original_fit = [self.algo.bu, self.algo.bi, self.algo.pu, self.algo.qi]
        self.new_ratings = pd.DataFrame(columns=['userId', 'movieId', 'rating'])
        self.random_ratio = 0
        self.movies_not_rated = list(self.original_trainset.movieId.unique())
        self.movies_rated = {}

    def get_top_n(self, predictions, n=10):
        '''Return the top-N recommendation for each user from a set of predictions.
        Args:
            predictions(list of Prediction objects): The list of predictions, as
                returned by the test method of an algorithm.
            n(int): The number of recommendation to output for each user. Default
                is 10.
        Returns:
        A dict where keys are user (raw) ids and values are lists of tuples:
            [(raw item id, rating estimation), ...] of size n.
        '''

        top_n = []
        for uid, iid, true_r, est, _ in predictions:
            top_n.append((iid, est))

        top_n.sort(key=lambda x: x[1], reverse=True)
        top_n = top_n[:n]

        return top_n

    def get_predictions_not_rated_any_user(self, user_id):
        random_subset = random.sample(self.movies_not_rated, min(50, len(self.movies_not_rated)))
        all_pairs = [(user_id, iid) for iid in random_subset]

        predictions = [self.algo.predict(uid,
                                         iid)
                       for (uid, iid) in all_pairs]
        return predictions

    def get_predictions_rated_other_users(self, user_id):
        all_movies_not_rated_user = [key for key, item in self.movies_rated.items() if
                                     user_id not in item]
        all_pairs = [(user_id, iid) for iid in all_movies_not_rated_user]

        predictions = [self.algo.predict(uid,
                                         iid)
                       for (uid, iid) in all_pairs]
        return predictions

    def get_n_to_rate(self, user_id):
        all_movies_not_rated_user = [key for key, item in self.movies_rated.items() if
                                     user_id not in item]
        return len(all_movies_not_rated_user)

    def reset_model(self):
        self.algo.bu = self.original_fit[0]
        self.algo.bi = self.original_fit[1]
        self.algo.pu = self.original_fit[2]
        self.algo.qi = self.original_fit[3]
        self.algo.trainset = self.original_dataset

    def new_session(self, random_ratio):
        self.reset_model()
        self.random_ratio = random_ratio
        self.new_ratings = pd.DataFrame(columns=['userId', 'movieId', 'rating'])
        self.movies_not_rated = list(self.original_trainset.movieId.unique())
        self.movies_rated = {}

    def get_movies_to_rate(self, user, nmovies):
        if len(self.new_ratings.userId.unique()) == 0:
            return random.sample(list(self.movies_not_rated), k=nmovies)
        else:
            nalready_rated = min(self.get_n_to_rate(user), math.floor(nmovies * (1 - self.random_ratio)))
            to_rate = self.get_top_n(self.get_predictions_rated_other_users(user_id=user), n=nalready_rated)
            to_rate.extend(
                self.get_top_n(self.get_predictions_not_rated_any_user(user_id=user), n=(nmovies - nalready_rated)))
            return [iid for (iid, _) in to_rate]

    def add_ratings(self, rating_df):
        self.reset_model()
        for index, row in rating_df.iterrows():
            if row.movieId in self.movies_rated:
                self.movies_rated[row.movieId].append(row.userId)
            else:
                self.movies_not_rated.remove(row.movieId)
                self.movies_rated[row.movieId] = [row.userId]
        user_adder.add_users(self.algo, self.original_trainset, self.new_ratings)
        self.new_ratings = pd.concat([self.new_ratings, rating_df])

    def check_approval(self, ratio_group):
        to_approve = math.ceil(len(self.new_ratings.userId.unique()) * ratio_group)
        best_movie = None
        best_approve = -1
        for movie, rated in self.movies_rated.items():
            if len(rated) >= to_approve:
                approve_this = len(
                    self.new_ratings.loc[(self.new_ratings["movieId"] == movie) & (self.new_ratings['rating'] == 5)])
                if approve_this >= to_approve and approve_this > best_approve:
                    best_movie = movie
                    best_approve = approve_this

        return best_movie
