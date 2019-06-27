from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import pandas as pd

import dataset_handler as data_handler


class User_simulator:

    def __init__(self, algo, full_dataset_df, threshold):
        self.full_dataset_df = full_dataset_df
        self.algo = algo
        print("Training User Simulator")
        self.algo.fit(data_handler.get_data_from_df(full_dataset_df).build_full_trainset())
        self.threshold = threshold

    def get_swipes_for_user(self, user, movies):
        res = pd.DataFrame(columns=['userId', 'movieId', 'rating'])

        for movieId in movies:
            if self.full_dataset_df["rating"][
                (self.full_dataset_df["userId"] == user) & (self.full_dataset_df["movieId"] == movieId)].any():
                prediction = self.full_dataset_df["rating"][
                    (self.full_dataset_df["userId"] == user) & (self.full_dataset_df["movieId"] == movieId)].iloc[0]
            else:
                prediction = self.algo.predict(user, movieId)[3]
            rating = 1 if prediction < self.threshold else 5
            res = res.append({'userId': user, 'movieId': movieId, 'rating': rating}, ignore_index=True)
        return res
