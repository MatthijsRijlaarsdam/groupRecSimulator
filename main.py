from collections import defaultdict
import random

from surprise import SVD
import numpy as np
import pandas as pd
import dataset_handler as data_handler
from user_simulator import User_simulator
from recommender import Recommender
from surprise.model_selection import GridSearchCV
from multiprocessing import Pool
import multiprocessing


def run_session(args):
    recommender = args[0]
    user_simulator = args[1]
    group_size = args[2]
    random_movie_ratio = args[3]
    ratio_agreement = args[4]
    movies_per_refresh = args[5]
    rating_threshold = args[6]

    group = random.sample(list(test_users), group_size)
    recommender.new_session(random_movie_ratio)
    while not recommender.check_approval(ratio_agreement) and not len(recommender.new_ratings) > group_size * 50:
        # print("new swipe round")
        new_ratings = pd.DataFrame(columns=['userId', 'movieId', 'rating'])

        for user in group:
            to_rate = recommender.get_movies_to_rate(user, movies_per_refresh)
            new_ratings = new_ratings.append(user_simulator.get_swipes_for_user(user, to_rate))

        recommender.add_ratings(new_ratings)
    # print(recommender.check_approval(ratio_agreement))
    # print("took {} swipes".format(len(recommender.new_ratings)))
    return pd.DataFrame([[len(recommender.new_ratings), len(recommender.new_ratings) / group_size, rating_threshold,
                          random_movie_ratio, group_size,
                          ratio_agreement, movies_per_refresh,
                          movie_names.loc[movie_names.movieId == recommender.check_approval(ratio_agreement)].title]],
                        columns=['number_of_swipes', 'swipes/user', 'rating_threshold', 'random_movie_ratio',
                                 'group_size',
                                 'ratio_agreement',
                                 'movies_per_refresh', 'recommended_movie'])


def do_grid_search(data):
    print("Doing gridsearch for best model.")
    param_grid = {'n_epochs': [10, 20, 30], 'n_factors': [100, 150, 200], 'lr_all': [0.001, 0.0025, 0.005, 0.001],
                  'reg_all': [0.2, 0.4, 0.6]}
    gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=5, joblib_verbose=5, n_jobs=-1)

    gs.fit(data_handler.get_data_from_df(data))
    # best RMSE score
    print(gs.best_score['rmse'])

    # combination of parameters that gave the best RMSE score
    print(gs.best_params['rmse'])

    # We can now use the algorithm that yields the best rmse:
    algo = gs.best_estimator['rmse']
    return algo


if __name__ == '__main__':
    movie_names = pd.read_csv("./ml-latest-small/movies.csv")
    df = pd.read_csv("./ml-latest-small/ratings.csv")
    train_set, test_users = data_handler.remove_users_from_trainset(df, 120)

    # only necessary once if values are not saved
    # algo = do_grid_search(df)
    # With saved best values, gives RSME of 0.8706802008822863 on full dataset
    algo = SVD()
    algo.n_factors = 150
    algo.n_epochs = 30
    algo.lr_all = 0.005
    algo.reg_all = 0.2

    with Pool(processes=multiprocessing.cpu_count()) as pool:
        # the minimal rating for a user to want to watch a movie
        _rating_threshold = 4
        # the amount of movies a user gets recommended before updating the model with new swipes
        _movies_per_refresh = 10
        # ratio of the movies per refresh that is random vs already rated by other users
        random_movie_ratio_range = np.arange(0.2, 1.1, .2)
        # the group sizes to test
        group_size_range = range(5, 15)
        # the minimal fraction of the grooup that needs to agree for a certain movie
        ratio_agreement_range = np.arange(1, 1.1, .25)
        # the amount of runs per setting
        n = 100

        total_sessions = len(random_movie_ratio_range) * len(group_size_range) * len(ratio_agreement_range) * n
        print("running {} sessions".format(total_sessions))

        res = pd.DataFrame(
            columns=['number_of_swipes', 'swipes/user', 'rating_threshold', 'random_movie_ratio', 'group_size',
                     'ratio_agreement',
                     'movies_per_refresh', 'recommended_movie'])

        _user_simulator = User_simulator(algo, df, _rating_threshold)
        _recommender = Recommender(algo, train_set)
        j = 0
        run_arguments = []
        for i in range(n):  # run every combination n times
            for _group_size in group_size_range:
                for _random_movie_ratio in random_movie_ratio_range:
                    for _ratio_agreement in ratio_agreement_range:
                        run_arguments.append(
                            [_recommender, _user_simulator, _group_size, _random_movie_ratio, _ratio_agreement,
                             _movies_per_refresh, _rating_threshold])

        res = []
        for session_res in pool.imap_unordered(run_session, run_arguments):
            print("\r{}/{}".format(len(res) + 1, total_sessions), end="")
            res.append(session_res)

        res = pd.concat(res, sort=False)
        res.to_excel("result_final_groupsize.xlsx")
