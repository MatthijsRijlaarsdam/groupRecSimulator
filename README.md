# netflix-met-veel
Dependencies:
suprise, pandas, numpy, openpyxl

Netflix met veel simulates users using an app that provides a swiping interface with movies to groups of users, in order to find the best recommendation to watch for that group.
It is meant to simulate the effects of various parameters on the performance of the application.

It is optimised for speed and parallelized. 


###Run instructions
In main.py, select the dataset and amount of users to put in to your test set.
```python
    movie_names = pd.read_csv("./ml-latest-small/movies.csv")
    df = pd.read_csv("./ml-latest-small/ratings.csv")
    train_set, test_users = data_handler.remove_users_from_trainset(df, 120)
```
    
Then either perform a gridsearch on the hyperparameters you want to test for you recommondation model, or leave the ones already there.

```python
     # only necessary once if values are not saved
     # algo = do_grid_search(df)
     # With saved best values, gives RSME of 0.8706802008822863 on full dataset
     algo = SVD()
     algo.n_factors = 150
     algo.n_epochs = 30
     algo.lr_all = 0.005
     algo.reg_all = 0.2   
```

There are multiple different parameters to test:

```python
    #the minimal rating for a user to want to watch a movie
    _rating_threshold = 4  
    #the amount of movies a user gets recommended before updating the model with new swipes
    _movies_per_refresh = 10 
    #ratio of the movies per refresh that is random vs already rated by other users
    random_movie_ratio_range = np.arange(0.2, 1.1, .2)
    #the group sizes to test
    group_size_range = range(5, 15)
    #the minimal fraction of the grooup that needs to agree for a certain movie
    ratio_agreement_range = np.arange(1, 1.1, .25)
    #the amount of runs per setting
    n = 100
```