import numpy as np
import dataset_handler as data_handler
import math

# updates biased SVD instance with a (small set) of new users. No new movies can be added


def add_users(svd_instance, old_trainset, new_trainset):

    svd_instance.trainset = data_handler.merge_datasets_df(old_trainset, new_trainset).build_full_trainset()
    n_new_users = len(new_trainset.userId.unique())


    # user biases
    bu = np.append(svd_instance.bu, np.zeros(n_new_users, np.double), axis=0)
    # item biases
    bi = svd_instance.bi
    # user factors
    pu = np.append(svd_instance.pu, np.random.normal(svd_instance.init_mean, svd_instance.init_std_dev,
                                                     (n_new_users, svd_instance.n_factors)), axis=0)
    # item factors
    qi = svd_instance.qi

    #print("Processing new users")
    #epochs relative to size of new ratings
    for current_epoch in range(max(5,math.floor(svd_instance.n_epochs*(len(new_trainset)/len(old_trainset))))):
        for index, rating in new_trainset.iterrows():
            u = svd_instance.trainset.to_inner_uid(int(rating['userId']))
            i = svd_instance.trainset.to_inner_iid(int(rating['movieId']))
            r = rating.rating
            # compute current error
            dot = 0  # <q_i, p_u>
            for f in range(svd_instance.n_factors):
                dot += qi[i, f] * pu[u, f]
            err = r - (svd_instance.trainset.global_mean + bu[u] + bi[i] + dot)

            # update biases
            bu[u] += svd_instance.lr_bu * (err - svd_instance.reg_bu * bu[u])
            bi[i] += svd_instance.lr_bi * (err - svd_instance.reg_bi * bi[i])

            # update only P factors for new users
            for f in range(svd_instance.n_factors):
                puf = pu[u, f]
                qif = qi[i, f]
                pu[u, f] += svd_instance.lr_pu * (err * qif - svd_instance.reg_pu * puf)

    svd_instance.bu = bu
    svd_instance.bi = bi
    svd_instance.pu = pu

    #print("added users")
