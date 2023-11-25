from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import loadmat
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import KFold


def train_val_test_split(kfold = 5, fold = 3, center_list = []):
    #n_sub = 1035
    n_sub = 871
    id = list(range(n_sub))
    
    import random
    #random.seed(666)
    #random.shuffle(id)
    
    
    kf = KFold(n_splits=kfold, random_state=123, shuffle = True)
    kfs = StratifiedKFold(n_splits=kfold, random_state=123, shuffle = True )
    kfs2 = StratifiedKFold(n_splits=8, random_state=123, shuffle = True )

    #kf = KFold(n_splits=kfold,shuffle = False)
    #kf2 = KFold(n_splits=kfold-1, shuffle=True, random_state = 666)


    test_index = list()
    train_index = list()
    val_index = list()

    for train, test in kfs.split(np.array(id), center_list):
        #test_index.append(te)
        test_index.append(test) #20% a test (si se usan 5 folds)


        center_list_train = [center_list[i] for i in train]
        train_final, valid = list(kfs2.split(train, center_list_train))[0]
        train_index.append(train[train_final])
        val_index.append(train[valid])
        

    train_id = train_index[fold]
    test_id = test_index[fold]
    val_id = val_index[fold]

    #modificaciión para bathc size
    train_id = np.append(train_id, test_id[-1])
    test_id = test_id[:-1]

    
    '''
    center_dist = {}

    for center in np.unique(center_list):
        center_dist[center] = np.sum(np.array(center_list)==center)/len(id)
    
    center_dist_train = {}
    
    for center in np.unique(center_list):
        train_centers = [center_list[i] for i in train_id]
        center_dist_train[center] = np.sum(np.array(train_centers)==center)/len(train_centers)

    center_dist_test = {}
    
    for center in np.unique(center_list):
        test_centers = [center_list[i] for i in test_id]
        center_dist_test[center] = np.sum(np.array(test_centers)==center)/len(test_centers)

    '''

    
    return train_id, test_id, val_id
    #return train_id,val_id,test_id