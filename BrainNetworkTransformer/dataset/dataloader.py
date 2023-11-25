import torch
import torch.utils.data as utils
from omegaconf import DictConfig, open_dict
from typing import List
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import torch.nn.functional as F
from scipy import stats
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold


def train_val_test_split(kfold = 5, fold = 4, center_list = []):
    n_sub = 846
    id = list(range(n_sub))
    
    import random    
    
    kf = KFold(n_splits=kfold, random_state=123, shuffle = True)
    kfs = StratifiedKFold(n_splits=kfold, random_state=123, shuffle = True )
    kfs2 = StratifiedKFold(n_splits=8, random_state=123, shuffle = True )

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
    
    return train_id, test_id, val_id


def init_stratified_dataloader(cfg: DictConfig,
                               final_timeseires: torch.tensor,
                               final_pearson: torch.tensor,
                               labels: torch.tensor,
                               stratified: np.array, fold: int) -> List[utils.DataLoader]:
    labels = F.one_hot(labels.to(torch.int64))
    length = final_timeseires.shape[0]
    train_length = int(length*cfg.dataset.train_set*1)
    val_length = int(length*cfg.dataset.val_set)
    test_length = length-train_length-val_length

    with open_dict(cfg):
        cfg.steps_per_epoch = (
            train_length - 1) // cfg.dataset.batch_size + 1
        cfg.total_steps = cfg.steps_per_epoch * cfg.training.epochs
    
    train_id, test_id, val_id = train_val_test_split(fold=fold, center_list=stratified)
    
    final_timeseires_train, final_pearson_train, labels_train = final_timeseires[
            train_id], final_pearson[train_id], labels[train_id]
    
    final_timeseires_val, final_pearson_val, labels_val = final_timeseires[
            val_id], final_pearson[val_id], labels[val_id]
    
    final_timeseires_test, final_pearson_test, labels_test = final_timeseires[
            test_id], final_pearson[test_id], labels[test_id]
    
    train_dataset = utils.TensorDataset(
        final_timeseires_train,
        final_pearson_train,
        labels_train
    )

    val_dataset = utils.TensorDataset(
        final_timeseires_val, final_pearson_val, labels_val
    )

    test_dataset = utils.TensorDataset(
        final_timeseires_test, final_pearson_test, labels_test
    )

    train_dataloader = utils.DataLoader(
        train_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=cfg.dataset.drop_last)

    val_dataloader = utils.DataLoader(
        val_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=False)

    test_dataloader = utils.DataLoader(
        test_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=False)

    return [train_dataloader, val_dataloader, test_dataloader]
