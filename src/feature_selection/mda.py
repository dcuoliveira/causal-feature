import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from time import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score


def mean_decrease_accuracy(df,
                           feature_names,
                           target_name,
                           n_splits,
                           random_state,
                           ModelClass=RandomForestClassifier(n_estimators=10)):
    """
    Using the classification model 'ModelClass'
    and the cross-validation object 'TimeSeriesSplit',
    we calculate the accuracy for
    "n_splits" test sets.

    EXPLAIN EXPLAIN EXPLAIN


    :param df: data
    :type df: pd.DataFrame
    :param feature_names: name of the feature columns
    :type feature_names: [str]
    :param target_name: name of the target column in 'df'
    :type target_name: str
    :param n_splits: number of cross-validation splits
    :type n_splits: int
    :param random_state: controls both the randomness of the
                         colum permutation and the model training
    :type random_state: int or RandomState
    :param ModelClass: any sklearn classification model
    :type ModelClass: RandomForestClassifier, LogisticRegression, etc
    :return: dfsfs
    :rtype: pd.DataFrame
    """
    np.random.seed(random_state)
    ModelClass.random_state = random_state
    scores = {f: [] for f in feature_names}
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_index, test_index in tscv.split(df):
        df_train = df.iloc[train_index]
        df_test = df.iloc[test_index]
        X_train, y_train = df_train[feature_names].values, df_train[target_name].values
        X_test, y_test = df_test[feature_names].values, df_test[target_name].values
        model = ModelClass.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc0 = accuracy_score(y_test, pred)
        for feature in feature_names:
            new_test = df_test[feature_names].copy()
            np.random.shuffle(new_test.loc[:, feature])
            new_X_test = new_test.values
            new_pred = model.predict(new_X_test)
            acc = accuracy_score(y_test, new_pred)
            acc_diff =  acc0 - acc
            imp = 1/(1.0 - acc_diff)
            scores[feature].append(imp)
            del new_test

    result = pd.DataFrame(scores).transpose().mean(1).reset_index()
    result.columns = ["feature", "feature_score"]
    result = result.sort_values("feature_score", ascending=False).reset_index(drop=True)
    return result
