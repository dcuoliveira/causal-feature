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
                           ModelClass=RandomForestClassifier(n_estimators=100)):
    """
    Using the classification model 'ModelClass'
    we calculate the misclassification error (1- accuracy)
    on the test set (with all variables intact),
    then, for each feature we calculate the misclassification
    error for the new version of the test set after we permutate
    the feature column. The importace of each feature is understood
    as the increase in error rate.

    The training and test splitting is done using
    the cross-validation object 'TimeSeriesSplit'.

    The final result is the mean increasing in error rate
    for each feature (mean over all splits).

    reference:
    Breiman Leo. 2001. Random Forests, Machine Learning, 45, 5-32.


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
    :return: sorted dataframe with scores for each feature
             (greater is better)
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
        miss_rate_0 = (1 - accuracy_score(y_test, pred))
        for feature in feature_names:
            new_test = df_test.loc[:, feature_names].copy()
            permutation = new_test[feature].sample(
                frac=1).reset_index(drop=True).values
            new_test.loc[:, feature] = permutation
            new_X_test = new_test.values
            new_pred = model.predict(new_X_test)
            miss_rate = (1 - accuracy_score(y_test, new_pred))
            if miss_rate_0 == 0:
                miss_pct = np.nan
            else:
                miss_pct = ((miss_rate - miss_rate_0) / miss_rate_0)
            scores[feature].append(miss_pct)
            del new_test
    result = pd.DataFrame(scores).transpose().mean(1).reset_index()
    result.columns = ["feature", "feature_score"]
    result = result.sort_values(
        "feature_score",
        ascending=False).reset_index(
        drop=True)
    return result


def get_mda_scores(merged_df,
                   target_name,
                   words,
                   max_lag,
                   random_state,
                   n_splits,
                   n_estimators=100,
                   verbose=True):
    """
    Get mda_score for all words in 'words' using lags from 1 to
    max_lag.

    :param merged_df: market and google trends data
    :type merged_df: pd.DataFrame
    :param target_name: name of the target column in 'merged_df'
    :type target_name: str
    :param words: list of words to create features
    :type words: [str]
    :param n_estimators: number of decision trees
                         used in the random forest model
    :type n_estimators: int
    :param random_state: controls both the randomness of the
                         colum permutation and the model training
    :type random_state: int or RandomState
    :param n_splits: number of cross-validation splits
    :type n_splits: int
    :param verbose: param to print iteration status
    :type verbose: bool
    :return: sorted dataframe with mean decrease impurity
            (greater is better)
    :rtype: pd.DataFrame
    """

    # add shift for all words

    feature_names = []

    for word in tqdm(words, disable=not verbose, desc="add shift"):
        for shift in range(1, max_lag + 1):
            new_feature = word.replace(" ", "_") + "_{}".format(shift)
            merged_df.loc[:, new_feature] = merged_df[word].shift(shift)
            feature_names.append(new_feature)

    # calculate mda for all words
    results = mean_decrease_accuracy(df=merged_df.dropna(),
                                     feature_names=feature_names,
                                     target_name=target_name,
                                     n_splits=n_splits,
                                     random_state=random_state,
                                     ModelClass=RandomForestClassifier(n_estimators=n_estimators))
    return results
