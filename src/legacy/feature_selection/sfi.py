import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from time import time
import statsmodels.formula.api as smf
from sklearn.model_selection import TimeSeriesSplit


def single_feature_importance_cv(df,
                                 feature_name,
                                 target_name,
                                 n_splits):
    """
    Using a linear model and the cross-validation
    object 'TimeSeriesSplit', we calculate the R2 OOS for
    "n_splits" test sets. The predictor is a simple linear
    model composed of the single feature 'feature_name'.

    The R2 is calculate using the formula in the paper

    "Empirical Asset Pricing via Machine
    Learning"

    In the case where there is only np.nan's
    in the dataframe, the funtion will also
    return np.nan

    :param df: data
    :type df: pd.DataFrame
    :param feature_name: name of the feature column in 'df'
    :type feature_name: str
    :param target_name: name of the target column in 'df'
    :type target_name: str
    :param n_splits: number of cross-validation splits
    :type n_splits: int
    :return: array of R2 values for each OOS slit
    :rtype: np.array
    """
    r2_OOS = []
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_index, test_index in tscv.split(df):
        formula = "{} ~ {}".format(target_name, feature_name)
        df_train = df.iloc[train_index]
        df_test = df.iloc[test_index]
        try:
            lr = smf.ols(formula=formula, data=df_train).fit()
            y_pred = lr.predict(df_test).values
            y_true = df_test[target_name].values
            erros = y_true - y_pred
            num = (erros).dot(erros)
            dem = (y_true).dot(y_true)
            r2 = 1 - (num / dem)
            r2_OOS.append(r2)
        except ValueError:
            v_train = df_train[feature_name].max()
            v_test = df_test[feature_name].max()
            print("Only np.nan's on dataframe\ntrain = {} | test = {}".format(v_train, v_test))
            r2_OOS.append(np.nan)

    return np.array(r2_OOS)


def get_sfi_scores(merged_df, target_name, words,
                   max_lag, n_splits=5, verbose=True):
    """
    Get sfi_score for all words in 'words' using lags from 1 to
    max_lag.

    :param merged_df: market and google trends data
    :type merged_df: pd.DataFrame
    :param target_name: name of the target column in 'merged_df'
    :type target_name: str
    :param words: list of words to create features
    :type words: [str]
    :param max_lag: number maximun lags to apply to word features
    :type max_lag: int
    :param n_splits: number of cross-validation splits
    :type n_splits: int
    :param verbose: param to print iteration status
    :type verbose: bool
    :return: sorted dataframe with R2 OOS values for each feature
            (greater is better)
    :rtype: pd.DataFrame
    """

    # add shift for all words

    feature_dict = {}

    for word in tqdm(words, disable=not verbose, desc="add shift"):
        new_features = []
        for shift in range(1, max_lag + 1):
            new_feature = word.replace(" ", "_") + "_{}".format(shift)
            merged_df.loc[:, new_feature] = merged_df[word].shift(shift)
            new_features.append(new_feature)
        feature_dict[word] = new_features

    # calculate R2 OOS for all words

    all_words_results = []

    for word in tqdm(words, disable=not verbose, desc="cv r2"):
        new_features = feature_dict[word]
        new_merged = merged_df[[target_name] + new_features]


        results = []
        # print(new_merged.shape, new_merged.dropna().shape)

        for new_feature in new_features:

            r2_arr = single_feature_importance_cv(df=new_merged,
                                                  feature_name=new_feature,
                                                  target_name=target_name,
                                                  n_splits=n_splits)

            results.append((new_feature, np.mean(r2_arr)))
        # feature_score = mean_r2 (greater is better)
        results = pd.DataFrame(results,
                               columns=["feature", "feature_score"])
        all_words_results.append(results)
    all_words_results = pd.concat(all_words_results)
    all_words_results = all_words_results.sort_values("feature_score",
                                                      ascending=False)
    all_words_results = all_words_results.reset_index(drop=True)
    return all_words_results
