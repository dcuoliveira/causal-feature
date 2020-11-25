import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor


def mdi_feature_importance(df,
                           feature_names,
                           target_name,
                           random_state,
                           n_estimators=100):
    """
    Using a random forest regression model
    we calculate how much of the
    "impurity" of the sample is eliminated
    by including the feature in the model.

    This implementation follows the one
    displayed on page 115 of Marcos' book

    "Advances in Financial Machine Learning"

    :param df: data
    :type df: pd.DataFrame
    :param feature_names: names of all feature columns in 'df'
    :type feature_names: [str]
    :param target_name: name of the target column in 'df'
    :type target_name: str
    :param random_state: controls both the randomness of the
                         bootstrapping of the samples used
                         when building trees
                         and the sampling of the
                         features to consider when looking
                         for the best split at each node
    :type random_state: int or RandomState
    :param n_estimators: number of decision trees
                         used in the random forest model
    :type n_estimators: int
    :return: dataframe with average mdi scores and std
             (each decision tree produces one mdi score)
    :rtype: pd.DataFrame
    """
    df_ = df[feature_names + [target_name]].dropna()
    X, y = df_[feature_names].values, df_[target_name].values
    rf = RandomForestRegressor(
        max_features=1,
        n_estimators=n_estimators,
        random_state=random_state)
    rf.fit(X, y)
    del X, y, df_
    fi_estimators = {
        i: dt.feature_importances_ for i,
        dt in enumerate(
            rf.estimators_)}
    fi_estimators = pd.DataFrame.from_dict(
        fi_estimators, orient="index", columns=feature_names)
    fi_estimators = fi_estimators.replace(0, np.nan)
    mean = fi_estimators.mean()
    std = fi_estimators.std()
    n = fi_estimators.shape[0]
    std = std * np.power(n, -0.5)
    imp = pd.concat({"mean": mean, "std": std}, axis=1)
    imp /= imp["mean"].sum()
    return imp


def get_mdi_scores(merged_df,
                   target_name,
                   words,
                   max_lag,
                   random_state,
                   n_estimators=100,
                   verbose=True):
    """
    Get mdi_score for all words in 'words' using lags from 1 to
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
                         bootstrapping of the samples used
                         when building trees
                         and the sampling of the
                         features to consider when looking
                         for the best split at each node
    :type random_state: int or RandomState
    :param verbose: param to print iteration status
    :type verbose: bool
    :return: sorted dataframe with mean decrease impurity
    :rtype: pd.DataFrame
    """

    # add shift for all words

    feature_names = []

    for word in tqdm(words, disable=not verbose, desc="add shift"):
        for shift in range(1, max_lag + 1):
            new_feature = word.replace(" ", "_") + "_{}".format(shift)
            merged_df.loc[:, new_feature] = merged_df[word].shift(shift)
            feature_names.append(new_feature)

    # calculate mdi for all words

    imp = mdi_feature_importance(df=merged_df,
                                 feature_names=feature_names,
                                 target_name=target_name,
                                 random_state=random_state,
                                 n_estimators=n_estimators)
    imp = imp.sort_values("mean", ascending=False)["mean"]
    imp = imp.reset_index()
    imp.columns = ["features", "mdi"]
    return imp
