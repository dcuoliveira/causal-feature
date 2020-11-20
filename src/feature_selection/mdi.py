from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np


def get_mdi_scores(df,
                   feature_names,
                   target_name,
                   n_estimators=100):
    """
    Using a random forest regression model
    we calculate the  how much of the
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
    :param n_estimators: number of decision trees
                         used in the random forest model
    :type n_estimators: int
    :return: dataframe with average mdi scores and std
             (each decision tree produces one mdi score)
    :rtype: pd.DataFrame
    """
    X, y = df[feature_names].values, df[target_name].values
    rf = RandomForestClassifier(max_features=1, n_estimators=n_estimators)
    rf.fit(X, y)
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
