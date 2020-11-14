import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from time import time
# from word_list.analysis import words
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

    :param df: data
    :type df: np.DataFrame
    :param feature_name: name of the feature column 'df'
    :type feature_name: str
    :param target_name: name of the target column 'df'
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
        lr = smf.ols(formula=formula, data=df_train).fit()
        y_pred = lr.predict(df_test).values
        y_true = df_test[target_name].values
        erros = y_true - y_pred
        num = (erros).dot(erros)
        dem = (y_true).dot(y_true)
        r2 = 1 - (num / dem)
        r2_OOS.append(r2)
    return np.array(r2_OOS)


# ## Naive Merging | esperar daniel

def merge_market_gtrends(market, gtrends):
    merged = pd.merge_asof(market, gtrends, left_index=True, right_index=True)
    return merged.dropna()


def get_sfi_scores(merged_df, target_name, words,
                   max_lag, n_splits=5, verbose=True):
    """
    fsfsffs

    """

    # add shift for all words

    feature_dict = {}

    for word in tqdm(words, disable=not verbose, desc="shift"):
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

        for new_feature in new_features:

            r2 = single_feature_importance_cv(df=new_merged,
                                              feature_name=new_feature,
                                              target_name=target_name,
                                              n_splits=n_splits)

            results.append((new_feature, np.mean(r2)))

        results = pd.DataFrame(results,
                               columns=["feature", "mean_r2"])
        all_words_results.append(results)
    all_words_results = pd.concat(all_words_results)
    all_words_results = all_words_results.sort_values("mean_r2",
                                                      ascending=False)
    all_words_results = all_words_results.reset_index(drop=True)
    return all_words_results


# if __name__ == '__main__':

#    # ## Loading trends
#     gtrends = pd.read_csv("data/gtrends.csv")
#     gtrends.loc[:, "date"] = pd.to_datetime(gtrends.date)
#     gtrends = gtrends.set_index("date")

#     # ## Loading and preprossesing market data

#     path = "data/crsp/nyse/CYN US Equity.csv"
#     name = path.split("/")[-1].split(".")[0]
#     target_name = name.replace(" ", "_") + "_return"
#     market = pd.read_csv(path)
#     market = market.drop([0, 1], 0)
#     market = market.rename(columns={"ticker": "date",
#                                     name: target_name})
#     market.loc[:, "date"] = pd.to_datetime(market.date)
#     market.loc[:, target_name] = market[target_name].astype("float") / 100
#     market = market.set_index("date")

#     # using only the training sample
#     market = market["2000":"2010"]

#     merged = merge_market_gtrends(market, gtrends)

#     a = get_sfi_scores(merged_df=merged,
#                        target_name=target_name,
#                        words=words[:3],
#                        max_lag=3,
#                        verbose=False)

#     print(a.head(4))
