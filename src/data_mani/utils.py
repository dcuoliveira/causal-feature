from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
import os
import numpy as np
from sys import platform

def target_ret_to_directional_movements(x, y_name):
    """
    discretize a series of returns into up (1) and down (0) movements

    :param x: target data
    :type x: data.frame
    :param y_name: target return to discretize
    :type words: str
    :return: full dataframe with the y_name variable discretized
    :rtype: dataframe
    """
    x[y_name] = [1 if r > 0 else 0 for r in x[y_name]]
    return x


def correlation_filter(data, threshold):
    """
    filter columns that has correlation higher than the threshold

    :param data: data to filter
    :type data: dataframe
    :param threshold: correlation threshold to apply the filter
    :type threshold: float
    :return: filtered data
    """
    col_corr = set()
    corr_matrix = data.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) or (corr_matrix.iloc[i, j] <= -threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
                if colname in data.columns:
                    del data[colname]

    return data


def make_shifted_df(df, words, verbose, max_lag):
    """
    make shift to specified words in the df

    :param df: data
    :type df: dataframe
    :param words: selected words to make shift
    :type words: list
    :param verbose:
    :type verbose: boolean
    :return: shifted dataframe and dicionary of words used to shift
    :rtype: dataframe and dict
    """
    feature_dict = {}

    for word in tqdm(words, disable=not verbose, desc="add shift"):
        new_features = []
        for shift in range(1, max_lag + 1):
            new_feature = word.replace(" ", "_") + "_{}".format(shift)
            df.loc[:, new_feature] = df[word].shift(shift)
            new_features.append(new_feature)
        feature_dict[word] = new_features

    return df, feature_dict


def get_ticker_name(path):
    """
    get ticker name from path

    :param path: path to market df
    :type path: str
    :return: ticker name
    :rtype: str
    """
    name = path.split("/")[-1].split(".")[0]
    name = name.replace(" ", "_")
    return name


def get_market_df(path):
    """
    Get ticker dataframe from path.

    We always drop the first two rows
    with the information

    'field, DAY_TO_DAY_TOT_RETURN_GROSS_DVDS
     date,'

    The total return is divided by 100.
    The return column named as
    "get_ticker_name(path)".

    :param path: path to market df
    :type path: str
    :return: market dataframe
    :rtype: pd.DataFrame
    """
    target_name = get_ticker_name(path)
    market = pd.read_csv(path)
    market = market.drop([0, 1], 0)  # drop first 2 lines
    market.columns = ["date", target_name]
    market.loc[:, "date"] = pd.to_datetime(market.date)
    market.loc[:, target_name] = market[target_name].astype("float") / 100
    return market.reset_index(drop=True)


def merge_data(df_list, freq='D'):
    """
    Merge all df's in the list 'df_list'.
    We assume that all df's are indexed
    by date.

    We resample all df's using the frequency
    "freq", and we concatenate them into
    a single dataframe.

    :param df_list: list of dataframes
    :type df_list: [pd.DataFrame]
    :param freq: frequency
    :type freq: str
    :return: merged dataframe
    :rtype: pd.DataFrame
    """
    list_out = []
    for df in df_list:
        df_loop = df.resample(freq).mean()
        list_out.append(df_loop)

    return pd.concat(list_out, axis=1)


def merge_market_and_gtrends(path,
                             test_size,
                             path_gt_list=["data", "gtrends.csv"]):
    """
    Merge market and google trends data.
    Market data is sliced using the
    parameter "test_size"

    :param path: path to market dataframe
    :type path: str
    :param test_size: value to split the data
                      into training and testing
    :type test_size: float in [0,1] or int
    :param path_gt_list: list of str to create gt path
    :type path_gt_list: [str]
    :return: merged dataframe train and tes
    :rtype: (pd.DataFrame,pd.DataFrame)
    """

    # loading google trends data
    path_gt = os.path.join(*path_gt_list)
    gtrends = pd.read_csv(path_gt)
    gtrends.loc[:, "date"] = pd.to_datetime(gtrends.date)
    gtrends = gtrends.set_index("date").sort_index()

    # loading market data
    market = get_market_df(path)
    name = get_ticker_name(path)
    market = market.rename(columns={"ticker": "date",
                                    name: "target_return"})
    market = market.set_index("date").sort_index()

    # merging
    merged = merge_data([market, gtrends])
    merged = merged.dropna()

    # if the merged data is null or has only one element
    # then both train and test are null
    if merged.shape[0] > 1:
        train, test = train_test_split(merged,
                                       test_size=test_size,
                                       shuffle=False)
        last_day_train = train.sort_index().index[-1]
        first_day_test = test.sort_index().index[0]
        assert last_day_train < first_day_test, "temporal ordering error"
    else:
        train, test = pd.DataFrame(), pd.DataFrame()
    return train, test


def path_filter(paths,
                threshold,
                verbose=True,
                path_gt_list=["data", "gtrends.csv"]):
    """
    filter each market data path by
    assessing the size of the associated
    merged dataframe.

    Remember,
    252 = business days in a year


    :param paths: list of paths to market data
    :type paths: [str]
    :param threshold: minimun number of days in
                      the merged dataframe
                      to not exclude a path
    :type threshold: int
    :param verbose: param to print iteration status
    :type verbose: bool
    :param path_gt_list: list of str to create gt path
    :type path_gt_list: [str]
    :return: list of filtered paths
    :rtype: [str]
    """
    
    new_paths = []
    for p in tqdm(paths, disable=not verbose, desc="filter"):
        df = pd.read_csv(p)
        if len(df.columns) > 1:
            train, test = merge_market_and_gtrends(p,
                                                   test_size=1,
                                                   path_gt_list=path_gt_list)
            df = pd.concat([train, test])
            if df.shape[0] >= threshold:
                new_paths.append(p)
    return new_paths


def check_constant_series(df,
                          target,
                          threshold):
    """
    Return a tag (True/False) if the target series is constant
    for more than the threshold

    :param df: data
    :type df: dataframe
    :param target: name of the column to check
    :type df: [str]
    :param threshold: percentual to check
    :type threshold: float
    :return: True/False boolean
    :rtype: boolean
    """
    target_df = df[[target]].pct_change(1)
    tot_point =df.shape[0]
    constant_points = np.sum(pd.isna(target_df)).iloc[0]
    perc = constant_points / tot_point

    if perc >= threshold:
        tag = True
    else:
        tag = False

    return tag

