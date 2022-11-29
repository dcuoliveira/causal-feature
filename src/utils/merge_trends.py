import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from tqdm import tqdm
from word_list.basic import politics1, politics2
from word_list.basic import business, preis, huang
from statsmodels.tsa.stattools import adfuller


def create_gtrends_data_using_base_path(base_path):
    """
    gather all data in the path "base_path" into
    a single dataframe.
    """
    trends_path = os.path.join(base_path, "*.csv")
    daily_dfs_path = sorted(glob(trends_path))
    daily_dfs = [pd.read_csv(path) for path in daily_dfs_path]
    daily_dfs_names = [i.split("/")[3] for i in daily_dfs_path]
    daily_dfs_names = [i.split(".")[0] for i in daily_dfs_names]

    trends_list = []
    for name, df in zip(daily_dfs_names, daily_dfs):
        df.index = pd.to_datetime(df.date)
        ts = df[name]
        new_name = name.replace(" ", "_")
        ts.name = name
        trends_list.append(ts)

    trends = pd.concat(trends_list, 1)
    trends = trends.fillna(0.0)
    columns = list(trends.columns)

    # checking the presence of the words
    # that will be used in analysis

    assert [c for c in huang if c not in preis] == []
    assert [c for c in preis if c not in columns] == []
    assert [c for c in huang if c not in columns] == []
    assert [c for c in politics1 if c not in columns] == []
    assert [c for c in politics2 if c not in columns] == []
    assert [c for c in business if c not in columns] == []

    selected_words = politics1 + politics2 + business + preis
    selected_words = sorted(set(selected_words))
    # removing words with 0's only
    selected_words.remove("notability")
    selected_words.remove("rare earths")

    trends = trends[selected_words]
    trends = trends[:"2020-12-31"]
    return trends


if __name__ == '__main__':

    # get the df's of all gtrends samples
    base_daily_dfs_paths = sorted(glob("data/all_daily_trends/*"))
    df_list = [create_gtrends_data_using_base_path(
        p) for p in base_daily_dfs_paths]

    # create median df
    word_list = df_list[0].columns.to_list()
    all_sequences = []

    for word in word_list:
        cut = [df[word] for df in df_list]
        ts = pd.concat(cut, 1)
        ts = ts.median(1)
        ts.name = word
        all_sequences.append(ts)

    trends = pd.concat(all_sequences, 1)
    for df in df_list:
        assert trends.shape == df.shape

    # sanity check
    # here we check if the first difference
    # of each gtrends series is an stationary series

    alpha = 0.01
    diff_param = 1
    obs = []
    for c in tqdm(trends.columns, desc="check for stationarity"):
        s = trends[c].diff(diff_param).dropna().values
        result_adfuller = adfuller(s)
        p_value = result_adfuller[1]
        obs.append((c, p_value, p_value < alpha))

    obs = pd.DataFrame(obs, columns=["column", "p_value", "test"])
    assert np.all(obs["test"])

    # final df with all features
    final_trends = trends.diff(1).dropna()
    path = os.path.join("../data", "gtrends.csv")
    final_trends.to_csv(path)
    print("\nfile saved")
    print("path = {}".format(path))

