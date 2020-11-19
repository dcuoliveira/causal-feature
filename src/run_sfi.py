import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from time import time
from multiprocessing import Pool
from word_list.analysis import words
from feature_selection.sfi import get_sfi_scores
from data_mani.utils import get_market_df, merge_data
from data_mani.utils import get_ticker_name


# Variables
N_SPLITS = 5
N_CORES = 2
MAX_LAG = 30
OUT_FOLDER = "nyse"
DEBUG = True
PATHS = sorted(glob("data/crsp/{}/*.csv".format(OUT_FOLDER)))

# debug condition
if DEBUG:
    words = words[:3]
    PATHS = PATHS[0:4]


def merge_market_and_gtrends(path,
                             init_train="2004-01-01",
                             final_train="2010-01-01"):
    """
    Merge market and google trends data.
    Market data is sliced using the
    training interval

    [init_train: final_train]


    :param path: path to market dataframe
    :type path: str
    :param init_train: initial timestamp for training
    :type init_train: str
    :param final_train: final timestamp for training
    :type final_train: str
    :return: merged dataframe
    :rtype: pd.DataFrame
    """

    # loading google trends data
    path_gt = os.path.join("data", "gtrends.csv")
    gtrends = pd.read_csv(path_gt)
    gtrends.loc[:, "date"] = pd.to_datetime(gtrends.date)
    gtrends = gtrends.set_index("date")

    # loading market data
    market = get_market_df(path)
    name = get_ticker_name(path)
    market = market.rename(columns={"ticker": "date",
                                    name: "target_return"})

    # using only the training sample
    market = market.set_index("date")
    market = market[init_train:final_train]

    # merging
    merged = merge_data([market, gtrends])
    merged = merged.dropna()

    return merged


def path_filter(paths, threshold=365):
    """
    filter each market data path by
    assessing the size of the associated
    merged dataframe.


    :param paths: list of paths to market data
    :type paths: [str]
    :param threshold: minimun number of days in
                      the merged dataframe
                      to not exclude a path
    :type threshold: int
    :return: list of filtered paths
    :rtype: [str]
    """
    new_paths = []
    for p in tqdm(paths, desc="filter"):
        df = pd.read_csv(p)
        if len(df.columns) > 1:
            df = merge_market_and_gtrends(p)
            if df.shape[0] >= threshold:
                new_paths.append(p)
    return new_paths


def sfi_vec(paths,
            out_folder=OUT_FOLDER,
            n_splits=N_SPLITS,
            words=words,
            max_lag=MAX_LAG):
    """
    vectorized version of the sfi function.
    for each path in 'paths' we:
        - merge with the gtrends data
        - run the sfi_scores functions
          using the parameters 'n_splits',
          'words' and 'max_lag'
        - save the results in the folder
          'results/sfi'

    :param paths: list of paths to market data
    :type paths: [str]
    :param out_folder: path to sabe the sfi results
    :type out_folder: str
    :param n_splits: number of cross-validation splits
    :type n_splits: int
    :param words: list of words to use in the gtrends data
    :type words: [str]
    :param max_lag: maximun number of lags to apply on gtrends
                    features
    :type max_lag: int
    """
    merged_dfs = [merge_market_and_gtrends(p) for p in paths]
    names = [get_ticker_name(p).replace("_", " ") for p in paths]
    results = []
    for merged, name in zip(merged_dfs, names):
        result = get_sfi_scores(merged_df=merged,
                                target_name="target_return",
                                words=words,
                                max_lag=max_lag,
                                verbose=False,
                                n_splits=n_splits)

        out_path = os.path.join("results", "sfi", out_folder, name + ".csv")
        result.to_csv(out_path, index=False)


def sfi_par(paths, n_cores=N_CORES):
    """
    parallelized version of the sfi_vec function

    :param paths: list of paths to market data
    :type paths: [str]
    :param n_cores: number of cores to use
    :type n_cores: int
    """
    path_split = np.array_split(paths, n_cores)
    pool = Pool(n_cores)
    result = pool.map(sfi_vec, path_split)
    pool.close()
    pool.join()
    return result


if __name__ == '__main__':
    paths = path_filter(PATHS)
    print("\nnumber of paths = {}".format(len(paths)))
    init = time()
    sfi_par(paths)
    tot_time = time() - init
    tot_time = tot_time / 60
    print(
        "total time = {:.3f} (minutes)\nusing {} cores".format(
            tot_time,
            N_CORES))
