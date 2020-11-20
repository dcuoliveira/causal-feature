import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from time import time
from multiprocessing import Pool
from word_list.analysis import words
from feature_selection.sfi import get_sfi_scores
from data_mani.utils import merge_market_and_gtrends
from data_mani.utils import get_ticker_name
from data_mani.utils import path_filter


# Variables
N_SPLITS = 5
N_CORES = 2
MAX_LAG = 30
OUT_FOLDER = "nyse"
DEBUG = True
PATHS = sorted(glob("data/crsp/{}/*.csv".format(OUT_FOLDER)))
# PATHS = PATHS[0:800]

# debug condition
if DEBUG:
    words = words[:3]
    PATHS = PATHS[0:4]


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
