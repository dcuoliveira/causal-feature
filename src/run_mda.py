import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from time import time
from multiprocessing import Pool
from word_list.analysis import words
from feature_selection.mda import get_mda_scores
from data_mani.utils import merge_market_and_gtrends
from data_mani.utils import get_ticker_name
from data_mani.utils import path_filter
from data_mani.utils import target_ret_to_directional_movements

# Variables
N_SPLITS = 2  # number of CV splits
N_ESTIMATORS = 10  # number of trees in the random forest model
N_CORES = 30  # number of cores to use
MAX_LAG = 20  # maximum number of lags to create
# google trends features
OUT_FOLDER = "nyse"  # name of the marked data folder
DEBUG = False  # param to debug the script
TEST_SIZE = 0.5  # pct of the train/test split
THRESHOLD = 252 * 2  # treshold to filted merged datframes
# 252 = business days in a year
PATHS = sorted(glob("data/crsp/{}/*.csv".format(OUT_FOLDER)))

# debug condition
if DEBUG:
    words = words[:3]
    PATHS = PATHS[10:20]


def mda_vec(paths,
            test_size=TEST_SIZE,
            out_folder=OUT_FOLDER,
            words=words,
            max_lag=MAX_LAG,
            n_estimators=N_ESTIMATORS,
            n_splits=N_SPLITS):
    """
    vectorized version of the mda function.
    for each path in 'paths' we:
        - merge with the gtrends data
        - run the mda_scores functions
          'words' and 'max_lag'
        - save the results in the folder
          'results/mda'

    :param paths: list of paths to market data
    :type paths: [str]
    :param out_folder: path to sabe the mda results
    :type out_folder: str
    :param words: list of words to use in the gtrends data
    :type words: [str]
    :param max_lag: maximun number of lags to apply on gtrends
                    features
    :type max_lag: int
    :param n_estimators: number of trees in the random forest model
    :type n_estimators: int
    :param n_splits: number of cross-validation splits
    :type n_splits: int
    """
    n = len(paths)
    rds = np.random.randint(1000000, size=n)
    names = [get_ticker_name(path) for path in paths]
    log = pd.DataFrame({"ticker": names,
                        "random_state": rds})
    first, last = names[0], names[-1]
    log_path = os.path.join(
        "logs",
        "mda",
        out_folder,
        "{}_to_{}.csv".format(
            first,
            last))
    log.to_csv(log_path, index=False)

    for path, random_state in zip(paths, rds):
        merged, _ = merge_market_and_gtrends(path, test_size=test_size)
        name = get_ticker_name(path).replace("_", " ")
        target_ret_to_directional_movements(merged, y_name="target_return")
        result = get_mda_scores(merged_df=merged,
                                target_name="target_return",
                                words=words,
                                max_lag=max_lag,
                                n_splits=n_splits,
                                n_estimators=n_estimators,
                                verbose=False,
                                random_state=random_state)

        out_path = os.path.join("results", "mda", out_folder, name + ".csv")
        result.to_csv(out_path, index=False)


def mda_par(paths, n_cores=N_CORES):
    """
    parallelized version of the mda_vec function

    :param paths: list of paths to market data
    :type paths: [str]
    :param n_cores: number of cores to use
    :type n_cores: int
    """
    path_split = np.array_split(paths, n_cores)
    pool = Pool(n_cores)
    result = pool.map(mda_vec, path_split)
    pool.close()
    pool.join()
    return result


if __name__ == '__main__':
    paths = path_filter(paths=PATHS,
                        threshold=THRESHOLD)
    pct = len(paths) / len(PATHS)

    print("\nnumber of paths = {}".format(len(paths)))
    print("({:.1%} of paths)".format(pct))

    init = time()
    mda_par(paths)
    tot_time = time() - init
    tot_time = tot_time / 60
    print(
        "total time = {:.3f} (minutes)\nusing {} cores".format(
            tot_time,
            N_CORES))

    # # Cleaning logs:
    log_paths = glob("logs/mda/{}/*.csv".format(OUT_FOLDER))
    log_paths = [lpath for lpath in log_paths if lpath.find(
        "random_states") == -1]

    if DEBUG:
        for lpath in log_paths:
            pass
            os.remove(lpath)
    else:
        final_log_path = os.path.join(
            "logs", "mda", OUT_FOLDER, "random_states.csv")
        logs = [pd.read_csv(lpath) for lpath in log_paths]
        log = pd.concat(logs)
        log.to_csv(final_log_path, index=False)
        for lpath in log_paths:
            os.remove(lpath)

    # Cleaning debug
    if DEBUG:
        for p in paths:
            name = get_ticker_name(p).replace("_", " ")
            out_path = os.path.join(
                "results", "mda", OUT_FOLDER, name + ".csv")
            os.remove(out_path)
