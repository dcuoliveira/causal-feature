from glob import glob
from time import time
import numpy as np
from multiprocessing import Pool
import os

from word_list.analysis import words
from data_mani.utils import path_filter
from data_mani.utils import merge_market_and_gtrends
from data_mani.utils import get_ticker_name
from feature_selection.PCMB import run_PCMB
import random


# variables
SIG_LEVEL = 0.01
MAX_LAG = 20 # maximum number of lags to create
N_CORES = 5 # number of cores to use
OUT_FOLDER = "spx" # name of the marked data folder
DEBUG = False # param to debug the script
TEST_SIZE = 0.5 # pct of the train/test split
THRESHOLD = 252 * 2 # treshold to filted merged datframes
                    # 252 = business days in a year
PAR = True # enable run in paralell
IS_DISCRETE = False

# ajuste pra path do windows
PATHS = sorted(glob("data/index/{}/*.csv".format(OUT_FOLDER)))

# debug condition
if DEBUG:
    words = words
    PATHS = PATHS[1:5]

def PCMB_fs_vec(paths,
                test_size=TEST_SIZE,
                out_folder=OUT_FOLDER,
                words=words,
                max_lag=MAX_LAG,
                sig_level=SIG_LEVEL,
                is_discrete=IS_DISCRETE,
                MB_algo_type=MB_ALGO_NAME):
    """
    TODO

    :param paths: list of paths to market data
    :type paths: [str]
    :param out_folder: path to sabe the sfi results
    :type out_folder: str
    :param words: list of words to use in the gtrends data
    :type words: [str]
    :param max_lag: maximun number of lags to apply on gtrends
                    features
    :type max_lag: int
    :param sig_level: significance level to use as threshold
    :type sig_level: int
    """

    for path in paths:
        merged, _ = merge_market_and_gtrends(path, test_size=test_size)

        name = get_ticker_name(path).replace("_", " ")
        result = run_PCMB(merged_df=merged,
                          target_name="target_return",
                          words=words,
                          max_lag=max_lag,
                          verbose=False,
                          sig_level=sig_level,
                          is_discrete=is_discrete)

        out_path = os.path.join("results",
                                "feature_selection",
                                "PCMB",
                                out_folder,
                                name + ".csv")
        result.to_csv(out_path, index=False)


def PCMB_fs_par(paths,
                n_cores=N_CORES,
                par=False):
    """
    TODO

    :param paths: list of paths to market data
    :type paths: [str]
    :param n_cores: number of cores to use
    :type n_cores: int
    """

    path_split = np.array_split(paths,
                                n_cores)
    if par:
        pool = Pool(n_cores)
        result = pool.map(PCMB_fs_vec,
                          path_split)
        pool.close()
        pool.join()
    else:
        for path in path_split:
            PCMB_fs_vec(path)
        result = None

    return result

if __name__ == '__main__':
    paths = path_filter(paths=PATHS,
                        threshold=THRESHOLD)
    pct = len(paths) / len(PATHS)

    print("\nnumber of paths = {}".format(len(paths)))
    print("({:.1%} of paths)".format(pct))
    init = time()
    PCMB_fs_par(paths=paths,
                par=PAR)
    tot_time = time() - init
    tot_time = tot_time / 60
    print(
        "total time = {:.3f} (minutes)\nusing {} cores".format(
            tot_time,
            N_CORES))

   #  Cleaning debug
    if DEBUG:
        for p in paths:
            name = get_ticker_name(p).replace("_", " ")
            out_path = os.path.join("results",
                                    "feature_selection", 
                                    "PCMB",
                                    OUT_FOLDER,
                                    name + ".csv")
            os.remove(out_path)
