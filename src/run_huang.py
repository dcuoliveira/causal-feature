from glob import glob
from time import time
import numpy as np
from multiprocessing import Pool
import os

from word_list.analysis import words
from data_mani.utils import path_filter
from data_mani.utils import merge_market_and_gtrends
from data_mani.utils import get_ticker_name
from feature_selection.huang import run_huang_methods


# variables
SIG_LEVEL = 0.01
MAX_LAG = 20 # maximum number of lags to create
N_CORES = 9 # number of cores to use
OUT_FOLDER = "indices" # name of the marked data folder
DEBUG = False # param to debug the script
TEST_SIZE = 0.5 # pct of the train/test split
THRESHOLD = 252 * 2 # treshold to filted merged datframes
                    # 252 = business days in a year
PAR = False # enable run in paralell
CORREL_THRESHOLD = 0.5 # correlation threshold to apply filter
IS_DISCRETE = True
CONSTANT_THRESHOLD = 0.9 # constant threshold to apply filter

# ajuste pra path do windows
PATHS = sorted(glob(os.path.join(os.path.dirname(__file__), "data/{}/*.csv".format(OUT_FOLDER))))
N_CORES = len(PATHS)  # number of cores to use

# # debug condition
# if DEBUG:
#     words = words[1:50]
#     PATHS = PATHS[1:5]

def huang_fs_vec(paths,
                 test_size=TEST_SIZE,
                 out_folder=OUT_FOLDER,
                 words=words,
                 max_lag=MAX_LAG,
                 sig_level=SIG_LEVEL,
                 correl_threshold=CORREL_THRESHOLD,
                 is_discrete=IS_DISCRETE,
                 constant_threshold=CONSTANT_THRESHOLD):
    """
    vectorized version of the Huang et al. (2019) feature selection techniques.
    for each path in 'paths' we:
        - merge with the gtrends data
        - run the huang_fs functions using
          using 'test_size', 'words' and 'max_lag'
        - save the results in the folder
          'results/huang_fs'

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
    :param correl_threshold: correl_threshold: correlation threshold to apply the filter (excluded
    high correlated series)
    :type correl_threshold: floast
    """

    for path in paths:
        merged, _ = merge_market_and_gtrends(path,
                                             test_size=test_size,
                                             is_discrete=is_discrete)

        name = get_ticker_name(path).replace("_", " ")
        result = run_huang_methods(merged_df=merged,
                                   target_name="target_return",
                                   words=words,
                                   max_lag=max_lag,
                                   verbose=False,
                                   sig_level=sig_level,
                                   constant_threshold=constant_threshold)
        if result is not None:
            out_path = os.path.join(os.path.dirname(__file__),
                                    "results",
                                    "feature_selection",
                                    "huang",
                                    out_folder)

            # check if output dir exists
            if not os.path.isdir(os.path.join(out_path)):
                os.mkdir(os.path.join(out_path))

            out_path = os.path.join(out_path,
                                    name + ".csv")
            result.to_csv(out_path, index=False)


def huang_fs_par(paths, n_cores=N_CORES, par=False):
    """
    parallelized version of the Huang et al. (2019) feature selection techniques.

    :param paths: list of paths to market data
    :type paths: [str]
    :param n_cores: number of cores to use
    :type n_cores: int
    """

    path_split = np.array_split(paths, n_cores)
    if par:
        pool = Pool(n_cores)
        result = pool.map(huang_fs_vec, path_split)
        pool.close()
        pool.join()
    else:
        for path in path_split:
            huang_fs_vec(path)

    return None

if __name__ == '__main__':
    paths = path_filter(paths=PATHS,
                        threshold=THRESHOLD)
    pct = len(paths) / len(PATHS)

    print("\nnumber of paths = {}".format(len(paths)))
    print("({:.1%} of paths)".format(pct))
    init = time()
    huang_fs_par(paths=paths,
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
            out_path = os.path.join(os.path.dirname(__file__),
                                    "results",
                                    "feature_selection",
                                    "huang",
                                    OUT_FOLDER,
                                    name + ".csv")
            os.remove(out_path)
