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
WINDOW_SIZE = 110
SIG_LEVEL = 0.05
MAX_LAG = 30 # maximum number of lags to create
N_CORES = 2 # number of cores to use
OUT_FOLDER = "nyse" # name of the marked data folder
DEBUG = True # param to debug the script
TEST_SIZE = 0.5 # pct of the train/test split
THRESHOLD = 252 * 2 # treshold to filted merged datframes
                    # 252 = business days in a year
PATHS = sorted(glob("data/crsp/{}/*.csv".format(OUT_FOLDER)))

# debug condition
if DEBUG:
    words = words[:100]
    PATHS = PATHS[10:20]

def huang_fs_vec(paths,
            test_size=TEST_SIZE,
            out_folder=OUT_FOLDER,
            words=words,
            max_lag=MAX_LAG,
            sig_level=SIG_LEVEL,
            window_size=WINDOW_SIZE):
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
    """
    for path in paths:
        merged, _ = merge_market_and_gtrends(path, test_size=test_size)

        name = get_ticker_name(path).replace("_", " ")
        result = run_huang_methods(merged_df=merged, target_name="target_return",
                                   words=words, max_lag=max_lag, verbose=False,
                                   sig_level=sig_level, window_size=window_size)

        out_path = os.path.join("results", "sfi", out_folder, name + ".csv")
        result.to_csv(out_path, index=False)


def huang_fs_par(paths, n_cores=N_CORES):
    """
    parallelized version of the Huang et al. (2019) feature selection techniques.

    :param paths: list of paths to market data
    :type paths: [str]
    :param n_cores: number of cores to use
    :type n_cores: int
    """
    path_split = np.array_split(paths, n_cores)
    pool = Pool(n_cores)
    result = pool.map(huang_fs_vec, path_split)
    pool.close()
    pool.join()
    return result

if __name__ == '__main__':
    paths = path_filter(paths=PATHS,
                        threshold=THRESHOLD)

    init = time()
    huang_fs_par(paths)
    tot_time = time() - init
    tot_time = tot_time / 60
    print(
        "total time = {:.3f} (minutes)\nusing {} cores".format(
            tot_time,
            N_CORES))