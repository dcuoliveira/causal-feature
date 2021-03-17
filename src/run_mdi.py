import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from time import time
from multiprocessing import Pool
from word_list.analysis import words
from feature_selection.mdi import get_mdi_scores
from data_mani.utils import merge_market_and_gtrends
from data_mani.utils import get_ticker_name
from data_mani.utils import path_filter

# Variables
N_CORES = 2 # number of cores to use
MAX_LAG = 20 # maximum number of lags to create
             # google trends features
OUT_FOLDER = "indices" # name of the marked data folder
DEBUG = True # param to debug the script
TEST_SIZE = 0.5 # pct of the train/test split
THRESHOLD = 252 * 2 # treshold to filted merged datframes
                    # 252 = business days in a year
PATHS = sorted(glob("data/{}/*.csv".format(OUT_FOLDER)))

# debug condition
if DEBUG:
    words = words[:3]


def mdi_vec(paths,
            test_size=TEST_SIZE,
            out_folder=OUT_FOLDER,
            words=words,
            max_lag=MAX_LAG):
    """
    vectorized version of the mdi function.
    for each path in 'paths' we:
        - merge with the gtrends data
        - run the mdi_scores functions
          'words' and 'max_lag'
        - save the results in the folder
          'results/mdi'

    :param paths: list of paths to market data
    :type paths: [str]
    :param out_folder: path to sabe the mdi results
    :type out_folder: str
    :param words: list of words to use in the gtrends data
    :type words: [str]
    :param max_lag: maximun number of lags to apply on gtrends
                    features
    :type max_lag: int
    """
    n = len(paths)
    rds = np.random.randint(1000000, size=n)
    names = [get_ticker_name(path) for path in paths]
    log = pd.DataFrame({"ticker": names, 
                        "random_state":rds})
    first, last = names[0], names[-1]
    log_path = os.path.join("logs", "mdi", out_folder, "{}_to_{}.csv".format(first, last))
    log.to_csv(log_path, index=False)

    for path, random_state in zip(paths,rds):
        merged, _ = merge_market_and_gtrends(path, test_size=test_size)
        name = get_ticker_name(path).replace("_", " ")
        result = get_mdi_scores(merged_df=merged,
                                target_name="target_return",
                                words=words,
                                max_lag=max_lag,
                                verbose=False,
                                random_state=random_state)

        out_path = os.path.join("results",
                                "feature_selection",
                                "mdi", out_folder, name + ".csv")
        result.to_csv(out_path, index=False)


def mdi_par(paths, n_cores=N_CORES):
    """
    parallelized version of the mdi_vec function

    :param paths: list of paths to market data
    :type paths: [str]
    :param n_cores: number of cores to use
    :type n_cores: int
    """
    path_split = np.array_split(paths, n_cores)
    pool = Pool(n_cores)
    result = pool.map(mdi_vec, path_split)
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
    mdi_par(paths)
    tot_time = time() - init
    tot_time = tot_time / 60
    print(
        "total time = {:.3f} (minutes)\nusing {} cores".format(
            tot_time,
            N_CORES))

    # Cleaning logs:
    log_paths = glob("logs/mdi/{}/*.csv".format(OUT_FOLDER))
    log_paths = [lpath for lpath in log_paths if lpath.find("random_states") ==-1]

    if DEBUG:
        for lpath in log_paths: 
                os.remove(lpath)
    else:
        final_log_path =  os.path.join("logs", "mdi", OUT_FOLDER, "random_states.csv")
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
                "results",
                "feature_selection",
                "mdi", OUT_FOLDER, name + ".csv")
            os.remove(out_path)

