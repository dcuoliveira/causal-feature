import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from time import time
from multiprocessing import Pool
from word_list.analysis import words
from feature_selection.sfi import get_sfi_scores

words = words[:3]



def merge_market_gtrends(market, gtrends):
    """
    naive version esperar daniel

    """
    merged = pd.merge_asof(market, gtrends, left_index=True, right_index=True)
    return merged.dropna()


def read_and_merge(path, init_train="2000", final_train="2010"):

    # ## Loading trends
    gtrends = pd.read_csv("data/gtrends.csv")
    gtrends.loc[:, "date"] = pd.to_datetime(gtrends.date)
    gtrends = gtrends.set_index("date")

    # ## Loading and preprossesing market data
    name = path.split("/")[-1].split(".")[0]
    # target_name = name.replace(" ", "_") + "_return"
    target_name = "target_return"
    market = pd.read_csv(path)
    market = market.drop([0, 1], 0)
    market = market.rename(columns={"ticker": "date",
                                    name: target_name})
    market.loc[:, "date"] = pd.to_datetime(market.date)
    market.loc[:, target_name] = market[target_name].astype("float") / 100
    market = market.set_index("date")

    # using only the training sample
    market = market[init_train:final_train]

    return merge_market_gtrends(market, gtrends), name


def filter_(paths):
    new_paths = []
    for p in tqdm(paths, desc="filter"):
        df,_ = read_and_merge(p)
        if df.shape[0] >= 252*2 :
            new_paths.append(p)
    return new_paths


def sfi_vec(paths, out_folder="nasdaq", n_splits=5, words=words, max_lag=max_lag):
    m_ts = [read_and_merge(p) for p in paths]
    results = []
    for m, name in m_ts:
        result = get_sfi_scores(merged_df=m,
                                target_name="target_return",
                                words=words,
                                max_lag=max_lag,
                                verbose=False,
                                n_splits=n_splits)

        out_path = os.path.join("results","sfi",out_folder, name + ".csv")
        result.to_csv(out_path, index=False)


def sfi_par(paths, n_cores):
    """
    parallelized version of the sfi function
    """
    path_split = np.array_split(paths, n_cores)
    pool = Pool(n_cores)
    result = pool.map(sfi_vec, path_split)
    pool.close()
    pool.join()
    return result


# SPLIT = 5
N_CORES = 4
paths = sorted(glob("data/crsp/nasdaq/*.csv"))
paths = paths[0:30]
paths = filter_(paths)
print("\nnumber of paths = {}\n".format(len(paths)))


init = time()
sfi_par(paths, n_cores=N_CORES)
tot_time = time() - init
tot_time = tot_time / 60
print("\nPARALLEL: total time = {:.3f} (minutes)".format(tot_time))
