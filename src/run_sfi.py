import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from time import time
from multiprocessing import Pool
from word_list.analysis import words
from feature_selection.sfi import get_sfi_scores



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


# ## 1 version

# In[4]:

def filter_(paths):
    new_paths = []
    for p in tqdm(paths, desc="filter"):
        df,_ = read_and_merge(p)
        if df.shape[0] >= 252*2 :
            new_paths.append(p)
    return new_paths

SPLIT = 5
paths = sorted(glob("data/crsp/nyse/*.csv"))
paths = paths[0:30]
paths = filter_(paths)
print("\nnumber of paths = {}\n".format(len(paths)))

#path1 = "data/crsp/nyse/CYN US Equity.csv"
#path2 = "data/crsp/nyse/C US Equity.csv"
#path3 = "data/crsp/nyse/CA US Equity.csv"
#path4 = "data/crsp/nyse/DRE US Equity.csv"
#paths = [path1, path2, path3, path4]



init = time()
for p in paths:

    m, name = read_and_merge(p)

    result = get_sfi_scores(merged_df=m,
                            target_name="target_return",
                            words=words,
                            max_lag=30,
                            verbose=True,
                            n_splits=SPLIT)
    #name = t.replace("_return", "")
    #name = name.replace("_", " ")
    out_path = os.path.join("results", "sfi", name + ".csv")
    result.to_csv(out_path, index=False)

tot_time = time() - init
tot_time = tot_time / 60
print("\nNAIVE : total time = {:.3f} (minutes)".format(tot_time))


# ## Parallel

def sfi_vec(paths):
    m_ts = [read_and_merge(p) for p in paths]
    results = []
    for m, name in m_ts:
        result = get_sfi_scores(merged_df=m,
                                target_name="target_return",
                                words=words,
                                max_lag=30,
                                verbose=False,
                                n_splits=SPLIT)

        #name = t.replace("_return", "")
        #name = name.replace("_", " ")
        out_path = os.path.join("results","sfi","nyse", name + ".csv")
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


init = time()
sfi_par(paths, n_cores=8)
tot_time = time() - init
tot_time = tot_time / 60
print("\nPARALLEL: total time = {:.3f} (minutes)".format(tot_time))
