from pytrends.request import TrendReq
import pandas as pd
from word_list.basic import politics
from multiprocessing import Pool
import numpy as np
import time

REFWORD = "google"
TARGET = politics
SLEEPTIME = 1


def get_df_from_word_list(kw_list, reference_word=REFWORD):
    """
    get a google trends word frequency df
    from a list of key words. All words have frequencies relative
    to the high searched word "reference_word".
    https://towardsdatascience.com/using-google-trends-at-scale-1c8b902b6bfa

    :param kw_list: word list
    :type kw_list: [str]
    :param reference_word: reference_word
    :type reference_word: str
    :return: word frequency dataframe
    :rtype: pd.DataFrame
    """
    dfList = []
    for kw in kw_list:
        time.sleep(SLEEPTIME)
        trends = TrendReq(hl='en-US', tz=360)
        trends.build_payload(kw_list=[kw, reference_word],
                             geo='US')
        df = trends.interest_over_time()
        dfList.append(df[kw])
    return pd.concat(dfList, axis=1)


def get_df_from_word_list_parallel(kw_list, n_cores):
    """
    parallelized version of get_df_from_word_list
    :param n_cores: number of cores to use
    :type n_cores: int
    :return: word frequency dataframe
    :rtype: pd.DataFrame
    """
    kw_list_split = np.array_split(kw_list, n_cores)
    pool = Pool(n_cores)
    result = pool.map(get_df_from_word_list, kw_list_split)
    result = pd.concat(result, 1)
    pool.close()
    pool.join()
    return result[kw_list]


if __name__ == '__main__':
    init = time.time()
    TARGET = list(set(TARGET))
    size = len(TARGET)
    print("Collecting {} words with 4 cores (reference word = '{}')....".format(size, REFWORD))
    df = get_df_from_word_list_parallel(TARGET, n_cores=4)
    df.to_csv("data/politics_{}.csv".format(REFWORD))
    final = time.time() - init
    final = final / 60
    print("process duration = {:.2f} minutes".format(final))
