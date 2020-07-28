from pytrends.request import TrendReq
import pandas as pd
from word_list.basic import politics, toy
from multiprocessing import Pool
import numpy as np
import time
from datetime import date
from copy import copy


# global parameters (used for multiprocessing)
REFWORD = "google"
TARGET = politics
# TARGET = toy
# TARGET = ["republican", "democrat"]
SLEEPTIME = 3
INIT_DATE = "2010-01-01"
# INIT_DATE = "2018-01-01"
N_CORES = 4


def get_time_intervals(init_date, timedelta="180d"):
    """
    get time intevals starting from the date "init_date"
    until today. Each date interval has "timedelta" days.
    For example, "timedelta=180d" correspond to 6 months.
    Each interval is in the format of the gtrends library.

    :param init_date: initial date
    :type init_date: str
    :param timedelta: time to add
    :type timedelta: str
    :return: list of dates intervals to be used
    :rtype: pd.DataFrame
    """
    intervals = []
    today = date.today()
    final_date = today.strftime("%Y-%m-%d")
    var_date = init_date

    while pd.Timestamp(var_date) < pd.Timestamp(final_date):
        next_date = pd.Timestamp(var_date) + pd.Timedelta(timedelta)
        next_date = str(next_date.date())
        timeframe = "{} {}".format(var_date, next_date)
        intervals.append(timeframe)
        var_date = copy(next_date)
    return intervals


# all time intervals
INTERVALS = get_time_intervals(INIT_DATE)


def get_daily_trend_from_word_list(kw_list,
                                   reference_word=REFWORD):
    """
    get a google trends word frequency df
    from a list of key words. All words have frequencies relative
    to the high searched word "reference_word".
    https://towardsdatascience.com/using-google-trends-at-scale-1c8b902b6bfa

    We are using a list of intervals in 'INTERVALS' to obtain
    daily information about each word in "kw_list". To combine repeated
    information among the different daily data frames we take the mean -
    this is done to circunvent the time limit in the google trends API.


    :param kw_list: word list
    :type kw_list: [str]
    :param reference_word: reference_word
    :type reference_word: str
    :return: word frequency dataframe
    :rtype: pd.DataFrame
    """
    dfList = []
    for kw in kw_list:
        daily_dfs = []
        for timeframe in INTERVALS:
            print(kw, timeframe)
            time.sleep(SLEEPTIME)
            trends = TrendReq(hl='en-US', tz=360)
            trends.build_payload(kw_list=[kw, REFWORD],
                                 geo='US', timeframe=timeframe)
            df = trends.interest_over_time()
            daily_dfs.append(df)
        daily_ts = pd.concat(daily_dfs).reset_index().groupby("date").mean()
        daily_ts.to_csv("data/daily/{}.csv".format(kw))
        dfList.append(daily_ts[kw])
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
    result = pool.map(get_daily_trend_from_word_list,
                      kw_list_split)
    # print(len(result))
    result = pd.concat(result, 1)
    pool.close()
    pool.join()
    return result[kw_list]


if __name__ == '__main__':
    init = time.time()
    TARGET = list(set(TARGET))
    size = len(TARGET)
    print(
        "Collecting {} words with {} cores (reference word = '{}')....".format(
            size,
            N_CORES,
            REFWORD,))
    # df = get_df_from_word_list_parallel(TARGET, n_cores=N_CORES)
    df = get_daily_trend_from_word_list(TARGET)
    df.to_csv("data/temp_daily_politics_{}.csv".format(REFWORD))
    final = time.time() - init
    final = final / 60
    print("process duration = {:.2f} minutes".format(final))
