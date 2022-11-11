import os
import time
import pandas as pd
from copy import copy
from glob import glob
from tqdm import tqdm
from datetime import date
from pytrends.request import TrendReq
from word_list.sanity_check import preis

# global parameters
TARGET = preis
SLEEPTIME = 60
INIT_DATE = "2004-01-01"
FINAL_DATE = "2021-01-01"
TIMEZONE_OFFSET = 360
HOST_LANGUAGE = 'en-US'
COUNTRY_ABBREVIATION = 'US'
SAMPLES = 5

# To add current time
# today = date.today()
# FINAL_DATE = today.strftime("%Y-%m-%d")


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
    final_date = FINAL_DATE
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


def get_daily_trend_from_word_list(kw_list):
    """
    get a google trends word frequency df
    from a list of key words. All words have frequencies relative
    to the high searched word "reference_word".
    https://towardsdatascience.com/using-google-trends-at-scale-1c8b902b6bfa

    We are using a list of intervals in 'INTERVALS' to obtain
    daily information about each word in "kw_list". To combine repeated
    information among the different daily data frames we take the mean -
    this is done to circunvent the time limit in the google trends API.

    Each moment a time series is completed, it is saved in "data/daily"

    :param kw_list: word list
    :type kw_list: [str]
    :param reference_word: reference_word
    :type reference_word: str
    """

    for kw in kw_list:

        daily_sample_dfs = []
        for sample in range(SAMPLES):

            daily_dfs = []
            for timeframe in tqdm(INTERVALS, desc="'{}': time intervals".format(kw + " " + str(sample))):
                success = 'no'
                while success == 'no':
                    try:
                        trends = TrendReq(hl=HOST_LANGUAGE, tz=TIMEZONE_OFFSET)
                        trends.build_payload(kw_list=[kw],
                                             geo=COUNTRY_ABBREVIATION,
                                             timeframe=timeframe)
                        df = trends.interest_over_time()
                        daily_dfs.append(df.drop(['isPartial'], axis=1))
                        success = 'yes'
                    except BaseException:
                        time.sleep(SLEEPTIME)

            daily_agg_df = pd.concat(daily_dfs, axis=0).groupby("date").mean()

            target_path = os.path.join("data", "all_daily_trends", "daily_trends{}".format(sample))

            # check if output dir exists
            if not os.path.isdir(os.path.join(target_path)):
                os.mkdir(os.path.join(target_path))

            target_path = os.path.join(target_path, "{}.csv".format(kw))

            daily_agg_df.to_csv(target_path)


if __name__ == '__main__':
    # main problem of this script
    # https://stackoverflow.com/questions/50571317/pytrends-the-request-failed-google-returned-a-response-with-code-429

    # possible solution
    # pip3 install --upgrade --user
    # git+https://github.com/GeneralMills/pytrends

    # Run code in different computers in different times to get all words

    already_collected = glob(os.path.join("data", "daily_trend", "*.csv"))
    already_collected = [i.split("/")[2] for i in already_collected]
    already_collected = [i.split(".")[0] for i in already_collected]
    NEW_TARGET = [i for i in TARGET if i not in already_collected]
    init = time.time()
    NEW_TARGET = sorted(set(NEW_TARGET))
    size = len(NEW_TARGET)
    df = get_daily_trend_from_word_list(NEW_TARGET)
    final = time.time() - init
    final = final / 60
    print("process duration = {:.2f} minutes".format(final))
