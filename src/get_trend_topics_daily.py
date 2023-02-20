import os
import time
import pandas as pd
from copy import copy
from glob import glob
from tqdm import tqdm
from pytrends.request import TrendReq
from word_list.topics import topics

# global parameters
SLEEPTIME = 60
INIT_DATE = "2004-01-01"
FINAL_DATE = "2022-08-31"
TIMEZONE_OFFSET = 360
HOST_LANGUAGE = 'en-US'
COUNTRY_ABBREVIATION = 'US'
SAMPLES = 5

pytrends = TrendReq(hl=HOST_LANGUAGE, tz=TIMEZONE_OFFSET)

suggs_list = []
for topic in topics:
    suggs = pytrends.suggestions(topic)
    tmp_suggs_df = pd.DataFrame(suggs)
    tmp_suggs_df = tmp_suggs_df.loc[tmp_suggs_df["type"] == "Topic"]

    suggs_list.append(tmp_suggs_df)

suggs_df = pd.concat(suggs_list, axis=0)
suggs_df = suggs_df.reset_index(drop=True).drop_duplicates()

TARGET = list(suggs_df["mid"].unique())

def get_time_intervals(init_date, timedelta="360d"):
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
        final_kw = suggs_df.loc[suggs_df["mid"] == kw]["title"].item()
        
        for sample in range(SAMPLES):

            target_path = os.path.join(os.path.dirname(__file__), "data", "all_daily_trends", "daily_trends{}".format(sample))
            target_path_check = os.path.exists(os.path.join(target_path, "{}.csv".format(final_kw)))
            if target_path_check:
                continue

            daily_dfs = []
            for timeframe in tqdm(INTERVALS, desc="'{}': time intervals".format(final_kw + " " + str(sample))):
                success = 'no'
                while success == 'no':
                    try:
                        trends = TrendReq(hl=HOST_LANGUAGE, tz=TIMEZONE_OFFSET)
                        trends.build_payload(kw_list=[kw],
                                             geo=COUNTRY_ABBREVIATION,
                                             timeframe=timeframe)
                        df = trends.interest_over_time()

                        if df.shape[0] == 0:
                            success = 'yes'
                            continue

                        daily_dfs.append(df.drop(['isPartial'], axis=1))
                        success = 'yes'
                    except BaseException:
                        time.sleep(SLEEPTIME)
            if len(daily_dfs) == 0:
                print("No GT data for {}".format(final_kw))
                continue
            daily_agg_df = pd.concat(daily_dfs, axis=0).groupby("date").mean()

            target_path = os.path.join(os.path.dirname(__file__), "data", "all_daily_trends", "daily_trends{}".format(sample))
            # check if output dir exists
            if not os.path.isdir(os.path.join(target_path)):
                os.mkdir(os.path.join(target_path))

            target_path = os.path.join(os.path.dirname(__file__), target_path, "{}.csv".format(final_kw))
            daily_agg_df.columns = [final_kw]
            daily_agg_df.to_csv(target_path)


if __name__ == '__main__':
    init = time.time()
    TARGET = sorted(set(TARGET))
    size = len(TARGET)
    df = get_daily_trend_from_word_list(TARGET)
    final = time.time() - init
    final = final / 60
    print("process duration = {:.2f} minutes".format(final))