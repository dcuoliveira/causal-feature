import os
import time
import pandas as pd
from copy import copy
from glob import glob
from tqdm import tqdm
from datetime import date, datetime, timedelta
from pytrends.request import TrendReq
from word_list.basic import base
from sklearn import preprocessing

# global parameters
REFWORD = "google"
TARGET = base
SLEEPTIME = 20
INIT_DATE = "2004-01-01"
FINAL_DATE = "2020-07-28"
TIMEZONE_OFFSET = 360
HOST_LANGUAGE = 'en-US'
COUNTRY_ABBREVIATION = 'US'
FREQ_TO_DOWNLOAD = 'daily'


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

    Each moment a time series is completed, it is saved in "data/daily"

    :param kw_list: word list
    :type kw_list: [str]
    :param reference_word: reference_word
    :type reference_word: str
    """
    dfList = []
    for kw in tqdm(kw_list, desc='word'):
        daily_dfs = []
        for timeframe in tqdm(
                INTERVALS, desc="'{}': time intervals".format(kw)):
            time.sleep(SLEEPTIME)
            trends = TrendReq(hl=HOST_LANGUAGE, tz=TIMEZONE_OFFSET)
            trends.build_payload(kw_list=[kw, REFWORD],
                                 geo=COUNTRY_ABBREVIATION,
                                 timeframe=timeframe)
            df = trends.interest_over_time()
            daily_dfs.append(df)
        daily_ts = pd.concat(daily_dfs).reset_index().groupby("date").mean()
        ts_path = os.path.join("data", "daily_trend", "{}.csv".format(kw))
        daily_ts.to_csv(ts_path)


def get_scaled_gtrends(kw_list, dt):
    target_path = os.path.join('data', 'daily_trend_normalized')
    if os.path.exists(target_path) == False:
        os.makedirs(target_path)

    gtrends_df = []
    timeframe_df = {}
    scaling_df = {}
    min_max_scaler = preprocessing.MinMaxScaler()
    for kw in kw_list:
        print(kw)
        scaling_series = []
        timeframe_series = []

        kw = [kw, REFWORD]

        pytrend = TrendReq()

        today = pd.to_datetime(FINAL_DATE)
        old_date = today

        new_date = today - dt

        timeframe = new_date.strftime(time_fmt) + ' ' + old_date.strftime(time_fmt)
        timeframe_series.append(timeframe)
        print(timeframe)
        pytrend.build_payload(kw_list=kw, timeframe=timeframe)
        interest_over_time_df = pytrend.interest_over_time()

        while new_date > pd.to_datetime(INIT_DATE):

            ### Save the new date from the previous iteration.
            # Overlap == 1 would mean that we start where we
            # stopped on the iteration before, which gives us
            # indeed overlap == 1.
            if daily == True:
                old_date = new_date + timedelta(days=overlap - 1)
            else:
                old_date = new_date + timedelta(hours=overlap - 1)

            ### Update the new date to take a step into the past
            # Since the timeframe that we can apply for daily data
            # is limited, we use step = maxstep - overlap instead of
            # maxstep.
            new_date = new_date - dt  # timedelta(hours=step)
            # If we went past our start_date, use it instead
            if new_date < pd.to_datetime(INIT_DATE):
                new_date = pd.to_datetime(INIT_DATE)

            # New timeframe
            timeframe = new_date.strftime(time_fmt) + ' ' + old_date.strftime(time_fmt)
            timeframe_series.append(timeframe)
            print(timeframe)

            # Download data
            pytrend.build_payload(kw_list=kw, timeframe=timeframe)
            temp_df = pytrend.interest_over_time()
            if (temp_df.empty):
                raise ValueError(
                    'Google sent back an empty dataframe. Possibly there were no searches at all during the this period! Set start_date to a later date.')
            # Renormalize the dataset and drop last line
            beg = new_date
            if daily == True:
                end = old_date - timedelta(days=1)
            else:
                end = old_date - timedelta(hours=1)

            # Since we might encounter zeros, we loop over the
            # overlap until we find a non-zero element
            for t in range(1, overlap + 1):
                if temp_df[kw[0]].iloc[-t] != 0:
                    scaling = float(interest_over_time_df[kw[0]].iloc[t - 1]) / temp_df[kw[0]].iloc[-t]
                    print(scaling)
                    break
                elif t == overlap:
                    print('Did not find non-zero overlap, set scaling to zero! Increase Overlap!')
                    scaling = 0
            # Apply scaling
            scaling_series.append(scaling)
            temp_df.loc[beg:end, kw[0]] = temp_df.loc[beg:end, kw[0]] * scaling
            interest_over_time_df = pd.concat([temp_df[:-overlap], interest_over_time_df])
            time.sleep(SLEEPTIME)
        x_scaled = min_max_scaler.fit_transform(interest_over_time_df[[kw[0]]])
        scaled_df = pd.DataFrame(x_scaled)
        scaled_df.index = interest_over_time_df.index
        scaled_df.columns = [kw[0]]
        ts_path = os.path.join("data", "daily_trend_normalized", "{}.csv".format(kw[0]))
        scaled_df.to_csv(ts_path)

        gtrends_df.append(scaled_df)
        scaling_df[kw[0]] = scaling_series
        timeframe_df[kw[0]] = timeframe_series

    return gtrends_df, pd.concat([pd.DataFrame.from_dict(scaling_df), pd.DataFrame.from_dict(timeframe_df)], axis=1)


if __name__ == '__main__':
    # main problem of this script
    # https://stackoverflow.com/questions/50571317/pytrends-the-request-failed-google-returned-a-response-with-code-429

    # possible solution
    # pip3 install --upgrade --user
    # git+https://github.com/GeneralMills/pytrends

    # Run code in different computers in different times to get all words
    type_coleta = 'normalized'

    if type_coleta != 'normalized':
        already_collected = glob(os.path.join("data", "daily_trend", "*.csv"))
        try:
            already_collected = [i.split("/")[2] for i in already_collected]
        except:
            already_collected = [i.split("\\")[2] for i in already_collected]
        already_collected = [i.split(".")[0] for i in already_collected]
        NEW_TARGET = [i for i in TARGET if i not in already_collected]
        init = time.time()
        NEW_TARGET = sorted(set(NEW_TARGET))

        size = len(NEW_TARGET)
        df = get_daily_trend_from_word_list(NEW_TARGET)
        final = time.time() - init
        final = final / 60
        print("process duration = {:.2f} minutes".format(final))

    else:
        already_collected = glob(os.path.join("data", "daily_trend_normalized", "*.csv"))
        try:
            already_collected = [i.split("/")[2] for i in already_collected]
        except:
            already_collected = [i.split("\\")[2] for i in already_collected]
        already_collected = [i.split(".")[0] for i in already_collected]
        NEW_TARGET = [i for i in TARGET if i not in already_collected]
        init = time.time()
        NEW_TARGET = sorted(set(NEW_TARGET))

        kw_list = NEW_TARGET
        daily = False
        hourly = False

        if FREQ_TO_DOWNLOAD == 'hourly':
            hourly = True
        elif FREQ_TO_DOWNLOAD == 'daily':
            daily = True
        else:
            print("Frequency not registered")
            exit(-1)

        if daily:
            maxstep = 269
            overlap = 100
            step = maxstep - overlap + 1
            dt = timedelta(days=step)
            time_fmt = '%Y-%m-%d'
        elif hourly:
            overlap = 50
            step = 168
            dt = timedelta(hours=step)
            time_fmt = '%Y-%m-%dT%H'
        gtrends_df, scaling_df = get_scaled_gtrends(kw_list, dt)
