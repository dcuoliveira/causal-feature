import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from word_list.analysis import words
import warnings

OUT_FOLDER = "data"
INPUT_FOLDER = os.path.join(os.path.dirname(__file__), "data", "all_daily_trends")
N_SAMPLES = 5
WORDS = words
START_DATE = "2010-01-01"

warnings.filterwarnings('ignore')

def build_gtrends_database_daily():

    gtrends = []
    for w in tqdm(WORDS, total=len(WORDS), desc="Generating daily gtrends database"):

        agg = []
        for sample in range(N_SAMPLES):
            try:
                df = pd.read_csv(os.path.join(INPUT_FOLDER, "daily_trends" + str(sample), w + ".csv"))
            except:
                continue

            # fix date format
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)

            # resample frequency
            df = df.resample("B").last()

            # compute log of values to adjust variance and forward fill values
            df["{} log".format(w)] = np.log(df[w]).replace(np.inf, np.nan).replace(-np.inf, np.nan).ffill()

            # compute yoy difference
            df["{} log yoy".format(w)] = df["{} log".format(w)].diff(periods=252)

            # fix start date and dropna
            df = df.loc[START_DATE:]

            agg.append(df[["{} log yoy".format(w)]].rename(columns={"{} log yoy".format(w): w}))
        # compute average across samples
        if len(agg) != 0:
            agg_df = pd.DataFrame(pd.concat(agg, axis=1).mean(axis=1), columns=[w])
        else:
            continue

        gtrends.append(agg_df)
    gtrends_df = pd.concat(gtrends, axis=1)
    gtrends_df.to_csv(os.path.join(os.path.dirname(__file__), OUT_FOLDER, "gtrends_daily.csv"))

def build_gtrends_database_weekly():

    gtrends = []
    for w in tqdm(WORDS, total=len(WORDS), desc="Generating weekly gtrends database"):

        agg = []
        for sample in range(N_SAMPLES):
            try:
                df = pd.read_csv(os.path.join(INPUT_FOLDER, "daily_trends" + str(sample), w + ".csv"))
            except:
                continue

            # fix date format
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)

            # resample frequency
            df = df.resample("W-FRI").last()

            # compute log of values to adjust variance and forward fill values
            df["{} log".format(w)] = np.log(df[w]).replace(np.inf, np.nan).replace(-np.inf, np.nan).ffill()

            # compute yoy difference
            df["{} log yoy".format(w)] = df["{} log".format(w)].diff(periods=54)

            # fix start date and dropna
            df = df.loc[START_DATE:]

            agg.append(df[["{} log yoy".format(w)]].rename(columns={"{} log yoy".format(w): w}))
        # compute average across samples
        if len(agg) != 0:
            agg_df = pd.DataFrame(pd.concat(agg, axis=1).mean(axis=1), columns=[w])
        else:
            continue

        gtrends.append(agg_df)
    gtrends_df = pd.concat(gtrends, axis=1)
    gtrends_df.to_csv(os.path.join(os.path.dirname(__file__), OUT_FOLDER, "gtrends_weekly.csv"))


if __name__ == '__main__':
    build_gtrends_database_daily()
    build_gtrends_database_weekly()