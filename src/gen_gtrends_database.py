import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from word_list.sanity_check import words

OUT_FOLDER = "data"
INPUT_FOLDER = os.path.join("data", "all_daily_trends")
N_SAMPLES = 5
WORDS = words
START_DATE = "2007-01-01"

def build_gtrends_database():

    gtrends = []
    for w in tqdm(WORDS, total=len(WORDS), desc="Generating gtrends database"):

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

            # compute log diff 1y diff 1w
            df[w] = np.log(df[w]).replace(np.inf, 0).replace(-np.inf, 0).diff(periods=252).diff(periods=5)

            # fix start date and dropna
            df = df.loc[START_DATE:].ffill().dropna()

            agg.append(df)
        # compute average across samples
        if len(agg) != 0:
            agg_df = pd.DataFrame(pd.concat(agg, axis=1).mean(axis=1), columns=[w])
        else:
            continue

        gtrends.append(agg_df)
    gtrends_df = pd.concat(gtrends, axis=1)
    gtrends_df.to_csv(os.path.join(OUT_FOLDER, "gtrends.csv"))


if __name__ == '__main__':
    build_gtrends_database()