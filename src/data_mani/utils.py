import pandas as pd
from tqdm import tqdm
import os


def get_ticker_name(path):
    """
    get ticker name from path

    :param path: path to market df
    :type path: str
    :return: ticker name
    :rtype: str
    """
    name = path.split("/")[-1].split(".")[0]
    name = name.replace(" ", "_")
    return name


def get_market_df(path):
    """
    Get ticker dataframe from path.

    We always drop the first two rows
    with the information

    'field, DAY_TO_DAY_TOT_RETURN_GROSS_DVDS
     date,'

    The total return is divided by 100.
    The return column named as
    "get_ticker_name(path)".

    :param path: path to market df
    :type path: str
    :return: market dataframe
    :rtype: pd.DataFrame
    """
    target_name = get_ticker_name(path)
    market = pd.read_csv(path)
    market = market.drop([0, 1], 0)  # drop first 2 lines
    market.columns = ["date", target_name]
    market.loc[:, "date"] = pd.to_datetime(market.date)
    market.loc[:, target_name] = market[target_name].astype("float") / 100
    return market.reset_index(drop=True)


def merge_data(df_list, freq='D'):
    """
    Merge all df's in the list 'df_list'.
    We assume that all df's are indexed
    by date.

    We resample all df's using the frequency
    "freq", and we concatenate them into
    a single dataframe.

    :param df_list: list of dataframes
    :type df_list: [pd.DataFrame]
    :param freq: frequency
    :type freq: str
    :return: merged dataframe
    :rtype: pd.DataFrame
    """
    list_out = []
    for df in df_list:
        df_loop = df.resample(freq).mean()
        list_out.append(df_loop)

    return pd.concat(list_out, axis=1)


def merge_market_and_gtrends(path,
                             init_train="2004-01-01",
                             final_train="2010-01-01"):
    """
    Merge market and google trends data.
    Market data is sliced using the
    training interval

    [init_train: final_train]


    :param path: path to market dataframe
    :type path: str
    :param init_train: initial timestamp for training
    :type init_train: str
    :param final_train: final timestamp for training
    :type final_train: str
    :return: merged dataframe
    :rtype: pd.DataFrame
    """

    # loading google trends data
    path_gt = os.path.join("data", "gtrends.csv")
    gtrends = pd.read_csv(path_gt)
    gtrends.loc[:, "date"] = pd.to_datetime(gtrends.date)
    gtrends = gtrends.set_index("date")

    # loading market data
    market = get_market_df(path)
    name = get_ticker_name(path)
    market = market.rename(columns={"ticker": "date",
                                    name: "target_return"})

    # using only the training sample
    market = market.set_index("date")
    market = market[init_train:final_train]

    # merging
    merged = merge_data([market, gtrends])
    merged = merged.dropna()

    return merged


def path_filter(paths, threshold=365):
    """
    filter each market data path by
    assessing the size of the associated
    merged dataframe.


    :param paths: list of paths to market data
    :type paths: [str]
    :param threshold: minimun number of days in
                      the merged dataframe
                      to not exclude a path
    :type threshold: int
    :return: list of filtered paths
    :rtype: [str]
    """
    new_paths = []
    for p in tqdm(paths, desc="filter"):
        df = pd.read_csv(p)
        if len(df.columns) > 1:
            df = merge_market_and_gtrends(p)
            if df.shape[0] >= threshold:
                new_paths.append(p)
    return new_paths