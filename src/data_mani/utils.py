import pandas as pd


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
