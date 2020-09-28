import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acovf
from statsmodels.tsa.stattools import acf, ccf
from matplotlib.ticker import MaxNLocator


def plot_acf(x, lag_range, reverse=True, figsize=(12, 5),
             title_fontsize=15, xlabel_fontsize=16, ylabel_fontsize=16):
    """
    plot autocorrelation of series x
    :param x: series that we will perform the lag
    :type x: pd.Series
    :param lag_range: range of lag
    :type lag_range: int
    :param out_path: path to save figure
    :type out_path: str
    :param ccf: cross-correlation function
    :type ccf: function
    :param reverse: param to reverse lags
    :type reverse: boolean
    :param figsize: figure size
    :type figsize: tuple
    :param title_fontsize: title font size
    :type title_fontsize: int
    :param xlabel_fontsize: x axis label size
    :type xlabel_fontsize: int
    :param ylabel_fontsize: y axis label size
    :type ylabel_fontsize: int
    """

    title = "{}".format(x.name)
    lags = range(lag_range)
    ac = acf(x,fft=False,nlags=lag_range)
    sigma = 1 / np.sqrt(x.shape[0])
    fig, ax = plt.subplots(figsize=figsize)
    ax.vlines(lags, [0], ac)
    plt.plot(lags, [0] * len(lags), c="black", linewidth=1.0)
    plt.plot(lags, [2 * sigma] * len(lags), '-.', c="blue", linewidth=0.6)
    plt.plot(lags, [-2 * sigma] * len(lags), '-.', c="blue", linewidth=0.6)
    ax.set_xlabel('Lag', fontsize=xlabel_fontsize)
    ax.set_ylabel('autocorrelation', fontsize=ylabel_fontsize)
    fig.suptitle(title, fontsize=title_fontsize, fontweight='bold', y=0.93)
    


def plot_ccf(x, y, lag_range,
             figsize=(12, 5),
             title_fontsize=15, xlabel_fontsize=16, ylabel_fontsize=16):
    """
    plot cross-correlation between series x and y
    :param x: series that we leads y on the left
    :type x: pd.Series
    :param y: series that we leads x on the right
    :type y: pd.Series
    :param lag_range: range of lag
    :type lag_range: int
    :param figsize: figure size
    :type figsize: tuple
    :param title_fontsize: title font size
    :type title_fontsize: int
    :param xlabel_fontsize: x axis label size
    :type xlabel_fontsize: int
    :param ylabel_fontsize: y axis label size
    :type ylabel_fontsize: int
    """

    title = "{} & {}".format(x.name, y.name)
    lags = range(-lag_range, lag_range + 1)
    left = ccf(y, x)[:lag_range + 1]
    right = ccf(x, y)[:lag_range]

    left = left[1:][::-1]
    cc = np.concatenate([left, right])

    sigma = 1 / np.sqrt(x.shape[0])
    fig, ax = plt.subplots(figsize=figsize)
    ax.vlines(lags, [0], cc)
    plt.plot(lags, [0] * len(lags), c="black", linewidth=1.0)
    plt.plot(lags, [2 * sigma] * len(lags), '-.', c="blue", linewidth=0.6)
    plt.plot(lags, [-2 * sigma] * len(lags), '-.', c="blue", linewidth=0.6)
    ax.set_xlabel('Lag', fontsize=xlabel_fontsize)
    ax.set_ylabel('cross-correlation', fontsize=ylabel_fontsize)    
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.suptitle(title, fontsize=title_fontsize, fontweight='bold', y=0.93)

        
def get_lead_matrix(lead_series, fixed_series, lag_range):
    
    """
    get ccf vector of size 'max_lag' for each ts in 'lead_series' in
    relation with 'fixed_series'. All the ccf results
    are arranged in matrix format.

    :param lead_series: list of series to be lagged. 
                        All series are indexed by time.
    :type lead_series: [pd.Series]
    :param fixed_series: list of series indexed by time.
    :type fixed_series: pd.Series
    :param lag_range: range of lag
    :type lag_range: int
    :return: matrix of ccf information
    :rtype: pd.DataFrame
    """

    ccf_rows = []
    for ts in lead_series:
        merged = pd.merge_asof(ts, fixed_series,
                               left_index=True, right_index=True)

        lagged_ts = merged[ts.name]
        fixed_ts = merged[fixed_series.name]
        row = ccf(fixed_ts, lagged_ts)[:lag_range +1]
        ccf_rows.append(row)



    ccf_matrix = np.array(ccf_rows)
    ccf_matrix = pd.DataFrame(ccf_matrix,
                              columns=["lag_{}".format(i) for i in range(lag_range +1)],
                              index=[ts.name for ts in lead_series]) 
    return ccf_matrix