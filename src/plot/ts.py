import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt


def get_sample_acvf(x, h):
    """
    x: series
    h: shift param
    return: autocovariance estimator
    """
    n = x.shape[0]
    shift_x = x.shift(h)
    mean = x.mean()
    result = (x - mean) * (shift_x - mean)
    return result.sum() / n


def autocovariance_f(x, nlags):
    """
    x: series
    nlags: range of lags param
    return: array of autocovariance estimators
    """
    results = np.array([get_sample_acvf(x, h) for h in range(nlags)])
    return results


def autocorrelation_f(x, nlags):
    """
    x: series
    nlags: range of lags param
    return: array of autocorrelation estimators
    """
    gammas = autocovariance_f(x, nlags)
    gamma_0 = get_sample_acvf(x, 0)
    return gammas / gamma_0


def get_sample_ccvf(x, y, h):
    """
    x: series
    y: series
    h: shift param
    return: cross-covariance estimator
    """
    n = x.shape[0]
    shift_x = x.shift(h)
    mean_x = x.mean()
    mean_y = y.mean()
    result = (shift_x - mean_x) * (y - mean_y)
    return result.sum() / n


def crosscorrelation_f(x, y, nlags):
    """
    x: series
    y: series
    nlags: range of lags param
    return: array of cross-correlation estimators
    """
    results = np.array([get_sample_ccvf(x, y, h) for h in range(nlags)])
    gamma_x_0 = get_sample_acvf(x, 0)
    gamma_y_0 = get_sample_acvf(y, 0)
    denominator = np.sqrt(gamma_x_0 * gamma_y_0)
    return results / denominator


def plot_acf(x, lag_range, out_path,
             acf_function=autocorrelation_f, reverse=True, figsize=(12, 5),
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
    ac = acf_function(x, nlags=lag_range)
    sigma = 1 / np.sqrt(x.shape[0])
    fig, ax = plt.subplots(figsize=figsize)
    ax.vlines(lags, [0], ac)
    plt.plot(lags, [0] * len(lags), c="black", linewidth=1.0)
    plt.plot(lags, [2 * sigma] * len(lags), '-.', c="blue", linewidth=0.6)
    plt.plot(lags, [-2 * sigma] * len(lags), '-.', c="blue", linewidth=0.6)
    ax.set_xlabel('Lag', fontsize=xlabel_fontsize)
    ax.set_ylabel('autocorrelation', fontsize=ylabel_fontsize)
    fig.suptitle(title, fontsize=title_fontsize, fontweight='bold', y=0.93)
    if out_path is not None:
        plt.savefig(out_path)


def plot_ccf(x, y, lag_range, out_path,
             ccf=crosscorrelation_f, reverse=True, figsize=(12, 5),
             title_fontsize=15, xlabel_fontsize=16, ylabel_fontsize=16):
    """
    plot cross-correlation between series x and y
    :param x: series that we will perform the lag
    :type x: pd.Series
    :param y: series
    :type y: pd.Series
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

    title = "{} & {}".format(x.name, y.name)
    lags = range(-lag_range, lag_range + 1)
    left = ccf(x, y, lag_range + 1)
    rigt = ccf(y, x, lag_range)
    left = left[1:][::-1]
    cc = np.concatenate([left, rigt])

    sigma = 1 / np.sqrt(x.shape[0])
    fig, ax = plt.subplots(figsize=figsize)
    ax.vlines(lags, [0], cc)
    plt.plot(lags, [0] * len(lags), c="black", linewidth=1.0)
    plt.plot(lags, [2 * sigma] * len(lags), '-.', c="blue", linewidth=0.6)
    plt.plot(lags, [-2 * sigma] * len(lags), '-.', c="blue", linewidth=0.6)
    ax.set_xlabel('Lag', fontsize=xlabel_fontsize)
    ax.set_ylabel('cross-correlation', fontsize=ylabel_fontsize)
    fig.suptitle(title, fontsize=title_fontsize, fontweight='bold', y=0.93)
    if out_path is not None:
        plt.savefig(out_path)
