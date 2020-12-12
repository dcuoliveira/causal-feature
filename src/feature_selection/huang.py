import statsmodels.api as sm
from tqdm import tqdm
import numpy as np


def target_ret_to_directional_movements(x, y_name):
    x[y_name] = [1 if r > 0 else 0 for r in x[y_name]]
    return x


def univariate_granger_causality_test(x, y_name, x_name, max_lag, verbose, sig_level):

    accept_tag = [None]

    # H0: second column does not granger causes the first column
    test_result = sm.tsa.stattools.grangercausalitytests(x=x[[y_name] + [x_name]], maxlag=max_lag, verbose=verbose)
    for lag in test_result.keys():
        pval = test_result[lag][0]['ssr_ftest'][1]
        if pval <= sig_level:
            accept_tag.append(x_name.replace(" ", "_") + '_' + str(lag))

    return accept_tag


def moving_window_granger_causality_test(x, y_name, x_name, max_lag, verbose, sig_level, window_size):
    accept_tag_dict = {}
    for step in range(0, x.shape[0]-1, 5):
        x_mw = x.iloc[(0+step):(window_size+step)]
        accept_tag = univariate_granger_causality_test(x=x_mw, y_name=y_name, x_name=x_name,
                                                       max_lag=max_lag, verbose=verbose, sig_level=sig_level)
        accept_tag_dict[step] = accept_tag
    return accept_tag_dict


def run_huang_methods(merged_df, target_name, words, max_lag, verbose, sig_level, window_size):

    merged_df = target_ret_to_directional_movements(x=merged_df, y_name=target_name)

    selected_words_list = []
    univariate_granger_causality_list = []
    moving_window_granger_causality_list = []
    for w in tqdm(merged_df.columns, desc="run huang feature selection"):
        if w in words and w != target_name:
            accept_tag = univariate_granger_causality_test(x=merged_df, y_name=target_name, x_name=w,
                                                           max_lag=max_lag, verbose=verbose, sig_level=sig_level)
            univariate_granger_causality_list += accept_tag

            accept_tag = moving_window_granger_causality_test(x=merged_df, y_name=target_name, x_name=w,
                                                         max_lag=max_lag, verbose=verbose, sig_level=sig_level,
                                                         window_size=window_size)
            moving_window_granger_causality_list += accept_tag


            # intersect between methods
            selected_words_list = univariate_granger_causality_list

    selected_words_list = [w for w in selected_words_list if w is not None]