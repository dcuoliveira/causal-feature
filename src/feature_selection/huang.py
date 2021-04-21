import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit
from tqdm import tqdm
import pandas as pd
import numpy as np

try:
    from data_mani.utils import make_shifted_df
    from data_mani.utils import correlation_filter
    from data_mani.utils import check_constant_series
    from data_mani.utils import target_ret_to_directional_movements
except ModuleNotFoundError:
    from src.data_mani.utils import make_shifted_df
    from src.data_mani.utils import correlation_filter
    from src.data_mani.utils import check_constant_series
    from src.data_mani.utils import target_ret_to_directional_movements


def univariate_granger_causality_test(x, y_name, x_name,
                                      max_lag, verbose, sig_level):
    """
    test univariate granger causality for each series in x_name
    and for each lag in 1:max_lag

    :param x: data
    :type x: dataframe
    :param y_name: name of the dependent variable to use in the test
    :type y_name: str
    :param x_name: list of independent variables to test
    :type x_name: list
    :param max_lag: number of max lags to test
    :type max_lag: int
    :param verbose:
    :type verbose: boolean
    :param sig_level: significance level to use as threshold of the test
    :type sig_level: float
    :return: exogenous variable names that passed the test of granger causality
    and a dictionary of each of the variables pvalues
    :rtype: list and dict
    """
    accept_tag = []
    pval_tag = []
    pval_dict = {}

    # H0: second column does not granger causes the first column
    test_result = sm.tsa.stattools.grangercausalitytests(x=x[[y_name] + [x_name]], maxlag=max_lag, verbose=verbose)
    for lag in test_result.keys():
        pval = test_result[lag][0]['ssr_ftest'][1]
        pval_dict[str(lag)] = pval
        if pval <= sig_level:
            accept_tag += [x_name.replace(" ", "_") + '_' + str(lag)]
            pval_tag += [pval]

    return accept_tag, pval_tag

def run_granger_causality(merged_df, target_name, words,
                          max_lag, sig_level, correl_threshold,
                          constant_threshold, verbose):
    """
    
    perform a simplified version of the huang feature selection method,
    that is, it performs univariate granger causality and logistic regression

    :param merged_df: data
    :type merged_df: dataframe
    :param target_name: name of the dependent variable
    :type target_name: str
    :param words: list of words to test as exogenous variables
    :type words: list
    :param max_lag: number of max lags to test granger
    :type max_lag: int
    :param verbose:
    :type verbose: boolean
    :param sig_level: significance level to use as threshold of the test
    :type sig_level: float
    :param correl_threshold: correlation threshold to apply the filter (excluded
    high correlated series)
    :type correl_threshold: float
    :param constant_threshold: constant threshold to apply the filter (exclude series that
    have more than x% of constant values)
    :type constant_threshold: float
    :return: dataframe with the words selectect using huangs method and the
    respective pvalues
    :rtype: dataframe
    """
    
    univariate_granger_causality_list = []
    pvals_list = []
    words_to_shift = []
    for w in tqdm(merged_df.columns, disable=not verbose, desc="run huang feature selection", ):
        if w in words and w != target_name:
            tag = check_constant_series(df=merged_df,
                                        target = w,
                                        threshold=constant_threshold)
            if not tag:
                accept_tag, pvals = univariate_granger_causality_test(x=merged_df, y_name=target_name, x_name=w,
                                                                max_lag=max_lag, verbose=verbose, sig_level=sig_level)
                univariate_granger_causality_list += accept_tag
                pvals_list += pvals
                if len(accept_tag) > 1:
                    words_to_shift.append(w)
            else:
                continue

    selected_words_list = [w for w in univariate_granger_causality_list if w is not None]
    out_df = pd.DataFrame(data={'feature': selected_words_list, 'feature_score': pvals_list})
    
    return out_df



def run_huang_methods(merged_df, target_name, words,
                      max_lag, verbose, sig_level,
                      correl_threshold, constant_threshold,
                      asset_name=None):
    """
    perform huang feature selection procedure, that is, univariate granger
    causality and logistic regression

    :param merged_df: data
    :type merged_df: dataframe
    :param target_name: name of the dependent variable
    :type target_name: str
    :param words: list of words to test as exogenous variables
    :type words: list
    :param max_lag: number of max lags to test granger
    :type max_lag: int
    :param verbose:
    :type verbose: boolean
    :param sig_level: significance level to use as threshold of the test
    :type sig_level: float
    :param correl_threshold: correlation threshold to apply the filter (excluded
    high correlated series)
    :type correl_threshold: float
    :return: dataframe with the words selectect using huangs method and the
    respective pvalues
    :rtype: dataframe
    """

    univariate_granger_causality_list = []
    words_to_shift = []
    for w in tqdm(merged_df.columns, disable=not verbose, desc="run huang feature selection", ):
        if w in words and w != target_name:
            tag = check_constant_series(df=merged_df,
                                        target = w,
                                        threshold=constant_threshold)
            if not tag:
                accept_tag, _ = univariate_granger_causality_test(x=merged_df, y_name=target_name, x_name=w,
                                                                max_lag=max_lag, verbose=verbose, sig_level=sig_level)
                univariate_granger_causality_list += accept_tag
                if len(accept_tag) >= 1:
                    words_to_shift.append(w)
            else:
                continue

    selected_words_list = [w for w in univariate_granger_causality_list if w is not None]

    merged_df, _ = make_shifted_df(df=merged_df, verbose=verbose,
                                                words=words_to_shift, max_lag=max_lag)

    if len(selected_words_list) != 0:
        logit_var_df = merged_df[[target_name] + selected_words_list].dropna()
        try:
            logit_model = Logit(endog=logit_var_df[[target_name]], exog=logit_var_df[selected_words_list]).fit()
        except:
            return None
        final_selected_words = list(logit_model.pvalues[logit_model.pvalues <= sig_level].index)
        words_not_selected_to_add = pd.DataFrame(list(set(merged_df.columns) - set(final_selected_words)))
        words_not_selected_to_add.columns = ['feature']
        words_not_selected_to_add['feature_score'] = np.nan
        logit_granger_result = pd.DataFrame(logit_model.pvalues[logit_model.pvalues <= sig_level])
        logit_granger_result = logit_granger_result.reset_index()
        logit_granger_result.columns = ['feature', 'feature_score']

        logit_granger_result = pd.concat([logit_granger_result, words_not_selected_to_add], axis=0)
    else:
        words_not_selected_to_add = pd.DataFrame(list(set(merged_df) - set(selected_words_list)))
        words_not_selected_to_add.columns = ['feature']
        words_not_selected_to_add['feature_score'] = np.nan
        logit_granger_result = words_not_selected_to_add

    # TODO - Acrescentar selecao pelo metodo de Mallows C_p

    return logit_granger_result