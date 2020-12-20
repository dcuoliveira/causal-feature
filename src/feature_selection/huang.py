import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit
from tqdm import tqdm
from data_mani.utils import make_shifted_df
import pandas as pd
import numpy as np

def target_ret_to_directional_movements(x, y_name):
    x[y_name] = [1 if r > 0 else 0 for r in x[y_name]]
    return x


def univariate_granger_causality_test(x, y_name, x_name,
                                      max_lag, verbose, sig_level):
    accept_tag = [None]
    pval_dict = {}

    # H0: second column does not granger causes the first column
    test_result = sm.tsa.stattools.grangercausalitytests(x=x[[y_name] + [x_name]], maxlag=max_lag, verbose=verbose)
    for lag in test_result.keys():
        pval = test_result[lag][0]['ssr_ftest'][1]
        pval_dict[str(lag)] = pval
        if pval <= sig_level:
            accept_tag.append(x_name.replace(" ", "_") + '_' + str(lag))

    return accept_tag, pval_dict


def run_huang_methods(merged_df, target_name, words,
                      max_lag, verbose, sig_level):

    merged_df = target_ret_to_directional_movements(x=merged_df, y_name=target_name)

    univariate_granger_causality_list = []
    words_to_shift = []
    for w in tqdm(merged_df.columns, desc="run huang feature selection"):
        if w in words and w != target_name:
            accept_tag, pvals = univariate_granger_causality_test(x=merged_df, y_name=target_name, x_name=w,
                                                           max_lag=max_lag, verbose=verbose, sig_level=sig_level)
            univariate_granger_causality_list += accept_tag
            if len(accept_tag) > 1:
                words_to_shift.append(w)

    selected_words_list = [w for w in univariate_granger_causality_list if w is not None]

    merged_df, _ = make_shifted_df(df=merged_df, verbose=verbose,
                                              words=words_to_shift, max_lag=max_lag)

    if len(selected_words_list) != 0:
        logit_var_df = merged_df[[target_name] + selected_words_list].dropna()
        logit_model = Logit(endog=logit_var_df[[target_name]], exog=logit_var_df[selected_words_list]).fit()
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