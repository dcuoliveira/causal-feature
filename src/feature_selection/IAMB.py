import numpy as np
try:
    from data_mani.utils import *
except ModuleNotFoundError:
    from src.data_mani.utils import *


def IAMB(data,
         target,
         alpha,
         is_discrete=True):
    """
    Implementation of the IAMB algorithm originally proposed by Tsamardinos and Aliferis (2003) and
    implemented by Kui Yu (https://github.com/kuiy/pyCausalFS).

    :param data: data
    :type data: dataframe
    :param target: name of the dependent variable
    :type target: str
    :param alpha: alpha (pvalue) threshold to reject the null hypothesis of independence
    :type alpha: float
    :param is_discrete: states if the target (exogenous) variable is continuos (if False) or discrete
    :type is_discrete: boolean
    """
    number, kVar = np.shape(data)
    CMB = []
    ci_number = 0
    # forward circulate phase
    circulate_Flag = True
    while circulate_Flag:
        # if not change, forward phase of IAMB is finished.
        circulate_Flag = False
        # tem_dep pre-set infinite negative.
        temp_dep = -(float)("inf")
        y = None
        variables = [i for i in range(kVar) if i != target and i not in CMB]

        for x in variables:                  
            ci_number += 1
            pval, dep = cond_indep_test(data, target, x, CMB, is_discrete)
            # print("target is:",target,",x is: ", x," CMB is: ", CMB," ,pval is: ",pval," ,dep is: ", dep)

            # chose maxsize of f(X:T|CMB)
            if pval <= alpha:
                if dep > temp_dep:
                    temp_dep = dep
                    y = x

        # if not condition independence the node,appended to CMB
        if y is not None:
            # print('appended is :'+str(y))
            CMB.append(y)
            circulate_Flag = True

    # backward circulate phase
    CMB_temp = CMB.copy()
    for x in CMB_temp:
        # exclude variable which need test p-value
        condition_Variables = [i for i in CMB if i != x]
        ci_number += 1
        pval, dep = cond_indep_test(data, target, x, condition_Variables, is_discrete)
        # print("target is:", target, ",x is: ", x, " condition_Variables is: ", condition_Variables, " ,pval is: ", pval, " ,dep is: ", dep)
        if pval > alpha:
            # print("removed variables is: " + str(x))
            CMB.remove(x)

    return list(set(CMB)), ci_number


def run_IAMB(merged_df,
             target_name,
             words,
             max_lag,
             verbose,
             sig_level,
             is_discrete,
             constant_threshold):
    """
    Perform IAMB algorithm originally proposed by Tsamardinos and Aliferis (2003) and
    implemented by Kui Yu (https://github.com/kuiy/pyCausalFS).

    For more information, see the work of Yu et al. (2019) available at https://arxiv.org/abs/1911.07147.

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
    :param is_discrete: states if the target (exogenous) variable is continuos (if False) or discrete
    :type is_discrete: boolean
    :param constant_threshold: constant threshold to apply the filter
    :type constant_threshold: float
    """

    list_df = []
    for w in [target_name] + words:
        list_df.append(merged_df[[w]])
    merged_df = pd.concat(list_df, axis=1)

    merged_df, _ = make_shifted_df(df=merged_df,
                                   verbose=verbose,
                                   words=words,
                                   max_lag=max_lag)

    for w in words:
        w_idx = list(merged_df.columns).index(w)
        del merged_df[merged_df.columns[w_idx]]

    not_constant_df = []
    for col in merged_df.columns:
        tag = check_constant_series(df=merged_df,
                                    target=col,
                                    threshold=constant_threshold)
        if not tag or col == target_name:
            not_constant_df.append(merged_df[[col]])
    merged_df = pd.concat(not_constant_df, axis=1)
    del not_constant_df

    target_name_index = list(merged_df.columns).index(target_name)
    MBs, ci_number = IAMB(data=merged_df.dropna(),
                          target=target_name_index,
                          alpha=sig_level,
                          is_discrete=is_discrete)

    if len(MBs) == 0:
        features = list(merged_df.columns)
        features.remove(target_name)
        MBs_df = pd.DataFrame(data={'feature': features,
                                    'feature_score': np.nan})
    else:
        features = []
        for mb in MBs:
            features.append(merged_df.columns[mb])
        MBs_df = pd.DataFrame(data={'feature': features,
                                    'feature_score': np.nan})

    return MBs_df
