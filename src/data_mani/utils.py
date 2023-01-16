from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
import os
import numpy as np
from scipy.stats import chi2, norm
import itertools
from itertools import combinations, chain
from statsmodels.discrete.discrete_model import Logit
from statsmodels.regression.linear_model import OLS

def target_ret_to_directional_movements(x, y_name):
    """
    discretize a series of returns into up (1) and down (0) movements

    :param x: target data
    :type x: data.frame
    :param y_name: target return to discretize
    :type words: str
    :return: full dataframe with the y_name variable discretized
    :rtype: dataframe
    """
    # alternativa
    # x.loc[:, y_name] = (x[y_name]>0).astype(int).values
    
    x[y_name] = [1 if r > 0 else 0 for r in x[y_name]]

    return x


def correlation_filter(data, threshold):
    """
    filter columns that has correlation higher than the threshold

    :param data: data to filter
    :type data: dataframe
    :param threshold: correlation threshold to apply the filter
    :type threshold: float
    :return: filtered data
    """
    col_corr = set()
    corr_matrix = data.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) or (corr_matrix.iloc[i, j] <= -threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
                if colname in data.columns:
                    del data[colname]

    return data


def make_shifted_df(df, words, verbose, max_lag):
    """
    make shift to specified words in the df

    :param df: data
    :type df: dataframe
    :param words: selected words to make shift
    :type words: list
    :param verbose:
    :type verbose: boolean
    :return: shifted dataframe and dicionary of words used to shift
    :rtype: dataframe and dict
    """
    feature_dict = {}

    for word in tqdm(words, disable=not verbose, desc="add shift"):
        new_features = []
        for shift in range(1, max_lag + 1):
            new_feature = word.replace(" ", "_") + "_{}".format(shift)
            df.loc[:, new_feature] = df[word].shift(shift)
            new_features.append(new_feature)
        feature_dict[word] = new_features

    return df, feature_dict


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
                             test_size,
                             is_discrete=False,
                             path_gt_list=[os.path.dirname(os.path.dirname(__file__)), "data", "gtrends.csv"]):
    """
    Merge market and google trends data.
    Market data is sliced using the
    parameter "test_size"

    :param path: path to market dataframe
    :type path: str
    :param test_size: value to split the data
                      into training and testing
    :type test_size: float in [0,1] or int
    :param path_gt_list: list of str to create gt path
    :type path_gt_list: [str]
    :return: merged dataframe train and tes
    :rtype: (pd.DataFrame,pd.DataFrame)
    """
    
    # loading google trends data
    path_gt = os.path.join(*path_gt_list)
    gtrends = pd.read_csv(path_gt)
    gtrends.loc[:, "date"] = pd.to_datetime(gtrends.date)
    gtrends = gtrends.set_index("date").sort_index()

    # loading market data
    market = get_market_df(path)
    name = get_ticker_name(path)
    market = market.rename(columns={"ticker": "date",
                                    name: "target_return"})
    market = market.set_index("date").sort_index()

    # merging
    merged = merge_data([market, gtrends])
    merged = merged.dropna()
    if is_discrete:
        merged.loc[:, 'target_return'] = target_ret_to_directional_movements(x=merged,
                                                                             y_name='target_return')

    # if the merged data is null or has only one element
    # then both train and test are null
    if merged.shape[0] > 1:
        train, test = train_test_split(merged,
                                       test_size=test_size,
                                       shuffle=False)
        last_day_train = train.sort_index().index[-1]
        first_day_test = test.sort_index().index[0]
        assert last_day_train < first_day_test, "temporal ordering error"
    else:
        train, test = pd.DataFrame(), pd.DataFrame()
    return train, test


def path_filter(paths,
                threshold,
                verbose=True,
                path_gt_list=[os.path.dirname(os.path.dirname(__file__)), "data", "gtrends.csv"]):
    """
    filter each market data path by
    assessing the size of the associated
    merged dataframe.

    Remember,
    252 = business days in a year


    :param paths: list of paths to market data
    :type paths: [str]
    :param threshold: minimun number of days in
                      the merged dataframe
                      to not exclude a path
    :type threshold: int
    :param verbose: param to print iteration status
    :type verbose: bool
    :param path_gt_list: list of str to create gt path
    :type path_gt_list: [str]
    :return: list of filtered paths
    :rtype: [str]
    """
    new_paths = []
    for p in tqdm(paths, disable=not verbose, desc="filter"):
        df = pd.read_csv(p)
        if len(df.columns) > 1:
            train, test = merge_market_and_gtrends(p,
                                                   test_size=1,
                                                   path_gt_list=path_gt_list)
            df = pd.concat([train, test])
            if df.shape[0] >= threshold:
                new_paths.append(p)
    return new_paths


def check_constant_series(df,
                          target,
                          threshold):
    """
    Return a tag (True/False) if the target series is constant
    for more than the threshold

    :param df: data
    :type df: dataframe
    :param target: name of the column to check
    :type df: [str]
    :param threshold: percentual to check
    :type threshold: float
    :return: True/False boolean
    :rtype: boolean
    """
    target_df = df[[target]].pct_change(1)
    tot_point =df.shape[0]
    constant_points = np.sum(pd.isna(target_df)).iloc[0]
    perc = constant_points / tot_point

    if perc >= threshold:
        tag = True
    else:
        tag = False

    return tag


def ns(data,
       MB):
    qi = []
    # if MBs == []:
    #     qi.append(1)
    for i in MB:
        try:
            i = str(i)
            q_temp = len(np.unique(data[i]))
        except:
            i = int(i)
            q_temp = len(np.unique(data[data.columns[i]]))
        qi.append(q_temp)
    return qi


def get_partial_matrix(S,
                       X,
                       Y):
    S = S[X, :]
    S = S[:, Y]
    return S


def partial_corr_coef(S,
                      i,
                      j,
                      Y):
    S = np.matrix(S)
    X = [i, j]
    inv_syy = np.linalg.inv(get_partial_matrix(S, Y, Y))
    i2 = 0
    j2 = 1
    S2 = get_partial_matrix(S, X, X) - get_partial_matrix(S, X, Y) * inv_syy * get_partial_matrix(S, Y, X)
    c = S2[i2, j2]
    r = c / np.sqrt((S2[i2, i2] * S2[j2, j2]))

    return r


def subsets(nbrs, k):
    return set(combinations(nbrs, k))


def getMinDep(data, target, x, CPC, alpha, is_discrete):

    """this function is to chose min dep(association) about Target,x|(subsets of CPC)"""

    ci_number = 0
    dep_min = float("inf")
    max_k = 3

    if len(CPC) > max_k:
        k_length = max_k
    else:
        k_length = len(CPC)
        
    for i in range(k_length+1):
        SS = subsets(CPC, i)
        for S in SS:
            ci_number += 1
            pval, dep = cond_indep_test(data, target, x, S, is_discrete)
            # this judge about target and x whether or not is condition independence ,if true,dep must be zero,
            # and end operating of function of getMinDep
            if pval > alpha:
                return 0, S, ci_number
            if dep_min > dep:
                dep_min = dep
    return dep_min, tuple(), ci_number


def cond_indep_fisher_z(data,
                        var1,
                        var2,
                        cond=[],
                        alpha=0.05):

    """
    COND_INDEP_FISHER_Z Test if var1 indep var2 given cond using Fisher's Z test
    CI = cond_indep_fisher_z(X, Y, S, C, N, alpha)
    C is the covariance (or correlation) matrix
    N is the sample size
    alpha is the significance level (default: 0.05)
    transfromed from matlab
    See p133 of T. Anderson, "An Intro. to Multivariate Statistical Analysis", 1984

    Parameters
    ----------
    data: pandas Dataframe
        The dataset on which to test the independence condition.

    var1: str
        First variable in the independence condition.

    var2: str
        Second variable in the independence condition

    cond: list
        List of variable names in given variables.

    Returns
    -------

    CI: int
        The  conditional independence of the fisher z test.
    r: float
        partial correlation coefficient
    p_value: float
        The p-value of the test
    """

    N, k_var = np.shape(data)
    list_z = [var1, var2] + list(cond)
    list_new = []
    for a in list_z:
        list_new.append(int(a))
    data_array = np.array(data)
    array_new = np.transpose(np.matrix(data_array[:, list_new]))
    cov_array = np.cov(array_new)
    size_c = len(list_new)
    X1 = 0
    Y1 = 1
    S1 = [i for i in range(size_c) if i != 0 and i != 1]
    r = partial_corr_coef(cov_array, X1, Y1, S1)
    z = 0.5 * np.log((1+r) / (1-r))
    z0 = 0
    W = np.sqrt(N - len(S1) - 3) * (z - z0)
    cutoff = norm.ppf(1 - 0.5 * alpha)
    if abs(W) < cutoff:
        CI = 1
    else:
        CI = 0
    p = norm.cdf(W)
    r = abs(r)

    return CI, r, p


def g_square_dis(dm,
                 x,
                 y,
                 s,
                 alpha,
                 levels):
    """G square test for discrete data.

    Args:
        dm: the data matrix to be used (as a numpy.ndarray).
        x: the first node (as an integer).
        y: the second node (as an integer).
        s: the set of neibouring nodes of x and y (as a set()).
        levels: levels of each column in the data matrix
            (as a list()).

    Returns:
        p_val: the p-value of conditional independence.
    """

    def _calculate_tlog(x, y, s, dof, levels, dm):
        prod_levels = np.prod(list(map(lambda x: levels[x], s)))
        nijk = np.zeros((levels[x], levels[y], prod_levels))
        s_size = len(s)
        z = []
        for z_index in range(s_size):
            z.append(s.pop())
            pass
        for row_index in range(dm.shape[0]):
            i = dm[row_index, x]
            j = dm[row_index, y]
            k = []
            k_index = 0
            for s_index in range(s_size):
                if s_index == 0:
                    k_index += dm[row_index, z[s_index]]
                else:
                    lprod = np.prod(list(map(lambda x: levels[x], z[:s_index])))
                    k_index += (dm[row_index, z[s_index]] * lprod)
                    pass
                pass
            nijk[i, j, k_index] += 1
            pass
        nik = np.ndarray((levels[x], prod_levels))
        njk = np.ndarray((levels[y], prod_levels))
        for k_index in range(prod_levels):
            nik[:, k_index] = nijk[:, :, k_index].sum(axis = 1)
            njk[:, k_index] = nijk[:, :, k_index].sum(axis = 0)
            pass
        nk = njk.sum(axis = 0)
        tlog = np.zeros((levels[x], levels[y], prod_levels))
        tlog.fill(np.nan)
        for k in range(prod_levels):
            tx = np.array([nik[:, k]]).T
            ty = np.array([njk[:, k]])
            tdijk = tx.dot(ty)
            tlog[:, :, k] = nijk[:, :, k] * nk[k] / tdijk
            pass
        return (nijk, tlog)

    #_logger.debug('Edge %d -- %d with subset: %s' % (x, y, s))
    row_size = dm.shape[0]
    s_size = len(s)
    dof = ((levels[x] - 1) * (levels[y] - 1)
           * np.prod(list(map(lambda x: levels[x], s))))

    # row_size_required = 5 * dof
    # if row_size < row_size_required:
    #     _logger.warning('Not enough samples. %s is too small. Need %s.'
    #                     % (str(row_size), str(row_size_required)))
    #     p_val = 1
    #     dep = 0
    #     return p_val, dep

    nijk = None
    if s_size < 5:
        if s_size == 0:
            nijk = np.zeros((levels[x], levels[y]))
            for row_index in range(row_size):
                i = dm[row_index, x]
                j = dm[row_index, y]
                nijk[i, j] += 1
            tx = np.array([nijk.sum(axis = 1)]).T
            ty = np.array([nijk.sum(axis = 0)])
            tdij = tx.dot(ty)
            tlog = nijk * row_size / tdij
            pass
        if s_size > 0:
            nijk, tlog = _calculate_tlog(x, y, s, dof, levels, dm)
            pass
        pass
    else:
        # s_size >= 5
        nijk = np.zeros((levels[x], levels[y], 1))
        i = dm[0, x]
        j = dm[0, y]
        k = []
        for z in s:
            k.append(dm[:, z])
            pass
        k = np.array(k).T
        parents_count = 1
        parents_val = np.array([k[0, :]])
        nijk[i, j, parents_count - 1] = 1
        for it_sample in range(1, row_size):
            is_new = True
            i = dm[it_sample, x]
            j = dm[it_sample, y]
            tcomp = parents_val[:parents_count, :] == k[it_sample, :]
            for it_parents in range(parents_count):
                if np.all(tcomp[it_parents, :]):
                    nijk[i, j, it_parents] += 1
                    is_new = False
                    break
                pass
            if is_new is True:
                parents_count += 1
                parents_val = np.r_[parents_val, [k[it_sample, :]]]
                nnijk = np.zeros((levels[x], levels[y], parents_count))
                for p in range(parents_count - 1):
                    nnijk[:, :, p] = nijk[:, :, p]
                    pass
                nnijk[i, j, parents_count - 1] = 1
                nijk = nnijk
                pass
            pass
        nik = np.ndarray((levels[x], parents_count))
        njk = np.ndarray((levels[y], parents_count))
        for k_index in range(parents_count):
            nik[:, k_index] = nijk[:, :, k_index].sum(axis = 1)
            njk[:, k_index] = nijk[:, :, k_index].sum(axis = 0)
            pass
        nk = njk.sum(axis = 0)
        tlog = np.zeros((levels[x], levels[y], parents_count))
        tlog.fill(np.nan)
        for k in range(parents_count):
            tx = np.array([nik[:, k]]).T
            ty = np.array([njk[:, k]])
            tdijk = tx.dot(ty)
            tlog[:, :, k] = nijk[:, :, k] * nk[k] / tdijk
            pass
        pass
    log_tlog = np.log(tlog)
    G2 = np.nansum(2 * nijk * log_tlog)
    # _logger.debug('dof = %d' % dof)
    # _logger.debug('nijk = %s' % nijk)
    # _logger.debug('tlog = %s' % tlog)
    # _logger.debug('log(tlog) = %s' % log_tlog)
    # _logger.debug('G2 = %f' % G2)
    if dof == 0:
        # dof can be 0 when levels[x] or levels[y] is 1, which is
        # the case that the values of columns x or y are all 0.
        p_val = 1
        G2 = 0
    else:
        p_val = chi2.sf(G2, dof)
        # print("p-value:", p_val)
    # _logger.info('p_val = %s' % str(p_val))

    if p_val > alpha:
        dep = 0
    else:
        dep = abs(G2)
    return p_val, dep


def g2_test_dis(data_matrix,
                x,
                y,
                s,
                alpha,
                **kwargs):
    s1 = sorted([i for i in s])
    levels = []
    data_matrix = np.array(data_matrix, dtype=int)
    # print(data_matrix)
    # print("x: ", x, " ,y: ", y, " ,s: ", s1)
    if 'levels' in kwargs:
        levels = kwargs['levels']
    else:
        levels = np.amax(data_matrix, axis=0) + 1
    return g_square_dis(data_matrix, x, y, s1, alpha, levels)


def logistic_reg(data,
                 target,
                 var,
                 cond_set,
                 alpha):
    cond_set = list(cond_set)
    model_fit = Logit(endog=data.iloc[:, [target]], exog=data.iloc[:, [var] + cond_set]).fit(disp=0)
    pval = model_fit.pvalues[data.columns[var]]
    dep = abs(model_fit.params[data.columns[var]])
    
    return pval, dep


def linear_gaussian_ols_reg(data,
                            target,
                            var,
                            cond_set,
                            alpha):
    cond_set = list(cond_set)
    model_fit = OLS(endog=data.iloc[:, [target]], exog=data.iloc[:, [var] + cond_set]).fit(disp=0)
    pval = model_fit.pvalues[data.columns[var]]
    dep = abs(model_fit.params[data.columns[var]])

    return pval, dep


def cond_indep_test(data,
                    target,
                    var,
                    cond_set=[],
                    is_discrete=False,
                    alpha=0.01):
    """
    Applies conditional independence test with the followgin hypothesis:
    
    H0: var \indep cond_set |data \ {var, cond_set}
    H1: var \indep cond_set |data \ {var, cond_set}
    
    If is_discrete==True, apply G2 test, otherwise apply Fisher Z test.

    :param paths: list of paths to market data
    :type paths: [str]
    :param n_cores: number of cores to use
    :type n_cores: int
    """
    if is_discrete:
        ## old function for all discrete variables in matrix
        # pval, dep = g2_test_dis(data, target, var, cond_set, alpha)
        
        ## new function to test conditional independence using a logistic model
        if len(data.iloc[:, target].unique()) == 2:
            pval, dep = logistic_reg(data=data, 
                                     target=target, 
                                     var=var, 
                                     cond_set=cond_set, 
                                     alpha=alpha)
        else:
            pval, dep = linear_gaussian_ols_reg(data=data, 
                                                target=target, 
                                                var=var, 
                                                cond_set=cond_set, 
                                                alpha=alpha)
            
    else:
        CI, dep, pval = cond_indep_fisher_z(data, target, var, cond_set, alpha)
    return pval, dep

