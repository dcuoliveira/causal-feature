import numpy as np
from scipy.stats import chi2, norm
import pandas as pd
#from CBD.MBs.common.chi_square_test import chi_square_test
#from CBD.MBs.common.chi_square_test import chi_square
try:
    from data_mani.utils import make_shifted_df
except ModuleNotFoundError:
    from src.data_mani.utils import make_shifted_df
    

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
                pass
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


def cond_indep_test(data,
                    target,
                    var,
                    cond_set=[],
                    is_discrete=False,
                    alpha=0.01):
    if is_discrete:
        pval, dep = g2_test_dis(data, target, var, cond_set,alpha)
        # if selected:
        #     _, pval, _, dep = chi_square_test(data, target, var, cond_set, alpha)
        # else:
        # _, _, dep, pval = chi_square(target, var, cond_set, data, alpha)
    else:
        CI, dep, pval = cond_indep_fisher_z(data, target, var, cond_set, alpha)
    return pval, dep


def IAMB(data,
         target,
         alaph,
         is_discrete=False):
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
            if pval <= alaph:
                if dep > temp_dep:
                    temp_dep=dep
                    y=x

        # if not condition independence the node,appended to CMB
        if y is not None:
            # print('appended is :'+str(y))
            CMB.append(y)
            circulate_Flag = True

    # backward circulate phase
    CMB_temp = CMB.copy()
    for x in CMB_temp:
        # exclude variable which need test p-value
        condition_Variables=[i for i in CMB if i != x]
        ci_number += 1
        pval, dep = cond_indep_test(data,target, x, condition_Variables, is_discrete)
        # print("target is:", target, ",x is: ", x, " condition_Variables is: ", condition_Variables, " ,pval is: ", pval, " ,dep is: ", dep)
        if pval > alaph:
            # print("removed variables is: " + str(x))
            CMB.remove(x)

    return list(set(CMB)), ci_number 


def fast_IAMB(data, target, alaph, is_discrete=True):
    number, kVar = np.shape(data)
    ci_number = 0

    #BT present B(T) and set null,according to pseudocode
    MB = []

    # set a dictionary to store variables and their pval,but it temporary memory
    S_variables=[]
    MBvariables = [i for i in range(kVar) if i != target ]
    repeat_in_set = [0 for i in range(kVar)]
    num_reapeat = 10
    no_in_set = []
    for x in MBvariables:
        ci_number += 0
        pval, dep = cond_indep_test(data, target, x, MB, is_discrete)
        if(pval <= alaph):
            S_variables.append([x,dep])
    BT_temp = -1

    # preset value
    attributes_removed_Flag = False

    while S_variables != []:
        flag_repeat_set = [False for i in range(kVar)]
        # S sorted according to pval
        S_variables = sorted(S_variables, key=lambda x: x[1], reverse=True)
        # print(S_variables)

        """Growing phase"""
        # print("growing phase begin!")
        S_length=len(S_variables)
        insufficient_data_Flag=False
        attributes_removed_Flag = False
        for y in range(S_length):
            x = S_variables[y][0]
            # number = number
            # print("MBs is: " + str(MBs))
            qi = ns(data, MB)
            # print("qi is: " + str(qi))
            tmp = [1]
            temp1 = []
            if len(qi) > 1:
                temp1 = np.cumprod(qi[0:-1])
            # print("temp1 is: " + str(temp1))
            for i in temp1:
                tmp.append(i)
            # qs = 1 + ([i-1 for i in qi]) * tmp

            # qs = np.array([i-1 for i in qi])* np.array(tmp).reshape(len(tmp),1) + 1
            # print("qi is: " + str(qi) + " ,tmp is: " + str(tmp))
            qs = 0
            if qi == []:
                qs = 0
            else:
                for i in range(len(qi)):
                    qs += (qi[i]-1)*tmp[i]
                qs += 1

            # print("qs is: " + str(qs))
            qxt = ns(data, [x, target])
            # print("length of qs is:" + str(len(list(qs))))
            # print("qxt is: " + str(qxt))
            if qs == 0 :
                df = np.prod(np.mat([i-1 for i in qxt])) * np.prod(np.mat(qi))
                # print("1 = " + str(np.prod(np.array([i-1 for i in qxt]))) + " , 2 = " + str(np.prod(np.array(qi))))
            else:
                df = np.prod(np.mat([i-1 for i in qxt])) * qs
                # print("1 = " + str(np.prod(np.array([i-1 for i in qxt])))+" , 22 = " + str(qs))
            # print("df = " + str(df))
            if number >= 5 * df:
                # S_sort = [(key,value),....],and BT append is key
                MB.append(S_variables[y][0])
                flag_repeat_set[S_variables[y][0]] =True
                # print("BT append is: " + str(S_variables[y][0]))
            else:
                # print('1')
                insufficient_data_Flag=True
                # due to insufficient data, then go to shrinking phase
                break

        """shrinking phase"""
        # print("shrinking phase begin")
        if BT_temp == MB:
            break
        BT_temp = MB.copy()
        # print(BT)
        for x in BT_temp:

            subsets_BT = [i for i in MB if i != x]
            ci_number += 1
            pval_sp, dep_sp = cond_indep_test(data, target, x, subsets_BT, is_discrete)

            if pval_sp > alaph:
                MB.remove(x)
                if flag_repeat_set[x] == True:
                    repeat_in_set[x] += 1
                    if repeat_in_set[x] > num_reapeat:
                        no_in_set.append(x)
                        # print("x not in again is: " + str(x))
                # print("BT remove is: "+str(x))
                attributes_removed_Flag = True

        # if no variable will add to S_variables, circulate will be break,and output the result
        if (insufficient_data_Flag == True) and (attributes_removed_Flag == False):
            # print("circulate end!")
            break
        else:
            # set a new S_variables ,and add variable which match the condition
            S_variables = []
            # print("circulate should continue,so S_variable readd variables")
            BTT_variables =[i for i in range(kVar) if i != target and i not in MB and i not in no_in_set]
            # print(BTT_variables)
            for x in BTT_variables:
                ci_number += 1
                pval, dep = cond_indep_test(data, target, x, MB, is_discrete)
                if pval <= alaph:
                    # print([x,dep])
                    S_variables.append([x,dep])
                    # print("sv is: " + str(S_variables))

    return list(set(MB)), ci_number  


def run_markov_blanket(merged_df,
                       target_name,
                       words,
                       max_lag,
                       verbose,
                       sig_level,
                       is_discrete,
                       MB_algo_type):
    
    list_df = []
    for w in [target_name] + words:
        list_df.append(merged_df[[w]])
    merged_df = pd.concat(list_df, axis=1)
    
    merged_df, _ = make_shifted_df(df=merged_df,
                                   verbose=verbose,
                                   words=words,
                                   max_lag=max_lag)
    
    target_name_index = list(merged_df.columns).index(target_name)
    if MB_algo_type == 'IAMB':
        MBs, ci_number = IAMB(data=merged_df.dropna(),
                              target=target_name_index,
                              alaph=sig_level,
                              is_discrete=is_discrete) 
    elif MB_algo_type == 'FIAMB':
        MBs, ci_number = fast_IAMB(data=merged_df.dropna(),
                                   target=target_name_index,
                                   alaph=sig_level,
                                   is_discrete=is_discrete) 
    else:
        raise Exception('MB algo nao cadastrado')
    
    if len(MBs) == 0:
        features = list(merged_df.columns)
        features.remove( )
        MBs_df = pd.DataFrame(data={'feature': features,
                                    'feature_score': np.nan})
    else:
        features = []
        for mb in MBs:
            features.append(merged_df.columns[mb])
        MBs_df = pd.DataFrame(data={'feature': features,
                                    'feature_score': np.nan})
    
    return MBs_df