import numpy as np
try:
    from data_mani.utils import *
except ModuleNotFoundError:
    from src.data_mani.utils import *

def MMPC(data, target, alpha, is_discrete):
    number, kVar = np.shape(data)
    ci_number = 0
    CPC = []
    deoZeroSet = []
    sepset = [[] for i in range(kVar)]

    while True:
        M_variables = [i for i in range(kVar) if i != target and i not in CPC and i not in deoZeroSet]
        vari_all_dep_max = -float("inf")
        vari_chose = 0

        # according to pseudocode, <F,assocF> = MaxMinFeuristic(T;CPC)
        for x in M_variables:
            # use a function of getMinDep to chose min dep of x
            x_dep_min, sepset_temp, ci_num2 = getMinDep(data, target, x, CPC, alpha, is_discrete)
            ci_number += ci_num2
            # print(str(x)+" dep min is: " + str(x_dep_min))

            # if x chose min dep is 0, it never append to CPC and should not test from now on,
            if x_dep_min == 0:
                deoZeroSet.append(x)
                sepset[x] = [j for j in sepset_temp]

            elif x_dep_min > vari_all_dep_max:
                vari_chose = x
                vari_all_dep_max = x_dep_min

        # print("x chosed is: " + str(vari_chose)+" and its dep is: " + str(vari_all_dep_max))
        if vari_all_dep_max >= 0:
            # print("CPC append is: "+ str(vari_chose))
            CPC.append(vari_chose)
        else:
            # CPC has not changed(In other world,CPC not append new), circulate should be break
            break
    # print("CPC is:" +str(CPC))
    """phaseII :Backward"""
    # print("shrinking phase begin")

    CPC_temp = CPC.copy()
    max_k = 3
    for a in CPC_temp:
        C_subsets = [i for i in CPC if i != a]

        # please see explanation of the function of getMinDep() explanation
        # the chinese annotation ,if you see,you will know.
        if len(C_subsets) > max_k:
            C_length = max_k
        else:
            C_length = len(C_subsets)

        breakFlag = False
        for length in range(C_length+1):
            if breakFlag:
                break
            SS = subsets(C_subsets, length)
            for S in SS:
                ci_number += 1
                pval, dep = cond_indep_test(data, target, a, S, is_discrete)
                if pval > alpha:
                    CPC.remove(a)
                    breakFlag = True
                    break

    return list(set(CPC)), sepset, ci_number


def MMMB(data, target, alaph, is_discrete=True):
    
    ci_number = 0
    PC, sepset, ci_num2 = MMPC(data, target, alaph, is_discrete)
    ci_number += ci_num2
    MB = PC.copy()
    for x in PC:
        PCofPC, _, ci_num3 = MMPC(data, x, alaph, is_discrete)
        ci_number += ci_num3
        for y in PCofPC:
            if y != target and y not in PC:
                conditions_Set = [i for i in sepset[y]]
                conditions_Set.append(x)
                conditions_Set = list(set(conditions_Set))
                ci_number += 1
                pval, dep = cond_indep_test(
                    data, target, y, conditions_Set, is_discrete)
                if pval <= alaph:
                    MB.append(y)
                    break
    return list(set(MB)), ci_number

def run_MMMB(merged_df,
             target_name,
             words,
             max_lag,
             verbose,
             sig_level,
             is_discrete,):
    
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