import numpy as np
try:
    from data_mani.utils import *
except ModuleNotFoundError:
    from src.data_mani.utils import *
    

def getPCD(data, target, alaph, is_discrete):
    number, kVar = np.shape(data)
    max_k = 3
    PCD = []
    ci_number = 0

    # use a list of sepset[] to store a condition set which can make target and the variable condition independence
    # the above-mentioned variable will be remove from CanPCD or PCD
    sepset = [[] for i in range(kVar)]

    while True:
        variDepSet = []
        CanPCD = [i for i in range(kVar) if i != target and i not in PCD]
        CanPCD_temp = CanPCD.copy()

        for vari in CanPCD_temp:
            breakFlag = False
            dep_gp_min = float("inf")
            vari_min = -1

            if len(PCD) >= max_k:
                Plength = max_k
            else:
                Plength = len(PCD)

            for j in range(Plength+1):
                SSubsets = subsets(PCD, j)
                for S in SSubsets:
                    ci_number += 1
                    pval_gp, dep_gp = cond_indep_test(data, target, vari, S, is_discrete)

                    if pval_gp > alaph:
                        vari_min = -1
                        CanPCD.remove(vari)
                        sepset[vari] = [i for i in S]
                        breakFlag = True
                        break
                    elif dep_gp < dep_gp_min:
                        dep_gp_min = dep_gp
                        vari_min = vari

                if breakFlag:
                    break

            # use a list of variDepset to store list, like [variable, its dep]
            if vari_min in CanPCD:
                variDepSet.append([vari_min, dep_gp_min])

        # sort list of variDepSet by dep from max to min
        variDepSet = sorted(variDepSet, key=lambda x: x[1], reverse=True)

        # if variDepset is null ,that meaning PCD will not change
        if variDepSet != []:
            y =variDepSet[0][0]
            PCD.append(y)
            pcd_index = len(PCD)
            breakALLflag = False
            while pcd_index >=0:
                pcd_index -= 1
                x = PCD[pcd_index]
                breakFlagTwo = False

                conditionSetALL = [i for i in PCD if i != x]
                if len(conditionSetALL) >= max_k:
                    Slength = max_k
                else:
                    Slength = len(conditionSetALL)

                for j in range(Slength+1):
                    SSubsets = subsets(conditionSetALL, j)
                    for S in SSubsets:
                        ci_number += 1
                        pval_sp, dep_sp = cond_indep_test(data, target, x, S, is_discrete)

                        if pval_sp > alaph:

                            PCD.remove(x)
                            if x == y:
                                breakALLflag = True

                            sepset[x] = [i for i in S]
                            breakFlagTwo = True
                            break
                    if breakFlagTwo:
                        break

                if breakALLflag:
                    break
        else:
            break
    return list(set(PCD)), sepset, ci_number


def getPC(data, target, alaph, is_discrete):
    ci_number = 0
    PC = []
    PCD, sepset, ci_num2 = getPCD(data, target, alaph, is_discrete)
    ci_number += ci_num2
    for x in PCD:
        variSet, _, ci_num3 = getPCD(data, x, alaph, is_discrete)
        ci_number += ci_num3
        # PC of target ,whose PC also has the target, must be True PC
        if target in variSet:
            PC.append(x)

    return list(set(PC)), sepset, ci_number

def PCMB(data,
         target,
         alaph,
         is_discrete=True):
    ci_number = 0
    PC, sepset, ci_num2 = getPC(data, target, alaph, is_discrete)
    ci_number += ci_num2
    # print(PC)
    # print(sepset)
    MB = PC.copy()

    for x in PC:
        PCofPC_temp, _, ci_num3 = getPC(data, x, alaph, is_discrete)
        ci_number += ci_num3
        # print(" pc of pc_temp is: " + str(PCofPC_temp))
        PCofPC = [i for i in PCofPC_temp if i != target and i not in MB]
        # print(" pc of pc is: " + str(PCofPC))
        for y in PCofPC:
            conditionSet = [i for i in sepset[y]]
            conditionSet.append(x)
            conditionSet = list(set(conditionSet))
            ci_number += 1
            pval, dep = cond_indep_test(
                data, target, y, conditionSet, is_discrete)
            if pval <= alaph:
                MB.append(y)
                break
    return list(set(MB)), ci_number


def run_PCMB(merged_df,
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
    MBs, ci_number = PCMB(data=merged_df.dropna(),
                          target=target_name_index,
                          alaph=sig_level,
                          is_discrete=is_discrete) 
    
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