import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter


def fs_results_aggregation(fs_paths, n):
    """
    get df with feature selection results.
    the df will present the ticker name,
    the top n features, and the bottom n
    features. We assume that the feature
    selection csv is sorted from best to worst.
    

    :param fs_paths: paths with feature selection results
    :type fs_paths: [str]
    :param n: number features to display
    :type n: int
    :return: dataframe with feature selection information
    :rtype: pd.DataFrame
    """

    result = []
    columns = ["ticker"]
    columns += ["top_{}".format(i+1) for i in range(n)]


    for path in tqdm(fs_paths):
        df = pd.read_csv(path).dropna()
        top_n = list(df.head(n).feature.values)
        name = path.split("/")[-1].split(".")[0]
        obs = [name] + top_n
        result.append(obs)
    return pd.DataFrame(result, columns=columns)



def get_top_features_from_fs_results(fs_results, top_k=10):
    """
    get the word, lag, and frequency from the
    top1 features in the dataframe 'fs_results'.
    
    :param fs_results: feature selection dataframe
    :type fs_results: DataFrame
    :param top_k: number features to display
    :type top_k: int
    :return: dataframe with top 1 feature information
    :rtype: pd.DataFrame
    """

    size = fs_results.shape[0] 
    tops = Counter(fs_results.top_1.values).most_common(top_k)
    new_tops = []
    for obs in tops:
        feature = obs[0]
        count =  obs[1]
        word = " ".join(feature.split("_")[:-1]) 
        lag = int(feature.split("_")[-1])
        new_tops.append((word, lag, count))
    top_features = pd.DataFrame(new_tops, columns=["word", "lags", "frequency"])
    top_features.loc[:, "frequency"] = (top_features.loc[:, "frequency"] / size)
    top_features.loc[:, "frequency"] = top_features.frequency.map(lambda x: "{:.1%}".format(x))
    return top_features