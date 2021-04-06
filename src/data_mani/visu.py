import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter 


def highlight_max(s):
    if s.dtype == np.object:
        is_neg = [False for _ in range(s.shape[0])]
    else:
        is_neg = s < 0
    return ['color: red;' if cell else 'color:white' 
            for cell in is_neg]
       
    
def plot_df_to_table(df,
                     index,
                     columns,
                     values,
                     apply_factor_to_table=100):
    """
    plot tables from dataframe

    :param df: melted dataframe
    :type df: dataframe
    :param target_asset_returns: name of the columns to use as pivot index
    :type target_asset_returns: list
    :param target_asset_returns: name of the columns to use as pivot table columns
    :type target_asset_returns: list
    :param target_asset_returns: name of the column to use as pivot table values
    :type target_asset_returns: list
    :param target_asset_returns: factor to multiply in the table
    :type target_asset_returns: int
    """
    
    tb_df = df.pivot_table(index=index, columns=columns, values=values) * apply_factor_to_table
    tb_df.columns = tb_df.columns.droplevel()
    tb_df.index.name = None
    
    agg_fs_tb_df = pd.concat([tb_df.sum(axis=1), tb_df.median(axis=1)], axis=1)
    agg_fs_tb_df = pd.concat([agg_fs_tb_df, tb_df.median(axis=1) / tb_df.std(axis=1)], axis=1)
    agg_fs_tb_df.columns = ['sum', 'median', 'median_std_adj']
    
    agg_fore_tb_df = pd.concat([tb_df.sum(axis=0), tb_df.median(axis=0)], axis=1)
    agg_fore_tb_df = pd.concat([agg_fore_tb_df, tb_df.median(axis=0) / tb_df.std(axis=0)], axis=1)
    agg_fore_tb_df.columns = ['sum', 'median', 'median_std_adj']
    
    return tb_df.round(2).style.apply(highlight_max), agg_fs_tb_df.round(2).style.apply(highlight_max), agg_fore_tb_df.round(2).style.apply(highlight_max)


def plot_cum_ret(pred_ret_df,
                 benchmark_df,
                 level_to_subset,
                 show=True):
    """
    plot tables from dataframe

    :param df: melted dataframe containing the predictions of each fs method and pred. model
    :type df: dataframe
    :param target_asset_returns: melted dataframe containing the benchmark returns of each ticker
    :type target_asset_returns: daframe
    :param target_asset_returns: level to fix in the plot (i.e. plot by fs, plot by pred. model)
    :type target_asset_returns: str
    :param target_asset_returns: show or not show plot
    :type target_asset_returns: boolean
    """
    ret_df = pd.concat([pred_ret_df, benchmark_df], axis=0)

    cum_ret = []
    for ticker in ret_df['ticker'].unique():
        for level in ret_df[level_to_subset].unique():
            if level == ticker:
                continue
            loop_df = ret_df.loc[(ret_df['ticker'] == ticker)&((ret_df[level_to_subset] == level)|(ret_df[level_to_subset] == ticker)|(ret_df['variable'] == 'return'))]
            pivot_level_to_add = list(loop_df.columns.drop(['date', 'ticker', 'value', 'variable'] + [level_to_subset]))
            pivot_all_ret_df = loop_df.pivot_table(index=['date'], columns=['ticker', 'variable'] + pivot_level_to_add, values=['value'])
            pivot_all_ret_df.columns = pivot_all_ret_df.columns.droplevel()

            cum_all_ret = (1 + pivot_all_ret_df).cumprod()
            cum_ret.append(cum_all_ret)
            if show:
                cum_all_ret.plot(figsize=(15, 10), title=level_to_subset + ' = ' + level.upper())
    cum_ret_df = pd.concat(cum_ret, axis=1)
    return cum_ret_df


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