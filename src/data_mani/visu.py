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
    
    tb_df = df.pivot_table(index=index, columns=columns, values=values) * apply_factor_to_table
    tb_df.columns = tb_df.columns.droplevel()
    tb_df.index.name = None
    
    agg_fs_tb_df = pd.concat([tb_df.sum(axis=1), tb_df.mean(axis=1)], axis=1)
    agg_fs_tb_df = pd.concat([agg_fs_tb_df, tb_df.mean(axis=1) / tb_df.std(axis=1)], axis=1)
    agg_fs_tb_df.columns = ['sum', 'mean', 'mean_std_adj']
    
    agg_fore_tb_df = pd.concat([tb_df.sum(axis=0), tb_df.mean(axis=0)], axis=1)
    agg_fore_tb_df = pd.concat([agg_fore_tb_df, tb_df.mean(axis=0) / tb_df.std(axis=0)], axis=1)
    agg_fore_tb_df.columns = ['sum', 'mean', 'mean_std_adj']
    
    return tb_df.style.apply(highlight_max), agg_fs_tb_df.style.apply(highlight_max), agg_fore_tb_df.style.apply(highlight_max)


def plot_ret_from_predictions(predictions_df,
                              forecast_model,
                              benchmark_name,
                              benchmark_alias,
                              plot_title='Cummulative Returns for each Feature Selection Method given a Prediction Model'):

    # Benchmark
    benchmark_buynhold_df = predictions_df.loc[(predictions_df['model'] == forecast_model)&
                                               (predictions_df['variable'] == benchmark_name)&
                                               (predictions_df['fs'] == 'all')]
    benchmark_buynhold_df = benchmark_buynhold_df.pivot_table(index=['date'], columns=['variable'], values=['value'])
    fs_model_pred_df = predictions_df.loc[(predictions_df['model'] == forecast_model)&
                                          (predictions_df['variable'] != benchmark_name)]
    fs_model_pred_df = fs_model_pred_df.pivot_table(index=['date'], columns=['model', 'fs'], values=['value'])
    names = fs_model_pred_df.columns.droplevel()
    
    # Positions
    positions_df = pd.DataFrame(np.where(fs_model_pred_df > 0, 1, -1))
    positions_df.columns = names
    positions_df.index = benchmark_buynhold_df.index
    
    # Strategy 
    fs_model_ret_df = pd.DataFrame(positions_df.values * benchmark_buynhold_df.values)
    fs_model_ret_df.columns = names
    fs_model_ret_df.index = benchmark_buynhold_df.index

    # Concat benchmark + strategy
    all_ret = pd.concat([fs_model_ret_df, benchmark_buynhold_df], axis=1)
    all_ret.columns = all_ret.columns.droplevel()
    all_ret.rename(columns={benchmark_name: benchmark_alias}, inplace=True)
    
    cum_ret = (1 + all_ret).cumprod()
    cum_ret_fig = cum_ret.plot(figsize=(15, 10),
                               title=plot_title)
    
    return cum_ret_fig, cum_ret, all_ret, positions_df


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