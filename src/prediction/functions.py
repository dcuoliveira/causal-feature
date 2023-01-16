import os
from glob import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

from data_mani.utils import merge_market_and_gtrends, target_ret_to_directional_movements
from data_mani.visu import *


def aggregate_prediction_results(prediction_models,
                                 fs_models,
                                 evaluation_start_date,
                                 evaluation_end_date,
                                 ticker_names,
                                 metric_name,
                                 tag='oos',
                                 benchmark_name='return'):
    """
    aggreagate prediction results of the specified models
    and feature selection methods.

    :param prediction_models: list of predcition model names (must match with results dir)
    :type prediction_models: list of strs
    :param fs_models: list of fs model names (must match with results dir)
    :type fs_models: list of strs
    :param evaluation_start_date: date to start computing oos results
    :type evaluation_start_date: str (yyyy-mm-dd)
    :param evaluation_end_date: date to end computing oos results
    :type evaluation_end_date:  str (yyyy-mm-dd)
    :param ticker_names: list of predcition model names (must match with data dir) 
    :type ticker_names: list of strs
    :param metric_name: name of the evaluation metric to be used
    :type metric_name: str
    :param tag: string tag to add with the metric name 
    :type tag: str
    :param benchmark_name: name of the benchmark in the files of the data and results directory
    :type benchmark_name: str
    """
    
    predictions = []
    metrics = []
    for fs in fs_models:
        
        fs_name = fs.upper()
        
        for model in prediction_models:
            
            if model == 'random_forest':
                model_name = 'RF'
            elif model == 'lgb':
                model_name = 'GB'
            else:
                model_name = model.upper()
            
            for ticker in ticker_names:
                df = pd.read_csv('results/forecast/' + fs + '/indices/' + model + '/' + ticker + '.csv')
                df.set_index('date', inplace=True)
                df = df.loc[evaluation_start_date:evaluation_end_date]
                df = df.reset_index()

                metric_eval_df = df.copy()
                metric = roc_auc_score(metric_eval_df[benchmark_name].values, metric_eval_df['prediction'].values)
                metric_df = pd.DataFrame([{'ticker': ticker,
                                           'model': model_name,
                                           'fs': fs_name,
                                            tag + metric_name: metric}])
                metrics.append(metric_df)

                melt_df = df.melt('date')
                melt_df['model'] = model_name
                melt_df['fs'] = fs_name
                melt_df['ticker'] = ticker
                predictions.append(melt_df)

    predictions_df = pd.concat(predictions, axis=0)
    benchmark_df = predictions_df.loc[(predictions_df['variable']==benchmark_name)&
                                    (predictions_df['fs']==fs_name)&
                                    (predictions_df['model']==model_name)]
    benchmark_df['model'] = benchmark_df['ticker']
    benchmark_df['fs'] = 'raw'
    predictions_df = predictions_df.loc[(predictions_df['variable']!=benchmark_name)]
    metric_df = pd.concat(metrics, axis=0)

    return predictions_df, benchmark_df, metric_df


def ann_avg_returns_tb(returns_df,
                       level_to_subset,
                       rf=.0):
    if level_to_subset == 'fs':
        other_level = 'model'
    else:
        other_level = 'fs'
    mean = returns_df.pivot_table(index=['date'], columns=['ticker', 'variable'] + [other_level] + [level_to_subset],
                                  values=['value']).dropna().mean()

    ann_avg_ret_df = pd.DataFrame((mean - .0) * 252)
    ann_avg_ret_df.index = ann_avg_ret_df.index.droplevel()
    ann_avg_ret_df.rename(columns={0: 'Ann. Avg. Return'}, inplace=True)
    rank_df = ann_avg_ret_df.sort_values('Ann. Avg. Return', ascending=False)

    pivot_tb = rank_df.reset_index().pivot_table(index=['ticker'] + [level_to_subset], columns=[other_level],
                                                 values=['Ann. Avg. Return'])

    agg_pivot_tb = pd.concat([pivot_tb.sum(axis=1), pivot_tb.median(axis=1)], axis=1)
    agg_pivot_tb = pd.concat([agg_pivot_tb, pivot_tb.median(axis=1) / pivot_tb.std(axis=1)], axis=1)
    agg_pivot_tb.columns = ['sum', 'median', 'median_std_adj']

    return rank_df, pivot_tb.fillna(0), agg_pivot_tb


def ann_vol_tb(returns_df,
               level_to_subset,
               rf=.0):
    if level_to_subset == 'fs':
        other_level = 'model'
    else:
        other_level = 'fs'
    std = returns_df.pivot_table(index=['date'], columns=['ticker', 'variable'] + [other_level] + [level_to_subset],
                                 values=['value']).dropna().std()

    vol_df = pd.DataFrame(std * np.sqrt(252))
    vol_df.index = vol_df.index.droplevel()
    vol_df.rename(columns={0: 'Ann. Volatility'}, inplace=True)
    rank_df = vol_df.sort_values('Ann. Volatility', ascending=False)

    pivot_tb = rank_df.reset_index().pivot_table(index=['ticker'] + [level_to_subset], columns=[other_level],
                                                 values=['Ann. Volatility'])

    agg_pivot_tb = pd.concat([pivot_tb.sum(axis=1), pivot_tb.median(axis=1)], axis=1)
    agg_pivot_tb = pd.concat([agg_pivot_tb, pivot_tb.median(axis=1) / pivot_tb.std(axis=1)], axis=1)
    agg_pivot_tb.columns = ['sum', 'median', 'median_std_adj']

    return rank_df, pivot_tb.fillna(0), agg_pivot_tb


def sharpe_ratio_tb(returns_df,
                    level_to_subset,
                    rf=.0):
    """
    generate sharpe ratio table for each "level to subset"

    :param returns_df: melted dataframe containing the returns of a strategy based on
    of each fs method and pred. model
    :type returns_df: dataframe
    :param level_to_subset: name of the columns to fix so as to generate the table (i.e. fs or model)
    :type level_to_subset: str
    """
    if level_to_subset == 'fs':
        other_level = 'model'
    else:
        other_level = 'fs'
    mean = returns_df.pivot_table(index=['date'], columns=['ticker', 'variable'] + [other_level] + [level_to_subset], values=['value']).dropna().mean()
    std = returns_df.pivot_table(index=['date'], columns=['ticker', 'variable'] + [other_level] + [level_to_subset], values=['value']).dropna().std()

    mean = mean.sort_index()
    std = std.sort_index()

    sr_df = pd.DataFrame((mean - .0) / std * np.sqrt(252))
    sr_df.index = sr_df.index.droplevel()
    sr_df.rename(columns={0: 'Sharpe ratio'}, inplace=True)
    rank_df = sr_df.sort_values('Sharpe ratio', ascending=False)

    pivot_tb = rank_df.reset_index().pivot_table(index=['ticker'] + [level_to_subset], columns=[other_level], values=['Sharpe ratio'])
    
    agg_pivot_tb = pd.concat([pivot_tb.sum(axis=1), pivot_tb.median(axis=1)], axis=1)
    agg_pivot_tb = pd.concat([agg_pivot_tb, pivot_tb.median(axis=1) / pivot_tb.std(axis=1)], axis=1)
    agg_pivot_tb.columns = ['sum', 'median', 'median_std_adj']

    return rank_df, pivot_tb.fillna(0), agg_pivot_tb


def max_drawdown_tb(returns_df,
                    level_to_subset):
    """
    generate max drawdown table for each "level to subset"

    :param pivot_ret_all_df: pivot dataframe containing in each columns the return of each of
    the fs x (pred. model) combinations
    :type pivot_ret_all_df: dataframe
    :param level_to_subset: name of the columns to fix so as to generate the table (i.e. fs or model)
    :type level_to_subset: str
    """
    if level_to_subset == 'fs':
        other_level = 'model'
    else:
        other_level = 'fs'

    pivot_rets = (returns_df.pivot_table(index=['date'], columns=['ticker', 'variable'] + ['model'] + ['fs'],
                                         values=['value']).dropna() / 100)
    cum_prod_df = (1 + pivot_rets).cumprod()
    previous_peaks_df = cum_prod_df.cummax()
    drawdown_df = (cum_prod_df - previous_peaks_df) / previous_peaks_df
    rank_df = pd.DataFrame(drawdown_df.min().sort_values(ascending=False))
    rank_df.index = rank_df.index.droplevel()
    rank_df.rename(columns={0: 'Max. drawdown'}, inplace=True)
    rank_df = rank_df.sort_values('Max. drawdown', ascending=False) * 100

    tb_df = rank_df.reset_index()
    tb_df = tb_df.pivot_table(index=['ticker'] + [level_to_subset], columns=[other_level],
                              values=['Max. drawdown']).fillna(0)

    agg_pivot_tb = pd.concat([tb_df.sum(axis=1), tb_df.median(axis=1)], axis=1)
    agg_pivot_tb = pd.concat([agg_pivot_tb, tb_df.median(axis=1) / tb_df.std(axis=1)], axis=1)
    agg_pivot_tb.columns = ['sum', 'median', 'median_std_adj']

    return rank_df, tb_df, agg_pivot_tb


def calmar_ratio_tb(returns_df,
                    level_to_subset):
    """
    generate sharpe ratio table for each "level to subset"

    :param returns_df: melted dataframe containing the returns of a strategy based on
    of each fs method and pred. model
    :type returns_df: dataframe
    :param level_to_subset: name of the columns to fix so as to generate the table (i.e. fs or model)
    :type level_to_subset: str
    """
    if level_to_subset == 'fs':
        other_level = 'model'
    else:
        other_level = 'fs'

    # pivot returns
    pivot_rets = (returns_df.pivot_table(index=['date'], columns=['ticker', 'variable'] + ['model'] + ['fs'],
                                         values=['value']).dropna() / 100)

    # mean returns
    mean = returns_df.pivot_table(index=['date'], columns=['ticker', 'variable'] + [other_level] + [level_to_subset],
                                  values=['value']).dropna().mean()
    mean.index = mean.index.droplevel()
    mean.columns = ['mean']

    #max drawdown
    cum_prod_df = (1 + pivot_rets).cumprod()
    previous_peaks_df = cum_prod_df.cummax()
    drawdown_df = (cum_prod_df - previous_peaks_df) / previous_peaks_df
    maxdrawdown_df = drawdown_df.min().sort_values(ascending=False)
    maxdrawdown_df.index = maxdrawdown_df.index.droplevel()
    maxdrawdown_df.columns = ['mdd']

    # calmar ratio df
    mean = mean.sort_index()
    maxdrawdown_df = maxdrawdown_df.sort_index()
    cr_df = pd.DataFrame((mean - .0) / maxdrawdown_df * -1)
    # cr_df.index = cr_df.index.droplevel()
    cr_df.rename(columns={0: 'Calmar ratio'}, inplace=True)
    rank_df = cr_df.sort_values('Calmar ratio', ascending=False)

    pivot_tb = rank_df.reset_index().pivot_table(index=['ticker'] + [level_to_subset], columns=[other_level],
                                                 values=['Calmar ratio'])

    agg_pivot_tb = pd.concat([pivot_tb.sum(axis=1), pivot_tb.median(axis=1)], axis=1)
    agg_pivot_tb = pd.concat([agg_pivot_tb, pivot_tb.median(axis=1) / pivot_tb.std(axis=1)], axis=1)
    agg_pivot_tb.columns = ['sum', 'median', 'median_std_adj']

    return rank_df, pivot_tb.fillna(0), agg_pivot_tb


def gen_strat_positions_and_ret_from_pred(predictions_df,
                                          target_asset_returns,
                                          class_threshold=None):
    """
    generate strategy positions (simple or ranking) from each of the
    fs x (pred model) predictions.

    :param predictions_df: melted dataframe containing the predictions of each fs method and pred. model
    :type predictions_df: dataframe
    :param class_threshold: threshold such that if "vec_val" > threshold => 1; otherwise => -1
    :type class_threshold: float
    :param target_asset_returns: melted dataframe containing daily returns of the benchmark indices
    :type target_asset_returns: dataframe
    """

    if 'strat_type' not in predictions_df.columns:
        predictions_df['strat_type'] = 'simple'
    
    
    for ticker in predictions_df['ticker'].unique():
        pred_positions = []
        pred_returns = []
        for strat in predictions_df['strat_type'].unique():
            strat_df = predictions_df.loc[predictions_df['strat_type'] == strat].drop(labels='strat_type', axis=1)
            if strat == 'simple':
                for ticker in strat_df['ticker'].unique():
                    ticker_strat_df = strat_df.loc[strat_df['ticker'] == ticker]
                    ticker_strat_pivot_df = ticker_strat_df.pivot_table(index=['date'], columns=['variable', 'ticker', 'model', 'fs'], values=['value'])
                    
                    if class_threshold is not None:
                        colnames = ticker_strat_pivot_df.columns
                        rownames = ticker_strat_pivot_df.index
                        ticker_strat_pivot_df = pd.DataFrame(np.where(ticker_strat_pivot_df > class_threshold,
                                                                      1,
                                                                      -1))
                        ticker_strat_pivot_df.columns = colnames
                        ticker_strat_pivot_df.index = rownames
                    names = ticker_strat_pivot_df.columns.droplevel().droplevel()

                    # Benchmark
                    benchmark_df = target_asset_returns.loc[target_asset_returns['ticker'] == ticker]
                    pivot_benchmark_df = benchmark_df.pivot_table(index=['date'], columns=['variable', 'ticker', 'model', 'fs'], values=['value'])

                    # Positions
                    positions_df = ticker_strat_pivot_df
                    positions_df.columns = names
                    positions_df.index = ticker_strat_pivot_df.index
                    melt_positions_df = positions_df.reset_index().melt('date')
                    melt_positions_df['variable'] = 'prediction'
                    melt_positions_df['ticker'] = ticker
                    pred_positions.append(melt_positions_df)
                    
                    pivot_benchmark_df = pivot_benchmark_df.loc[positions_df.index[0]:positions_df.index[len(positions_df)-1]]
                    # Strategy 
                    pred_ret_df = pd.DataFrame(positions_df.values * pivot_benchmark_df.values)
                    pred_ret_df.columns = names
                    pred_ret_df.index = ticker_strat_pivot_df.index
                    melt_pred_df = pred_ret_df.reset_index().melt('date')
                    melt_pred_df['variable'] = 'prediction'
                    melt_pred_df['ticker'] = ticker
                    pred_returns.append(melt_pred_df)
                pred_ret_df = pd.concat(pred_returns, axis=0)
                pred_pos_df = pd.concat(pred_positions)
            elif strat == 'rank':
                pass

    return pred_ret_df, pred_pos_df


def get_features_IAMB_MMMB(ticker_name,
                           out_folder,
                           fs_method,
                           path_list):
    """
    Select a subset of features using
    either the IAMB or the MMMB feature selection method.
    We select all features that appear on the resulting
    dataframe

    :param ticker_name: ticker name (without extension)
    :type ticker_name: str
    :param out_folder: folder with market data
    :type out_folder: str
    :param fs_method: folder with feature selection
                      results
    :type fs_method: str
    :param path_list: list of str to create feature path
    :type path_list: [str]
    :return: list of feature names
    :rtype: [str]
    """
    assert fs_method in ["IAMB", "MMMB"]
    ticker_name = "{}.csv".format(ticker_name)
    if len(path_list) > 2:
        path = os.path.join(*[path_list[0],
                              "src", "results", "feature_selection",
                              fs_method, out_folder, ticker_name])
    else:
        path = os.path.join(*["results", "feature_selection",
                              fs_method, out_folder, ticker_name])
    scores = pd.read_csv(path)
    scores = scores.feature.to_list()
    return scores


def get_features_granger_huang(ticker_name,
                               out_folder,
                               fs_method,
                               path_list):
    """
    Select a subset of features using
    either the granger or the huang feature selection method.
    we select all features that has some p-value. All the
    others, with nans, are excluded.

    :param ticker_name: ticker name (without extension)
    :type ticker_name: str
    :param out_folder: folder with market data
    :type out_folder: str
    :param fs_method: folder with feature selection
                      results
    :type fs_method: str
    :param path_list: list of str to create feature path
    :type path_list: [str]
    :return: list of feature names
    :rtype: [str]
    """
    assert fs_method in ["huang", "granger"]
    ticker_name = "{}.csv".format(ticker_name)
    if len(path_list) > 2:
        path = os.path.join(*[path_list[0],
                              "src", "results", "feature_selection",
                              fs_method, out_folder, ticker_name])
    else:
        path = os.path.join(*["results", "feature_selection",
                              fs_method, out_folder, ticker_name])
    scores = pd.read_csv(path).dropna()
    scores = scores.feature.to_list()
    return scores


def get_selected_features(ticker_name,
                          out_folder,
                          fs_method,
                          path_list):
    """
    Select a subset of features using a feature
    selection method. As suggested in AFML,
    we select the top k ranked features with importance scores
    higher than the mean importance scores across all features.


    :param ticker_name: ticker name (without extension)
    :type ticker_name: str
    :param out_folder: folder with market data
    :type out_folder: str
    :param fs_method: folder with feature selection
                      results
    :type fs_method: str
    :param path_list: list of str to create feature path
    :type path_list: [str]

    :return: list of feature names
    :rtype: [str]
    """
    assert fs_method in ["sfi", "mdi", "mda"]
    ticker_name = "{}.csv".format(ticker_name)

    if len(path_list) > 2:
        path = os.path.join(*[path_list[0],
                              "src", "results", "feature_selection",
                              fs_method, out_folder, ticker_name])
    else:
        path = os.path.join(*["results", "feature_selection",
                              fs_method, out_folder, ticker_name])
    scores = pd.read_csv(path)
    cut_point = scores.feature_score.mean()
    scores = scores.loc[scores.feature_score >= cut_point]
    scores = scores.feature.to_list()
    return scores


def new_r2(y_true,
           y_pred):
    """
    The R2 is calculate using the formula in the paper

    "Empirical Asset Pricing via Machine
    Learning"

    :param y_true: true returns
    :type y_true: np.array
    :param y_pred: model predictions
    :type y_pred: np.array
    :return: R2 value
    :rtype: float
    """
    erros = y_true - y_pred
    num = (erros).dot(erros)
    dem = (y_true).dot(y_true)
    r2 = 1 - (num / dem)
    return r2


def add_shift(merged_df,
              words,
              max_lag,
              verbose):
    """
    add shift for all words in 'words' using
    lags from 1 to 'max_lag'

    :param merged_df: df with market and gtrends data
    :type merged_df: pd.DataFrame
    :param words: list of words
    :type words: [str]
    :param max_lag: maximun number of lags
    :type max_lag: int
    """
    for word in tqdm(words, disable=not verbose, desc="add shift"):
        for shift in range(1, max_lag + 1):
            new_feature = word.replace(" ", "_") + "_{}".format(shift)
            merged_df.loc[:, new_feature] = merged_df[word].shift(shift)


def hyper_params_search(df,
                        wrapper,
                        n_iter,
                        n_splits,
                        n_jobs,
                        verbose,
                        seed,
                        target_name="target_return"):
    """
    Use the dataframe 'df' to search for the best
    params for the model 'wrapper'.

    The CV split is performed using the TimeSeriesSplit
    class.

    We can define the size of the test set using the formula

    ``n_samples//(n_splits + 1)``,


    where ``n_samples`` is the number of samples. Hence,
    we can define

    n_splits = (n - test_size) // test_size


    :param df: train data
    :type df: pd.DataFrame
    :param wrapper: predictive model
    :type wrapper: sklearn model wrapper
    :param n_iter: number of hyperparameter searchs
    :type n_iter: int
    :param n_splits: number of splits for the cross-validation
    :type n_splits: int
    :param n_jobs: number of concurrent workers
    :type n_jobs: int
    :param verbose: param to print iteration status
    :type verbose: bool, int
    :param target_name: name of the target column in 'df'
    :type target_name: str
    :return: R2 value
    :rtype: float
    """

    X = df.drop(labels=target_name, axis=1).values
    y = df[target_name].values

    time_split = TimeSeriesSplit(n_splits=n_splits)
    roc_auc_scorer = make_scorer(roc_auc_score)

    if wrapper.search_type == 'random':
        model_search = RandomizedSearchCV(estimator=wrapper.ModelClass,
                                          param_distributions=wrapper.param_grid,
                                          n_iter=n_iter,
                                          cv=time_split,
                                          verbose=verbose,
                                          n_jobs=n_jobs,
                                          scoring=roc_auc_scorer,
                                          random_state=seed)
    elif wrapper.search_type == 'grid':
        model_search = GridSearchCV(estimator=wrapper.ModelClass,
                                    param_grid=wrapper.param_grid,
                                    cv=time_split,
                                    verbose=verbose,
                                    n_jobs=n_jobs,
                                    scoring=roc_auc_scorer)
    else:
        raise Exception('search type method not registered')

    model_search = model_search.fit(X, y)

    return model_search


def annualy_fit_and_predict(df,
                            init_steps,
                            predict_steps,
                            Wrapper,
                            n_iter,
                            n_splits,
                            n_jobs,
                            verbose,
                            seed,
                            target_name="target_return"):
    """
     We recursively increase the training sample, periodically refitting
     the entire model once per year, and making
     out-of-sample predictions for the subsequent year.

     On each fit, to perform hyperparameter search,
     we perform cross-validation on a rolling basis.

     :param df: train and test data combined
     :type df: pd.DataFrame
     :param Wrapper: predictive model class
     :type Wrapper: sklearn model wrapper class
     :param n_iter: number of hyperparameter searchs
     :type n_iter: int
     :param n_splits: number of splits for the cross-validation
     :type n_splits: int
     :param n_jobs: number of concurrent workers
     :type n_jobs: int
     :param verbose: param to print iteration status
     :type verbose: bool, int
     :param target_name: name of the target column in 'df'
     :type target_name: str
     :return: dataframe with the date, true return
              and predicted return.
     :rtype: pd.DataFrame
     """

    all_preds = []

    features = sorted(df.drop(labels=target_name, axis=1).columns.to_list())
    df = df[features + [target_name]]
    df = target_ret_to_directional_movements(df, target_name)

    for t in tqdm(range(init_steps, df.shape[0] - init_steps, predict_steps),
                  desc="Running TSCV"):

        train_ys = df[:t]
        test_ys = df[t:(t + predict_steps)]
        store_train_target = train_ys[target_name].values
        store_test_target = test_ys[target_name].values

        scaler = StandardScaler()
        train_ys_v = scaler.fit_transform(train_ys)
        train_ys = pd.DataFrame(train_ys_v,
                                columns=train_ys.columns,
                                index=train_ys.index)
        train_ys.loc[:, target_name] = store_train_target

        test_ys_v = scaler.transform(test_ys)
        test_ys = pd.DataFrame(test_ys_v,
                               columns=test_ys.columns,
                               index=test_ys.index)
        test_ys.loc[:, target_name] = store_test_target
        y_test = test_ys[target_name].values

        # we have some roles in the time interval
        # for some tickers, for example,
        # "SBUX UA Equity"
        if test_ys.shape[0] > 0:
            model_wrapper = Wrapper()
            model_search = hyper_params_search(df=train_ys,
                                               wrapper=model_wrapper,
                                               n_jobs=n_jobs,
                                               n_splits=n_splits,
                                               n_iter=n_iter,
                                               seed=seed,
                                               verbose=verbose)
            X_test = test_ys.drop(labels=target_name, axis=1).values
            test_pred = model_search.best_estimator_.predict_proba(X_test)[:, 1]
            dict_ = {"date": test_ys.index,
                     "return_direction": y_test,
                     "prediction": test_pred}
            result = pd.DataFrame(dict_)
            all_preds.append(result)
        else:
            pass
    pred_results = pd.concat(all_preds).reset_index(drop=True)
    return pred_results


def forecast(ticker_name,
             db_name,
             fs_method,
             init_steps,
             predict_steps,
             Wrapper,
             n_iter,
             n_splits,
             n_jobs,
             seed,
             verbose=1,
             target_name="target_return",
             max_lag=20):
    """
    Function to perform the predition using one ticker,
    one feature selection method, and one prediction model.
    IT ALWAYS PERFORMS CLASSIFICATION


    :param ticker_name: ticker name (without extension)
    :type ticker_name: str
    :param fs_method: folder with feature selection
                      results
    :type fs_method: str
    :param Wrapper: predictive model class
    :type Wrapper: sklearn model wrapper class
    :param n_iter: number of hyperparameter searchs
    :type n_iter: int
    :param n_splits: number of splits for the cross-validation
    :type n_splits: int
    :param n_jobs: number of concurrent workers
    :type n_jobs: int
    :param verbose: param to print iteration status
    :type verbose: bool, int
    :param target_name: name of the target column in 'df'
    :type target_name: str
    :param max_lag: maximun number of lags
    :type max_lag: int
    :return: dataframe with the date, true return
            and predicted return.
    :rtype: pd.DataFrame
    """
    path_list = ["data", "index"]
    ticker_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/indices/{}.csv".format(ticker_name))
    train, test = merge_market_and_gtrends(ticker_path,
                                           test_size=0.5,
                                           db_name=db_name)
    words = train.drop(labels=target_name, axis=1).columns.to_list()
    complete = pd.concat([train, test])
    del train, test

    add_shift(merged_df=complete,
              words=words,
              max_lag=max_lag,
              verbose=verbose)
    complete = complete.fillna(0.0)

    if fs_method in ["sfi", "mdi", "mda"]:

        select = get_selected_features(ticker_name=ticker_name,
                                       out_folder="indices",
                                       fs_method=fs_method,
                                       path_list=path_list)

    elif fs_method in ["granger", "huang"]:

        select = get_features_granger_huang(ticker_name=ticker_name,
                                            out_folder="indices",
                                            fs_method=fs_method,
                                            path_list=path_list)

    elif fs_method in ["IAMB", "MMMB"]:

        select = get_features_IAMB_MMMB(ticker_name=ticker_name,
                                        out_folder="indices",
                                        fs_method=fs_method,
                                        path_list=path_list)

    else:
        assert fs_method == "all"
        select = complete.drop(labels=words + [target_name], axis=1).columns.to_list()

    complete_selected = complete[[target_name] + select]

    pred_results = annualy_fit_and_predict(df=complete_selected,
                                           init_steps=init_steps,
                                           predict_steps=predict_steps,
                                           Wrapper=Wrapper,
                                           n_iter=n_iter,
                                           n_jobs=n_jobs,
                                           n_splits=n_splits,
                                           target_name=target_name,
                                           seed=seed,
                                           verbose=verbose)

    return pred_results


def forecast_comb(ticker_name,
                  fs_method,
                  model,
                  start_date,
                  end_date,
                  metric_name,
                  comb_model,
                  verbose=1,
                  target_name="target_return",
                  benchmark_name='return_direction',
                  Wrapper=None,
                  n_iter=None,
                  n_jobs=None,
                  n_splits=None,
                  seed=None):

    melted_predictions_df, melted_benchmark_df, _ = aggregate_prediction_results(prediction_models=model,
                                                                                 fs_models=[fs_method],
                                                                                 evaluation_start_date=start_date,
                                                                                 evaluation_end_date=end_date,
                                                                                 ticker_names=[ticker_name],
                                                                                 metric_name=metric_name,
                                                                                 benchmark_name=benchmark_name)

    # turn predictions into a pivot table
    predictions_df = melted_predictions_df.pivot_table(index=['date'], columns=['model'], values=['value'])
    predictions_df.columns = predictions_df.columns.droplevel()

    # turn benchmark into a pivot table
    benchmark_df = melted_benchmark_df.pivot_table(index=['date'], columns=['model'], values=['value'])
    benchmark_df.columns = benchmark_df.columns.droplevel()

    df = pd.concat([benchmark_df, predictions_df], axis=1)
    df.index = pd.to_datetime(df.index)
    df.rename(columns={ticker_name: target_name}, inplace=True)

    years = df.index.map(lambda x: x.year)
    years = range(np.min(years), np.max(years))
    features = sorted(df.drop(lebels=target_name, axis=1).columns.to_list())
    df = df[features + [target_name]]

    if comb_model == 'nncomb':
        comb_predictions_out = annualy_fit_and_predict(df=df,
                                                       Wrapper=Wrapper,
                                                       n_iter=n_iter,
                                                       n_jobs=n_jobs,
                                                       n_splits=n_splits,
                                                       target_name=target_name,
                                                       seed=seed,
                                                       verbose=verbose)
        comb_predictions_out.set_index('date', inplace=True)
    else:
        comb_predictions_list = []
        for y in tqdm(years,
                      disable=not verbose,
                      desc="annual training and prediction"):
            train_ys = df.loc[:str(y)]
            test_ys = df.loc[str(y + 1)]

            target_train = train_ys[target_name]
            train_ys = train_ys.drop(labels=target_name, axis=1)
            target_test = test_ys[target_name]
            test_ys = test_ys.drop(labels=target_name, axis=1)

            if comb_model == 'average':
                comb_predictions = test_ys.mean(axis=1)
            elif comb_model == 'median':
                comb_predictions = test_ys.median(axis=1)
            elif comb_model == 'bates_granger':
                test_ys_plus = pd.concat([pd.DataFrame(train_ys.iloc[-1]).T, test_ys])
                target_test_plus = pd.concat([pd.DataFrame(target_train.iloc[-1], columns=[target_name], index=[target_train.index[-1]]), pd.DataFrame(target_test)])
                m = train_ys.shape[1]
                l_m = np.ones((m, 1))

                comb_dict = {}
                for i in range(test_ys_plus.shape[0]):
                    y_pred = test_ys_plus.to_numpy()[i]
                    y_target = target_test_plus.values[i]

                    e = l_m.dot(y_target) - y_pred.reshape(-1, 1)
                    cov_e = e.dot(e.transpose())

                    if np.linalg.det(cov_e) != 0:
                        inv_cov_e = np.linalg.inv(cov_e)
                    else:
                        inv_cov_e = np.linalg.pinv(cov_e)

                    # bates and granger optimal weights
                    if np.linalg.det(l_m.transpose().dot(inv_cov_e).dot(l_m)) != 0:
                        w_star = np.linalg.inv(l_m.transpose().dot(inv_cov_e).dot(l_m)) * inv_cov_e.dot(l_m)
                    else:
                        w_star = np.linalg.pinv(l_m.transpose().dot(inv_cov_e).dot(l_m)) * inv_cov_e.dot(l_m)

                    if i + 1 <= test_ys_plus.reset_index().index[-1]:
                        y_pred_t1 = test_ys_plus.to_numpy()[i + 1]
                        comb_pred = y_pred_t1.dot(w_star)

                        comb_dict[i + 1] = {'date': test_ys_plus.iloc[i + 1].name,
                                            'return_direction': target_test_plus.iloc[i + 1].values[0],
                                            'prediction': comb_pred[0]}
                comb_predictions = pd.DataFrame(comb_dict).T
                comb_predictions = comb_predictions['prediction'].values
            else:
                raise Exception('Combination method ' + comb_model + ' not registered')

            comb_predictions_df = pd.DataFrame({'date': target_test.index,
                                                'return_direction': target_test,
                                                'prediction': comb_predictions})
            comb_predictions_df.set_index('date', inplace=True)
            comb_predictions_list.append(comb_predictions_df)

        comb_predictions_out = pd.concat(comb_predictions_list, axis=0)
    return comb_predictions_out























