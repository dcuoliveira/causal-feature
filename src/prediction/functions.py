import os
from glob import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from data_mani.visu import *

try:
    from data_mani.utils import merge_market_and_gtrends
except ModuleNotFoundError:
    from src.data_mani.utils import merge_market_and_gtrends
    
 
def sharpe_ratio_tb(returns_df,
                    rf=.0):
    
    ranking = (((pivot_metrics_df.mean() - rf) / pivot_metrics_df.std()) * np.sqrt(252)).sort_values(ascending=False)
    pivot_tb = (((pivot_metrics_df.mean() - rf) / pivot_metrics_df.std()) * np.sqrt(252)).reset_index().pivot_table(index=['fs'], columns=['model'], values=[0])

    return ranking, pivot_tb.style.apply(highlight_max)

def aggregate_prediction_positions(predictions_df,
                                   forecast_model,
                                   benchmark_name,
                                   benchmark_alias):

    benchmark_buynhold_df = predictions_df.loc[(predictions_df['model'] == forecast_model[0])&
                                               (predictions_df['variable'] == benchmark_name)&
                                               (predictions_df['fs'] == 'all')]
    benchmark_buynhold_df = benchmark_buynhold_df.pivot_table(index=['date'], columns=['variable'], values=['value'])
    melt_benchmark_buynhold_df = benchmark_buynhold_df.reset_index().melt('date')
    melt_benchmark_buynhold_df.rename(columns={melt_benchmark_buynhold_df.columns[1]: 'model',
                                            melt_benchmark_buynhold_df.columns[2]: 'fs'}, inplace=True)

    metrics_df = []
    positions_df = []
    for model in forecast_model:
        _, _, _, positons_fs_model_df = plot_ret_from_predictions(predictions_df=predictions_df,
                                                                  forecast_model=model,
                                                                  benchmark_name=benchmark_name,
                                                                  benchmark_alias=benchmark_alias)
        positions_df.append(positons_fs_model_df.reset_index().melt('date'))

    positions_df = pd.concat(positions_df, axis=0)
    pivot_positions_df = positions_df.pivot_table(index=['date'], columns=['model', 'fs'], values=['value'])
    metrics_ex_ret_df = pd.concat(metrics_df, axis=0)
    metrics_df = pd.concat([metrics_ex_ret_df, melt_benchmark_buynhold_df], axis=0)
    pivot_metrics_df = metrics_df.pivot_table(index=['date'], columns=['model', 'fs'], values=['value'])
    
    return pivot_metrics_df
 
 
def aggregate_prediction_results(prediction_models,
                                 fs_models,
                                 evaluation_start_date,
                                 ticker_name):
    predictions = []
    r2s = []

    for model in prediction_models:
        for fs in fs_models:
            assets = glob('results/forecast/' + fs + '/indices/' + model + '/' + ticker_name + '.csv')
            for asset_path in assets:
                df = pd.read_csv(asset_path)
                df.set_index('date', inplace=True)
                df = df.loc[evaluation_start_date:]
                df = df.reset_index()

                r2_eval_df = df.copy()
                r2 = new_r2(r2_eval_df['return'].values, r2_eval_df['prediction'].values)
                r2_df = pd.DataFrame([{'model': model,
                                    'fs': fs,
                                    'r2': r2}])
                r2s.append(r2_df)

                melt_df = df.melt('date')
                melt_df['model'] = model
                melt_df['fs'] = fs
                predictions.append(melt_df)

    predictions_df = pd.concat(predictions, axis=0)
    r2_df = pd.concat(r2s, axis=0)
    
    return predictions_df, r2_df


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


def get_selected_features(ticker_name, out_folder, fs_method, path_list):
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


def new_r2(y_true, y_pred):
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


def add_shift(merged_df, words, max_lag, verbose):
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

    X = df.drop(target_name, 1).values
    y = df[target_name].values

    time_split = TimeSeriesSplit(n_splits=n_splits)
    r2_scorer = make_scorer(new_r2)

    if wrapper.search_type == 'random':
        model_search = RandomizedSearchCV(estimator=wrapper.ModelClass,
                                          param_distributions=wrapper.param_grid,
                                          n_iter=n_iter,
                                          cv=time_split,
                                          verbose=verbose,
                                          n_jobs=n_jobs,
                                          scoring=r2_scorer,
                                          random_state=seed)
    elif wrapper.search_type == 'grid':
        model_search = GridSearchCV(estimator=wrapper.ModelClass,
                                    param_grid=wrapper.param_grid,
                                    cv=time_split,
                                    verbose=verbose,
                                    n_jobs=n_jobs,
                                    scoring=r2_scorer)
    else:
        raise Exception('search type method not registered')

    model_search = model_search.fit(X, y)

    return model_search


def annualy_fit_and_predict(df,
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

    years = df.index.map(lambda x: x.year)
    years = range(np.min(years), np.max(years))
    features = sorted(df.drop(target_name, 1).columns.to_list())
    df = df[features + [target_name]]

    for y in tqdm(years,
                  disable=not verbose,
                  desc="anual training and prediction"):
        train_ys = df.loc[:str(y)]
        test_ys = df.loc[str(y + 1)]

        scaler = StandardScaler()
        train_ys_v = scaler.fit_transform(train_ys)
        train_ys = pd.DataFrame(train_ys_v,
                                columns=train_ys.columns,
                                index=train_ys.index)
        y_test = test_ys[target_name].values
        test_ys_v = scaler.transform(test_ys)
        test_ys = pd.DataFrame(test_ys_v,
                               columns=test_ys.columns,
                               index=test_ys.index)

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
            X_test = test_ys.drop(target_name, 1).values
            test_pred = model_search.best_estimator_.predict(X_test)
            X_to_go = np.hstack([X_test, test_pred.reshape(-1, 1)])
            X_to_go = scaler.inverse_transform(X_to_go)
            df_to_go = pd.DataFrame(X_to_go,
                                    columns=test_ys.columns,
                                    index=test_ys.index)
            test_pred = df_to_go[target_name].values

            dict_ = {"date": test_ys.index,
                     "return": y_test,
                     "prediction": test_pred}
            result = pd.DataFrame(dict_)
            all_preds.append(result)
        else:
            pass

    pred_results = pd.concat(all_preds).reset_index(drop=True)
    return pred_results


def forecast(ticker_name,
             fs_method,
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
    ticker_path = "data/indices/{}.csv".format(ticker_name)
    train, test = merge_market_and_gtrends(
        ticker_path, test_size=0.5)
    words = train.drop(target_name, 1).columns.to_list()
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
        select = complete.drop(words + [target_name], 1).columns.to_list()

    complete_selected = complete[[target_name] + select]

    pred_results = annualy_fit_and_predict(df=complete_selected,
                                           Wrapper=Wrapper,
                                           n_iter=n_iter,
                                           n_jobs=n_jobs,
                                           n_splits=n_splits,
                                           target_name=target_name,
                                           seed=seed,
                                           verbose=verbose)

    return pred_results
