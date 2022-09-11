import pandas as pd
import numpy as np
from glob import glob
from data_mani.visu import *
from prediction.functions import *

PREDICTION_MODEL = ['logit']
FS_METHODS = ['all', 'granger', 'huang', 'IAMB', 'mda', 'mdi', 'MMMB', 'sfi']
EVALUATION_START = '2012-07-03'
TICKER_NAMES = ['SPX Index', 'CCMP Index', 'RTY Index']
TITLE = 'OOS Cummulative Returns for each Feature Selection Method given a Prediction Model'
BENCHMARK_NAME = 'return_direction'
METRIC_NAME = 'auc'

if __name__ == '__main__':
    oos_melt_predictions_df, oos_melt_benchmark_df, oos_melt_auc_df = aggregate_prediction_results(prediction_models=PREDICTION_MODEL,
                                                                                                   fs_models=FS_METHODS,
                                                                                                   evaluation_start_date='2012-07-03',
                                                                                                   evaluation_end_date='2020-12-31',
                                                                                                   ticker_names=TICKER_NAMES,
                                                                                                   metric_name=METRIC_NAME,
                                                                                                   tag='oos',
                                                                                                   benchmark_name=BENCHMARK_NAME)
    
    oos_pred_ret_df, oos_pred_pos_df = gen_strat_positions_and_ret_from_pred(predictions_df=oos_melt_predictions_df,
                                                                             class_threshold=0.5,
                                                                             target_asset_returns=oos_melt_benchmark_df)
    
    benchmarks = glob('data/indices/*.csv')
    bench_list = []
    for b in benchmarks:
        ticker = b.replace('data/indices/', '').replace('.csv', '')
        bench_ret_df = pd.read_csv(b)[3:]
        bench_ret_df.columns = ['date', 'return']
        bench_ret_df = bench_ret_df.melt('date')
        bench_ret_df['model'] = bench_ret_df['ticker'] = ticker
        bench_ret_df['fs'] = 'raw'

        bench_list.append(bench_ret_df)
    benchmark_df = pd.concat(bench_list, axis=0)

    benchmark_df['value'] = benchmark_df['value'].astype(float)
    cum_ret_df2 = plot_cum_ret(pred_ret_df=oos_pred_ret_df,
                               benchmark_df=benchmark_df,
                               level_to_subset='fs',
                               show=True)
    
    
    