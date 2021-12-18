import os
from prediction.functions import forecast_comb

comb_models = ['nncomb'] # ['average', 'median', 'bates_granger', 'nncomb']
models = ['logit', 'ridge', 'lasso', 'enet', 'random_forest', 'lgb', 'nn3']
fs_methods = ['all', 'sfi', 'mdi', 'mda', 'granger', 'huang', 'IAMB', 'MMMB']
tickers = ['SPX Index', 'CCMP Index','RTY Index', 'SPX Basic Materials',
           'SPX Communications', 'SPX Consumer Cyclical',
           'SPX Consumer Non cyclical', 'SPX Energy', 'SPX Financial',
           'SPX Industrial', 'SPX Technology', 'SPX Utilities']
metric = "auc"
start_date = '2005-01-03'
end_date = '2020-12-31'

if __name__ == '__main__':
    for f in fs_methods:
        for c in comb_models:
            for t in tickers:
                pred_results = forecast_comb(ticker_name=t,
                                             fs_method=f,
                                             model=models,
                                             start_date=start_date,
                                             end_date=end_date,
                                             metric_name=metric,
                                             comb_model=c,
                                             target_name='target_return',
                                             benchmark_name='return_direction')
                pred_results.to_csv(os.path.join("results",
                                                 "forecast",
                                                 f,
                                                 "indices",
                                                 c,
                                                 t + ".csv"))
