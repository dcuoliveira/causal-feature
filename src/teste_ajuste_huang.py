from feature_selection.huang import run_huang_methods
from data_mani.utils import merge_market_and_gtrends
from word_list.analysis import words

test_size = 0.5
max_lag = 20
sig_level = 0.05
path = 'data/crsp/nyse/0544801D US Equity.csv'
merged, _ = merge_market_and_gtrends(path, test_size=test_size)
result = run_huang_methods(merged_df=merged, target_name="target_return",
                           words=words, max_lag=max_lag, verbose=False,
                           sig_level=sig_level, correl_threshold=0.8)