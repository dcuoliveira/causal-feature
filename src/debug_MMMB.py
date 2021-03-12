from glob import glob
from data_mani.utils import path_filter
from word_list.analysis import words
from data_mani.utils import merge_market_and_gtrends
from data_mani.utils import get_ticker_name
from feature_selection.MMMB import run_MMMB

THRESHOLD = 252 * 2
TEST_SIZE = 0.5
IS_DISCRETE = False
SIG_LEVEL = 0.01
MAX_LAG = 20
CONSTANT_THRESHOLD = 0.9

if __name__ == '__main__':
    result_tickers = glob("results/feature_selection/MMMB/spx/*.csv")
    result_tickers = [r.replace('results/feature_selection/MMMB/spx/', '') for r in result_tickers]

    all_tickers = glob("data/index/spx/*.csv")
    all_tickers = [r.replace('data/index/spx/', '') for r in all_tickers]
    
    error_tickers = [r for r in all_tickers if r not in result_tickers]
    error_tickers = ['data/index/spx/' + r for r in error_tickers]
    
    paths = path_filter(paths=error_tickers,
                         threshold=THRESHOLD)
     
    path = paths[0]

    merged, _ = merge_market_and_gtrends(path, test_size=TEST_SIZE)

    name = get_ticker_name(path).replace("_", " ")
    result = run_MMMB(merged_df=merged,
                        target_name="target_return",
                        words=words,
                        max_lag=MAX_LAG,
                        verbose=False,
                        sig_level=SIG_LEVEL,
                        is_discrete=IS_DISCRETE,
                        constant_threshold=CONSTANT_THRESHOLD)
    a=1
    