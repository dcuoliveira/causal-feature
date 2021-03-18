import pandas as pd
import os
import sys
import inspect
import unittest
from glob import glob

currentdir = os.path.dirname(
    os.path.abspath(
        inspect.getfile(
            inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from src.data_mani.utils import merge_market_and_gtrends  # noqa
from src.prediction.models import RandomForestWrapper  # noqa
from src.prediction.functions import forecast  # noqa


class Test_forecast(unittest.TestCase):
    def test_ticker_with_all_years(self):
        path_gt_list = [parentdir,
                        "src",
                        "data",
                        "gtrends.csv"]
        ticker_name = 'SPX_Financial'

        path_t_list = [parentdir,
                        "src", "data",
                        "indices",
                        "{}.csv".format(ticker_name)]
        ticker1_path = os.path.join(*path_t_list)
        train, test = merge_market_and_gtrends(ticker1_path,
                                               test_size=0.5,
                                               path_gt_list=path_gt_list)
        complete = pd.concat([train, test])
        print(complete.shape)
        exit()
        pred_results = forecast(ticker_name=ticker_name,
                                fs_method="sfi",
                                Wrapper=RandomForestWrapper,
                                n_iter=1,
                                n_splits=2,
                                n_jobs=2,
                                verbose=0,
                                path_list=path_gt_list)
        self.assertTrue(complete["2005":].shape[0] == pred_results.shape[0])

if __name__ == '__main__':
    unittest.main(verbosity=2)
