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
from src.feature_selection.huang import run_huang_methods  # noqa


words = ['BUY AND HOLD',
         'DOW JONES',
         'act',
         'arts',
         'bank',
         'banking',
         'blacklist',
         'bonds',
         'bubble',
         'business']


class Test_huang(unittest.TestCase):
    def test_huang_basic1(self):
        path_m = os.path.join(parentdir,
                              "src", "data",
                              "crsp", "nyse",
                              "0700161D US Equity.csv")
        merged, _ = merge_market_and_gtrends(path_m,
                                             test_size=0.5,
                                             path_gt_list=[parentdir,
                                                           "src",
                                                           "data",
                                                           "gtrends.csv"])
        result = run_huang_methods(merged_df=merged, target_name="target_return",
                                   words=words, max_lag=20, verbose=False,
                                   sig_level=0.05)
        result = list(result[~result.feature_score.isna()].feature)

        self.assertTrue('banking_1' in result)
        self.assertTrue('bonds_1' in result)
        self.assertTrue(len(result) == 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
