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


class Test_merger(unittest.TestCase):
    def test_market_gtrend_merge(self):
        path_m = os.path.join(parentdir,
                              "src", "data",
                              "crsp", "nyse",
                              "0062761Q US Equity.csv")
        for size in [0.1, 0.25, 0.5, 0.75, 0.9]:
            train, test = merge_market_and_gtrends(path_m,
                                                   path_gt_list=[parentdir,
                                                                 "src",
                                                                 "data",
                                                                 "gtrends.csv"],
                                                   test_size=size)

            train_shape = train.shape
            test_shape = test.shape
            last_day_train = train.sort_index().index[-1]
            first_day_test = test.sort_index().index[0]

            self.assertTrue(train_shape[0] > 0)
            self.assertTrue(train_shape[1] == 183)
            self.assertTrue(test_shape[0] > 0)
            self.assertTrue(test_shape[1] == 183)
            self.assertTrue(train.index[0] > pd.Timestamp("2003-12-01"))
            self.assertTrue(last_day_train < first_day_test)


if __name__ == '__main__':
    unittest.main(verbosity=2)
