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

from src.feature_selection.mdi import get_mdi_scores  # noqa
from src.data_mani.utils import merge_market_and_gtrends  # noqa


class Test_MDI(unittest.TestCase):
    def test_mdi_reproducibility(self):
        path_m = os.path.join(parentdir,
                              "src", "data",
                              "toy", "ticker2.csv")
        train, test = merge_market_and_gtrends(path_m,
                                               path_gt_list=[parentdir,
                                                             "src",
                                                             "data",
                                                             "gtrends.csv"],
                                               test_size=0.5)
        result = get_mdi_scores(merged_df=train,
                                target_name="target_return",
                                words=["bank", "return", "food"],
                                max_lag=30,
                                random_state=2927,
                                verbose=False,
                                classification=False)

        self.assertAlmostEqual(
            result.iloc[0, 1], 0.017442061511909843, places=3)

    def test_mdi_classifiction(self):
        path_m = os.path.join(parentdir,
                              "src", "data",
                              "toy", "ticker2.csv")
        train, test = merge_market_and_gtrends(path_m,
                                               path_gt_list=[parentdir,
                                                             "src",
                                                             "data",
                                                             "gtrends.csv"],
                                               test_size=0.5)
        result = get_mdi_scores(merged_df=train,
                                target_name="target_return",
                                words=["bank", "return", "food"],
                                max_lag=30,
                                random_state=2927,
                                verbose=False,
                                classification=True)

        self.assertEqual(result.iloc[0, 0], "bank_4")
        self.assertAlmostEqual(
            result.iloc[0, 1], 0.015275348798034095, places=3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
