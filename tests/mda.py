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

from src.feature_selection.mda import get_mda_scores  # noqa
from src.data_mani.utils import merge_market_and_gtrends  # noqa
from src.data_mani.utils import target_ret_to_directional_movements  # noqa


class Test_mda(unittest.TestCase):
    def test_mda_reproducibility(self):
        path_m = os.path.join(parentdir,
                              "src", "data",
                              "toy", "ticker3.csv")
        train, _ = merge_market_and_gtrends(path_m,
                                            path_gt_list=[parentdir,
                                                          "src",
                                                          "data",
                                                          "gtrends.csv"],
                                            test_size=0.5)
        target_ret_to_directional_movements(train, y_name="target_return")
        words = ["bank", "BUY AND HOLD", "act", "DOW JONES"]
        results = get_mda_scores(merged_df=train,
                                 target_name="target_return",
                                 words=words,
                                 max_lag=5,
                                 random_state=114,
                                 n_splits=2,
                                 n_estimators=10,
                                 verbose=False)
        self.assertEqual(results.iloc[0, 0], 'bank_1')
        self.assertAlmostEqual(
            results.iloc[0, 1], 0.028174249182277657, places=3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
