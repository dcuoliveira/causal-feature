import numpy as np
import pandas as pd
from time import time
import shutil
import os
import sys
import inspect
import unittest
import statsmodels.formula.api as smf
from sklearn.model_selection import TimeSeriesSplit

currentdir = os.path.dirname(
    os.path.abspath(
        inspect.getfile(
            inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from src.feature_selection.sfi import get_sfi_scores  # noqa


def merge_market_gtrends(market, gtrends):
    merged = pd.merge_asof(market, gtrends, left_index=True, right_index=True)
    return merged.dropna()


class SFI_Tester(unittest.TestCase):
    @classmethod
    def setUp(cls):
        path_gt = os.path.join(parentdir,
                               "src", "data", "gtrends.csv")
        gtrends = pd.read_csv(path_gt)
        gtrends.loc[:, "date"] = pd.to_datetime(gtrends.date)
        gtrends = gtrends.set_index("date")
        path_m = os.path.join(parentdir,
                              "src", "data",
                              "toy", "ticker1.csv")
        name = path_m.split("/")[-1].split(".")[0]
        target_name = name.replace(" ", "_") + "_return"
        market = pd.read_csv(path_m)
        market = market.drop([0, 1], 0)
        market = market.rename(columns={"ticker": "date",
                                        name: target_name})
        market.loc[:, "date"] = pd.to_datetime(market.date)
        market.loc[:, target_name] = market[target_name].astype("float") / 100
        market = market.set_index("date")
        market = market["2000":"2010"]
        cls.merged = merge_market_gtrends(market, gtrends)
        cls.target_name = target_name

    def test_sfi_basic_run_reg(self):
        words = ["short selling", "texas", "return"]
        max_lag = 6
        n_splits = 4
        result = get_sfi_scores(merged_df=self.merged.copy(),
                                target_name=self.target_name,
                                words=words,
                                max_lag=max_lag,
                                verbose=False,
                                classification=False,
                                n_splits=n_splits)
        word, lag = result.feature[0].split("_")
        lag = int(lag)
        feature_name = "{}_{}".format(word, lag)
        self.merged.loc[:, "{}_{}".format(
            word, lag)] = self.merged[word].shift(lag)
        r2_OOS = []
        tscv = TimeSeriesSplit(n_splits=n_splits)
        for train_index, test_index in tscv.split(self.merged):
            formula = "{} ~ {}".format(self.target_name, feature_name)
            df_train = self.merged.iloc[train_index]
            df_test = self.merged.iloc[test_index]
            lr = smf.ols(formula=formula, data=df_train).fit()
            y_pred = lr.predict(df_test).values
            y_true = df_test[self.target_name].values
            num = np.sum((y_true - y_pred)**2)
            dem = np.sum((y_true)**2)
            r2 = 1 - (num / dem)
            r2_OOS.append(r2)

        calulated = np.mean(r2_OOS)
        function = result["feature_score"][0]
        self.assertTrue(
            np.isclose(
                calulated,
                function),
            "problem in calculating r2")

    def test_sfi_basic_run_class(self):
        words = ["short selling", "texas", "return"]
        max_lag = 6
        n_splits = 4
        result = get_sfi_scores(merged_df=self.merged.copy(),
                                target_name=self.target_name,
                                words=words,
                                max_lag=max_lag,
                                verbose=False,
                                classification=True,
                                n_splits=n_splits)
        self.assertEqual(result.iloc[0, 0], "return_4")
        self.assertAlmostEqual(result.iloc[0, 1], 0.507112, places=3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
