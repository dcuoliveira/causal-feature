import pandas as pd
import os
import sys
import inspect
import unittest
from glob import glob
from sklearn.metrics import roc_auc_score
from warnings import simplefilter

currentdir = os.path.dirname(
    os.path.abspath(
        inspect.getfile(
            inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from src.data_mani.utils import merge_market_and_gtrends  # noqa
from src.prediction.models import RandomForestWrapper  # noqa
from src.prediction.models import LogisticRegWrapper  # noqa
from src.prediction.models import LassoWrapper  # noqa
from src.prediction.models import RidgeWrapper  # noqa
from src.prediction.models import ElasticNetWrapper  # noqa
from src.prediction.models import LGBWrapper  # noqa
from src.prediction.models import NN3Wrapper  # noqa
from src.prediction.functions import get_selected_features  # noqa
from src.prediction.functions import get_features_granger_huang  # noqa
from src.prediction.functions import get_features_IAMB_MMMB  # noqa
from src.prediction.functions import add_shift, annualy_fit_and_predict  # noqa


class Test_forecast(unittest.TestCase):

    @classmethod
    def setUp(cls):
        cls.ticker_name = "SPX Utilities"
        market_folder = "toy"
        cls.path_gt_list = [parentdir,
                            "src",
                            "data",
                            "gtrends.csv"]
        path_t_list = [parentdir,
                       "src", "data",
                       market_folder,
                       "{}.csv".format(cls.ticker_name)]
        ticker_path = os.path.join(*path_t_list)
        max_lag = 20
        verbose = False
        cls.target_name = "target_return"
        train, test = merge_market_and_gtrends(ticker_path,
                                               test_size=0.5,
                                               path_gt_list=cls.path_gt_list)
        words = train.drop(cls.target_name, 1).columns.to_list()
        complete = pd.concat([train, test])
        del train, test
        add_shift(merged_df=complete,
                  words=words,
                  max_lag=max_lag,
                  verbose=verbose)
        cls.complete = complete.fillna(0.0)

    def test_forecast_auc(self):
        fs_method = "mdi"
        select = get_selected_features(ticker_name=self.ticker_name,
                                       out_folder="indices",
                                       fs_method=fs_method,
                                       path_list=self.path_gt_list)
        complete_selected = self.complete[[self.target_name] + select]

        pred_results = annualy_fit_and_predict(df=complete_selected,
                                               Wrapper=NN3Wrapper,
                                               n_iter=2,
                                               n_jobs=2,
                                               n_splits=2,
                                               seed=44,
                                               target_name=self.target_name,
                                               verbose=False)
        y_true = pred_results["return_direction"]
        y_pred = pred_results["prediction"]
        auc = roc_auc_score(y_true, y_pred)
        self.assertTrue(0.0 < auc < 1.0)

    def test_forecast_models(self):
        fs_method = "mdi"
        select = get_selected_features(ticker_name=self.ticker_name,
                                       out_folder="indices",
                                       fs_method=fs_method,
                                       path_list=self.path_gt_list)
        complete_selected = self.complete[[self.target_name] + select[:2]]
        linear_models = [LogisticRegWrapper, LassoWrapper,
                         RidgeWrapper, ElasticNetWrapper]
        for Wrapper in linear_models:
            pred_results = annualy_fit_and_predict(df=complete_selected,
                                                   Wrapper=Wrapper,
                                                   n_iter=1,
                                                   n_jobs=2,
                                                   n_splits=2,
                                                   seed=123,
                                                   target_name=self.target_name,
                                                   verbose=False)

            self.assertTrue(2 < complete_selected.shape[1] < 3641)
            self.assertEqual(pred_results.shape[0],
                             complete_selected.loc["2005":].shape[0])

    def test_forecast_ml_method(self):
        fs_method = "mdi"
        select = get_selected_features(ticker_name=self.ticker_name,
                                       out_folder="indices",
                                       fs_method=fs_method,
                                       path_list=self.path_gt_list)
        complete_selected = self.complete[[self.target_name] + select]
        pred_results = annualy_fit_and_predict(df=complete_selected,
                                               Wrapper=LGBWrapper,
                                               n_iter=1,
                                               n_jobs=2,
                                               n_splits=2,
                                               seed=123,
                                               target_name=self.target_name,
                                               verbose=False)

        self.assertTrue(2 < complete_selected.shape[1] < 3641)
        self.assertEqual(pred_results.shape[0],
                         complete_selected.loc["2005":].shape[0])

    def test_forecast_granger_huang(self):
        fs_method = "granger"
        select = get_features_granger_huang(ticker_name=self.ticker_name,
                                            out_folder="indices",
                                            fs_method=fs_method,
                                            path_list=self.path_gt_list)

        complete_selected = self.complete[[self.target_name] + select]
        pred_results = annualy_fit_and_predict(df=complete_selected,
                                               Wrapper=RandomForestWrapper,
                                               n_iter=1,
                                               n_jobs=2,
                                               n_splits=2,
                                               seed=123,
                                               target_name=self.target_name,
                                               verbose=False)

        self.assertTrue(2 < complete_selected.shape[1] < 3641)
        self.assertEqual(pred_results.shape[0],
                         complete_selected.loc["2005":].shape[0])

    def test_forecast_IAMB_MMB(self):
        fs_method = "IAMB"
        select = get_features_IAMB_MMMB(ticker_name=self.ticker_name,
                                        out_folder="indices",
                                        fs_method=fs_method,
                                        path_list=self.path_gt_list)
        complete_selected = self.complete[[self.target_name] + select]
        pred_results = annualy_fit_and_predict(df=complete_selected,
                                               Wrapper=RandomForestWrapper,
                                               n_iter=1,
                                               n_jobs=2,
                                               n_splits=2,
                                               seed=123,
                                               target_name=self.target_name,
                                               verbose=False)

        self.assertTrue(2 < complete_selected.shape[1] < 3641)
        self.assertEqual(pred_results.shape[0],
                         complete_selected.loc["2005":].shape[0])


if __name__ == '__main__':
    unittest.main(verbosity=2)
