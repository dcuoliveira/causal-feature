import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
import os
import sys
import inspect
import unittest

currentdir = os.path.dirname(
    os.path.abspath(
        inspect.getfile(
            inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from src.feature_selection.mdi import mdi_feature_importance  # noqa
from src.feature_selection.sfi import single_feature_importance_cv  # noqa
from src.feature_selection.mda import mean_decrease_accuracy  # noqa


class Sanity_check_test_feature_selection(unittest.TestCase):
    def test_fs_artifcial_dataset(self):

        # creating artificial dataset
        n_features = 4
        n_informative = 3
        top_n = n_features
        X, y, coef = make_regression(n_samples=5000,
                                     n_features=n_features,
                                     n_informative=n_informative,
                                     random_state=1233,
                                     coef=True)
        feature_names = ["f{}".format(i) for i in range(n_features)]
        columns = feature_names + ["target_return"]

        y = y.reshape(-1, 1)
        df = pd.DataFrame(np.hstack([X, y]), columns=columns)
        true_imp = pd.DataFrame({"feature": feature_names,
                                 "feature_score": coef}).sort_values("feature_score",
                                                                     ascending=False).reset_index(drop=True)

        # SFI test
        sfi_results = []

        for f in feature_names:
            r2_arr = single_feature_importance_cv(df=df,
                                                  feature_name=f,
                                                  target_name="target_return",
                                                  n_splits=5)
            sfi_results.append((f, np.mean(r2_arr)))
        sfi_results = pd.DataFrame(sfi_results,
                                   columns=["feature", "feature_score"])
        sfi_results = sfi_results.sort_values("feature_score",
                                              ascending=False)
        sfi_results = sfi_results.reset_index(drop=True)
        self.assertTrue(np.all(true_imp.feature.head()
                               == sfi_results.feature.head()))

        # MDI test
        mdi_result = mdi_feature_importance(df=df,
                                            feature_names=feature_names,
                                            target_name="target_return",
                                            random_state=12)

        mdi_result = mdi_result.sort_values("mean", ascending=False)["mean"]
        mdi_result = mdi_result.reset_index()
        mdi_result.columns = ["feature", "feature_score"]
        self.assertTrue(np.all(true_imp.feature.head()
                               == mdi_result.feature.head()))

        # MDA test
        y_class = np.where(y < y.mean(), 0, 1)
        y_class = y_class.astype(int)
        df = pd.DataFrame(np.hstack([X, y_class]), columns=columns)
        mda_results = mean_decrease_accuracy(df=df,
                                             feature_names=feature_names,
                                             target_name="target_return",
                                             random_state=27973,
                                             n_splits=2)

        self.assertTrue(np.all(true_imp.feature == mda_results.feature))


if __name__ == '__main__':
    unittest.main(verbosity=2)
