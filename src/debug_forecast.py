import os
import pandas as pd
import numpy as np
import argparse
from time import time

from prediction.models import RandomForestWrapper
from prediction.models import LassoWrapper
from prediction.models import RidgeWrapper
from prediction.models import ElasticNetWrapper
from prediction.models import LGBWrapper
from prediction.models import NN3Wrapper
from prediction.models import LogisticRegWrapper
from prediction.functions import forecast

ticker_name = "SPX Index"
fs_method = "all"
Wrapper = LogisticRegWrapper
model_name = Wrapper().model_name
n_iter = 50
n_splits = 5
n_jobs = 1
verbose = True
seed = 2294

if __name__ == '__main__':
    init = time()
    pred_results = forecast(ticker_name=ticker_name,
                            fs_method=fs_method,
                            Wrapper=Wrapper,
                            n_iter=n_iter,
                            n_splits=n_splits,
                            n_jobs=n_jobs,
                            verbose=verbose,
                            seed=seed)

    # saving forecast on the results folder
    out_path_list = ["results",
                     "forecast",
                     fs_method,
                     "indices",
                     model_name,
                     "{}.csv".format(ticker_name)]
    out_folder = os.path.join(*out_path_list[:-1])
    out_path = os.path.join(*out_path_list)
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    pred_results.to_csv(out_path, index=False)

    tempo = (time() - init) / 60
    print("\nDONE\ntotal run time = ", np.round(tempo, 2), "min")
