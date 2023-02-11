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
fs_method = "granger"
init_steps=252
predict_steps=5
Wrapper = LogisticRegWrapper
model_name = Wrapper().model_name
n_iter = 1
n_splits = 5
n_jobs = 1
verbose = False
seed = 2294
dynamic_fs = True

if __name__ == '__main__':
    init = time()
    pred_results, fs_results= forecast(ticker_name=ticker_name,
                                       fs_method=fs_method,
                                       init_steps=init_steps,
                                       predict_steps=predict_steps,
                                       Wrapper=Wrapper,
                                       n_iter=n_iter,
                                       n_splits=n_splits,
                                       n_jobs=n_jobs,
                                       dynamic_fs=dynamic_fs,
                                       verbose=verbose,
                                       seed=seed)

    # saving forecast on the results folder
    out_path_list = [os.path.dirname(__file__),
                     "results",
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

    # saving features on the results folder
    out_path_list = [os.path.dirname(__file__),
                     "results",
                     "features",
                     fs_method,
                     "indices",
                     model_name,
                     "{}.csv".format(ticker_name)]
    out_folder = os.path.join(*out_path_list[:-1])
    out_path = os.path.join(*out_path_list)
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    fs_results.to_csv(out_path, index=False)

    tempo = (time() - init) / 60
    print("\nDONE\ntotal run time = ", np.round(tempo, 2), "min")
