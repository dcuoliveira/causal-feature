import os
import pandas as pd
import numpy as np
import argparse
from time import time

from prediction.models import RandomForestWrapper
from prediction.models import LassoWrapper
from prediction.models import RidgeWrapper
from prediction.models import ElasticNetWrapper
from prediction.models import XGBWrapper
from prediction.functions import forecast


def main():
    """
    Script to run ticker forecast using one FS method and one ML model.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('ticker_name',
                        type=str, help='ticker name (without extension)')
    parser.add_argument('fs_method',
                        type=str, help='name of the feature selection method')
    parser.add_argument('model_name',
                        type=str, help='name of the ML model')

    parser.add_argument('-i',
                        '--n_iter',
                        type=int,
                        help='number of hyperparameter searchs, (default=50)',
                        default=50)
    parser.add_argument('-s',
                        '--n_splits',
                        type=int,
                        help='number of splits for the cross-validation, (default=5)',
                        default=5)
    parser.add_argument('-j',
                        '--n_jobs',
                        type=int,
                        help='number of concurrent workers, (default=-1)',
                        default=-1)
    parser.add_argument('-S',
                        '--seed',
                        type=int,
                        help='random seed, (default=None)',
                        default=None)
    parser.add_argument("-v",
                        "--verbose",
                        action="store_true",
                        default=True,
                        help="print training results stages (default=True)")  # noqa
    args = parser.parse_args()

    # selecting the ML model
    model_dict = {"random_forest": RandomForestWrapper,
                  "lasso": LassoWrapper,
                  "ridge": RidgeWrapper,
                  "enet": ElasticNetWrapper,
                  "xgb": XGBWrapper}

    assert args.model_name in model_dict, "no model with this name"
    Wrapper = model_dict[args.model_name]

    init = time()
    pred_results = forecast(ticker_name=args.ticker_name,
                            fs_method=args.fs_method,
                            Wrapper=Wrapper,
                            n_iter=args.n_iter,
                            n_splits=args.n_splits,
                            n_jobs=args.n_jobs,
                            verbose=args.verbose,
                            seed=args.seed)

    # saving forecast on the results folder
    out_path_list = ["results", "forecast", args.fs_method, "indices",
                     args.model_name, "{}.csv".format(args.ticker_name)]
    out_folder = os.path.join(*out_path_list[:-1])
    out_path = os.path.join(*out_path_list)
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    pred_results.to_csv(out_path, index=False)

    tempo = (time() - init) / 60
    print("\nDONE\ntotal run time = ", np.round(tempo, 2), "min")
if __name__ == '__main__':
    main()
