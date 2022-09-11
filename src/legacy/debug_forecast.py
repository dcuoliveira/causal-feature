import os
import pandas as pd
import numpy as np
from time import time

from data_mani.utils import merge_market_and_gtrends
from prediction.models import RandomForestWrapper, LassoWrapper, LogisticRegWrapper
from prediction.functions import forecast

if __name__ == '__main__':
    
    init = time()
    pred_results = forecast(ticker_name="SPX Index",
                            fs_method="sfi",
                            Wrapper=LogisticRegWrapper,
                            n_iter=10,
                            n_splits=5,
                            n_jobs=-1,
                            seed=2294,
                            verbose=1)
    pred_results.to_csv(os.path.join("results",
                                     "forecast",
                                     "sfi",
                                     "indices",
                                     "SPX Index.csv"))
    tempo = (time() - init) / 60
    print("total run time = ", np.round(tempo, 2), "min")  