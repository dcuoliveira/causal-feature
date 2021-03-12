import os
import pandas as pd
import numpy as np
from time import time

from data_mani.utils import merge_market_and_gtrends
from prediction.models import RandomForestWrapper
from prediction.functions import new_forecast

if __name__ == '__main__':
    
    init = time()
    pred_results = new_forecast(ticker_name="SBUX UA Equity",
                                fs_method="sfi",
                                step_size=2,
                                Wrapper=RandomForestWrapper,
                                n_iter=10,
                                n_splits=3,
                                n_jobs=-1,
                                verbose=1)
    tempo = (time() - init) / 60
    print("total run time = ", np.round(tempo, 2), "min")    