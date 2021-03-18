import os
import numpy as np
import pandas as pd
from data_mani.utils import get_market_df
from tqdm import tqdm
from glob import glob
from time import time
from data_mani.utils import merge_market_and_gtrends
from prediction.models import RandomForestWrapper
from prediction.functions import forecast


pred_results = forecast(ticker_name="SPX Financial",
                        fs_method="sfi",
                        Wrapper=RandomForestWrapper,
                        n_iter=10,
                        n_splits=5,
                        n_jobs=-1,
                        verbose=1)

print(pred_results.head())