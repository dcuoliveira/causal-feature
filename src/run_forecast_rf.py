import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from time import time
from data_mani.utils import merge_market_and_gtrends
from sklearn.ensemble import RandomForestRegressor
from prediction.util import new_r2, add_shift
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import make_scorer


# VARIABLES

N_ITER = 5
N_SPLITS = 5
N_JOBS = 2
TEST_SIZE = 0.5 # pct of the train/test split

# iNITIAL OBJECTS
init = time()

ticker_name =  "AMZN US Equity"
ticker_file ="{}.csv".format(ticker_name)
fs_method = "sfi"
out_folder = "nasdaq"

model_name = "random_forest"

model_class = RandomForestRegressor()

model_params =  {"max_features":list(range(1, int(np.sqrt(3640)+1))),
                 "n_estimators": list(range(2, 60)),
                 "max_depth": list(range(2, 21))}


# GET SCORES
score_path = ["results", "feature_selection", fs_method,
              out_folder, ticker_file]
score_path = os.path.join(*score_path)
scores =  pd.read_csv(score_path)
cut = scores.feature_score.mean()
scores = (scores.loc[scores.feature_score > cut]).feature.to_list()

# GET MERGED DATAFRAME
ticker_path = ["data", "crsp", out_folder, ticker_file]
ticker_path = os.path.join(*ticker_path)
train, test = merge_market_and_gtrends(ticker_path, test_size=TEST_SIZE)


# ADD WORDS AND LAGS
words = train.drop("target_return",1).columns.to_list()

add_shift(merged_df=train, words=words, max_lag=20)
train = train[["target_return"] + scores]
train = train.fillna(0)

add_shift(merged_df=test, words=words, max_lag=20)
test = test[["target_return"] + scores]
test = test.fillna(0)

# GET TRAIN/TEST DATA MATRIX
X_train = train.drop("target_return",1).values
y_train = train.target_return.values

X_test = test.drop("target_return",1).values
y_test = test.target_return.values


# PERFORM RANDOM SEARCH
time_split = TimeSeriesSplit(n_splits=N_SPLITS)
r2_scorer = make_scorer(new_r2)

model_search = RandomizedSearchCV(estimator=model_class,
                               param_distributions=model_params,
                               n_iter=N_ITER,
                               cv=time_split,
                               verbose=1,
                               n_jobs=N_JOBS,
                               scoring=r2_scorer)
model_search = model_search.fit(X_train,y_train)

# PERFORM PREDICTION
train_pred  = model_search.best_estimator_.predict(X_train)
test_pred = model_search.best_estimator_.predict(X_test)

# SAVE FORECAST
train.loc[:, "prediction"] = train_pred
train.loc[:, "type"] = "train"
test.loc[:, "prediction"] = test_pred
test.loc[:, "type"] = "test"

both = [train[["target_return", "prediction", "type"]],
        test[["target_return", "prediction", "type"]]]

complete_forecast = pd.concat(both)
out_path = ["results", "forecast",fs_method,model_name,out_folder, ticker_file]
out_path = os.path.join(*out_path)
complete_forecast.to_csv(out_path)

