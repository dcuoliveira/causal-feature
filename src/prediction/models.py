from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
import numpy as np
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


class RandomForestWrapper():
    def __init__(self, model_params=None):
        self.model_name = "random_forest"
        self.search_type = 'random'
        self.param_grid = {"max_features": ['auto', 'sqrt', 'log2'],
                           "min_samples_split": sp_randint(2, 31),
                           "n_estimators": sp_randint(2, 301),
                           "max_depth": sp_randint(2, 20)}
        if model_params is None:
            self.ModelClass = RandomForestRegressor()
        else:
            self.ModelClass = RandomForestRegressor(**model_params)


class LassoWrapper():
    def __init__(self, model_params={'fit_intercept': True}):
        self.model_name = "lasso"
        self.search_type = 'random'
        self.param_grid = {'alpha': np.linspace(0, 1, 100)}
        if model_params is None:
            self.ModelClass = Lasso()
        else:
            self.ModelClass = Lasso(**model_params)


class RidgeWrapper():
    def __init__(self, model_params={'fit_intercept': True}):
        self.model_name = "ridge"
        self.search_type = 'random'
        self.param_grid = {'alpha': np.linspace(0, 1, 100)}
        if model_params is None:
            self.ModelClass = Ridge()
        else:
            self.ModelClass = Ridge(**model_params)


class ElasticNetWrapper():
    def __init__(self, model_params={'fit_intercept': True}):
        self.model_name = "elastic_net"
        self.search_type = 'random'
        self.param_grid = {'alpha': np.linspace(0, 1, 100),
                           'l1_ratio': np.linspace(0, 1, 100)}
        if model_params is None:
            self.ModelClass = ElasticNet()
        else:
            self.ModelClass = ElasticNet(**model_params)


class XGBWrapper():
    def __init__(self, model_params=None):
        self.model_name = "xgb_regression"
        self.search_type = 'random'
        self.param_grid = {'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
                           'subsample': sp_uniform(loc=0.2, scale=0.8),
                           "n_estimators": sp_randint(10, 300),
                           "max_depth": sp_randint(3, 100),
                           "learning_rate": np.linspace(0, 1, 100),
                           'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
                           'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
                           'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100],
                           "objective": ["reg:pseudohubererror"]}
        if model_params is None:
            self.ModelClass = XGBRegressor()
        else:
            self.ModelClass = XGBRegressor(**model_params)
