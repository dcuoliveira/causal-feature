from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
import numpy as np
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor


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
    def __init__(self, model_params=None):
        self.model_name = "lasso"
        self.search_type = 'grid'
        self.param_grid = {'alpha': np.logspace(-4, 0, 50)}
        if model_params is None:
            self.ModelClass = Lasso()
        else:
            self.ModelClass = Lasso(**model_params)
