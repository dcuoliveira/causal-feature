import os
import numpy as np
import argparse


parser = argparse.ArgumentParser()


parser.add_argument('model',
                    type=str,
                    help='forecast model')

args = parser.parse_args()
model = args.model

seeds = range(1, 50000)

fs_methods = ["all", "sfi", "mdi", "mda", "huang", "granger", "IAMB", "MMMB"]
indices = ["SPX Index", "CCMP Index", "RTY Index", "SPX Basic Materials",
           "SPX Communications", "SPX Consumer Cyclical", "SPX Consumer Non cyclical",
           "SPX Energy", "SPX Financial", "SPX Industrial", "SPX Technology",
           "SPX Utilities"]


with open("{}_forecast_script.sh".format(model), "w") as file:
    for indice in indices:
        comment = "# commands for {}\n".format(indice)
        file.write(comment)
        for fs_method in fs_methods:
            seed = np.random.choice(seeds)
            command = "python3 forecast.py '{}' {} {} -S {}\n".format(indice,
                                                                      fs_method,
                                                                      model,
                                                                      seed)
            file.write(command)
