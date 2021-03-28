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



with open("{}_forecast_script.sh".format(model), "w") as file:

    for fs_method in fs_methods:
        seed = np.random.choice(seeds)
        command = "python3 forecast.py 'SPX Index' {} {} -S {}\n".format(fs_method,
                                                                         model,
                                                                         seed)
        file.write(command)