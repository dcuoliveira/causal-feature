import pandas as pd
import os
import sys
import inspect
import unittest
from glob import glob

currentdir = os.path.dirname(
    os.path.abspath(
        inspect.getfile(
            inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from src.data_mani.utils import path_filter  # noqa


class Test_filter(unittest.TestCase):
    def test_filter_nasdaq(self):
        path_1 = os.path.join(parentdir,
                              "src", "data",
                              "toy",
                              "ticker5.csv")
        path_2 = os.path.join(parentdir,
                              "src", "data",
                              "toy",
                              "ticker6.csv")
        paths = path_filter(paths=[path_1, path_2],
                            threshold=252,
                            verbose=False,
                            path_gt_list=[parentdir,
                                          "src",
                                          "data",
                                          "gtrends.csv"])
        self.assertTrue(len(paths) == 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
