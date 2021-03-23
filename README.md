# causal_feature

## Installing

Installing [miniconda on linux](https://dev.to/waylonwalker/installing-miniconda-on-linux-from-the-command-line-4ad7).

Installing [multiprocess on conda](https://anaconda.org/conda-forge/multiprocess).

## Procedure

1) Install all packages

`bash install.sh`


2) Run tests

`bash test.sh`


3) If needed, run the script to create the google trends dataset

`cd src`

`python3 get_trends.py`


4) If needed, run the script to create sector time series:

`cd src`

`python3 create_sectors.py`


5) Run scripts for feature selection

`cd src`

`python3 run_sfi.py`

`python3 run_mdi.py`

`python3 run_mda.py`

`python3 run_granger.py`

`python3 run_huang.py`

`python3 run_IAMB.py`

`python3 run_MMMB.py`
