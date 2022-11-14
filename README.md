# causal_feature

## Installing

Installing [miniconda on linux](https://dev.to/waylonwalker/installing-miniconda-on-linux-from-the-command-line-4ad7).

Installing [multiprocess on conda](https://anaconda.org/conda-forge/multiprocess).

## Procedure

1) Install all packages

`bash install.sh`


2) Run tests

`bash test.sh`

3) If needed, run the script to sample the google trends data

`cd src`

`python3 get_trends_daily.py`

4After we gather multiples samples from gtrends, we combine all of them
by taking the mean and creating the file `data\gtrends.csv`:

`python3 gen_gtrends_dabase.py`

5) If needed, run the script to create sector time series:

`cd src`

`python3 create_sectors.py`


6) Run scripts for feature selection

`cd src`

`python3 run_granger.py`

`python3 run_huang.py`

7) Run script for forecast based on one feature selection method and one machine learning model. For example:

`cd src`

`python3 forecast.py "SPX Utilities" granger random_forest -i 1 -s 2 -j 2
`