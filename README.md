# causal_feature


## Procedure

1) Install all packages

`bash install.sh`


2) Run tests

`bash test.sh`


3) If needed, run the script to create the google trends dataset

`cd src`

`python3 get_trends.py`


4) Run script for feature selection using the SFI method

`cd src`

`python3 run_sfi.py`
