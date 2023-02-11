# commands for SPX Index
python3 forecast.py 'SPX Index' huang logit 252 5 -S 21230 -i 1 -d true
python3 forecast.py 'SPX Index' granger logit 252 5 -S 18212 -i 1 -d true