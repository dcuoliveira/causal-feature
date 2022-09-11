# 1) feature selection phase

# python3 run_sfi.py
# python3 run_mdi.py
# python3 run_mda.py
# python3 run_granger.py
# python3 run_huang.py
# python3 run_IAMB.py
# python3 run_MMMB.py

# Logit

# commands for SPX Index -i 1
python3 forecast.py 'SPX Index' all logit -S 331 -i 1
python3 forecast.py 'SPX Index' sfi logit -S 44219 -i 1
python3 forecast.py 'SPX Index' mdi logit -S 8098 -i 1
python3 forecast.py 'SPX Index' mda logit -S 44432 -i 1
python3 forecast.py 'SPX Index' huang logit -S 28051 -i 1
python3 forecast.py 'SPX Index' granger logit -S 41149 -i 1
python3 forecast.py 'SPX Index' IAMB logit -S 47411 -i 1
python3 forecast.py 'SPX Index' MMMB logit -S 29109 -i 1
# commands for CCMP Index -i 1
python3 forecast.py 'CCMP Index' all logit -S 34066 -i 1
python3 forecast.py 'CCMP Index' sfi logit -S 16975 -i 1
python3 forecast.py 'CCMP Index' mdi logit -S 27519 -i 1
python3 forecast.py 'CCMP Index' mda logit -S 5319 -i 1
python3 forecast.py 'CCMP Index' huang logit -S 37578 -i 1
python3 forecast.py 'CCMP Index' granger logit -S 20243 -i 1
python3 forecast.py 'CCMP Index' IAMB logit -S 34963 -i 1
python3 forecast.py 'CCMP Index' MMMB logit -S 16884 -i 1
# commands for RTY Index -i 1
python3 forecast.py 'RTY Index' all logit -S 34171 -i 1
python3 forecast.py 'RTY Index' sfi logit -S 17954 -i 1
python3 forecast.py 'RTY Index' mdi logit -S 39535 -i 1
python3 forecast.py 'RTY Index' mda logit -S 31226 -i 1
python3 forecast.py 'RTY Index' huang logit -S 15585 -i 1
python3 forecast.py 'RTY Index' granger logit -S 5072 -i 1
python3 forecast.py 'RTY Index' IAMB logit -S 42620 -i 1
python3 forecast.py 'RTY Index' MMMB logit -S 14171 -i 1
# commands for SPX Basic Materials -i 1
python3 forecast.py 'SPX Basic Materials' all logit -S 42084 -i 1
python3 forecast.py 'SPX Basic Materials' sfi logit -S 42743 -i 1
python3 forecast.py 'SPX Basic Materials' mdi logit -S 44975 -i 1
python3 forecast.py 'SPX Basic Materials' mda logit -S 33146 -i 1
python3 forecast.py 'SPX Basic Materials' huang logit -S 49015 -i 1
python3 forecast.py 'SPX Basic Materials' granger logit -S 36213 -i 1
python3 forecast.py 'SPX Basic Materials' IAMB logit -S 28149 -i 1
python3 forecast.py 'SPX Basic Materials' MMMB logit -S 31859 -i 1
# commands for SPX Communications -i 1
python3 forecast.py 'SPX Communications' all logit -S 20625 -i 1
python3 forecast.py 'SPX Communications' sfi logit -S 22707 -i 1
python3 forecast.py 'SPX Communications' mdi logit -S 34654 -i 1
python3 forecast.py 'SPX Communications' mda logit -S 5716 -i 1
python3 forecast.py 'SPX Communications' huang logit -S 1125 -i 1
python3 forecast.py 'SPX Communications' granger logit -S 11575 -i 1
python3 forecast.py 'SPX Communications' IAMB logit -S 566 -i 1
python3 forecast.py 'SPX Communications' MMMB logit -S 27766 -i 1
# commands for SPX Consumer Cyclical -i 1
python3 forecast.py 'SPX Consumer Cyclical' all logit -S 34092 -i 1
python3 forecast.py 'SPX Consumer Cyclical' sfi logit -S 23271 -i 1
python3 forecast.py 'SPX Consumer Cyclical' mdi logit -S 1461 -i 1
python3 forecast.py 'SPX Consumer Cyclical' mda logit -S 34416 -i 1
python3 forecast.py 'SPX Consumer Cyclical' huang logit -S 27373 -i 1
python3 forecast.py 'SPX Consumer Cyclical' granger logit -S 5594 -i 1
python3 forecast.py 'SPX Consumer Cyclical' IAMB logit -S 30624 -i 1
python3 forecast.py 'SPX Consumer Cyclical' MMMB logit -S 49822 -i 1
# commands for SPX Consumer Non cyclical -i 1
python3 forecast.py 'SPX Consumer Non cyclical' all logit -S 22699 -i 1
python3 forecast.py 'SPX Consumer Non cyclical' sfi logit -S 21068 -i 1
python3 forecast.py 'SPX Consumer Non cyclical' mdi logit -S 11937 -i 1
python3 forecast.py 'SPX Consumer Non cyclical' mda logit -S 30310 -i 1
python3 forecast.py 'SPX Consumer Non cyclical' huang logit -S 3948 -i 1
python3 forecast.py 'SPX Consumer Non cyclical' granger logit -S 44075 -i 1
python3 forecast.py 'SPX Consumer Non cyclical' IAMB logit -S 21623 -i 1
python3 forecast.py 'SPX Consumer Non cyclical' MMMB logit -S 43750 -i 1
# commands for SPX Energy -i 1
python3 forecast.py 'SPX Energy' all logit -S 4721 -i 1
python3 forecast.py 'SPX Energy' sfi logit -S 9288 -i 1
python3 forecast.py 'SPX Energy' mdi logit -S 32704 -i 1
python3 forecast.py 'SPX Energy' mda logit -S 10344 -i 1
python3 forecast.py 'SPX Energy' huang logit -S 11808 -i 1
python3 forecast.py 'SPX Energy' granger logit -S 914 -i 1
python3 forecast.py 'SPX Energy' IAMB logit -S 47119 -i 1
python3 forecast.py 'SPX Energy' MMMB logit -S 6544 -i 1
# commands for SPX Financial -i 1
python3 forecast.py 'SPX Financial' all logit -S 25920 -i 1
python3 forecast.py 'SPX Financial' sfi logit -S 1198 -i 1
python3 forecast.py 'SPX Financial' mdi logit -S 9687 -i 1
python3 forecast.py 'SPX Financial' mda logit -S 3299 -i 1
python3 forecast.py 'SPX Financial' huang logit -S 29121 -i 1
python3 forecast.py 'SPX Financial' granger logit -S 42402 -i 1
python3 forecast.py 'SPX Financial' IAMB logit -S 28589 -i 1
python3 forecast.py 'SPX Financial' MMMB logit -S 23042 -i 1
# commands for SPX Industrial -i 1
python3 forecast.py 'SPX Industrial' all logit -S 3139 -i 1
python3 forecast.py 'SPX Industrial' sfi logit -S 15591 -i 1
python3 forecast.py 'SPX Industrial' mdi logit -S 8733 -i 1
python3 forecast.py 'SPX Industrial' mda logit -S 43185 -i 1
python3 forecast.py 'SPX Industrial' huang logit -S 27830 -i 1
python3 forecast.py 'SPX Industrial' granger logit -S 14212 -i 1
python3 forecast.py 'SPX Industrial' IAMB logit -S 7876 -i 1
python3 forecast.py 'SPX Industrial' MMMB logit -S 40937 -i 1
# commands for SPX Technology -i 1
python3 forecast.py 'SPX Technology' all logit -S 15012 -i 1
python3 forecast.py 'SPX Technology' sfi logit -S 7439 -i 1
python3 forecast.py 'SPX Technology' mdi logit -S 41077 -i 1
python3 forecast.py 'SPX Technology' mda logit -S 48262 -i 1
python3 forecast.py 'SPX Technology' huang logit -S 10194 -i 1
python3 forecast.py 'SPX Technology' granger logit -S 334 -i 1
python3 forecast.py 'SPX Technology' IAMB logit -S 44767 -i 1
python3 forecast.py 'SPX Technology' MMMB logit -S 38107 -i 1
# commands for SPX Utilities -i 1
python3 forecast.py 'SPX Utilities' all logit -S 6041 -i 1
python3 forecast.py 'SPX Utilities' sfi logit -S 21931 -i 1
python3 forecast.py 'SPX Utilities' mdi logit -S 25944 -i 1
python3 forecast.py 'SPX Utilities' mda logit -S 5110 -i 1
python3 forecast.py 'SPX Utilities' huang logit -S 21393 -i 1
python3 forecast.py 'SPX Utilities' granger logit -S 20629 -i 1
python3 forecast.py 'SPX Utilities' IAMB logit -S 9413 -i 1
python3 forecast.py 'SPX Utilities' MMMB logit -S 823 -i 1


