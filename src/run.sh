# 1) feature selection phase

python3 run_sfi.py
python3 run_mdi.py
python3 run_mda.py
python3 run_granger.py
python3 run_huang.py
python3 run_IAMB.py
python3 run_MMMB.py

# # 2) lasso forecast

# # commands for SPX Index
# # python3 forecast.py 'SPX Index' all lasso -S 17142
# # python3 forecast.py 'SPX Index' sfi lasso -S 21262
# # python3 forecast.py 'SPX Index' mdi lasso -S 39774
# # python3 forecast.py 'SPX Index' mda lasso -S 44105
# # python3 forecast.py 'SPX Index' huang lasso -S 6998
# # python3 forecast.py 'SPX Index' granger lasso -S 6045
# # python3 forecast.py 'SPX Index' IAMB lasso -S 3635
# # python3 forecast.py 'SPX Index' MMMB lasso -S 25778
# # commands for CCMP Index
# # python3 forecast.py 'CCMP Index' all lasso -S 47705
# # python3 forecast.py 'CCMP Index' sfi lasso -S 8127
# # python3 forecast.py 'CCMP Index' mdi lasso -S 596
# # python3 forecast.py 'CCMP Index' mda lasso -S 42131
# # python3 forecast.py 'CCMP Index' huang lasso -S 11042
# # python3 forecast.py 'CCMP Index' granger lasso -S 21780
# # python3 forecast.py 'CCMP Index' IAMB lasso -S 19391
# # python3 forecast.py 'CCMP Index' MMMB lasso -S 28896
# # commands for RTY Index
# # python3 forecast.py 'RTY Index' all lasso -S 36679
# # python3 forecast.py 'RTY Index' sfi lasso -S 908
# # python3 forecast.py 'RTY Index' mdi lasso -S 28161
# # python3 forecast.py 'RTY Index' mda lasso -S 8294
# # python3 forecast.py 'RTY Index' huang lasso -S 18992
# # python3 forecast.py 'RTY Index' granger lasso -S 28724
# # python3 forecast.py 'RTY Index' IAMB lasso -S 21216
# # python3 forecast.py 'RTY Index' MMMB lasso -S 8342
# # commands for SPX Basic Materials
# python3 forecast.py 'SPX Basic Materials' all lasso -S 35973
# python3 forecast.py 'SPX Basic Materials' sfi lasso -S 18611
# python3 forecast.py 'SPX Basic Materials' mdi lasso -S 17641
# python3 forecast.py 'SPX Basic Materials' mda lasso -S 41257
# python3 forecast.py 'SPX Basic Materials' huang lasso -S 41908
# python3 forecast.py 'SPX Basic Materials' granger lasso -S 33719
# python3 forecast.py 'SPX Basic Materials' IAMB lasso -S 46979
# python3 forecast.py 'SPX Basic Materials' MMMB lasso -S 14404
# # commands for SPX Communications
# python3 forecast.py 'SPX Communications' all lasso -S 17638
# python3 forecast.py 'SPX Communications' sfi lasso -S 45290
# python3 forecast.py 'SPX Communications' mdi lasso -S 23047
# python3 forecast.py 'SPX Communications' mda lasso -S 36622
# python3 forecast.py 'SPX Communications' huang lasso -S 41003
# python3 forecast.py 'SPX Communications' granger lasso -S 36155
# python3 forecast.py 'SPX Communications' IAMB lasso -S 44854
# python3 forecast.py 'SPX Communications' MMMB lasso -S 35404
# # commands for SPX Consumer Cyclical
# python3 forecast.py 'SPX Consumer Cyclical' all lasso -S 34750
# python3 forecast.py 'SPX Consumer Cyclical' sfi lasso -S 27691
# python3 forecast.py 'SPX Consumer Cyclical' mdi lasso -S 44464
# python3 forecast.py 'SPX Consumer Cyclical' mda lasso -S 3954
# python3 forecast.py 'SPX Consumer Cyclical' huang lasso -S 24581
# python3 forecast.py 'SPX Consumer Cyclical' granger lasso -S 13537
# python3 forecast.py 'SPX Consumer Cyclical' IAMB lasso -S 23403
# python3 forecast.py 'SPX Consumer Cyclical' MMMB lasso -S 23409
# # commands for SPX Consumer Non cyclical
# python3 forecast.py 'SPX Consumer Non cyclical' all lasso -S 15756
# python3 forecast.py 'SPX Consumer Non cyclical' sfi lasso -S 37662
# python3 forecast.py 'SPX Consumer Non cyclical' mdi lasso -S 25438
# python3 forecast.py 'SPX Consumer Non cyclical' mda lasso -S 27439
# python3 forecast.py 'SPX Consumer Non cyclical' huang lasso -S 47742
# python3 forecast.py 'SPX Consumer Non cyclical' granger lasso -S 583
# python3 forecast.py 'SPX Consumer Non cyclical' IAMB lasso -S 39165
# python3 forecast.py 'SPX Consumer Non cyclical' MMMB lasso -S 47306
# # commands for SPX Energy
# python3 forecast.py 'SPX Energy' all lasso -S 22803
# python3 forecast.py 'SPX Energy' sfi lasso -S 13965
# python3 forecast.py 'SPX Energy' mdi lasso -S 36163
# python3 forecast.py 'SPX Energy' mda lasso -S 15530
# python3 forecast.py 'SPX Energy' huang lasso -S 5174
# python3 forecast.py 'SPX Energy' granger lasso -S 46353
# python3 forecast.py 'SPX Energy' IAMB lasso -S 1239
# python3 forecast.py 'SPX Energy' MMMB lasso -S 39514
# # commands for SPX Financial
# python3 forecast.py 'SPX Financial' all lasso -S 26425
# python3 forecast.py 'SPX Financial' sfi lasso -S 17547
# python3 forecast.py 'SPX Financial' mdi lasso -S 34104
# python3 forecast.py 'SPX Financial' mda lasso -S 19876
# python3 forecast.py 'SPX Financial' huang lasso -S 26042
# python3 forecast.py 'SPX Financial' granger lasso -S 49019
# python3 forecast.py 'SPX Financial' IAMB lasso -S 27572
# python3 forecast.py 'SPX Financial' MMMB lasso -S 20247
# # commands for SPX Industrial
# python3 forecast.py 'SPX Industrial' all lasso -S 49805
# python3 forecast.py 'SPX Industrial' sfi lasso -S 30731
# python3 forecast.py 'SPX Industrial' mdi lasso -S 7646
# python3 forecast.py 'SPX Industrial' mda lasso -S 34230
# python3 forecast.py 'SPX Industrial' huang lasso -S 43550
# python3 forecast.py 'SPX Industrial' granger lasso -S 43170
# python3 forecast.py 'SPX Industrial' IAMB lasso -S 41616
# python3 forecast.py 'SPX Industrial' MMMB lasso -S 9891
# # commands for SPX Technology
# python3 forecast.py 'SPX Technology' all lasso -S 20031
# python3 forecast.py 'SPX Technology' sfi lasso -S 28580
# python3 forecast.py 'SPX Technology' mdi lasso -S 29659
# python3 forecast.py 'SPX Technology' mda lasso -S 17880
# python3 forecast.py 'SPX Technology' huang lasso -S 14257
# python3 forecast.py 'SPX Technology' granger lasso -S 31277
# python3 forecast.py 'SPX Technology' IAMB lasso -S 9950
# python3 forecast.py 'SPX Technology' MMMB lasso -S 27716
# # commands for SPX Utilities
# python3 forecast.py 'SPX Utilities' all lasso -S 21181
# python3 forecast.py 'SPX Utilities' sfi lasso -S 46300
# python3 forecast.py 'SPX Utilities' mdi lasso -S 43264
# python3 forecast.py 'SPX Utilities' mda lasso -S 6689
# python3 forecast.py 'SPX Utilities' huang lasso -S 1061
# python3 forecast.py 'SPX Utilities' granger lasso -S 7689
# python3 forecast.py 'SPX Utilities' IAMB lasso -S 30260
# python3 forecast.py 'SPX Utilities' MMMB lasso -S 14143

# # 3) ridge forecast

# # commands for SPX Index
# # python3 forecast.py 'SPX Index' all ridge -S 30448
# # python3 forecast.py 'SPX Index' sfi ridge -S 15771
# # python3 forecast.py 'SPX Index' mdi ridge -S 40534
# # python3 forecast.py 'SPX Index' mda ridge -S 18918
# # python3 forecast.py 'SPX Index' huang ridge -S 13885
# # python3 forecast.py 'SPX Index' granger ridge -S 15726
# # python3 forecast.py 'SPX Index' IAMB ridge -S 11240
# # python3 forecast.py 'SPX Index' MMMB ridge -S 40731
# # commands for CCMP Index
# # python3 forecast.py 'CCMP Index' all ridge -S 1490
# # python3 forecast.py 'CCMP Index' sfi ridge -S 16155
# # python3 forecast.py 'CCMP Index' mdi ridge -S 30865
# # python3 forecast.py 'CCMP Index' mda ridge -S 48886
# # python3 forecast.py 'CCMP Index' huang ridge -S 48116
# # python3 forecast.py 'CCMP Index' granger ridge -S 30927
# # python3 forecast.py 'CCMP Index' IAMB ridge -S 41299
# # python3 forecast.py 'CCMP Index' MMMB ridge -S 26485
# # commands for RTY Index
# # python3 forecast.py 'RTY Index' all ridge -S 38714
# # python3 forecast.py 'RTY Index' sfi ridge -S 18814
# # python3 forecast.py 'RTY Index' mdi ridge -S 3147
# # python3 forecast.py 'RTY Index' mda ridge -S 17805
# # python3 forecast.py 'RTY Index' huang ridge -S 40819
# # python3 forecast.py 'RTY Index' granger ridge -S 7915
# # python3 forecast.py 'RTY Index' IAMB ridge -S 27475
# # python3 forecast.py 'RTY Index' MMMB ridge -S 30368
# # commands for SPX Basic Materials
# python3 forecast.py 'SPX Basic Materials' all ridge -S 23655
# python3 forecast.py 'SPX Basic Materials' sfi ridge -S 41645
# python3 forecast.py 'SPX Basic Materials' mdi ridge -S 1370
# python3 forecast.py 'SPX Basic Materials' mda ridge -S 19636
# python3 forecast.py 'SPX Basic Materials' huang ridge -S 2464
# python3 forecast.py 'SPX Basic Materials' granger ridge -S 15026
# python3 forecast.py 'SPX Basic Materials' IAMB ridge -S 1491
# python3 forecast.py 'SPX Basic Materials' MMMB ridge -S 34295
# # commands for SPX Communications
# python3 forecast.py 'SPX Communications' all ridge -S 26011
# python3 forecast.py 'SPX Communications' sfi ridge -S 4262
# python3 forecast.py 'SPX Communications' mdi ridge -S 38411
# python3 forecast.py 'SPX Communications' mda ridge -S 34521
# python3 forecast.py 'SPX Communications' huang ridge -S 6385
# python3 forecast.py 'SPX Communications' granger ridge -S 7289
# python3 forecast.py 'SPX Communications' IAMB ridge -S 15170
# python3 forecast.py 'SPX Communications' MMMB ridge -S 700
# # commands for SPX Consumer Cyclical
# python3 forecast.py 'SPX Consumer Cyclical' all ridge -S 27058
# python3 forecast.py 'SPX Consumer Cyclical' sfi ridge -S 1835
# python3 forecast.py 'SPX Consumer Cyclical' mdi ridge -S 46991
# python3 forecast.py 'SPX Consumer Cyclical' mda ridge -S 24197
# python3 forecast.py 'SPX Consumer Cyclical' huang ridge -S 10140
# python3 forecast.py 'SPX Consumer Cyclical' granger ridge -S 12969
# python3 forecast.py 'SPX Consumer Cyclical' IAMB ridge -S 6216
# python3 forecast.py 'SPX Consumer Cyclical' MMMB ridge -S 35835
# # commands for SPX Consumer Non cyclical
# python3 forecast.py 'SPX Consumer Non cyclical' all ridge -S 13051
# python3 forecast.py 'SPX Consumer Non cyclical' sfi ridge -S 29097
# python3 forecast.py 'SPX Consumer Non cyclical' mdi ridge -S 22170
# python3 forecast.py 'SPX Consumer Non cyclical' mda ridge -S 33736
# python3 forecast.py 'SPX Consumer Non cyclical' huang ridge -S 22680
# python3 forecast.py 'SPX Consumer Non cyclical' granger ridge -S 22528
# python3 forecast.py 'SPX Consumer Non cyclical' IAMB ridge -S 38987
# python3 forecast.py 'SPX Consumer Non cyclical' MMMB ridge -S 31916
# # commands for SPX Energy
# python3 forecast.py 'SPX Energy' all ridge -S 42636
# python3 forecast.py 'SPX Energy' sfi ridge -S 12460
# python3 forecast.py 'SPX Energy' mdi ridge -S 7233
# python3 forecast.py 'SPX Energy' mda ridge -S 19883
# python3 forecast.py 'SPX Energy' huang ridge -S 24442
# python3 forecast.py 'SPX Energy' granger ridge -S 29168
# python3 forecast.py 'SPX Energy' IAMB ridge -S 17804
# python3 forecast.py 'SPX Energy' MMMB ridge -S 4363
# # commands for SPX Financial
# python3 forecast.py 'SPX Financial' all ridge -S 24659
# python3 forecast.py 'SPX Financial' sfi ridge -S 14301
# python3 forecast.py 'SPX Financial' mdi ridge -S 48807
# python3 forecast.py 'SPX Financial' mda ridge -S 30250
# python3 forecast.py 'SPX Financial' huang ridge -S 47853
# python3 forecast.py 'SPX Financial' granger ridge -S 3521
# python3 forecast.py 'SPX Financial' IAMB ridge -S 13716
# python3 forecast.py 'SPX Financial' MMMB ridge -S 35984
# # commands for SPX Industrial
# python3 forecast.py 'SPX Industrial' all ridge -S 39468
# python3 forecast.py 'SPX Industrial' sfi ridge -S 19394
# python3 forecast.py 'SPX Industrial' mdi ridge -S 6699
# python3 forecast.py 'SPX Industrial' mda ridge -S 10554
# python3 forecast.py 'SPX Industrial' huang ridge -S 3810
# python3 forecast.py 'SPX Industrial' granger ridge -S 20567
# python3 forecast.py 'SPX Industrial' IAMB ridge -S 6057
# python3 forecast.py 'SPX Industrial' MMMB ridge -S 35579
# # commands for SPX Technology
# python3 forecast.py 'SPX Technology' all ridge -S 48066
# python3 forecast.py 'SPX Technology' sfi ridge -S 33288
# python3 forecast.py 'SPX Technology' mdi ridge -S 27750
# python3 forecast.py 'SPX Technology' mda ridge -S 12631
# python3 forecast.py 'SPX Technology' huang ridge -S 49489
# python3 forecast.py 'SPX Technology' granger ridge -S 22139
# python3 forecast.py 'SPX Technology' IAMB ridge -S 12515
# python3 forecast.py 'SPX Technology' MMMB ridge -S 9590
# # commands for SPX Utilities
# python3 forecast.py 'SPX Utilities' all ridge -S 39428
# python3 forecast.py 'SPX Utilities' sfi ridge -S 23303
# python3 forecast.py 'SPX Utilities' mdi ridge -S 10616
# python3 forecast.py 'SPX Utilities' mda ridge -S 39464
# python3 forecast.py 'SPX Utilities' huang ridge -S 13825
# python3 forecast.py 'SPX Utilities' granger ridge -S 32206
# python3 forecast.py 'SPX Utilities' IAMB ridge -S 7406
# python3 forecast.py 'SPX Utilities' MMMB ridge -S 24827

# # 4) enet forecast

# # commands for SPX Index
# # python3 forecast.py 'SPX Index' all enet -S 49640
# # python3 forecast.py 'SPX Index' sfi enet -S 5569
# # python3 forecast.py 'SPX Index' mdi enet -S 2099
# # python3 forecast.py 'SPX Index' mda enet -S 22669
# # python3 forecast.py 'SPX Index' huang enet -S 45211
# # python3 forecast.py 'SPX Index' granger enet -S 1014
# # python3 forecast.py 'SPX Index' IAMB enet -S 47793
# # python3 forecast.py 'SPX Index' MMMB enet -S 15559
# # commands for CCMP Index
# # python3 forecast.py 'CCMP Index' all enet -S 45124
# # python3 forecast.py 'CCMP Index' sfi enet -S 4071
# # python3 forecast.py 'CCMP Index' mdi enet -S 10593
# # python3 forecast.py 'CCMP Index' mda enet -S 16800
# # python3 forecast.py 'CCMP Index' huang enet -S 34175
# # python3 forecast.py 'CCMP Index' granger enet -S 5689
# # python3 forecast.py 'CCMP Index' IAMB enet -S 29163
# # python3 forecast.py 'CCMP Index' MMMB enet -S 19457
# # commands for RTY Index
# # python3 forecast.py 'RTY Index' all enet -S 34175
# # python3 forecast.py 'RTY Index' sfi enet -S 13896
# # python3 forecast.py 'RTY Index' mdi enet -S 9972
# # python3 forecast.py 'RTY Index' mda enet -S 28901
# # python3 forecast.py 'RTY Index' huang enet -S 39340
# # python3 forecast.py 'RTY Index' granger enet -S 48110
# # python3 forecast.py 'RTY Index' IAMB enet -S 3139
# # python3 forecast.py 'RTY Index' MMMB enet -S 23097
# # commands for SPX Basic Materials
# python3 forecast.py 'SPX Basic Materials' all enet -S 40897
# python3 forecast.py 'SPX Basic Materials' sfi enet -S 17538
# python3 forecast.py 'SPX Basic Materials' mdi enet -S 29825
# python3 forecast.py 'SPX Basic Materials' mda enet -S 35884
# python3 forecast.py 'SPX Basic Materials' huang enet -S 13929
# python3 forecast.py 'SPX Basic Materials' granger enet -S 36874
# python3 forecast.py 'SPX Basic Materials' IAMB enet -S 47678
# python3 forecast.py 'SPX Basic Materials' MMMB enet -S 25437
# # commands for SPX Communications
# python3 forecast.py 'SPX Communications' all enet -S 11494
# python3 forecast.py 'SPX Communications' sfi enet -S 48368
# python3 forecast.py 'SPX Communications' mdi enet -S 2980
# python3 forecast.py 'SPX Communications' mda enet -S 45350
# python3 forecast.py 'SPX Communications' huang enet -S 34888
# python3 forecast.py 'SPX Communications' granger enet -S 892
# python3 forecast.py 'SPX Communications' IAMB enet -S 6683
# python3 forecast.py 'SPX Communications' MMMB enet -S 33436
# # commands for SPX Consumer Cyclical
# python3 forecast.py 'SPX Consumer Cyclical' all enet -S 28555
# python3 forecast.py 'SPX Consumer Cyclical' sfi enet -S 43545
# python3 forecast.py 'SPX Consumer Cyclical' mdi enet -S 27930
# python3 forecast.py 'SPX Consumer Cyclical' mda enet -S 7298
# python3 forecast.py 'SPX Consumer Cyclical' huang enet -S 28815
# python3 forecast.py 'SPX Consumer Cyclical' granger enet -S 26155
# python3 forecast.py 'SPX Consumer Cyclical' IAMB enet -S 27350
# python3 forecast.py 'SPX Consumer Cyclical' MMMB enet -S 49368
# # commands for SPX Consumer Non cyclical
# python3 forecast.py 'SPX Consumer Non cyclical' all enet -S 14040
# python3 forecast.py 'SPX Consumer Non cyclical' sfi enet -S 27440
# python3 forecast.py 'SPX Consumer Non cyclical' mdi enet -S 834
# python3 forecast.py 'SPX Consumer Non cyclical' mda enet -S 41507
# python3 forecast.py 'SPX Consumer Non cyclical' huang enet -S 41766
# python3 forecast.py 'SPX Consumer Non cyclical' granger enet -S 39315
# python3 forecast.py 'SPX Consumer Non cyclical' IAMB enet -S 7206
# python3 forecast.py 'SPX Consumer Non cyclical' MMMB enet -S 25831
# # commands for SPX Energy
# python3 forecast.py 'SPX Energy' all enet -S 23256
# python3 forecast.py 'SPX Energy' sfi enet -S 43448
# python3 forecast.py 'SPX Energy' mdi enet -S 43034
# python3 forecast.py 'SPX Energy' mda enet -S 8556
# python3 forecast.py 'SPX Energy' huang enet -S 4562
# python3 forecast.py 'SPX Energy' granger enet -S 4017
# python3 forecast.py 'SPX Energy' IAMB enet -S 24553
# python3 forecast.py 'SPX Energy' MMMB enet -S 38623
# # commands for SPX Financial
# python3 forecast.py 'SPX Financial' all enet -S 4618
# python3 forecast.py 'SPX Financial' sfi enet -S 40111
# python3 forecast.py 'SPX Financial' mdi enet -S 48489
# python3 forecast.py 'SPX Financial' mda enet -S 11996
# python3 forecast.py 'SPX Financial' huang enet -S 1832
# python3 forecast.py 'SPX Financial' granger enet -S 23486
# python3 forecast.py 'SPX Financial' IAMB enet -S 42745
# python3 forecast.py 'SPX Financial' MMMB enet -S 3873
# # commands for SPX Industrial
# python3 forecast.py 'SPX Industrial' all enet -S 40405
# python3 forecast.py 'SPX Industrial' sfi enet -S 3355
# python3 forecast.py 'SPX Industrial' mdi enet -S 33517
# python3 forecast.py 'SPX Industrial' mda enet -S 43446
# python3 forecast.py 'SPX Industrial' huang enet -S 4769
# python3 forecast.py 'SPX Industrial' granger enet -S 11282
# python3 forecast.py 'SPX Industrial' IAMB enet -S 43363
# python3 forecast.py 'SPX Industrial' MMMB enet -S 38309
# # commands for SPX Technology
# python3 forecast.py 'SPX Technology' all enet -S 39787
# python3 forecast.py 'SPX Technology' sfi enet -S 30648
# python3 forecast.py 'SPX Technology' mdi enet -S 32343
# python3 forecast.py 'SPX Technology' mda enet -S 41680
# python3 forecast.py 'SPX Technology' huang enet -S 30625
# python3 forecast.py 'SPX Technology' granger enet -S 15821
# python3 forecast.py 'SPX Technology' IAMB enet -S 41576
# python3 forecast.py 'SPX Technology' MMMB enet -S 5962
# # commands for SPX Utilities
# python3 forecast.py 'SPX Utilities' all enet -S 35374
# python3 forecast.py 'SPX Utilities' sfi enet -S 8096
# python3 forecast.py 'SPX Utilities' mdi enet -S 7622
# python3 forecast.py 'SPX Utilities' mda enet -S 40441
# python3 forecast.py 'SPX Utilities' huang enet -S 18879
# python3 forecast.py 'SPX Utilities' granger enet -S 9419
# python3 forecast.py 'SPX Utilities' IAMB enet -S 21413
# python3 forecast.py 'SPX Utilities' MMMB enet -S 15315

# # 5) random forest forecast

# # commands for SPX Index
# # python3 forecast.py 'SPX Index' all random_forest -S 4192
# # python3 forecast.py 'SPX Index' sfi random_forest -S 30926
# # python3 forecast.py 'SPX Index' mdi random_forest -S 12989
# # python3 forecast.py 'SPX Index' mda random_forest -S 48646
# # python3 forecast.py 'SPX Index' huang random_forest -S 35856
# # python3 forecast.py 'SPX Index' granger random_forest -S 31699
# # python3 forecast.py 'SPX Index' IAMB random_forest -S 25071
# # python3 forecast.py 'SPX Index' MMMB random_forest -S 16398
# # commands for CCMP Index
# # python3 forecast.py 'CCMP Index' all random_forest -S 4575
# # python3 forecast.py 'CCMP Index' sfi random_forest -S 45157
# # python3 forecast.py 'CCMP Index' mdi random_forest -S 935
# # python3 forecast.py 'CCMP Index' mda random_forest -S 26370
# # python3 forecast.py 'CCMP Index' huang random_forest -S 44782
# # python3 forecast.py 'CCMP Index' granger random_forest -S 11654
# # python3 forecast.py 'CCMP Index' IAMB random_forest -S 22157
# # python3 forecast.py 'CCMP Index' MMMB random_forest -S 9831
# # commands for RTY Index
# # python3 forecast.py 'RTY Index' all random_forest -S 23324
# # python3 forecast.py 'RTY Index' sfi random_forest -S 49640
# # python3 forecast.py 'RTY Index' mdi random_forest -S 20569
# # python3 forecast.py 'RTY Index' mda random_forest -S 39479
# # python3 forecast.py 'RTY Index' huang random_forest -S 33517
# # python3 forecast.py 'RTY Index' granger random_forest -S 2740
# # python3 forecast.py 'RTY Index' IAMB random_forest -S 22459
# # python3 forecast.py 'RTY Index' MMMB random_forest -S 37887
# # commands for SPX Basic Materials
# python3 forecast.py 'SPX Basic Materials' all random_forest -S 31493
# python3 forecast.py 'SPX Basic Materials' sfi random_forest -S 15931
# python3 forecast.py 'SPX Basic Materials' mdi random_forest -S 32241
# python3 forecast.py 'SPX Basic Materials' mda random_forest -S 24041
# python3 forecast.py 'SPX Basic Materials' huang random_forest -S 49205
# python3 forecast.py 'SPX Basic Materials' granger random_forest -S 26073
# python3 forecast.py 'SPX Basic Materials' IAMB random_forest -S 8607
# python3 forecast.py 'SPX Basic Materials' MMMB random_forest -S 21329
# # commands for SPX Communications
# python3 forecast.py 'SPX Communications' all random_forest -S 21792
# python3 forecast.py 'SPX Communications' sfi random_forest -S 8247
# python3 forecast.py 'SPX Communications' mdi random_forest -S 7534
# python3 forecast.py 'SPX Communications' mda random_forest -S 29339
# python3 forecast.py 'SPX Communications' huang random_forest -S 48749
# python3 forecast.py 'SPX Communications' granger random_forest -S 34559
# python3 forecast.py 'SPX Communications' IAMB random_forest -S 39807
# python3 forecast.py 'SPX Communications' MMMB random_forest -S 10803
# # commands for SPX Consumer Cyclical
# python3 forecast.py 'SPX Consumer Cyclical' all random_forest -S 23254
# python3 forecast.py 'SPX Consumer Cyclical' sfi random_forest -S 9855
# python3 forecast.py 'SPX Consumer Cyclical' mdi random_forest -S 1922
# python3 forecast.py 'SPX Consumer Cyclical' mda random_forest -S 19744
# python3 forecast.py 'SPX Consumer Cyclical' huang random_forest -S 43794
# python3 forecast.py 'SPX Consumer Cyclical' granger random_forest -S 21027
# python3 forecast.py 'SPX Consumer Cyclical' IAMB random_forest -S 3748
# python3 forecast.py 'SPX Consumer Cyclical' MMMB random_forest -S 10130
# # commands for SPX Consumer Non cyclical
# python3 forecast.py 'SPX Consumer Non cyclical' all random_forest -S 11033
# python3 forecast.py 'SPX Consumer Non cyclical' sfi random_forest -S 24145
# python3 forecast.py 'SPX Consumer Non cyclical' mdi random_forest -S 37564
# python3 forecast.py 'SPX Consumer Non cyclical' mda random_forest -S 46701
# python3 forecast.py 'SPX Consumer Non cyclical' huang random_forest -S 42946
# python3 forecast.py 'SPX Consumer Non cyclical' granger random_forest -S 34788
# python3 forecast.py 'SPX Consumer Non cyclical' IAMB random_forest -S 13210
# python3 forecast.py 'SPX Consumer Non cyclical' MMMB random_forest -S 47375
# # commands for SPX Energy
# python3 forecast.py 'SPX Energy' all random_forest -S 43872
# python3 forecast.py 'SPX Energy' sfi random_forest -S 20130
# python3 forecast.py 'SPX Energy' mdi random_forest -S 38172
# python3 forecast.py 'SPX Energy' mda random_forest -S 24907
# python3 forecast.py 'SPX Energy' huang random_forest -S 42395
# python3 forecast.py 'SPX Energy' granger random_forest -S 33727
# python3 forecast.py 'SPX Energy' IAMB random_forest -S 29028
# python3 forecast.py 'SPX Energy' MMMB random_forest -S 41590
# # commands for SPX Financial
# python3 forecast.py 'SPX Financial' all random_forest -S 13233
# python3 forecast.py 'SPX Financial' sfi random_forest -S 49947
# python3 forecast.py 'SPX Financial' mdi random_forest -S 1684
# python3 forecast.py 'SPX Financial' mda random_forest -S 46372
# python3 forecast.py 'SPX Financial' huang random_forest -S 26589
# python3 forecast.py 'SPX Financial' granger random_forest -S 33130
# python3 forecast.py 'SPX Financial' IAMB random_forest -S 30329
# python3 forecast.py 'SPX Financial' MMMB random_forest -S 15837
# # commands for SPX Industrial
# python3 forecast.py 'SPX Industrial' all random_forest -S 22122
# python3 forecast.py 'SPX Industrial' sfi random_forest -S 28916
# python3 forecast.py 'SPX Industrial' mdi random_forest -S 38551
# python3 forecast.py 'SPX Industrial' mda random_forest -S 15087
# python3 forecast.py 'SPX Industrial' huang random_forest -S 12010
# python3 forecast.py 'SPX Industrial' granger random_forest -S 39759
# python3 forecast.py 'SPX Industrial' IAMB random_forest -S 7150
# python3 forecast.py 'SPX Industrial' MMMB random_forest -S 26016
# # commands for SPX Technology
# python3 forecast.py 'SPX Technology' all random_forest -S 32226
# python3 forecast.py 'SPX Technology' sfi random_forest -S 13945
# python3 forecast.py 'SPX Technology' mdi random_forest -S 20183
# python3 forecast.py 'SPX Technology' mda random_forest -S 8622
# python3 forecast.py 'SPX Technology' huang random_forest -S 46218
# python3 forecast.py 'SPX Technology' granger random_forest -S 19791
# python3 forecast.py 'SPX Technology' IAMB random_forest -S 35529
# python3 forecast.py 'SPX Technology' MMMB random_forest -S 11553
# # commands for SPX Utilities
# python3 forecast.py 'SPX Utilities' all random_forest -S 45743
# python3 forecast.py 'SPX Utilities' sfi random_forest -S 5911
# python3 forecast.py 'SPX Utilities' mdi random_forest -S 22195
# python3 forecast.py 'SPX Utilities' mda random_forest -S 23772
# python3 forecast.py 'SPX Utilities' huang random_forest -S 46625
# python3 forecast.py 'SPX Utilities' granger random_forest -S 47856
# python3 forecast.py 'SPX Utilities' IAMB random_forest -S 13503
# python3 forecast.py 'SPX Utilities' MMMB random_forest -S 42158

# # 6) lgb forecast

# # commands for SPX Index
# # python3 forecast.py 'SPX Index' all lgb -S 30938
# # python3 forecast.py 'SPX Index' sfi lgb -S 49796
# # python3 forecast.py 'SPX Index' mdi lgb -S 21902
# # python3 forecast.py 'SPX Index' mda lgb -S 42268
# # python3 forecast.py 'SPX Index' huang lgb -S 47489
# # python3 forecast.py 'SPX Index' granger lgb -S 49743
# # python3 forecast.py 'SPX Index' IAMB lgb -S 30755
# # python3 forecast.py 'SPX Index' MMMB lgb -S 44595
# # commands for CCMP Index
# # python3 forecast.py 'CCMP Index' all lgb -S 48517
# # python3 forecast.py 'CCMP Index' sfi lgb -S 1184
# # python3 forecast.py 'CCMP Index' mdi lgb -S 14604
# # python3 forecast.py 'CCMP Index' mda lgb -S 18788
# # python3 forecast.py 'CCMP Index' huang lgb -S 27592
# # python3 forecast.py 'CCMP Index' granger lgb -S 47596
# # python3 forecast.py 'CCMP Index' IAMB lgb -S 43791
# # python3 forecast.py 'CCMP Index' MMMB lgb -S 32608
# # commands for RTY Index
# # python3 forecast.py 'RTY Index' all lgb -S 21130
# # python3 forecast.py 'RTY Index' sfi lgb -S 46556
# # python3 forecast.py 'RTY Index' mdi lgb -S 41273
# # python3 forecast.py 'RTY Index' mda lgb -S 31097
# # python3 forecast.py 'RTY Index' huang lgb -S 39820
# # python3 forecast.py 'RTY Index' granger lgb -S 9897
# # python3 forecast.py 'RTY Index' IAMB lgb -S 34880
# # python3 forecast.py 'RTY Index' MMMB lgb -S 14656
# # commands for SPX Basic Materials
# python3 forecast.py 'SPX Basic Materials' all lgb -S 45187
# python3 forecast.py 'SPX Basic Materials' sfi lgb -S 11905
# python3 forecast.py 'SPX Basic Materials' mdi lgb -S 10086
# python3 forecast.py 'SPX Basic Materials' mda lgb -S 18424
# python3 forecast.py 'SPX Basic Materials' huang lgb -S 45315
# python3 forecast.py 'SPX Basic Materials' granger lgb -S 3600
# python3 forecast.py 'SPX Basic Materials' IAMB lgb -S 45162
# python3 forecast.py 'SPX Basic Materials' MMMB lgb -S 32690
# # commands for SPX Communications
# python3 forecast.py 'SPX Communications' all lgb -S 3291
# python3 forecast.py 'SPX Communications' sfi lgb -S 36347
# python3 forecast.py 'SPX Communications' mdi lgb -S 424
# python3 forecast.py 'SPX Communications' mda lgb -S 48741
# python3 forecast.py 'SPX Communications' huang lgb -S 5743
# python3 forecast.py 'SPX Communications' granger lgb -S 36557
# python3 forecast.py 'SPX Communications' IAMB lgb -S 40444
# python3 forecast.py 'SPX Communications' MMMB lgb -S 40184
# # commands for SPX Consumer Cyclical
# python3 forecast.py 'SPX Consumer Cyclical' all lgb -S 6506
# python3 forecast.py 'SPX Consumer Cyclical' sfi lgb -S 6497
# python3 forecast.py 'SPX Consumer Cyclical' mdi lgb -S 47135
# python3 forecast.py 'SPX Consumer Cyclical' mda lgb -S 23339
# python3 forecast.py 'SPX Consumer Cyclical' huang lgb -S 18612
# python3 forecast.py 'SPX Consumer Cyclical' granger lgb -S 36206
# python3 forecast.py 'SPX Consumer Cyclical' IAMB lgb -S 6813
# python3 forecast.py 'SPX Consumer Cyclical' MMMB lgb -S 3790
# # commands for SPX Consumer Non cyclical
# python3 forecast.py 'SPX Consumer Non cyclical' all lgb -S 9949
# python3 forecast.py 'SPX Consumer Non cyclical' sfi lgb -S 10208
# python3 forecast.py 'SPX Consumer Non cyclical' mdi lgb -S 49770
# python3 forecast.py 'SPX Consumer Non cyclical' mda lgb -S 38020
# python3 forecast.py 'SPX Consumer Non cyclical' huang lgb -S 25934
# python3 forecast.py 'SPX Consumer Non cyclical' granger lgb -S 43635
# python3 forecast.py 'SPX Consumer Non cyclical' IAMB lgb -S 5247
# python3 forecast.py 'SPX Consumer Non cyclical' MMMB lgb -S 7193
# # commands for SPX Energy
# python3 forecast.py 'SPX Energy' all lgb -S 4902
# python3 forecast.py 'SPX Energy' sfi lgb -S 11119
# python3 forecast.py 'SPX Energy' mdi lgb -S 45228
# python3 forecast.py 'SPX Energy' mda lgb -S 37617
# python3 forecast.py 'SPX Energy' huang lgb -S 25697
# python3 forecast.py 'SPX Energy' granger lgb -S 12374
# python3 forecast.py 'SPX Energy' IAMB lgb -S 6734
# python3 forecast.py 'SPX Energy' MMMB lgb -S 22986
# # commands for SPX Financial
# python3 forecast.py 'SPX Financial' all lgb -S 27243
# python3 forecast.py 'SPX Financial' sfi lgb -S 29851
# python3 forecast.py 'SPX Financial' mdi lgb -S 19782
# python3 forecast.py 'SPX Financial' mda lgb -S 36338
# python3 forecast.py 'SPX Financial' huang lgb -S 1976
# python3 forecast.py 'SPX Financial' granger lgb -S 23324
# python3 forecast.py 'SPX Financial' IAMB lgb -S 5304
# python3 forecast.py 'SPX Financial' MMMB lgb -S 42540
# # commands for SPX Industrial
# python3 forecast.py 'SPX Industrial' all lgb -S 13746
# python3 forecast.py 'SPX Industrial' sfi lgb -S 22486
# python3 forecast.py 'SPX Industrial' mdi lgb -S 10262
# python3 forecast.py 'SPX Industrial' mda lgb -S 33361
# python3 forecast.py 'SPX Industrial' huang lgb -S 8801
# python3 forecast.py 'SPX Industrial' granger lgb -S 33885
# python3 forecast.py 'SPX Industrial' IAMB lgb -S 46012
# python3 forecast.py 'SPX Industrial' MMMB lgb -S 29538
# # commands for SPX Technology
# python3 forecast.py 'SPX Technology' all lgb -S 24633
# python3 forecast.py 'SPX Technology' sfi lgb -S 22664
# python3 forecast.py 'SPX Technology' mdi lgb -S 13625
# python3 forecast.py 'SPX Technology' mda lgb -S 8658
# python3 forecast.py 'SPX Technology' huang lgb -S 25318
# python3 forecast.py 'SPX Technology' granger lgb -S 37252
# python3 forecast.py 'SPX Technology' IAMB lgb -S 41648
# python3 forecast.py 'SPX Technology' MMMB lgb -S 43439
# # commands for SPX Utilities
# python3 forecast.py 'SPX Utilities' all lgb -S 37622
# python3 forecast.py 'SPX Utilities' sfi lgb -S 9214
# python3 forecast.py 'SPX Utilities' mdi lgb -S 34296
# python3 forecast.py 'SPX Utilities' mda lgb -S 38725
# python3 forecast.py 'SPX Utilities' huang lgb -S 28476
# python3 forecast.py 'SPX Utilities' granger lgb -S 13663
# python3 forecast.py 'SPX Utilities' IAMB lgb -S 41671
# python3 forecast.py 'SPX Utilities' MMMB lgb -S 46377

# # 7) nn3 forecast

# # commands for SPX Index
# # python3 forecast.py 'SPX Index' all nn3 -S 20905
# # python3 forecast.py 'SPX Index' sfi nn3 -S 17136
# # python3 forecast.py 'SPX Index' mdi nn3 -S 1027
# # python3 forecast.py 'SPX Index' mda nn3 -S 49898
# # python3 forecast.py 'SPX Index' huang nn3 -S 29430
# # python3 forecast.py 'SPX Index' granger nn3 -S 1557
# # python3 forecast.py 'SPX Index' IAMB nn3 -S 32950
# # python3 forecast.py 'SPX Index' MMMB nn3 -S 34235
# # commands for CCMP Index
# # python3 forecast.py 'CCMP Index' all nn3 -S 32131
# # python3 forecast.py 'CCMP Index' sfi nn3 -S 316
# # python3 forecast.py 'CCMP Index' mdi nn3 -S 15729
# # python3 forecast.py 'CCMP Index' mda nn3 -S 37590
# # python3 forecast.py 'CCMP Index' huang nn3 -S 46415
# # python3 forecast.py 'CCMP Index' granger nn3 -S 17986
# # python3 forecast.py 'CCMP Index' IAMB nn3 -S 10958
# # python3 forecast.py 'CCMP Index' MMMB nn3 -S 27688
# # commands for RTY Index
# # python3 forecast.py 'RTY Index' all nn3 -S 48439
# # python3 forecast.py 'RTY Index' sfi nn3 -S 35182
# # python3 forecast.py 'RTY Index' mdi nn3 -S 39039
# # python3 forecast.py 'RTY Index' mda nn3 -S 27689
# # python3 forecast.py 'RTY Index' huang nn3 -S 15830
# # python3 forecast.py 'RTY Index' granger nn3 -S 39330
# # python3 forecast.py 'RTY Index' IAMB nn3 -S 48948
# # python3 forecast.py 'RTY Index' MMMB nn3 -S 33189
# # commands for SPX Basic Materials
# python3 forecast.py 'SPX Basic Materials' all nn3 -S 1775
# python3 forecast.py 'SPX Basic Materials' sfi nn3 -S 33003
# python3 forecast.py 'SPX Basic Materials' mdi nn3 -S 26131
# python3 forecast.py 'SPX Basic Materials' mda nn3 -S 49729
# python3 forecast.py 'SPX Basic Materials' huang nn3 -S 22059
# python3 forecast.py 'SPX Basic Materials' granger nn3 -S 36229
# python3 forecast.py 'SPX Basic Materials' IAMB nn3 -S 23832
# python3 forecast.py 'SPX Basic Materials' MMMB nn3 -S 21155
# # commands for SPX Communications
# python3 forecast.py 'SPX Communications' all nn3 -S 49601
# python3 forecast.py 'SPX Communications' sfi nn3 -S 7826
# python3 forecast.py 'SPX Communications' mdi nn3 -S 30968
# python3 forecast.py 'SPX Communications' mda nn3 -S 31796
# python3 forecast.py 'SPX Communications' huang nn3 -S 33250
# python3 forecast.py 'SPX Communications' granger nn3 -S 23107
# python3 forecast.py 'SPX Communications' IAMB nn3 -S 39950
# python3 forecast.py 'SPX Communications' MMMB nn3 -S 2510
# # commands for SPX Consumer Cyclical
# python3 forecast.py 'SPX Consumer Cyclical' all nn3 -S 37275
# python3 forecast.py 'SPX Consumer Cyclical' sfi nn3 -S 2100
# python3 forecast.py 'SPX Consumer Cyclical' mdi nn3 -S 23841
# python3 forecast.py 'SPX Consumer Cyclical' mda nn3 -S 34635
# python3 forecast.py 'SPX Consumer Cyclical' huang nn3 -S 39008
# python3 forecast.py 'SPX Consumer Cyclical' granger nn3 -S 40350
# python3 forecast.py 'SPX Consumer Cyclical' IAMB nn3 -S 15847
# python3 forecast.py 'SPX Consumer Cyclical' MMMB nn3 -S 30100
# # commands for SPX Consumer Non cyclical
# python3 forecast.py 'SPX Consumer Non cyclical' all nn3 -S 5889
# python3 forecast.py 'SPX Consumer Non cyclical' sfi nn3 -S 30883
# python3 forecast.py 'SPX Consumer Non cyclical' mdi nn3 -S 21520
# python3 forecast.py 'SPX Consumer Non cyclical' mda nn3 -S 8931
# python3 forecast.py 'SPX Consumer Non cyclical' huang nn3 -S 14259
# python3 forecast.py 'SPX Consumer Non cyclical' granger nn3 -S 48123
# python3 forecast.py 'SPX Consumer Non cyclical' IAMB nn3 -S 24555
# python3 forecast.py 'SPX Consumer Non cyclical' MMMB nn3 -S 10246
# # commands for SPX Energy
# python3 forecast.py 'SPX Energy' all nn3 -S 36416
# python3 forecast.py 'SPX Energy' sfi nn3 -S 27674
# python3 forecast.py 'SPX Energy' mdi nn3 -S 17129
# python3 forecast.py 'SPX Energy' mda nn3 -S 31
# python3 forecast.py 'SPX Energy' huang nn3 -S 6963
# python3 forecast.py 'SPX Energy' granger nn3 -S 49794
# python3 forecast.py 'SPX Energy' IAMB nn3 -S 11170
# python3 forecast.py 'SPX Energy' MMMB nn3 -S 28902
# # commands for SPX Financial
# python3 forecast.py 'SPX Financial' all nn3 -S 2024
# python3 forecast.py 'SPX Financial' sfi nn3 -S 28521
# python3 forecast.py 'SPX Financial' mdi nn3 -S 6967
# python3 forecast.py 'SPX Financial' mda nn3 -S 26229
# python3 forecast.py 'SPX Financial' huang nn3 -S 39100
# python3 forecast.py 'SPX Financial' granger nn3 -S 41051
# python3 forecast.py 'SPX Financial' IAMB nn3 -S 17168
# python3 forecast.py 'SPX Financial' MMMB nn3 -S 34661
# # commands for SPX Industrial
# python3 forecast.py 'SPX Industrial' all nn3 -S 41175
# python3 forecast.py 'SPX Industrial' sfi nn3 -S 27208
# python3 forecast.py 'SPX Industrial' mdi nn3 -S 48313
# python3 forecast.py 'SPX Industrial' mda nn3 -S 28397
# python3 forecast.py 'SPX Industrial' huang nn3 -S 32904
# python3 forecast.py 'SPX Industrial' granger nn3 -S 42226
# python3 forecast.py 'SPX Industrial' IAMB nn3 -S 43761
# python3 forecast.py 'SPX Industrial' MMMB nn3 -S 43416
# # commands for SPX Technology
# python3 forecast.py 'SPX Technology' all nn3 -S 27143
# python3 forecast.py 'SPX Technology' sfi nn3 -S 19504
# python3 forecast.py 'SPX Technology' mdi nn3 -S 41296
# python3 forecast.py 'SPX Technology' mda nn3 -S 34939
# python3 forecast.py 'SPX Technology' huang nn3 -S 6500
# python3 forecast.py 'SPX Technology' granger nn3 -S 35387
# python3 forecast.py 'SPX Technology' IAMB nn3 -S 19048
# python3 forecast.py 'SPX Technology' MMMB nn3 -S 28226
# # commands for SPX Utilities
# python3 forecast.py 'SPX Utilities' all nn3 -S 26424
# python3 forecast.py 'SPX Utilities' sfi nn3 -S 36800
# python3 forecast.py 'SPX Utilities' mdi nn3 -S 23256
# python3 forecast.py 'SPX Utilities' mda nn3 -S 15682
# python3 forecast.py 'SPX Utilities' huang nn3 -S 25386
# python3 forecast.py 'SPX Utilities' granger nn3 -S 28141
# python3 forecast.py 'SPX Utilities' IAMB nn3 -S 46309
# python3 forecast.py 'SPX Utilities' MMMB nn3 -S 47192
