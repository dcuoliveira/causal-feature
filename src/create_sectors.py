import numpy as np
import pandas as pd
from data_mani.utils import get_market_df
from tqdm import tqdm
from glob import glob



if __name__ == '__main__':
    path = "data/tickers/spx_group_sector.csv"
    df = pd.read_csv(path)
    df = df.loc[df.field == "INDUSTRY_SECTOR"].reset_index(drop=True)
    df.loc[:, "ticker"] = df.ticker.map(lambda x: x.replace("/", " "))
    sectors = df.value.unique().tolist()
    df_ord = df.set_index("value")

    for sector in tqdm(sectors, desc='sector'):
        base_list = df_ord.xs(sector).ticker.values.tolist()
        path_list = ['data/tickers/spx/{}.csv'.format(n) for n in base_list]
        dfs = []
        for p in tqdm(path_list, desc='ticker'):
            try:
                df_m = get_market_df(p).set_index("date").dropna()
                dfs.append(df_m)
            except ValueError:
                pass
        complete = pd.concat(dfs,1)
        new_name = sector.replace(", ", "_").replace("-", "_")
        sector_df = complete.mean(1).to_frame().rename(columns={0: new_name})
        sector_df = sector_df * 100
        out_path = "data/indices/SPX_{}.csv".format(new_name) 
        sector_df.to_csv(out_path)
        # creating the same format as the other csv's
        file_in = open(out_path, "r")
        prefix = ["ticker,SPX_{}\n".format(new_name),
                  "field,DAY_TO_DAY_TOT_RETURN_GROSS_DVDS\n",
                  "date,\n"] 
        lines = prefix + file_in.readlines()[1:]
        file_out = open(out_path,"w") 
        file_out.writelines(lines) 
        file_out.close()


    # list of problematic tickers

    # problematic = ["ADI UN Equity", "ADP UN Equity", "AMD UN Equity",
    #                "BMC UW Equity", "CA UN Equity", "CAR UN Equity",
    #                "CIEN UA Equity", "CMCSA UA Equity", "CME UN Equity",
    #                "CMVT UQ Equity", "COST UA Equity", "CSX UN Equity",
    #                "DVN UA Equity", "EHC UN Equity", "ETFC UN Equity",
    #                "ETFC UW Equity", "FTRCQ UN Equity", "GENZ UA Equity",
    #                "GT UN Equity", "HBAN UA Equity", "JOY UN Equity",
    #                "MAR UN Equity", "MAT UN Equity", "MDLZ UN Equity",
    #                "MU UN Equity", "MYL UN Equity", "NBR UA Equity",
    #                "NVDA UA Equity", "RRD UN Equity", "SBUX UA Equity",
    #                "SCHW UN Equity", "SLM UN Equity", "SPLS UA Equity",
    #                "TFCFA UN Equity", "TMUS UN Equity", "TXN UN Equity",
    #                "UAL UW Equity", "VIAB UN Equity", "WBA UN Equity",
    #                "WDC UN Equity", "WINMQ UN Equity"]

    # problematic_sectors = df.loc[df.ticker.isin(
    #     problematic)].value.unique().tolist()
    # non_problematic_sectors = [s for s in sectors if s not in problematic_sectors]
    # print("non problematic sectors = ", non_problematic_sectors)
