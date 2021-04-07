import numpy as np
import pandas as pd
from data_mani.utils import get_market_df
from tqdm import tqdm
from glob import glob



if __name__ == '__main__':
   path = "data/tickers/spx_group_sector.csv"
    dicionario_df = pd.read_csv(path)
    dicionario_df = dicionario_df.loc[dicionario_df.field == "INDUSTRY_SECTOR"].reset_index(drop=True)
    dicionario_df.loc[:, "ticker"] = dicionario_df.ticker.map(lambda x: x.replace("/", " "))
    sectors = dicionario_df.value.unique().tolist()
    dicionario_df.rename(columns={'value': 'sectors'}, inplace=True)
    
    spx_adj_ticker = glob('data/tickers/spx_adj/*.csv')

    melt_out = []
    for ticker in tqdm(spx_adj_ticker, desc='Build melted dataframe with stocks'):
        df = pd.read_csv(ticker)
        melt_df = df.melt('date')
        melt_out.append(melt_df)
    melt_df = pd.concat(melt_out, axis=0)
    melt_df.rename(columns={'variable': 'ticker'}, inplace=True)
    merge_melt_df = pd.merge(melt_df, dicionario_df, on='ticker', how='left')
    merge_melt_df = merge_melt_df[['date', 'ticker', 'value', 'sectors']]
    
    assert  merge_melt_df.shape[0] == melt_df.shape[0]
    assert len(sectors) == len(merge_melt_df['sectors'].unique())
    
    for sector in tqdm(merge_melt_df['sectors'].unique(), desc='agg by sectors and save data'):
        sector = merge_melt_df['sectors'].unique()[0]
        complete = merge_melt_df.loc[merge_melt_df['sectors'] == sector].drop('sectors', 1).pivot_table(index=['date'], columns=['ticker'])
        new_name = sector.replace(", ", " ").replace("-", " ")
        sector_df = complete.mean(1).to_frame().rename(columns={0: new_name})
        sector_df = sector_df
        out_path = "data/indices/SPX {}.csv".format(new_name) 
        sector_df.to_csv(out_path)
        # creating the same format as the other csv's
        file_in = open(out_path, "r")
        prefix = ["ticker,SPX {}\n".format(new_name),
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
