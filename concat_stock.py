import random
import pandas as pd
import numpy as np
import timeit
import math
import get_data as gd

def concat_row(date):
    gd.get_data(date)

    stock_data1 = pd.read_csv(r'D:\wuziyang\workfile\STK_MKT_Dalyr.csv', sep='\t')

    stock_data = pd.read_csv(r'D:\wuziyang\workfile\stock_tmp3.csv', sep=',')

    # stock_data1['TradingDate'] = stock_data1['TradingDate'].map(lambda x: x.replace('-', ''))
    stock_data = stock_data.append(stock_data1)
    # stock_data.drop_duplicates(subset=['Symbol', 'TradingDate'], inplace=True)

    stock_data = stock_data.sort_values(['Symbol', 'TradingDate'], ascending=True)

    stock_data.to_csv(r'D:\wuziyang\workfile\stock_tmp3.csv', index=False)


def concat_col():
    ori_df = pd.read_csv(r'D:\wuziyang\workfile\stock_latest.csv', sep=',')
    df_group = ori_df.groupby(by="TradingDate")
    dates = list(df_group.groups.keys())
    basic_df = pd.DataFrame(columns = ["Symbol","TradingDate","turnover_rate","turnover_rate_f","volume_ratio", \
    "pe","pe_ttm","pb","ps","ps_ttm","dv_ratio","dv_ttm","total_share","float_share","free_share","total_mv","circ_mv"])
    for i in dates:
        basic_df = basic_df.append(gd.get_basic(int(i)), ignore_index=True)
        # basic_df = pd.concat(basic_df, gd.get_basic(int(i)))

    basic_df.to_csv(r'D:\wuziyang\workfile\stock_basic_tmp.csv', index=False)
    print(basic_df.dtypes)
    ori_df = pd.merge(ori_df, basic_df, on=["Symbol", "TradingDate"])

    ori_df.to_csv(r'D:\wuziyang\workfile\stock_tmp.csv', index=False)

# concat_col()
# ori_df = pd.read_csv(r'D:\wuziyang\workfile\stock_latest.csv', sep=',')
# basic_df = pd.read_csv(r'D:\wuziyang\workfile\stock_basic_tmp2.csv', sep=',')
# ori_df = pd.merge(ori_df, basic_df, on=["Symbol", "TradingDate"])
# ori_df.to_csv(r'D:\wuziyang\workfile\stock_tmp3.csv', index=False)
concat_row(20200427)

# stock_data = pd.read_csv(r'D:\wuziyang\stock_latest.csv', sep=',')
# # 获取所有交易日
# df_group = stock_data.groupby(by="TradingDate")
# tradingdate = list(df_group.groups.keys())
# tradingdate = [e for e in tradingdate if e > 20170000]
# for d in tradingdate:
#     concat_data(str(int(d)))
