import tushare as ts
import pandas as pd
import numpy as np
# import Vaex

mytoken = "e631dfd1df584ec161d6a2449ac8e11d246448e3ef3bdfb956059f92"
token = "b15148f5ca285bd0e85bbc3f659daefff549ade3bba06fae6a037f03"

# 用tushare每天读取最新数据，方便处理
# 读取数据
def get_data(date):
    ts.set_token(token)
    pro = ts.pro_api()
    df = pro.daily(trade_date=date)
    # 只要这三个
    df = df[['ts_code', 'trade_date', 'pct_chg', 'open', 'close', 'high', 'low']]
    # 去后缀
    df['ts_code'] = df.apply(lambda row: row['ts_code'].split(".")[0], axis=1)
    # 利润率去百分号
    df['pct_chg'] = df.apply(lambda row: row['pct_chg'] / 100, axis=1)
    df.columns = ['Symbol', 'TradingDate', 'ChangeRatio', 'Open', 'Close', 'Max', 'Min']

    basic_df = get_basic(date)
    df = pd.merge(df, basic_df, on=["Symbol", "TradingDate"])

    df.to_csv(r'D:\wuziyang\workfile\STK_MKT_Dalyr.csv', sep='\t', index=False)


def get_basic(date):
    ts.set_token(token)
    pro = ts.pro_api()
    basic_df = pro.daily_basic(trade_date=date)
    basic_df['ts_code'] = basic_df.apply(lambda row: row['ts_code'].split(".")[0], axis=1)
    # print(basic_df.columns)
    basic_df = basic_df.drop('close', axis=1)
    basic_df.rename(columns={"ts_code":"Symbol", "trade_date":"TradingDate"}, inplace=True)
    return basic_df
