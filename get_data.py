import tushare as ts
import pandas as pd
import numpy as np
# import Vaex

token = "e631dfd1df584ec161d6a2449ac8e11d246448e3ef3bdfb956059f92"

# 用tushare每天读取最新数据，方便处理
# 读取数据
def get_data(date):
    ts.set_token(token)
    pro = ts.pro_api()
    df = pro.daily(trade_date=date)
    # 只要这三个
    df = df[['ts_code', 'trade_date', 'pct_chg', 'open', 'close']]
    # 去后缀
    df['ts_code'] = df.apply(lambda row: row['ts_code'].split(".")[0], axis=1)
    # 利润率去百分号
    df['pct_chg'] = df.apply(lambda row: row['pct_chg'] / 100, axis=1)
    df.columns = ['Symbol', 'TradingDate', 'ChangeRatio', 'Open', 'Close']

    df.to_csv(r'C:\Users\wuziyang\Documents\PyWork\trading_simulation\data\add_data\STK_MKT_Dalyr.csv', sep='\t', index=False)
