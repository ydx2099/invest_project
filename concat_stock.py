import random
import pandas as pd
import numpy as np
import timeit
import math
import get_data as gd


gd.get_data("20200220")

stock_data1 = pd.read_csv(r'C:\Users\wuziyang\Documents\PyWork\trading_simulation\data\add_data\STK_MKT_Dalyr.csv', sep='\t')

stock_data = pd.read_csv(r'C:\Users\wuziyang\Documents\PyWork\trading_simulation\data\stockdata\stock_latest.csv', sep=',')

# stock_data1['TradingDate'] = stock_data1['TradingDate'].map(lambda x: x.replace('-', ''))
stock_data = stock_data.append(stock_data1)
stock_data.drop_duplicates(subset=['Symbol', 'TradingDate'], inplace=True)

stock_data = stock_data.sort_values(['Symbol', 'TradingDate'], ascending=True)

stock_data.to_csv(r'C:\Users\wuziyang\Documents\PyWork\trading_simulation\data\stockdata\stock_latest.csv', index=False)

