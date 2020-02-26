import numpy as np
import pandas as pd
import os
import datetime
from dateutil.relativedelta import relativedelta
import time
import test_data_produce as ts
import random


# 当前日期
today = int(time.strftime("%Y%m%d", time.localtime()))


class HoldStock():
    # 计算卖出日期与利率
    def cal_sell(self):
        for i in range(0, 20):
            self.p *= data.iloc[i]["ChangeRatio"]
            self.enddate = data.iloc[i]["TradingDate"]
            # 满足条件提前终止循环
            if self.p > 1.15:
                break


    def __init__(self, symbol:str, date:str, amount:float, data):
        self.symbol = symbol
        self.startdate = date
        self.amount = amount
        self.p = 1.0
        self.data = data
        self.sell_day = 0
        self.cal_sell()
        

    def __del__(self):
        del self.symbol
        del self.startdate
        del self.amount
        del self.p
        del self.data
        del self.sell_day
    

# 根据给定数据计算最终收益
def get_end_profit(data):
    end_p = 1.0
    max_p = 1.0
    # 去重
    data.drop_duplicates(subset=['TradingDate'], inplace=True)
    # 排序
    data = data.sort_values(['TradingDate'], ascending=True)
    # 计算
    for i in range(0, data.shape[0]):
        end_p = end_p * (1 + data.iloc[i]['ChangeRatio'])
        if end_p > max_p:
            max_p = end_p
    return end_p, max_p

# 回测
# 对所有策略以最近n年数据进行测试
# 设置最大同时持股数，当前为5，后续随资金量增长可设置更大值

# 需要一种方法衡量策略的风险
def back_test(data, test_years, max_stockhold, tradingdate):
    # 交易频率(天)
    up10_trading_freq = 10 
    down_trading_freq = 20
    # 计算日期
    cal_date = tradingdate[4:]
    # 创建策略总值map,余额总值map
    total = {"5_10": 1.0, "nnn":1.0, "7":1.0}
    balance = {"5_10": 1.0, "nnn":1.0, "7":1.0}
    # 记录持股
    stock_hold_5_10 = []
    stock_hold_7 = []
    stock_hold_nnn = []
    # 记录每次卖出收益
    sell = {"5_10": [], "7": [], "nnn": []}
    # 回测开始
    for day in cal_date:
        # 持股不满时才需要计算
        if len(stock_hold_5_10) < 5:
            # 获取筛选结果
            sel_5_10 = cal_5_10(data, day, tradingdate)
            for i in range(0, 5 - len(stock_hold_5_10)):
                stock_data = data[data['Symbol'] == sel_5_10[i]]
                stock_data = stock_data['TradingDate' >= day]
                spend = min(balance["5_10"], total["5_10"] / 5)
                balance["5_10"] -= spend
                stock_hold_5_10.append(HoldStock(sel_5_10[i], day, spend, stock_data))
        if len(stock_hold_7) < 5:
            # 获取筛选结果
            sel_7 = cal_7(data, day)
            for i in range(0, 5 - len(stock_hold_7)):
                stock_data = data[data['Symbol'] == sel_7[i]]
                stock_data = stock_data['TradingDate' >= day]
                spend = min(balance["7"], total["7"] / 5)
                balance["7"] -= spend
                stock_hold_7.append(HoldStock(sel_7[i], day, spend, stock_data))
        if len(stock_hold_nnn) < 5:
            # 获取筛选结果
            sel_nnn = cal_nnn(data, day, tradingdate)
            for i in range(0, 5 - len(stock_hold_nnn)):
                stock_data = data[data['Symbol'] == sel_nnn[i]]
                stock_data = stock_data['TradingDate' >= day]
                spend = min(balance["nnn"], total["nnn"] / 5)
                balance["nnn"] -= spend
                stock_hold_nnn.append(HoldStock(sel_nnn[i], day, spend, stock_data))
            
        # 计算卖出
        for hold in stock_hold_5_10:
            if day == hold.enddate:
                total["5_10"] += hold.amount * hold.p
                balance["5_10"] += hold.amount * hold.p
                stock_hold_5_10.remove(hold)
                del hold

        for hold in stock_hold_7:
            if day == hold.enddate:
                total["7"] += hold.amount * hold.p
                balance["7"] += hold.amount * hold.p
                stock_hold_7.remove(hold)
                del hold

        for hold in stock_hold_nnn:
            if day == hold.enddate:
                total["nnn"] += hold.amount * hold.p
                balance["nnn"] += hold.amount * hold.p
                stock_hold_nnn.remove(hold)
                del hold
   
    return


def cal_5_10(data, date, tradingdate):
    # 提取最近五天数据
    today_indexes = tradingdate.index(date)
    cal_date = tradingdate[today_indexes - 4: today_indexes]
    data = data[data['TradingDate'] in cal_date]
    # 计算
    filter_data = list()
    df_group = data.groupby(by="Symbol")
    stock_list = list(df_group.groups.keys())
    for i in stock_list:
        cur_data = data[data['Symbol'] == i]
        cur_data.sort_values(by='TradingDate', ascending=True)
        # 计算
        cur_p = 1.0
        for j in range(0, 5):
            cur_p *= (cur_data.iloc[j]['ChangeRatio'] + 1)
        cur_p = cur_p - 1
        if cur_p > 0.1:
            filter_data.append([i, cur_p])
    filter_data.sort(key=(lambda x: x[-1]))
    ret_data = list()
    for i in range(0, 5): ret_data.append(filter_data[i][0])
    return ret_data


# n天连涨策略
def cal_nnn(data, date, tradingdate):
    # 提取最近三天数据
    today_indexes = tradingdate.index(date)
    cal_date = tradingdate[today_indexes - 2: today_indexes]
    data = data[data['TradingDate'] in cal_date]
    # 计算
    filter_data = list()
    df_group = data.groupby(by="Symbol")
    stock_list = list(df_group.groups.keys())
    for i in stock_list:
        cur_data = data[data['Symbol'] == i]
        cur_data.sort_values(by='TradingDate', ascending=True)
        if cur_data.iloc[0]['ChangeRatio'] > 0 and cur_data.iloc[1]['ChangeRatio'] > 0 and cur_data.iloc[2]['ChangeRatio'] > 0:
            filter_data.append(cur_data.iloc[2]['Symbol'], cur_data.iloc[2]['TradingDate'])

    return filter_data


# 涨7策略
def cal_7(data, date):
    # 算法计算得到的数据
    filter_data = list()
    df_group = data.groupby(by="Symbol")
    stock_list = list(df_group.groups.keys())
    for i in stock_list:
        cur_data = data[data['Symbol'] == i]
        if cur_data.iloc[0]['ChangeRatio'] >= 0.07:
            filter_data.append(cur_data.iloc[0]['Symbol'])

    return filter_data


if __name__ == "__main__":
    data_path = r'C:\Users\wuziyang\Documents\PyWork\trading_simulation\data\stockdata\stock_latest.csv'
    data = pd.read_csv(data_path, sep=',', low_memory=False)
    data = data[['Symbol', 'TradingDate', 'ChangeRatio']]
    
    testYear = 2
    # 获取最近两年数据
    tradedata = data[data['TradingDate'] >= today - testYear * 10000]
    # 获取所有交易日
    df_group = tradedata.groupby(by="TradingDate")
    tradingdate = list(df_group.groups.keys()).sort()
    # 回测
    # 评估公式: score = all_add(x - 1.06) / np.var(x)
    back_test(data, testYear, 5, tradingdate)
