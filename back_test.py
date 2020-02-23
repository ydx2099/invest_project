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
    def cal_sell(self):
        0


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
    # 回测开始
    for day in cal_date:
        # 持股不满时才需要计算
        if len(stock_hold_5_10) < 5:
            # 获取筛选结果
            sel_5_10 = cal_5_10(data, day, tradingdate)
            for i in range(0, 5 - len(stock_hold_5_10)):
                stock_hold_5_10.append(HoldStock(sel_5_10[i], day, ))        
        if len(stock_hold_7) < 5:
            # 获取筛选结果
            sel_7 = cal_7(data, day)
            for i in range(0, 5 - len(stock_hold_7)):
                stock_hold_7.append(sel_7[i])
        if len(stock_hold_nnn) < 5:
            # 获取筛选结果
            sel_nnn = cal_nnn(data, day, tradingdate)
            for i in range(0, 5 - len(stock_hold_nnn)):
                stock_hold_nnn.append(sel_nnn[i])
            
        # 新增股票也要计算当天盈利
        

    # n天涨跌n%策略
    # 测试方法为每两个月按策略进行买入卖出，计算最终收益
    # 一个月为买入期，一个月为卖出期
    for i in range(0, test_years * 6):
        # 计算收益
        if not i == 0:  
            cal_data = use_data[use_data['TradingDate'] <= start_date + i * 100]
            cur_profit_up10 = [0.0, 0.0]
            cur_profit_downc10 = [0.0, 0.0]
            # 计算每只持有股票的最终/最高收益
            for stock in stock_hold_up10:
                stock_data = cal_data[cal_data['Symbol'] == stock['Symbol']]
                stock_data = stock_data[stock_data['TradingDate'] > stock['TradingDate']]
                end_p, max_p = get_end_profit(stock_data)
                cur_profit_up10[0] += end_p
                cur_profit_up10[1] += max_p
                
            for stock in stock_hold_down10:
                stock_data = cal_data[cal_data['Symbol'] == stock['Symbol']]
                stock_data = stock_data[stock_data['TradingDate'] > stock['TradingDate']]
                end_p, max_p = get_end_profit(stock_data)
                cur_profit_downc10[0] += end_p
                cur_profit_downc10[1] += max_p

            cur_profit_up10[0] /= len(stock_hold_up10)
            cur_profit_up10[1] /= len(stock_hold_up10)
            cur_profit_downc10[0] /= len(stock_hold_down10)
            cur_profit_downc10[1] /= len(stock_hold_up10)
            # 更新profit
            profit_up10[0] *= cur_profit_up10[0]
            profit_up10[1] *= cur_profit_up10[1]
            profit_down10[0] *= cur_profit_downc10[0]
            profit_down10[1] *= cur_profit_downc10[1]
            
        # 取出当前批次数据
        cur_batch_data = use_data[use_data['TradingDate'] <= start_date + i * 100]
        cur_batch_data = cur_batch_data[cur_batch_data['TradingDate'] >= start_date + (i - 1) * 100]

        # 计算下一批次所选股票
        simu_cal5_data = cal_profit(cur_batch_data, 5, True)
        simu_cal5_up10_data = simu_cal5_data[simu_cal5_data['ChangeRatio'] >= 0.1]
        simu_cal5_down10_data = simu_cal5_data[simu_cal5_data['ChangeRatio'] <= -0.1]
        # simu_cal10_data = cal_profit(cur_batch_data, 10, True)
        # 按最终涨幅排序，从高到低
        simu_cal5_up10_data = simu_cal5_up10_data.sort_values(by='ChangeRatio', ascending=False)
        simu_cal5_down10_data = simu_cal5_down10_data.sort_values(by='ChangeRatio', ascending=False)
        # 先清空持股记录
        stock_hold_up10 = []
        stock_hold_down10 = []
        # 持有股票
        for j in range(0, min(max_stockhold, simu_cal5_up10_data.shape[0])):
            stock_hold_up10.append(simu_cal5_up10_data.iloc[j])

        for j in range(0, min(max_stockhold, simu_cal5_down10_data.shape[0])):
            stock_hold_down10.append(simu_cal5_down10_data.iloc[j])



    # # 涨7策略执行短线交易，每当拥有现金即进行交易
    # # 设置最大持有天数，暂为10个交易日
    # # 测试为简便起见，以10个交易日为一个交易周期
    # df_group = data.groupby(by="TradingDate")
    # date_list = list(df_group.groups.keys())
    # for i in date_list:
    #     cur_day_data = use_data[use_data['TradingDate'] == i]
    #     simu_7_data = cal_7(cur_batch_data, i)
   
    return profit_up10, profit_down10


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
