import numpy as np
import pandas as pd
import os
import datetime
from dateutil.relativedelta import relativedelta
import time
import test_data_produce as ts
import random


#TODO: nnn在年关附近测试
# 7 的过拟合测试

# 当前日期
today = int(time.strftime("%Y%m%d", time.localtime()))


class HoldStock():
    # 计算卖出日期与利率
    def cal_sell(self):
        hold_max_day = 5
        sell_max_profit = 1.15
        sell_min_profit = 0.93
        if self.data.shape[0] >= hold_max_day:
            for i in range(0, hold_max_day):
                self.p *= (self.data.iloc[i]["ChangeRatio"] + 1)
                self.enddate = self.data.iloc[i]["TradingDate"]
                # 满足条件提前终止循环
                if self.p > sell_max_profit:
                    break
                if self.p < sell_min_profit:
                    break


    def __init__(self, symbol:str, amount:float, data):
        self.symbol = symbol
        self.amount = amount
        self.p = 1.0
        self.data = data
        self.enddate = 0
        self.cal_sell()
        

    def __del__(self):
        del self.symbol
        del self.amount
        del self.p
        del self.data
        del self.enddate
    

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
    # 计算日期
    cal_date = tradingdate[4:-1]
    # 创建策略总值map,余额总值map
    total = {"5_10": 1.0, "nnn":1.0, "7":1.0}
    balance = {"5_10": 1.0, "nnn":1.0, "7":1.0}
    # 记录持股
    # stock_hold_5_10 = []
    # stock_hold_7 = []
    stock_hold_nnn = []
    # 记录每次卖出收益
    sell = {"5_10": [], "7": [], "nnn": []}
    # 回测开始
    for day in cal_date:
        # 持股不满时才需要计算
        # if len(stock_hold_5_10) < 5:
        #     # 获取筛选结果
        #     sel_5_10 = cal_5_10(data, day, tradingdate, -0.1, -0.16)
        #     for i in range(0, 5 - len(stock_hold_5_10)):
        #         if len(sel_5_10) > i:
        #             stock_data = data[data['Symbol'] == sel_5_10[i]]
        #             stock_data = stock_data[stock_data['TradingDate'] > day]
        #             spend = min(balance["5_10"], total["5_10"] / 5)
        #             balance["5_10"] -= spend
        #             stock_hold_5_10.append(HoldStock(sel_5_10[i], spend, stock_data))
        # if len(stock_hold_7) < 5:
        #     # 获取筛选结果
        #     sel_7 = cal_7(data, day)
        #     for i in range(0, 5 - len(stock_hold_7)):
        #         stock_data = data[data['Symbol'] == sel_7[i]]
        #         stock_data = stock_data[stock_data['TradingDate'] > day]
        #         spend = min(balance["7"], total["7"] / 5)
        #         balance["7"] -= spend
        #         stock_hold_7.append(HoldStock(sel_7[i], spend, stock_data))
        if len(stock_hold_nnn) < 5:
            # 获取筛选结果
            sel_nnn = cal_nnn(data, day, tradingdate)
            for i in range(0, 5 - len(stock_hold_nnn)):
                if len(sel_nnn) > i:
                    stock_data = data[data['Symbol'] == sel_nnn[i][0]]
                    stock_data = stock_data[stock_data['TradingDate'] > day]
                    spend = min(balance["nnn"], total["nnn"] / 5)
                    balance["nnn"] -= spend
                    stock_hold_nnn.append(HoldStock(sel_nnn[i], spend, stock_data))
            
        # # 计算卖出
        # for hold in stock_hold_5_10:
        #     if int(day) == int(hold.enddate):
        #         total["5_10"] += hold.amount * (hold.p - 1)
        #         balance["5_10"] += hold.amount * hold.p
        #         sell["5_10"].append(hold.p)
        #         stock_hold_5_10.remove(hold)
        #         del hold

        # for hold in stock_hold_7:
        #     if int(day) == int(hold.enddate):
        #         total["7"] += hold.amount * (hold.p - 1)
        #         balance["7"] += hold.amount * hold.p
        #         sell["7"].append(hold.p)
        #         stock_hold_7.remove(hold)
        #         del hold

        for hold in stock_hold_nnn:
            if int(day) == int(hold.enddate):
                total["nnn"] += hold.amount * (hold.p - 1)
                balance["nnn"] += hold.amount * hold.p
                sell["nnn"].append(hold.p)
                stock_hold_nnn.remove(hold)
                del hold
   
    return sell, total


def cal_5_10(data, date, tradingdate, high_limit, low_limit):
    # 获取对应日期数据
    today_indexes = tradingdate.index(date)
    cal_date = tradingdate[today_indexes - 4: today_indexes + 1]
    data = data[data['TradingDate'] >= cal_date[0]]
    data = data[data['TradingDate'] <= cal_date[4]]
    # 计算
    filter_data = list()
    df_group = data.groupby(by="Symbol")
    stock_list = list(df_group.groups.keys())
    for i in stock_list:
        cur_data = data[data['Symbol'] == i]
        cur_data.sort_values(by='TradingDate', ascending=True)
        # 计算
        cur_p = 1.0
        if cur_data.shape[0] == 5:
            for j in range(0, 5):
                cur_p *= (cur_data.iloc[j]['ChangeRatio'] + 1)
            cur_p = cur_p - 1
            if cur_p > low_limit and cur_p < high_limit:
                filter_data.append(i)
    # filter_data.sort(key=(lambda x: x[-1]))
    random.shuffle(filter_data)
    return filter_data[0:4]


# n天连涨策略
def cal_nnn(data, date, tradingdate):
    # 提取最近四天数据
    today_indexes = tradingdate.index(date)
    cal_date = tradingdate[today_indexes - 3: today_indexes + 1]
    data = data[data['TradingDate'] >= cal_date[0]]
    data = data[data['TradingDate'] <= cal_date[3]]
    # 计算
    filter_data = list()
    df_group = data.groupby(by="Symbol")
    stock_list = list(df_group.groups.keys())
    for i in stock_list:
        cur_data = data[data['Symbol'] == i]
        if cur_data.shape[0] == 4:
            cur_data.sort_values(by='TradingDate', ascending=True)
            if cur_data.iloc[0]['ChangeRatio'] < 0 and cur_data.iloc[1]['ChangeRatio'] > 0 and cur_data.iloc[2]['ChangeRatio'] > 0 and cur_data.iloc[3]['ChangeRatio'] > 0:
                filter_data.append([cur_data.iloc[2]['Symbol'], cur_data.iloc[2]['TradingDate']])

    random.shuffle(filter_data)
    return filter_data


# 涨7策略
def cal_7(data, date):
    # 获取对应日期数据
    data = data[data['TradingDate'] == date]
    # 算法计算得到的数据
    filter_data = list()
    df_group = data.groupby(by="Symbol")
    stock_list = list(df_group.groups.keys())
    for i in stock_list:
        cur_data = data[data['Symbol'] == i]
        if cur_data.iloc[0]['ChangeRatio'] >= 0.07:
            filter_data.append(cur_data.iloc[0]['Symbol'])

    random.shuffle(filter_data)
    return filter_data


if __name__ == "__main__":
    print("start...")

    data_path = r'D:\wuziyang\workfile\stock_latest.csv'
    data = pd.read_csv(data_path, sep=',', low_memory=False)
    data = data[['Symbol', 'TradingDate', 'ChangeRatio']]
    
    testYear = 2
    stockhold = 5
    # 用于test nnn
    for i in range(0, testYear):
        tradedata = data[data['TradingDate'] >= 10000*(2020-i) + 100]
        tradedata = tradedata[tradedata['TradingDate'] <= 10000*(2020-i) + 400]
    # 用于test7
    # tradedata = data[data['TradingDate'] >= today - (testYear + 2) * 10000]
    # tradedata = tradedata[tradedata['TradingDate'] <= today - testYear * 10000]
    #
    # # 获取最近n年数据
    # tradedata = data[data['TradingDate'] >= today - testYear * 10000]
        # 获取所有交易日
        df_group = tradedata.groupby(by="TradingDate")
        tradingdate = list(df_group.groups.keys())
        # 回测
        # 评估公式: score = all_add(x - 1.06) / np.var(x)
        p, t = back_test(data, testYear, stockhold, tradingdate)
        print("total:")
        # print("5-10:" + str(t["5_10"]))
        # print("7:" + str(t["7"]))
        print("nnn:" + str(t["nnn"]))

        # np5_10 = np.array(p["5_10"])
        # np7 = np.array(p["7"])
        npnnn = np.array(p["nnn"])
        print("mean:")
        # print(np.mean(np5_10))
        # print(np.mean(np7))
        print(np.mean(npnnn))
        # score5_10 = np.mean(np5_10) / np.var(np5_10)
        # score7 = np.mean(np7) / np.var(np7)
        scorennn = np.mean(npnnn) / np.var(npnnn)
        print("score:")
        # print("5-10:" + str(score5_10))
        # print("7:" + str(score7))
        print("nnn:" + str(scorennn))
