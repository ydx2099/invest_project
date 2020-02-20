import numpy as np
import pandas as pd
import os
import datetime
from dateutil.relativedelta import relativedelta
import time
import test_data_produce as ts
import random


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
def back_test(data, test_years, max_stockhold):
    # 交易频率(天)
    up10_trading_freq = 10 
    down_trading_freq = 20
    # 最小持股数 = (最大持股数 / 2) + 1
    # min_stockhold = max_stockhold / 2 + 1
    start_date = today - test_years * 10000
    use_data = data[data['TradingDate'] >= start_date - 200]
    # 记录持股
    stock_hold_up10 = []
    stock_hold_down10 = []
    stock_hold_7 = []
    # 记录收益
    profit_up10 = [1.0, 1.0]
    profit_down10 = [1.0, 1.0]
    profit_7 = 1.0

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


# 
def cal_profit(data, max_day, profit):
    date = ""
    df_group = data.groupby(by="Symbol")
    stock_list = list(df_group.groups.keys())
    dayn_profit = []
    for i in stock_list:
        cur_data = data.loc[data['Symbol'] == i]
        cur_data = cur_data.sort_values("TradingDate")
        for j in range(cur_data.shape[0] - max_day + 1):
            cur_profit = 0.0
            max_profit = 0.0
            date = cur_data.iloc[j]['TradingDate']
            stock_id = cur_data.iloc[j]['Symbol']
            # n天后同id才计算
            if cur_data.iloc[j]['Symbol'] == cur_data.iloc[j + max_day - 1]['Symbol']:
                for k in range(0, max_day): 
                    if cur_data.shape[0] <= k + j + 1:
                        continue
                    day_data = cur_data.iloc[k + j]['ChangeRatio']
                    cur_profit = (1 + cur_profit) * day_data + cur_profit
                    if cur_profit >= max_profit:
                        max_profit = cur_profit
                dayn_profit.append([stock_id, date, cur_profit, max_profit])
    
    # 转为dataframe输出
    df_profit = pd.DataFrame(dayn_profit, columns=['Symbol', 'TradingDate', 'ChangeRatio', 'max'])

    return df_profit


# 对单个策略进行回测
def single_test(data, test_years, trading_freq, max_hold, test_days):
    start_date = today - test_years * 10000
    use_data = data[data['TradingDate'] >= start_date - 200]
    # 记录持股
    stock_hold = []
    # 记录收益
    profit = [1.0, 1.0]

    # n天涨跌n%策略
    # 测试方法为每两个月按策略进行买入卖出，计算最终收益
    # 一个月为买入期，一个月为卖出期
    for i in range(0, test_years * 6):
        # 计算收益
        if not i == 0:  
            cal_data = use_data[use_data['TradingDate'] <= start_date + i * 100]
            cur_profit = [0.0, 0.0]
            # 计算每只持有股票的最终/最高收益
            for stock in stock_hold:
                stock_data = cal_data[cal_data['Symbol'] == stock['Symbol']]
                stock_data = stock_data[stock_data['TradingDate'] > stock['TradingDate']]
                end_p, max_p = get_end_profit(stock_data)
                cur_profit[0] += end_p
                cur_profit[1] += max_p
                
            cur_profit[0] /= len(stock_hold)
            cur_profit[1] /= len(stock_hold)
            # 更新profit
            profit[0] *= cur_profit[0]
            profit[1] *= cur_profit[1]
            
        # 取出当前批次数据
        cur_batch_data = use_data[use_data['TradingDate'] <= start_date + i * 100]
        cur_batch_data = cur_batch_data[cur_batch_data['TradingDate'] >= start_date + (i - 1) * 100]

        # 计算下一批次所选股票
        candidate_stocks = cal_profit(cur_batch_data, test_days, 0.1)
        # 先清空持股记录
        stock_hold = []
        # 随机选取五只
        for _ in range(0, min(max_hold, candidate_stocks.shape[0])):
            j = random.randint(0, candidate_stocks.shape[0])
            while candidate_stocks.iloc[j] in stock_hold:
                j = random.randint(0, candidate_stocks.shape[0])
            stock_hold.append(candidate_stocks.iloc[j])

    return profit

if __name__ == "__main__":
    data_path = r'C:\Users\wuziyang\Documents\PyWork\trading_simulation\data\stockdata\stock_latest.csv'
    data = pd.read_csv(data_path, sep=',', low_memory=False)
    data = data[['Symbol', 'TradingDate', 'ChangeRatio']]
    tdata = data[data['Symbol'] == 600000]
    tdata = tdata.sort_values("TradingDate", ascending=False)
    date_list = tdata['TradingDate'].tolist
    
    # 评估公式: score = all_add(x - 1.06) / np.var(x)