import numpy as np
import pandas as pd
import os
import datetime
from dateutil.relativedelta import relativedelta
import time
import test_data_produce as ts
import random
import threading as td
import lightgbm as lgb
from matplotlib import pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split,GridSearchCV,StratifiedKFold
import sklearn
from sklearn.externals import joblib
import multiprocessing as mltp
import tushare as ts
import heapq

# # 多进程
# def mult_pr(thread_num):
#     p = mltp.Pool(thread_num)
#     for i in range(0, thread_num):
#         p.apply_async(usrdata_func, args=(uid_lst[i], deviceModel, pkg, Vc, dis, dos,))
#     p.close()
#     p.join()

#TODO: 统计last跌next高开的比例
# 按条件生成特征，如涨7后的数据

# 当前日期
today = int(time.strftime("%Y%m%d", time.localtime()))


# 线程类
class myThread(td.Thread):
    def __init__(self, func, args, name=''):
        super(myThread, self).__init__()
        self.name = name
        self.func = func
        self.args = args
        
    def run(self):
        self.result = self.func(*self.args)

    def getRes(self):
        run()
        return self.result


# 持股交易类
class HoldStock():
    # 计算卖出日期与利率
    def cal_sell(self):
        hold_max_day = 60
        sell_max_profit = 1.10
        # sell_min_profit = 0.94
        drop_limit = 0.03
        max_profit = 1.0
        if self.data.shape[0] >= hold_max_day + 1:
            for i in range(1, hold_max_day):
                self.p *= (self.data.iloc[i]["ChangeRatio"] + 1)
                if self.p > max_profit:
                    max_profit = self.p
                self.enddate = self.data.iloc[i]["TradingDate"]
                # self.holdday += 1
                # 满足条件提前终止循环,由于backtest逻辑，因此不需要处理当日卖出情况
                if self.p > sell_max_profit:
                    if self.p <= max_profit - drop_limit:
                        # earn_logger.writelines("ID:{},GetDate:{},SellDate:{},Profit:{}"\
                        #     .format(self.symbol, self.data.iloc[0]['TradingDate'], self.enddate, self.p))
                        # earn_logger.writelines('\n')
                        break
                    # self.p *= data.iloc[i + 1]['Open'] / data.iloc[i]['Close']
                    # break
                # if self.p < sell_min_profit and self.holdday > 1:
                # if self.p < sell_min_profit or (self.holdday >= 3 and self.p <= 1.06):
                #     self.p *= data.iloc[i]['Close'] / data.iloc[i + 1]['Open']
                #     loss_logger.writelines("ID:{},GetDate:{},SellDate:{},Profit:{}"\
                #         .format(self.symbol, self.data.iloc[0]['TradingDate'], self.enddate, self.p))
                #     loss_logger.writelines('\n')
                #     # self.p *= data.iloc[i + 1]['Open'] / data.iloc[i]['Close']
                #     break
            # earn_logger.writelines("ID:{},GetDate:{},SellDate:{},Profit:{}"\
            #     .format(self.symbol, self.data.iloc[0]['TradingDate'], self.enddate, self.p))
            # earn_logger.writelines('\n')



    def __init__(self, symbol:str, amount:float, data):
        self.symbol = symbol
        self.amount = amount
        self.p = 1.0 * data.iloc[0]['Close'] / data.iloc[1]['Open']
        self.data = data
        self.enddate = 0
        self.holdday = 0
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
    cal_date = tradingdate[20:-2]
    # #
    # all_count = 0
    # all_contain = 0
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
    for day in cal_date[:-2]:
        # 持股不满时才需要计算
        # if len(stock_hold_5_10) < max_stockhold:
        #     # 获取筛选结果
        #     sel_5_10 = cal_5_10(data, day, tradingdate, -0.08, -0.12)
        #     for i in range(0, 5 - len(stock_hold_5_10)):
        #         if len(sel_5_10) > i:
        #             stock_data = data[data['Symbol'] == sel_5_10[i][0]]
        #             stock_data = stock_data[stock_data['TradingDate'] > day]
        #             spend = min(balance["5_10"], total["5_10"] / 5)
        #             balance["5_10"] -= spend
        #             stock_hold_5_10.append(HoldStock(sel_5_10[i][0], spend, stock_data))
        # if len(stock_hold_7) < max_stockhold:
        #     # 获取筛选结果
        #     sel_7 = cal_7(data, day, tradingdate)
        #     # all_count += temp_count
        #     # all_contain += temp_contain
        #     for i in range(0, min(len(sel_7), 5 - len(stock_hold_7))):
        #         stock_data = data[data['Symbol'] == sel_7[-i][0]]
        #         stock_data = stock_data[stock_data['TradingDate'] >= day]
        #         spend = min(balance["7"], total["7"] / 5)
        #         balance["7"] -= spend
        #         stock_hold_7.append(HoldStock(sel_7[-i][0], spend, stock_data))
        if len(stock_hold_nnn) < max_stockhold:
            # 获取筛选结果
            sel_nnn = cal_nnn(data, day, tradingdate)
            # sel_nnn = cal_uuu(data, day, tradingdate)
            for i in range(0, 5 - len(stock_hold_nnn)):
                if len(sel_nnn) > i:
                    stock_data = data[data['Symbol'] == sel_nnn[i]]
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

    # print("contain rate:" + str(all_contain / all_count))
    return sell, total


# 按lgbm特征提取数据特征
def cal_lgbm(data, date, tradingdate):
    models = []
    for i in range(0, 5):
        models.append(joblib.load(r'D:\wuziyang\workfile\model' + str(i) + r'.m'))
    res = []
    df_group = data.groupby(by="Symbol")
    stock_list = list(df_group.groups.keys())
    for s in stock_list:
        cur_data = data[data['Symbol'] == s]
        cur_data.sort_values(by='TradingDate', ascending=True)
        # 获取对应日期数据
        today_indexes = tradingdate.index(date)
        cal_date = tradingdate[today_indexes - 20: today_indexes]
        cur_data = cur_data[cur_data['TradingDate'] >= cal_date[0]]
        cur_data = cur_data[cur_data['TradingDate'] <= cal_date[-1]]
        # 计算
        if cur_data.shape[0] == 20:
            # 预测的p
            old_data = cur_data[:20]
            old_data = old_data['Close'].values
            av = np.mean(old_data)
            std = np.std(old_data)
            # 20天的p
            p20 = cur_data.iloc[-1]['Close'] / cur_data.iloc[0]['Close']
            # 10天的p
            p10 = cur_data.iloc[-1]['Close'] / cur_data.iloc[10]['Close']
            # 5天的p
            p5 = cur_data.iloc[-1]['Close'] / cur_data.iloc[15]['Close']
            # 超越均值的标准差倍数
            over_times = (cur_data.iloc[-1]['Close'] - av) / std
            # std比av
            std_rate = std / av

            res.append([s, date, p5, p10, p20, over_times])
    # 保存当天数据
    pd.DataFrame(res, columns=['Id', 'Date', 'P5', 'P10', 'P20', 'OverTimes']).to_csv(r'D:\wuziyang\workfile\lgbm_feature.csv')
    test_feature = np.array(res)[:, 2:]
    test_pred_prob = np.zeros((len(test_feature), 2))
    for m in models:
        test_pred_prob += m.predict(test_feature, num_iteration=m.best_iteration) / 5
    # 保存当天数据
    pd.DataFrame(test_pred_prob, columns=['pos', 'neg']).to_csv(r'D:\wuziyang\workfile\lgbm_class.csv')

    return


def cal_5_10(data, date, tradingdate, high_limit, low_limit):
    # 获取对应日期数据
    today_indexes = tradingdate.index(date)
    cal_date = tradingdate[today_indexes - 5: today_indexes + 1]
    today_data = data[data['TradingDate'] >= cal_date[0]]
    today_data = today_data[today_data['TradingDate'] <= cal_date[5]]
    tommorow_data = data[data['TradingDate'] == tradingdate[today_indexes + 1]]
    # 计算
    thread_num = 8
    res_list = list()
    df_group = data.groupby(by="Symbol")
    stock_list = list(df_group.groups.keys())
    st_lists = np.array_split(stock_list, thread_num)
    tds = []
    for i in range(0, thread_num):
        t = myThread(func_5_10, (st_lists[i], today_data, today_indexes, tommorow_data, high_limit, low_limit, data))
        tds.append(t)
        tds[i].start()
    for i in range(0, thread_num):
        tds[i].join()
        res_list += tds[i].getRes()

    res_list.sort(key=(lambda x: x[-1]))
    # random.shuffle(res_list)
    return res_list[0:4]

def func_5_10(stock_list, today_data, today_indexes, tommorow_data, high_limit, low_limit, data):
    filter_data = list()
    for i in stock_list:
        cur_data = today_data[today_data['Symbol'] == i]
        next_data = tommorow_data[tommorow_data['Symbol'] == i]
        cur_data.sort_values(by='TradingDate', ascending=True)
        # 计算
        cur_p = 1.0
        if next_data.shape[0] != 0 and cur_data.shape[0] == 6:
            cur_c = cur_data.iloc[5]['Close']
            final_p = cur_data.iloc[5]['ChangeRatio']
            next_p = next_data.iloc[0]['ChangeRatio']
            for j in range(0, 5):
                cur_p *= (cur_data.iloc[j]['ChangeRatio'] + 1)
            cur_p = cur_p - 1
            if low_limit < cur_p < high_limit and final_p >= 0.07:
                flag, over_rate = over_aver(data, today_indexes, cur_data.iloc[5]['Close'])
                print(over_rate)
                if next_p <= 1.095 * cur_c and flag:
                    filter_data.append([i, over_rate])

    return filter_data


# n天连涨策略
def cal_nnn(data, date, tradingdate):
    # 提取最近3天数据
    today_indexes = tradingdate.index(date)
    cal_date = tradingdate[today_indexes - 2: today_indexes + 1]
    today_data = data[data['TradingDate'] >= cal_date[0]]
    today_data = today_data[today_data['TradingDate'] <= cal_date[2]]
    tommorow_data = data[data['TradingDate'] == tradingdate[today_indexes + 1]]
    # 计算
    filter_data = list()
    df_group = today_data.groupby(by="Symbol")
    stock_list = list(df_group.groups.keys())
    for i in stock_list:
        cur_data = today_data[today_data['Symbol'] == i]
        next_data = tommorow_data[tommorow_data['Symbol'] == i]
        if next_data.shape[0] != 0 and cur_data.shape[0] == 3:
            cur_c = cur_data.iloc[0]['Close']
            next_h = next_data.iloc[0]['Max']
            next_l = next_data.iloc[0]['Min']
            next_c = next_data.iloc[0]['Close']
            next_o = next_data.iloc[0]['Open']
            cur_data.sort_values(by='TradingDate', ascending=True)
            if (1 + cur_data.iloc[0]['ChangeRatio']) * (1 + cur_data.iloc[1]['ChangeRatio']) * (1 + cur_data.iloc[2]['ChangeRatio']) >= 1.2:
            # if cur_data.iloc[0]['ChangeRatio'] > 0 and cur_data.iloc[1]['ChangeRatio'] > 0 and cur_data.iloc[2]['ChangeRatio'] > 0:
                if cur_c <= next_o <= 1.095 * cur_c:
                    filter_data.append(cur_data.iloc[2]['Symbol'])

    random.shuffle(filter_data)
    return filter_data


# n天止跌策略
def cal_uuu(data, date, tradingdate):
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
            if cur_data.iloc[0]['ChangeRatio'] < 0 and cur_data.iloc[1]['ChangeRatio'] < 0 and cur_data.iloc[2]['ChangeRatio'] < 0 and cur_data.iloc[3]['ChangeRatio'] > 0:
                filter_data.append([cur_data.iloc[2]['Symbol'], cur_data.iloc[2]['TradingDate']])

    random.shuffle(filter_data)
    return filter_data


# 涨7策略
def cal_7(data, date, tradate):
    # 提取最近3天数据
    today_indexes = tradate.index(date)
    cal_date = tradate[today_indexes - 2: today_indexes + 1]
    today_data = data[data['TradingDate'] >= cal_date[0]]
    today_data = today_data[today_data['TradingDate'] <= cal_date[2]]
    tommorow_data = data[data['TradingDate'] == tradate[today_indexes + 1]]
    # 计算
    thread_num = 8
    res_list = list()
    df_group = today_data.groupby(by="Symbol")
    stock_list = list(df_group.groups.keys())
    st_lists = np.array_split(stock_list, thread_num)
    tds = []
    for i in range(0, thread_num):
        t = myThread(func7, (st_lists[i], today_indexes, today_data, tommorow_data, data))
        tds.append(t)
        tds[i].start()
    for i in range(0, thread_num):
        tds[i].join()
        res_list += tds[i].getRes()

    res_list = sorted(res_list, key=lambda x: x[-1])
    random.shuffle(res_list)
    return res_list#, amount, contain_amount

def func7(stock_list, today_indexes, today_data, tommorow_data, data):
    filter_data = list()
    for i in stock_list:
        cur_data = today_data[today_data['Symbol'] == i]
        next_data = tommorow_data[tommorow_data['Symbol'] == i]
        if next_data.shape[0] != 0 and cur_data.shape[0] != 0:
            # cur_c = cur_data.iloc[0]['Close']
            # next_o = next_data.iloc[0]['Open']
            cur_data.sort_values(by='TradingDate', ascending=True)
            # if cur_data.iloc[0]['ChangeRatio'] > 0 and cur_data.iloc[1]['ChangeRatio'] > 0 and cur_data.iloc[2]['ChangeRatio'] >= 0.07:
            if 0.07 <= cur_data.iloc[0]['ChangeRatio'] <= 0.095:
                flag, over_rate = over_aver(data, today_indexes, cur_data.iloc[0]['Close'])
                # if next_o <= 1.03 * cur_c and flag:
                # if cur_c * 0.97 <= next_o <= cur_c * 1.03 and flag:
                if flag:
                    filter_data.append([cur_data.iloc[0]['Symbol'], over_rate])

    return filter_data


# 是否超越均线+方差
def over_aver(data, index, cur_price):
    cal_data = data[index - 20: index + 1]['Close'].values
    mean = np.mean(cal_data)
    std = np.std(cal_data)
    if mean + 3 * std <= cur_price:
        rate = (cur_price - mean) / std
        return True, rate
    else:
        return False, 0.0


# test feature extract
class extract_feature():
    def __init__(self, data):
        self.data = data
        self.thread_num = 16
        # 获取所有交易日
        df_group = data.groupby(by="TradingDate")
        self.date = list(df_group.groups.keys())
        # 所有代码
        df_group = data.groupby(by="Symbol")
        self.stock = list(df_group.groups.keys())


    def filter_func(self, dates:np.array):
        # 收集结果
        pos_res = list()
        neg_res = list()
        mid_res = list()
        for date in dates:
            # 获取对应日期数据
            today_indexes = self.date.index(date)
            cal_date = self.date[today_indexes - 20: today_indexes + 5]
            today_data = self.data[self.data['TradingDate'] >= cal_date[0]]
            today_data = today_data[today_data['TradingDate'] <= cal_date[-1]]
            for i in self.stock:
                cur_data = today_data[today_data['Symbol'] == i]
                cur_data.sort_values(by='TradingDate', ascending=True)
                # 计算
                if cur_data.shape[0] == 25:
                    cur_p = 1.0
                    buy_p = cur_data.iloc[-5]['Open'] / cur_data.iloc[-6]['Close']
                    if buy_p <= 1.095:
                        # 预测的p
                        cur_p = cur_data.iloc[-1]['Close'] / cur_data.iloc[-5]['Open']
                        old_data = cur_data[:20]
                        old_data = old_data['Close'].values
                        av = np.mean(old_data)
                        std = np.std(old_data)
                        # 20天的p
                        p20 = cur_data.iloc[-6]['Close'] / cur_data.iloc[0]['Close']
                        # 10天的p
                        p10 = cur_data.iloc[-6]['Close'] / cur_data.iloc[10]['Close']
                        # 5天的p
                        p5 = cur_data.iloc[-6]['Close'] / cur_data.iloc[15]['Close']
                        # 超越均值的标准差倍数
                        over_times = (cur_data.iloc[-6]['Close'] - av) / std
                        # std比av
                        std_rate = std / av
                        if cur_p >= 1.12:
                            # 记录
                            pos_res.append([i, date, p5, p10, p20, over_times, std_rate])
                            # pos_res.append([i, date, p5, over_times, std_rate])
                        if 1.12 > cur_p > 1:
                            # mid_res.append([i, date, p5, over_times, std_rate])
                            mid_res.append([i, date, p5, p10, p20, over_times, std_rate])
                        if cur_p <= 1:
                            # 记录
                            neg_res.append([i, date, p5, p10, p20, over_times, std_rate])
                            # neg_res.append([i, date, p5, over_times, std_rate])

        return pos_res, mid_res, neg_res


    # 获取正样本
    def cal_sample(self):
        # dates = np.array_split(self.date[20:], self.thread_num)
        pos_res = list()
        neg_res = list()
        mid_res = list()
        pos_res, mid_res, neg_res = self.filter_func(self.date[20:])
        # tds = []
        # for i in range(0, self.thread_num):
        #     t = myThread(self.filter_func, args=(dates[i],))
        #     tds.append(t)
        #     tds[i].start()
        # for i in range(0, self.thread_num):
        #     tds[i].join()
        #     pos_tmp = list()
        #     neg_tmp = list()
        #     mid_tmp = list()
        #     pos_tmp, mid_tmp, neg_tmp = tds[i].getRes()
        #     pos_res += pos_tmp
        #     neg_res += neg_tmp
        #     mid_res += mid_tmp

        # 写出
        # pd.DataFrame(pos_res, columns=['Id', 'Date', 'P5', 'OverTimes', 'Std']).to_csv(r'D:\wuziyang\pos.csv', index=False)
        # pd.DataFrame(neg_res, columns=['Id', 'Date', 'P5', 'OverTimes', 'Std']).to_csv(r'D:\wuziyang\neg.csv', index=False)
        # pd.DataFrame(mid_res, columns=['Id', 'Date', 'P5', 'OverTimes', 'Std']).to_csv(r'D:\wuziyang\mid.csv', index=False)
        pd.DataFrame(pos_res, columns=['Id', 'Date', 'P5', 'P10', 'P20', 'OverTimes', 'Std']).to_csv(r'D:\wuziyang\workfile\\pos3.csv', index=False)
        pd.DataFrame(mid_res, columns=['Id', 'Date', 'P5', 'P10', 'P20', 'OverTimes', 'Std']).to_csv(r'D:\wuziyang\workfile\\mid3.csv', index=False)
        pd.DataFrame(neg_res, columns=['Id', 'Date', 'P5', 'P10', 'P20', 'OverTimes', 'Std']).to_csv(r'D:\wuziyang\workfile\\neg3.csv', index=False)
        return


# lgbm with kfold
# 将数据根据不同策略进行分类，收益符合预期的为大于1的类别，失败例分为类别0
# 使用lgbm框架训练，对后续数据进行预测
# TODO：跟踪预测数据的结果，判断是否正确，重新加权训练
def lgbmKfold(datas, labels, classes, testData=None, testLabel=None):
    # iris = sklearn.datasets.load_iris()
    # datas = iris.data
    # labels = iris.target
    if testData == None and testLabel == None:
        X_train,X_test,y_train,y_test=train_test_split(datas, labels, test_size=0.2, random_state=2020)
    else:
        X_train = datas
        X_test = testData
        y_train = labels
        y_test = testLabel
    params={
        'boosting_type': 'gbdt',  
        'learning_rate':0.1,
        'lambda_l1':0.1,
        'lambda_l2':0.2,
        'max_depth':4,
        'objective':'multiclass',
        'num_class':classes,
    }
    # k折交叉验证
    folds_num = 5
    skf = StratifiedKFold(n_splits=folds_num, random_state=2020, shuffle=True)
    test_pred_prob = np.zeros((X_test.shape[0], classes))
    for i, (trn_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(len(trn_idx))

        print(len(val_idx))
        train_data = lgb.Dataset(X_train[trn_idx], label=y_train[trn_idx])
        validation_data = lgb.Dataset(X_train[val_idx], label=y_train[val_idx])
        # 训练
        clf = lgb.train(params, train_data, valid_sets=[validation_data], num_boost_round=1000)
        # 保存模型
        joblib.dump(clf, r'D:\wuziyang\workfile\model' + str(i) + r'.m')
        # # 当前折预测
        # clf.predict(X_train[val_idx], num_iteration=clf.best_iteration)
        # 加权计算预测分数
        test_pred_prob += clf.predict(X_test, num_iteration=clf.best_iteration) / folds_num
    
    pred_class = test_pred_prob.argmax(axis=1)
    # 获取预测正样本下标
    pos_indexes = np.argwhere(pred_class == 0)
    # 计算正样本预测正确概率
    correct_num = len(np.argwhere(y_test[pos_indexes] == 0))
    print(len(pos_indexes))
    print(correct_num)
    # 获取预测负样本下标
    neg_indexes = np.argwhere(pred_class == 1)
    # 计算负样本预测正确概率
    neg_error_num = len(np.argwhere(y_test[neg_indexes] == 0))
    print(len(neg_indexes))
    print(neg_error_num)

    # correct = sum(pred_class==y_test)
    # print("accuracy:" + str(correct / len(pred_class)))
    return pred_class

# machine learns
# all is over 12, and per is below 6
# in 5 or 10 days
def machine_learning():
    # you can use lgbm to classify successes compare to failures
    
    # get confirm data, select data and give label
    pos_data = pd.read_csv(r'D:\wuziyang\workfile\pos.csv', sep=',')
    pos_datas = pos_data[['P5', 'P10', 'P20', 'OverTimes']]
    labels = [0 for i in range(0, pos_datas.shape[0])]
    datas = pos_datas

    neg_data = pd.read_csv(r'D:\wuziyang\workfile\neg.csv', sep=',')
    neg_datas = neg_data[['P5', 'P10', 'P20', 'OverTimes']]
    labels += [1 for i in range(0, neg_datas.shape[0])]
    datas = datas.append(neg_datas)
    
    ret = lgbmKfold(np.array(datas), np.array(labels), 2)
    print(ret)
    return

# 统计测试均值回归
def AverRegression_test(data,tradingdate):
    # 计算日期
    cal_date = tradingdate[60:-60]
    # 创建策略总值map,余额总值map
    total = 1.0
    balance = 1.0
    # 记录持股
    stock_hold = []
    # 记录每次卖出收益
    sell = []
    # 回测开始
    for day in cal_date:
        if len(stock_hold) < 5:
            # 获取筛选结果
            sel_s = cal_AverRegression(data, day, tradingdate)
            for i in range(0, 5 - len(stock_hold)):
                if len(sel_s) > i:
                    stock_data = data[data['Symbol'] == sel_s[i][0]]
                    stock_data = stock_data[stock_data['TradingDate'] > day]
                    spend = min(balance, total / 5)
                    balance -= spend
                    stock_hold.append(HoldStock(sel_s[i][0], spend, stock_data))
        for hold in stock_hold:
            if int(day) == int(hold.enddate):
                total += hold.amount * (hold.p - 1)
                balance += hold.amount * hold.p
                sell.append(hold.p)
                stock_hold.remove(hold)
                del hold
    print(total)
    print(np.average(np.array(sell)))
    return
# 均值回归
def cal_AverRegression(data, day, tradingdate):
    res = []
    today_indexes = tradingdate.index(day)
    cur_data = data[data['TradingDate'] >= tradingdate[today_indexes - 60]]
    cur_data = cur_data[cur_data['TradingDate'] <= day]
    df_group = cur_data.groupby(by="Symbol")
    stocks = list(df_group.groups.keys())
    for i in stocks:
        this_data = cur_data[cur_data['Symbol'] == i]
        if this_data.shape[0] == 60:
            this_data = this_data[['Close']]
            data_array = np.array(this_data)
            post_aver = np.average(data_array[:60])
            if post_aver > data_array[-1]:
                res.append([i, post_aver / data_array[-1]])
    res = sorted(res, key=lambda x: x[-1], reverse=True)
    return res

# 统计测试drop rate
class drop_rate_test():
    def __init__(self, data):
        self.data = data
        self.thread_num = 16
        # 股票下跌卖出最高比例
        self.drop_rate = 0.02


    def test_drop_rate(self):
        df_group = data.groupby(by="Symbol")
        # 按线程分割原股票，进行多线程处理
        sts = np.array_split(list(df_group.groups.keys()), self.thread_num)
        tds = []
        res = list()
        # 建立多线程并执行
        for i in range(0, self.thread_num):
            t = myThread(self.filter_func, args=(sts[i],))
            tds.append(t)
            tds[i].start()
        for i in range(0, self.thread_num):
            tds[i].join()
            # 获取执行结果
            res += tds[i].getRes()

        return res

    def filter_func(self, stocks):
        # 存储结果
        res = list()
        # 处理所有股票
        for stock in stocks:
            cur_stock_data = self.data[self.data['Symbol'] == stock]
            # 获取所有交易日
            df_group = cur_stock_data.groupby(by="TradingDate")
            total_date = list(df_group.groups.keys())
            # 对一支股票，计算所有交易日期符合条件数据，统计结果
            for day in total_date[5:]:
                # 获取某天在所有日期中的位置
                day_index = total_date.index(day)
                early_data = cur_stock_data[cur_stock_data['TradingDate'] == total_date[day_index - 5]]
                today_data = cur_stock_data[cur_stock_data['TradingDate'] == day]
                if today_data.shape[0] == 1 and early_data.shape[0] == 1:
                    cur_p = today_data.iloc[0]['Close'] / early_data.iloc[0]['Open']
                    max_p = cur_p
                    # 只处理过去五天盈利过12的
                    if cur_p > 1.12:
                        # 计算接下来的交易日，按drop rate最终能够获得的盈利
                        for after_day in total_date[day_index:]:
                            cur_p *= 1 + cur_stock_data[cur_stock_data['TradingDate'] == after_day].iloc[0]['ChangeRatio']
                            if cur_p >= max_p:
                                max_p = cur_p
                            if max_p - self.drop_rate >= cur_p:
                                res.append(cur_p)
                                break

        return res


def test_func(data):
    # 配置参数
    testYear = 1
    stockhold = 5
    test_batch = 1
    # 汇总计算
    aver = 0.0
    score = 0.0
    # 多次测试求平均
    for i in range(0, test_batch):
        # 用于test nnn
        # for i in range(0, testYear):
        #     tradedata = data[data['TradingDate'] >= 10000*(2020-i) + 115]
        #     tradedata = tradedata[tradedata['TradingDate'] <= 10000*(2020-i) + 300]
        # 用于test7
        # tradedata = data[data['TradingDate'] >= today - (testYear + 2) * 10000]
        # tradedata = tradedata[tradedata['TradingDate'] <= today - testYear * 10000]
        #
        # 获取最近n年数据
        tradedata = data[data['TradingDate'] >= today - testYear * 10000]
        # 获取所有交易日
        df_group = tradedata.groupby(by="TradingDate")
        tradingdate = list(df_group.groups.keys())
        # 回测
        # 评估公式: score = all_add(x - 1.06) / np.var(x)
        p, t = back_test(data, testYear, stockhold, tradingdate)
        print("total:")
        # print("5-10:" + str(t["5_10"]))
        # print("7:" + str(t["7"]))
        # aver += t["5_10"]
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
        # score += score7
        print("nnn:" + str(scorennn))

    # print("aver:" + str(aver / test_batch))
    # print("score:" + str(score / test_batch))


def temp_use():
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    myfont=matplotlib.font_manager.FontProperties(fname='C:/Windows/Fonts/STXINWEI.TTF')

    pos = pd.read_csv(r'D:\wuziyang\pos.csv')
    neg = pd.read_csv(r'D:\wuziyang\neg.csv')
    pos_lis = pos[['OverTimes', 'Std']].values
    neg_lis = neg[['OverTimes', 'Std']].values

    # p5 = pos_lis[:, 0]
    # p5 = np.append(p5, neg_lis[:, 0])
    # std5 = np.std(p5)
    # p10 = pos_lis[:, 1]
    # p10 = np.append(p10, neg_lis[:, 1])
    # std10 = np.std(p10)
    # p20 = pos_lis[:, 2]
    # p20 = np.append(p20, neg_lis[:, 2])
    # std20 = np.std(p20)
    # ot = pos_lis[:, 3]
    # ot = np.append(ot, neg_lis[:, 3])
    # stdot = np.std(ot)
    # print(std5 / np.mean(p5))
    # print(std10 / np.mean(p10))
    # print(std20 / np.mean(p20))
    # print(stdot / np.mean(ot))


    # avx1 = np.mean(pos_lis[:, 0])
    # avy1 = np.mean(pos_lis[:, 1])
    # avx2 = np.mean(neg_lis[:, 0])
    # avy2 = np.mean(neg_lis[:, 1])
    # x1 = pos_lis[:, 0] / avx1
    # y1 = pos_lis[:, 1] / avy1
    # x2 = neg_lis[:, 0] / avx2
    # y2 = neg_lis[:, 1] / avy2
    # plt.scatter(x1, y1, label=u"Hive写入平均耗时", marker='o', alpha=1, s=32, c='', edgecolors='r')
    # plt.scatter(x2, y2, label=u"Hive写入平均耗时", marker='*', alpha=1, s=32, c='', edgecolors='b')

    plt.show()

#
def test(data, tradingdate):
    df_group = data.groupby(by="Symbol")
    stocks = list(df_group.groups.keys())
    num = 80# np.arrange(60,120,10)
    add = np.arange(0,200,10)# num/2
    print(tradingdate[num])
    print(len(stocks))
    for k in range(0, len(add)):
        count = 0
        p_count = 0
        over_count = 0
        res = []
        for i in stocks:
            c_d = data[data['Symbol'] == i]
            c_d = np.array(c_d[['Close']])
            if len(c_d) >= num * 1.5 + add[k]:
                c_d = c_d[add[k]: 2 * num - 1 + add[k]]
                p_aver = np.average(c_d[:num])
                p_std = np.std(c_d[:num])
                p_max = max(c_d[:num])
                p_min = min(c_d[:num])
                p_mid = (p_max + p_min) / 2
                if p_aver - 1 * p_std > c_d[num] and c_d[num] / c_d[num - 4] > 1.1:
                    count += 1
                    # res.append(i)
                    if max(c_d[num + 1:]) > c_d[num] * 1.1:
                        p_count += 1
                        res.append(1.1)
                    else:
                        res.append(min(c_d[num + 1:]) / c_d[num])
                    # if max(c_d[num + 1:]) > c_d[num] * 1.08:
                    #     over_count += 1
        print(count)
        print(p_count)
        print(over_count)
        print(np.average(np.array(res)))
        print('------')
    # pd.DataFrame(res, columns=['id']).to_csv(r'D:\wuziyang\test.csv', index=False)


# KNN
def knn(data):
    return


# 获取某日涨幅前5概念股
# 输入数据为单日数据
def get_concept_stock(data):
    # 获取股票所属行业
    # ts.get_industry_classified()
    # 获取股票所属概念
    concept_df = ts.get_concept_classified()
    concept_group = concept_df.groupby(by="c_name")
    c_name = list(concept_group.groups.keys())
    concept_changerate = []
    # 计算各概念
    for i in c_name:
        cur_df = concept_df[concept_df['c_name'] == i]
        code_group = cur_df.groupby(by="code")
        code = list(code_group.groups.keys())
        # 计算涨跌幅
        concept_open = 0
        concept_close = 0
        for j in code:
            this_data = data[data['Symbol'] == j]
            concept_close += this_data['Close']
            concept_open += this_data['Open']
        
        concept_changerate += concept_close / concept_open
    # 获取涨幅前5概念下标
    upper_concept = heapq.nlargest(5, range(len(concept_changerate)), concept_changerate.take)
    # 只计算前5概念的股票
    for i in upper_concept:
        # 要涨
        if concept_changerate[i] > 1:
            cur_df = concept_df[concept_df['c_name'] == c_name[i]]


            

    df[df[''].isin()]
    heapq.nlargest(5, range(len(a)), a.take)


if __name__ == "__main__":
    print("start...")
    startt = time.time()
    # 记录日志
    earn_logger = open(r'D:\wuziyang\workfile\s7log_earn', 'a+')
    loss_logger = open(r'D:\wuziyang\workfile\s7log_loss', 'a+')
    earn_logger.writelines('\n')
    loss_logger.writelines('\n')

    data_path = r'D:\wuziyang\workfile\stock_latest.csv'
    data = pd.read_csv(data_path, sep=',', low_memory=False)
    # 去除科创板和沪B
    data = data[data['Symbol'] < 680000]
    # data = data[['Symbol', 'TradingDate', 'ChangeRatio']]
    data = data[data['TradingDate'] <= 20200200]
    data = data[data['TradingDate'] >= 20190200]
    # 特征提取
    e = extract_feature(data)
    e.cal_sample()

    # temp_use()
    # 策略测试
    # test_func(data)
    # 最佳drop rate测试
    # f = drop_rate_test(data)
    # d = f.test_drop_rate()
    # print(np.mean(np.array(d)))
    # print(np.std(np.array(d)))
    
    # # 生成一天的ml测试数据
    # df_group = data.groupby(by="TradingDate")
    # tradingdate = list(df_group.groups.keys())
    # cal_lgbm(data, 20200306, tradingdate)

    # # ml
    # machine_learning()

    # AverRegression_test(data, tradingdate)
    # test(data, tradingdate)

    # earn_logger.close
    # loss_logger.close

    print("time:" + str(time.time() - startt))
