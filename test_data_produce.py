 # ToDo:配置可拓展，支持同天数内不同利率的筛选

 # 已完成：
 # 1.计算n天的最大和最终涨跌幅并设置阈值筛选
 # TODO：
 # 筛选n天连涨状态的股票 || n天连涨且涨幅到达阈值
 # 测试n天涨幅达阈值的股票方案是否可靠

import numpy as np
import pandas as pd
import os
import datetime
from dateutil.relativedelta import relativedelta
import time
import lightgbm as lgb
from sklearn import datasets
from sklearn.model_selection import train_test_split,GridSearchCV,StratifiedKFold
from sklearn.metrics import f1_score
import sklearn


# 当前日期
today = int(time.strftime("%Y%m%d", time.localtime()))
ori_start_date = int((datetime.datetime.today() + datetime.timedelta(weeks=-20)).strftime('%Y%m%d'))
# 测试与验证路径
path_5_5 = 'C:/Users/wuziyang/Documents/PyWork/trading_simulation/test/5_-5/'
path_5_10 = 'C:/Users/wuziyang/Documents/PyWork/trading_simulation/test/5_-10/'
path_10_10 = 'C:/Users/wuziyang/Documents/PyWork/trading_simulation/test/10_-10/'
path_10_15 = 'C:/Users/wuziyang/Documents/PyWork/trading_simulation/test/10_-15/'
_path_5_5 = 'C:/Users/wuziyang/Documents/PyWork/trading_simulation/test/5_5/'
_path_5_10 = 'C:/Users/wuziyang/Documents/PyWork/trading_simulation/test/5_10/'
_path_10_10 = 'C:/Users/wuziyang/Documents/PyWork/trading_simulation/test/10_10/'
_path_10_15 = 'C:/Users/wuziyang/Documents/PyWork/trading_simulation/test/10_15/'
path_over7 = 'C:/Users/wuziyang/Documents/PyWork/trading_simulation/test/7/'
path_nnn = 'C:/Users/wuziyang/Documents/PyWork/trading_simulation/test/nnn/'



# 寻找符合要求的测试数据，将数据写入csv
def test_data_produce(data, count_days, drop_rate, path, isIncrease):
    #  路径不存在则创建
    if not os.path.exists(path):
        os.mkdir(path)
        os.mkdir(path + 'test_data/')
        os.mkdir(path + 'confirm_data/')
    # # 获取已有数据的最晚日期
    # start_date = max(get_latest_test_date(path + 'test_data/'))
    # 获取最新数据
    df_group = data.groupby(by="TradingDate")
    date_list = list(df_group.groups.keys())
    cal_data = data[data['TradingDate'] >= date_list[-count_days]]

    ss_group = cal_data.groupby(by="TradingDate")
    ss_list = list(ss_group.groups.keys())
    for i in ss_list: print(i)

    # 计算5天涨跌
    target_data = cal_profit(cal_data, count_days, True)
    # 按日期排序,输出每天跌幅达到阈值的数据
    df_group = target_data.groupby(by="TradingDate")
    date_list = list(df_group.groups.keys())
    for i in date_list:
        temp_list = []
        everyday_data = target_data[target_data['TradingDate'] == i]
        for row in everyday_data.values:
            # 根据传入参数决定取大于设定比率还是小于设定比率
            if isIncrease:
                if row[2] >= drop_rate:
                    temp_list.append(list(row))
            elif row[2] <= drop_rate:
                temp_list.append(list(row))
        if len(temp_list):
            print(str(int(i)) + 'is output')
            testdata_df = pd.DataFrame(temp_list, columns=['Symbol', 'TradingDate', 'ChangeRatio', 'max'])
            testdata_df.to_csv(path + 'test_data/' + str(int(i)) + '.csv', index=False)

# 收集测试数据对应的验证数据,更新相应文件
# result: ['countDays', 'ID', 'Date', 'total', 'max']
def confirm_data_update(data, count_days, path):
    test_dates = get_latest_test_date(path + 'test_data/')
    confirm_dates = filter(lambda x: x >= ori_start_date, test_dates)
    # 获取需要计算的股票
    stock_list = list()
    for dates in confirm_dates:
        if os.path.exists(path + 'test_data/' + str(dates) + '.csv'):
            test_data = pd.read_csv(path + 'test_data/' + str(dates) + '.csv')
            df_group = test_data.groupby(by="Symbol")
            stock_list += list(df_group.groups.keys())
    stock_set = set(stock_list)

    data = data[data['Symbol'] in stock_set]
    # 只计算test股票
    confirm_data = cal_profit(data, count_days, False)
    for i in confirm_dates:
        day_data = confirm_data[confirm_data['TradingDate'] == i]
        # 更新数据
        out_data = []
        if os.path.exists(path + 'test_data/' + str(i) + '.csv'):
            test_data = pd.read_csv(path + 'test_data/' + str(i) + '.csv')
            df_group = test_data.groupby(by="Symbol")
            id_list = list(df_group.groups.keys())
            for test_id in id_list:
                out_data.append(list(day_data.loc[day_data['Symbol'] == test_id]))

        # 转为dataframe输出
        confirm_df = pd.DataFrame(out_data, columns=['Symbol', 'TradingDate', 'ChangeRatio', 'max'])
        confirm_df.to_csv(path + 'confirm_data/' + str(int(i)) + '.csv', index=False)                


# 获取最近三个月的数据，用于提取测试与验证数据
def get_data(data):
    recent_data = data[data['TradingDate'] >= ori_start_date]
    return recent_data

# 获取验证数据的最晚日期,没有数据则返回三个月前的日期
def get_latest_confirm_date():
    confirm_date = []
    confirm_files = os.listdir(r'C:\Users\wuziyang\Documents\PyWork\trading_simulation\test\confirm_data')
    if len(confirm_files) == 0:
        return today
    for f in confirm_files:
        confirm_date.append(int(f))
    return confirm_date

# 获取测试数据的所有日期
def get_latest_test_date(path):
    test_date = []
    if not os.path.exists(path):
        test_date.append(ori_start_date)
        return test_date
    test_files = os.listdir(path)
    if len(test_files) == 0:
        test_date.append(ori_start_date)
        return test_date
    for f in test_files:
        test_date.append(int(f.split('.')[0]))
    return test_date


# 去除科创板筛选函数
def filter_fn(id):
    return id / 1000 != 688


# 处理输入数据，返回选择天数n的累计值和最大值
def cal_profit(data, up_day:int, test:bool):
    date = ""
    df_group = data.groupby(by="Symbol")
    stock_list = list(df_group.groups.keys())
    stock_list = filter(filter_fn, stock_list)
    dayn_profit = []
    # count = 0
    count_days = up_day
    # print("all stock:" + str(len(stock_list)))
    for i in stock_list:
        # count += 1
        # if count % 50 == 0:
        #     print("now :" + str(count))
        cur_data = data.loc[data['Symbol'] == i]
        cur_data = cur_data.sort_values("TradingDate")
        # 验证数据消除日期计算限制
        if not test:
            up_day = 1
        for j in range(0, cur_data.shape[0] - up_day + 1):
            cur_profit = 0.0
            max_profit = 0.0
            date = cur_data.iloc[j]['TradingDate']
            stock_id = cur_data.iloc[j]['Symbol']
            # n天后同id才计算
            if cur_data.iloc[j]['Symbol'] == cur_data.iloc[j + up_day - 1]['Symbol']:
                for k in range(0, count_days): 
                    # if cur_data.shape[0] <= k + j + 1:
                    #     continue
                    day_data = cur_data.iloc[k + j]['ChangeRatio']
                    date = cur_data.iloc[k + j]['TradingDate']
                    cur_profit = (1 + cur_profit) * day_data + cur_profit
                    if cur_profit >= max_profit:
                        max_profit = cur_profit
                dayn_profit.append([stock_id, date, cur_profit, max_profit])
    
    # 转为dataframe输出
    df_profit = pd.DataFrame(dayn_profit, columns=['Symbol', 'TradingDate', 'ChangeRatio', 'max'])

    return df_profit


# lgbm with kfold
# 将数据根据不同策略进行分类，收益符合预期的为大于1的类别，失败例分为类别0
# 使用lgbm框架训练，对后续数据进行预测
# TODO：跟踪预测数据的结果，判断是否正确，重新加权训练
def lgbmKfold(datas, labels, classes, testData=None, testLabel=None):
    if testData == None and testLabel == None:
        X_train,X_test,y_train,y_test=train_test_split(datas, labels, test_size=0.3, random_state=2020)
    else:
        X_train = datas
        X_test = labels
        y_train = testData
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
    for _, (trn_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        train_data = lgb.Dataset(X_train[trn_idx], label=y_train[trn_idx])
        validation_data = lgb.Dataset(X_test[val_idx], label=y_test[val_idx])
        # 训练
        clf = lgb.train(params, train_data, valid_sets=[validation_data], num_boost_round=1000)
        # 当前折预测
        clf.predict(X_train[val_idx], num_iteration=clf.best_iteration)
        # 加权计算预测分数
        test_pred_prob += clf.predict(X_test, num_iteration=clf.best_iteration) / folds_num

    return test_pred_prob


# filter data behind window
# cal n days average, get variance
# via statistic set suitable window
# get stocks over window or below window
# return map (over:data, below:data)
def cal_window(data, count_num):
    over_data = list()
    below_data = list()
    for i in range(0, len(data)):
        varience = data.iloc[i: i + count_num, 2].var()
        avr = data.iloc[i: i + count_num, 2].mean()
        if data.iloc[i + count_num, 2] > avr + 3 * varience:
            over_data += data.iloc[i + count_num]
        if data.iloc[i + count_num, 2] < avr - 3 * varience:
            below_data += data.iloc[i + count_num]
    res_map = {"over": over_data, "below": below_data}
    return res_map


# get confirm data and give label
def get_data_and_label(path):
    pd_data = pd.read_csv(path)
    datas = list()
    labels = list()
    for p in pd_data:
        if p["p"] >= 0.12:
            labels.append(1)
        else:
            labels.append(0)
        datas.append(p)

    return datas, labels


# machine learns
# all is over 12, and per is below 6
# in 5 or 10 days
def machine_learning():
    # you can use lgbm to classify successes compare to failures
    
    # get confirm data, select data and give label
    datas, labels = get_data_and_label(r'')
    
    lgbmKfold(datas, labels, 2)
    return


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

    # 转为dataframe输出
    confirm_df = pd.DataFrame(filter_data, columns=['Symbol'])
    confirm_df.to_csv('D:/wuziyang/workfile/_7_' + str(date) + '.csv', index=False)                

    return filter_data


# n天连涨策略
def cal_nnn(data, date):
    filter_data = list()
    df_group = data.groupby(by="Symbol")
    stock_list = list(df_group.groups.keys())
    for i in stock_list:
        cur_data = data[data['Symbol'] == i]
        for j in range(0, cur_data.shape[0] - 2):
            if cur_data.iloc[j]['ChangeRatio'] > 0 and cur_data.iloc[j + 1]['ChangeRatio'] > 0 and cur_data.iloc[j + 2]['ChangeRatio'] > 0:
                filter_data.append(cur_data.iloc[j + 2]['Symbol'])

    # 转为dataframe输出
    confirm_df = pd.DataFrame(filter_data, columns=['Symbol'])
    confirm_df.to_csv('D:/wuziyang/workfile/nnn_' + str(date) + '.csv', index=False)                

    return filter_data

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
def back_test(data, test_years, max_stockhold, cal_days):
    # 交易频率(天)
    up10_trading_freq = 10 
    down_trading_freq = 20
    # 记录回测所有持股
    all_up_hold = []
    all_down_hold = []
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
    for y in range(0, test_years):
        for i in range(0, test_years * 6):
            # 计算收益
            if not i == 0:  
                cal_data = use_data[use_data['TradingDate'] <= start_date + y * 10000 + i * 100]
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
            cur_batch_data = use_data[use_data['TradingDate'] <= start_date + y * 10000 + i * 100]
            cur_batch_data = cur_batch_data[cur_batch_data['TradingDate'] >= start_date + y * 10000 + (i - 1) * 100]

            # 计算下一批次所选股票
            simu_cal_data = cal_profit(cur_batch_data, cal_days, True)
            simu_cal_up10_data = simu_cal_data[simu_cal_data['ChangeRatio'] >= 0.1]
            simu_cal_down10_data = simu_cal_data[simu_cal_data['ChangeRatio'] <= -0.1]
            # simu_cal10_data = cal_profit(cur_batch_data, 10, True)
            # 按最终涨幅排序，从高到低
            simu_cal_up10_data = simu_cal_up10_data.sort_values(by='ChangeRatio', ascending=False)
            simu_cal_down10_data = simu_cal_down10_data.sort_values(by='ChangeRatio', ascending=False)
            # 先清空持股记录
            stock_hold_up10 = []
            stock_hold_down10 = []
            # 持有股票
            for j in range(0, min(max_stockhold, simu_cal_up10_data.shape[0])):
                stock_hold_up10.append(simu_cal_up10_data.iloc[j])
                all_up_hold.append([simu_cal_up10_data.iloc[j]['Symbol'], simu_cal_up10_data.iloc[j]['TradingDate']])

            for j in range(0, min(max_stockhold, simu_cal_down10_data.shape[0])):
                stock_hold_down10.append(simu_cal_down10_data.iloc[j])
                all_down_hold.append([simu_cal_up10_data.iloc[j]['Symbol'], simu_cal_up10_data.iloc[j]['TradingDate']])



    # # 涨7策略执行短线交易，每当拥有现金即进行交易
    # # 设置最大持有天数，暂为10个交易日
    # # 测试为简便起见，以10个交易日为一个交易周期
    # df_group = data.groupby(by="TradingDate")
    # date_list = list(df_group.groups.keys())
    # for i in date_list:
    #     cur_day_data = use_data[use_data['TradingDate'] == i]
    #     simu_7_data = cal_7(cur_batch_data, i)

    # 打印回测过程的持股
    pd.DataFrame(all_up_hold, columns=['Symbol', 'TradingDate']).to_csv(r'C:\Users\wuziyang\Desktop\up' + str(cal_days) + '.csv')
    pd.DataFrame(all_down_hold, columns=['Symbol', 'TradingDate']).to_csv(r'C:\Users\wuziyang\Desktop\down' + str(cal_days) + '.csv')
        
    return profit_up10, profit_down10

# 计算最近一个交易日的5-10策略5
def cal_5_10(data):
    # 计算
    filter_data = list()
    df_group = data.groupby(by="Symbol")
    stock_list = list(df_group.groups.keys())
    for i in stock_list:
        cur_data = data[data['Symbol'] == i]
        cur_data.sort_values(by='TradingDate', ascending=True)
        # 计算
        cur_p = 1.0
        data_size = cur_data.shape[0]
        if data_size >= 65:
            for j in range(0, 5):
                # 取最近五天数据
                cur_p *= (cur_data.iloc[data_size + j - 5]['ChangeRatio'] + 1)
            cur_p = cur_p - 1
            if cur_p > 0.1:
                cur_close = cur_data[['Close']].values
                # 获取5天前的收盘价
                recent_close = cur_close[-65:-5]
                # 获取均值和方差
                recent_average = np.average(recent_close)
                recent_std = np.std(recent_close)
                # 获取最高价
                recent_max = max(recent_close)
                # 从最高那天起，寻找最低
                recent_close = recent_close[np.argmax(recent_close):]
                # recent_min = min(recent_close)
                # # 以2/3和1/3的权重根据min和max计算中间值
                # recent_mid = 1/3 * recent_max + 2/3 * recent_min
                if recent_average - recent_std > cur_close[-1] and np.argmin(recent_close) > len(recent_close) - 3:
                    filter_data.append([i, cur_p])
    # 按利率从高到低排序
    filter_data.sort(key=(lambda x: x[-1]))
    # 获取计算日期
    cal_date = list(data.groupby(by="TradingDate").groups.keys())[-1]
    # 转为dataframe输出
    filter_df = pd.DataFrame(filter_data, columns=['Symbol', 'Profit'])
    filter_df.to_csv('D:/wuziyang/workfile/_510_' + str(cal_date) + '.csv', index=False)             


if __name__ == "__main__":
    data_path = r'D:\wuziyang\workfile\stock_latest.csv'
    data = pd.read_csv(data_path, sep=',', low_memory=False)

    # # 合并下载数据出错时简单回退
    # data = data[data['TradingDate'] <= 20200209]
    # data.to_csv(r'C:\Users\wuziyang\Documents\PyWork\trading_simulation\data\stockdata\stock_latest.csv', index=False)
    
    # # 精简数据
    # data = data[['Symbol', 'TradingDate', 'ChangeRatio']]
    # 去除科创板和沪B
    data = data[data['Symbol'] < 680000]
    # 删除涨跌幅为nan的行
    indexs = list(data[np.isnan(data['ChangeRatio'])].index)
    data = data.drop(indexs)
    # 获取最近数据
    rec_data = get_data(data)

    # 获取最近1天和n天数据
    n = 3
    tdata = rec_data[rec_data['Symbol'] == 600000]
    tdata = tdata.sort_values("TradingDate", ascending=False)
    last_tradingdate = int(tdata.iloc[0]['TradingDate'])
    last_n_tradingdate = int(tdata.iloc[n]['TradingDate'])
    yestoday_data = rec_data[rec_data['TradingDate'] == last_tradingdate]
    last_n_data = rec_data[rec_data['TradingDate'] > last_n_tradingdate]

    # # 回测
    # ups = []
    # downs = []
    # stock_hold = 5
    # test_year = 2
    # for i in range(3, 9):
    #     print(i)
    #     up, down = back_test(data, 2, stock_hold, i)
    #     ups.append(up)
    #     downs.append(down)
    # print(ups)
    # print(downs)
    # up, down = back_test(data, test_year, stock_hold, 2)
    # print(up)
    # print(down)

    # # 涨7策略
    # cal_7(yestoday_data, last_tradingdate)

    # # 连涨策略
    # cal_nnn(last_n_data, last_tradingdate)

    # n天涨n策略
    data = data[data['TradingDate'] <= 20190700]
    cal_5_10(data)

    # # test 5days, rate -5, confirm 50days
    # test_data_produce(rec_data, 5, -0.05, path_5_5, False)
    # # confirm_data_update(rec_data, 50, path_5_5)
    # # test 10days, rate -10, confirm 50days
    # test_data_produce(rec_data, 10, -0.1, path_10_10, False)
    # # confirm_data_update(rec_data, 50, path_10_10)
    # # test 5days, rate 5, confirm 50days
    # test_data_produce(rec_data, 5, 0.05, _path_5_5, True)
    # # test 10days, rate 10, confirm 50days
    # test_data_produce(rec_data, 10, 0.1, _path_10_10, True)
    # test_data_produce(rec_data, 10, 0.15, _path_10_15, True)
    # test_data_produce(rec_data, 10, -0.15, path_10_15, False)

    # LGBM
    # machine_learning()

    # # 先执行需要多天数据的策略再取短期数据
    # test_data_produce(rec_data, 5, 0.1, _path_5_10, True)
    print("is end")
