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


class cal_para():
    cal_day = 0
    cal_profit = 0.0
    res_path = ""
    

# 当前日期
today = int(time.strftime("%Y%m%d", time.localtime()))
ori_start_date = int((datetime.datetime.today() + datetime.timedelta(weeks=-20)).strftime('%Y%m%d'))
# 测试与验证路径
path_5_5 = 'C:/Users/wuziyang/Documents/PyWork/trading_simulation/test/5_-5/'
path_5_10 = 'C:/Users/wuziyang/Documents/PyWork/trading_simulation/test/5_-10/'
path_10_10 = 'C:/Users/wuziyang/Documents/PyWork/trading_simulation/test/10_-10/'
path_10_15 = 'C:/Users/wuziyang/Documents/PyWork/trading_simulation/test/10_-15/'
_path_5_5 = 'C:/Users/wuziyang/Documents/PyWork/trading_simulation/test/5_5/'
_path_10_10 = 'C:/Users/wuziyang/Documents/PyWork/trading_simulation/test/10_10/'
_path_10_15 = 'C:/Users/wuziyang/Documents/PyWork/trading_simulation/test/10_15/'



# 寻找符合要求的测试数据，将数据写入csv
def test_data_produce(data, count_days, drop_rate, path, isIncrease):
    #  路径不存在则创建
    if not os.path.exists(path):
        os.mkdir(path)
        os.mkdir(path + 'test_data/')
        os.mkdir(path + 'confirm_data/')
    # 获取已有数据的最晚日期
    start_date = max(get_latest_test_date(path + 'test_data/'))
    # 获取最晚日期后的数据
    cal_data = data[data['TradingDate'] > start_date]
    # 计算5天涨跌
    target_data = cal_profit(cal_data, count_days, True)
    # 按日期排序,输出每天跌幅达到阈值的数据
    df_group = target_data.groupby(by="Date")
    date_list = list(df_group.groups.keys())
    for i in date_list:
        temp_list = []
        everyday_data = target_data[target_data['Date'] == i]
        for row in everyday_data.values:
            if isIncrease:
                if row[2] >= drop_rate:
                    temp_list.append(list(row))
            elif row[2] <= drop_rate:
                temp_list.append(list(row))
        if len(temp_list):
            testdata_df = pd.DataFrame(temp_list, columns=['ID', 'Date', 'total', 'max'])
            testdata_df.to_csv(path + 'test_data/' + str(i) + '.csv', index=False)

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
            df_group = test_data.groupby(by="ID")
            stock_list += list(df_group.groups.keys())
    stock_set = set(stock_list)

    data = data[data['Symbol'] in stock_set]
    # 只计算test股票
    confirm_data = cal_profit(data, count_days, False)
    for i in confirm_dates:
        day_data = confirm_data[confirm_data['Date'] == i]
        # 更新数据
        out_data = []
        if os.path.exists(path + 'test_data/' + str(i) + '.csv'):
            test_data = pd.read_csv(path + 'test_data/' + str(i) + '.csv')
            df_group = test_data.groupby(by="ID")
            id_list = list(df_group.groups.keys())
            for test_id in id_list:
                out_data.append(list(day_data.loc[day_data['ID'] == test_id]))

        # 转为dataframe输出
        confirm_df = pd.DataFrame(out_data, columns=['ID', 'Date', 'total', 'max'])
        confirm_df.to_csv(path + 'confirm_data/' + str(i) + '.csv', index=False)                


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


# 处理输入数据，返回选择天数n的累计值和最大值
def cal_profit(data, up_day:int, test:bool):
    date = ""
    df_group = data.groupby(by="Symbol")
    stock_list = list(df_group.groups.keys())
    dayn_profit = []
    count = 0
    count_days = up_day
    print("all stock:" + str(len(stock_list)))
    for i in stock_list:
        count += 1
        if count % 50 == 0:
            print("now :" + str(count))
        cur_data = data.loc[data['Symbol'] == i]
        cur_data = cur_data.sort_values("TradingDate")
        # 验证数据消除日期计算限制
        if not test:
            up_day = 1
        for j in range(cur_data.shape[0] - up_day):
            cur_profit = 0.0
            max_profit = 0.0
            date = cur_data.iloc[j]['TradingDate']
            stock_id = cur_data.iloc[j]['Symbol']
            # n天后同id才计算
            if cur_data.iloc[j]['Symbol'] == cur_data.iloc[j + up_day - 1]['Symbol']:
                for k in range(0, count_days): 
                    if cur_data.shape[0] <= k + j + 1:
                        continue
                    day_data = cur_data.iloc[k + j]['ChangeRatio']
                    cur_profit = (1 + cur_profit) * day_data + cur_profit
                    if cur_profit >= max_profit:
                        max_profit = cur_profit
                dayn_profit.append([stock_id, date, cur_profit, max_profit])
    
    # 转为dataframe输出
    df_profit = pd.DataFrame(dayn_profit, columns=['ID', 'Date', 'total', 'max'])

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
def cal_7(data):
    # 算法计算得到的数据
    filter_data = list()
    df_group = data.groupby(by="Symbol")
    stock_list = list(df_group.groups.keys())
    for i in stock_list:
        cur_data = data[data['Symbol'] == i]
        if cur_data['ChangeRatio'] >= 0.07:
            filter_data.append(cur_data['Symbol'], cur_data['TradingDate'])
    return filter_data


# n天连涨策略
def cal_nnn(data):
    filter_data = list()
    df_group = data.groupby(by="Symbol")
    stock_list = list(df_group.groups.keys())
    for i in stock_list:
        cur_data = data[data['Symbol'] == i]
        for j in range(0, cur_data.shape[0] - 2):
            if cur_data[j]['ChangeRatio'] and cur_data[j + 1]['ChangeRatio'] and cur_data[j + 2]['ChangeRatio']:
                filter_data.append(cur_data[j + 2]['Symbol'], cur_data[j + 2]['ChangeRatio'])
    return filter_data


# 回测
def back_test():

    return


if __name__ == "__main__":
    data_path = r'C:\Users\wuziyang\Documents\PyWork\trading_simulation\data\stockdata\stock_latest.csv'
    data = pd.read_csv(data_path, sep=',', low_memory=False)
    rec_data = get_data(data)

    # 获取最近1天和n天数据
    n = 3
    data = rec_data[rec_data['Symbol'] == 600001]
    data = data.sort_values("TradingDate", ascending=False)
    last_tradingdate = int(data.iloc[0]['TradingDate'].replace("_", ""))
    last_n_tradingdate = int(data.iloc[n]['TradingDate'].replace("_", ""))
    yestoday_data = rec_data[rec_data['TradingDate'] == last_tradingdate]
    last_n_data = rec_data[rec_data['TradingDate'] > last_n_tradingdate]

    # 回测
    back_test()

    # 涨7策略
    cal_7(yestoday_data)

    # 连涨策略
    cal_nnn(last_n_data)

    # n天涨n策略

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