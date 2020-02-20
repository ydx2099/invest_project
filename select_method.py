import random
import pandas as pd
import time

# 用于随机选股
def random_select(candidate_num):
    select = set()
    while len(select) < 15:
        s = random.randint(2, candidate_num)
        if s not in select: print(s)
        select.add(s)

# 用于取最值选股
def max_select(path):
    data = pd.read_csv(path, sep=',', low_memory=False)
    data = data.sort_values(by="ChangeRatio")
    for i in range(0, 15):
        print(int(data.iloc[i]['Symbol']))

# 7
print('this is 7')
random_select(24)
# # nnn
# print('this is nnn')
# random_select(350)
# 5天涨10
# print('this is 5_10')
# max_select(r'C:\Users\wuziyang\Documents\PyWork\trading_simulation\test\5_10\test_data\20200214.csv')

