import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt


# 使用K-means方法，对输入数据聚类
# 距离计算方法：distance = (x1 - x2)^2 + (y1 - y2)^2
# 输入：待分类数据(numpy.array[numpy.array])，分类类别数(int)，最大迭代次数(int)，迭代终止最大平均误差(float)
# 输出：分类结果(List[List[int]])
def classify(data, class_num, max_iter, stop_loss):
    # 保存结果
    res_list = []
    # 保存每次的中心点
    center_list = []
    # 保存每类所有点的和，用于计算下一次的中心点
    sum_point_list = []
    # 判断数据量是否符合聚类要求
    if data.shape[0] <= class_num:
        print("Are You Kidding Me?")
        return
    
    # 初始化
    for i in range(class_num):
        center_list.append(data[i])
        res_list.append(list())
        sum_point_list.append(list([0.0, 0.0]))

    # 聚类
    for i in range(max_iter):
        if i % 100 == 0:
            print("processing " + str(i) + " now...")
        # 每次清空缓存结果
        for m in range(5):
            res_list[m] = list()

        for index in range(len(data)):
            # 寻找最近中心点
            min_dis = np.sum(np.square(center_list[0] - data[index]))
            temp_dis = 0.0
            # 点所属的类别
            _class = 0
            for j in range(1, 5):
                temp_dis = np.sum(np.square(center_list[j] - data[index]))
                if temp_dis < min_dis:
                    min_dis = temp_dis
                    _class = j
            res_list[_class].append(index)
            sum_point_list[_class][0] += data[index][0]
            sum_point_list[_class][1] += data[index][1]
        # 计算与上一中心点的偏差并更新
        loss = 0.0
        for k in range(5):
            # 出现特殊情况，中心点周围一个点都没有，则保持原中心点并将误差记为0
            if len(res_list[k]) == 0:
                continue

            new_x = sum_point_list[k][0] / len(res_list[k])
            new_y = sum_point_list[k][1] / len(res_list[k])
            loss += math.sqrt(pow(new_x - center_list[k][0], 2) + pow(new_y - center_list[k][1], 2))
            center_list[k] = list([new_x, new_y])
        
        if loss / 5 <= stop_loss:
            break
            
    return res_list


# 读取待分类数据，数据存储在csv文件中，格式为：
def read_data(path):
    df = pd.read_csv(path)
    return df

classes = 5
# 读取df数据
df_data = read_data(r'C:\Users\wuziyang\Desktop\1.csv')
# 数据转成numpy数组
np_data = df_data.values
# plt.plot(np_data[:, 0], np_data[:, 1], 'o')
# plt.show()
# 聚类
result = classify(np_data, classes, 1000, 1e-5)
# 展示结果
colors = ['b', 'g', 'r', 'y', 'k', 'w', 'm', 'c']
for i in range(classes):
    plt.plot(np_data[result[i], 0], np_data[result[i], 1], marker='o', color=colors[i])
plt.xlabel('x')
plt.ylabel('y')
plt.show()


a = [[1,2,3],[4,5,6]]
print(a[0][1])


# a = np.array([1,2,3,4,5,6])
# b = np.array([3,5])
# print(a[b])