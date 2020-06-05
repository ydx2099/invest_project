from sklearn.externals import joblib
from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import time
import lightgbm as lgb
from sklearn import datasets
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.metrics import f1_score
import sklearn
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math

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
        # # 保存模型
        # joblib.dump(clf, r'D:\wuziyang\workfile\model' + str(i) + r'.m')
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


def lgbm(datas, labels, classes, testData=None, testLabel=None):
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
    # 加入验证集
    x_t, x_valid, y_t, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=2020)
    train_data = lgb.Dataset(x_t, label=y_t)
    validation_data = lgb.Dataset(x_valid, label=y_valid)
    # 训练
    clf = lgb.train(params, train_data, valid_sets=[validation_data], num_boost_round=1000)
    # # 保存模型
    # joblib.dump(clf, r'D:\wuziyang\workfile\model' + str(i) + r'.m')
    # # 当前折预测
    # clf.predict(X_train[val_idx], num_iteration=clf.best_iteration)
    # 加权计算预测分数
    # test_pred_prob += clf.predict(X_test, num_iteration=clf.best_iteration)
    
    # pred_class = test_pred_prob.argmax(axis=1)
    # # 获取预测正样本下标
    # pos_indexes = np.argwhere(pred_class == 0)
    # # 计算正样本预测正确概率
    # correct_num = len(np.argwhere(y_test[pos_indexes] == 0))
    # print(len(pos_indexes))
    # print(correct_num)
    # # 获取预测负样本下标
    # neg_indexes = np.argwhere(pred_class == 1)
    # # 计算负样本预测正确概率
    # neg_error_num = len(np.argwhere(y_test[neg_indexes] == 0))
    # print(len(neg_indexes))
    # print(neg_error_num)

    # correct = sum(pred_class==y_test)
    # print("accuracy:" + str(correct / len(pred_class)))
    return clf


# sklearn决策树
def DecisionTree(data, label, label_ls):
    # wine = load_wine()
    # print(wine.data.shape)
    # print(wine.target)
    # print(wine.feature_names)
    # print(wine.target_names)

    xtrain, xtest, ytrain, ytest = train_test_split(data, label, test_size=0.3)
    
    c_weight = {0:label_ls[0], 1:label_ls[1]}
    clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=3, class_weight=c_weight)
    clf.fit(xtrain, ytrain)
    score = clf.score(xtest, ytest)
    print(score)
    score = clf.score(xtrain, ytrain)
    print(score)

    # feature_name = ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']
    feature_name = ['P5', 'P10', 'P20', 'OverTimes', 'Std']

    import graphviz
    dot_data = tree.export_graphviz(clf, feature_names=feature_name, class_names=['1','2'], filled=True, rounded=True)
    graph = graphviz.Source(dot_data)
    
    # graph.view()
    print(list(zip(feature_name, clf.feature_importances_)))

if __name__ == "__main__":
    neg_rate = 5
    # 设置多组相同数据量的正负样本集
    dataset_num = 20
    # # get data
    # # 正样本
    # pos_data = pd.read_csv(r'D:\wuziyang\workfile\pos3.csv', sep=',')
    # pos_datas = pos_data[['P5', 'P10', 'P20', 'OverTimes', 'Std']]
    # labels = [0 for i in range(0, pos_datas.shape[0])]
    # datas = [[] for _ in range(dataset_num)]
    # # # 按样本集数填正样本数据
    # # for i in np.arange(0, dataset_num, 1):
    # #     datas[i].append(pos_datas.values)
    # # # mid样本
    # # mid_data = pd.read_csv(r'D:\wuziyang\workfile\mid3.csv', sep=',')
    # # mid_data = mid_data[['P5', 'P10', 'P20', 'OverTimes', 'Std']]

    # # 负样本
    # neg_data = pd.read_csv(r'D:\wuziyang\workfile\neg3.csv', sep=',')
    # neg_datas = neg_data[['P5', 'P10', 'P20', 'OverTimes', 'Std']]
    # # datas = datas.append(neg_datas)
    # for i in np.arange(0, dataset_num, 1):
    #     datas[i] = pos_datas.append(neg_datas.sample(n=neg_rate * pos_datas.shape[0], replace=False, axis=0)).values
    
    # # 此处label负标签设置与正标签相同数量，用于均衡预测
    # labels += [1 for i in range(0, neg_rate * pos_datas.shape[0])]
    # # labels += [2 for i in range(0, pos_datas.shape[0])]
    # # 直接使用lgbm，不用k折交叉验证
    # clfs = []
    # for i in np.arange(0, dataset_num, 1):
    #     clfs.append(lgbm(np.array(datas[i]), np.array(labels), 2))

    # for i in range(len(clfs)):
    #     # 保存模型
    #     joblib.dump(clfs[i], r'D:\wuziyang\workfile\cur_model' + str(i) + r'.m')

    clfs = []
    for i in range(dataset_num):
        # 加载模型
        clfs.append(joblib.load(filename=r'D:\wuziyang\workfile\cur_model' + str(i) + r'.m'))

    pos_valid = pd.read_csv(r'D:\wuziyang\workfile\pos3_valid.csv', sep=',')
    pos_valid = pos_valid[['P5', 'P10', 'P20', 'OverTimes', 'Std']]
    neg_valid = pd.read_csv(r'D:\wuziyang\workfile\neg3_valid.csv', sep=',')
    neg_valid = neg_valid[['P5', 'P10', 'P20', 'OverTimes', 'Std']]

    # 用验证集测试性能
    weights = []
    F2TRates = []
    for clf in clfs:
        amount = 1000
        f2tnum = 0
        res = clf.predict(neg_valid.iloc[100:1100])
        for i in range(1000):
            if res[i][0] > 0.8:
                f2tnum += 1

        F2TRates.append(f2tnum / amount)

    # 计算e
    expRates = []
    for i in F2TRates:
        expRates.append(math.exp(i))

    # 计算权重
    sum_exp = np.sum(np.array(expRates))
    for i in expRates:
        weights.append(i / sum_exp)
    best_clf = np.argmin(np.array(F2TRates))
    worst_clf = np.argmax(np.array(F2TRates))

    plot_range = list(np.arange(0, 1, 0.1))

    # 计算score
    score = 0.0
    neg_scores = np.zeros([1, neg_valid.shape[0]])
    # for i in range(neg_valid.shape[0]):
    for j in range(len(clfs)):
        # score += clfs[j].predict(neg_valid.iloc[i])[0] * weights[j]
        pred_res = clfs[j].predict(neg_valid.values)
        neg_scores += pred_res[:, 0] * weights[j]
    # score = clfs[best_clf].predict(neg_valid.iloc[i])[0]
    # score = clfs[worst_clf].predict(neg_valid.iloc[i])[0]

    # print(score)
    # neg_scores.append(score[0])
    # score = 0.0

    # plt.hist(neg_scores, bins=10, stacked=True)
    # neg_hist, _ = np.histogram(neg_scores, bins=10)
    # plt.show()

    pos_scores = np.zeros([1, pos_valid.shape[0]])
    # for i in range(pos_valid.shape[0]):
    for j in range(len(clfs)):
        # score += clfs[j].predict(pos_valid.iloc[i])[0] * weights[j]
        pos_scores += clfs[j].predict(pos_valid.values)[:, 0] * weights[j]
        # score = clfs[best_clf].predict(pos_valid.iloc[i])[0]
        # score = clfs[worst_clf].predict(pos_valid.iloc[i])[0]

    # print(score)
    # pos_scores.append(score[0])
    # score = 0.0

    sc_ls = list()
    sc_ls.append(neg_scores[0])
    sc_ls.append(pos_scores[0])
    plt.hist(sc_ls, bins=20, label=['neg', 'pos'], normed=True)
    # plt.hist(pos_scores, bins=10, label='pos', color='b', stacked=True)
    # pos_hist, _ = np.histogram(pos_scores, bins=10)
    # plt.bar(plot_range, neg_hist, label='neg', color='c')
    # plt.bar(plot_range, neg_hist, label='pos', color='b')
    plt.legend(loc='best')
    plt.show()

    # label_ls = [pos_datas.shape[0], pos_datas[0].shape[0]]
    # print(ret)
    # # 决策树
    # DecisionTree(datas, labels, label_ls)