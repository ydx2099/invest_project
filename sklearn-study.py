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
    train_data = lgb.Dataset(X_train, label=y_train)
    validation_data = lgb.Dataset(X_test, label=y_test)
    # 训练
    clf = lgb.train(params, train_data, valid_sets=[validation_data], num_boost_round=1000)
    # # 保存模型
    # joblib.dump(clf, r'D:\wuziyang\workfile\model' + str(i) + r'.m')
    # # 当前折预测
    # clf.predict(X_train[val_idx], num_iteration=clf.best_iteration)
    # # 加权计算预测分数
    # test_pred_prob += clf.predict(X_test, num_iteration=clf.best_iteration) / folds_num
    
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
    feature_name = ['P5', 'P10', 'P20', 'OverTimes']

    import graphviz
    dot_data = tree.export_graphviz(clf, feature_names=feature_name, class_names=['1','2'], filled=True, rounded=True)
    graph = graphviz.Source(dot_data)
    
    # graph.view()
    print(list(zip(feature_name, clf.feature_importances_)))

if __name__ == "__main__":
    # 设置多组相同数据量的正负样本集
    dataset_num = 10
    # get data
    # 正样本
    pos_data = pd.read_csv(r'D:\wuziyang\workfile\pos.csv', sep=',')
    pos_datas = pos_data[['P5', 'P10', 'P20', 'OverTimes']]
    labels = [0 for i in range(0, pos_datas.shape[0])]
    datas = [[] for _ in range(10)]
    # # 按样本集数填正样本数据
    # for i in np.arange(0, dataset_num, 1):
    #     datas[i].append(pos_datas.values)
    # 负样本
    neg_data = pd.read_csv(r'D:\wuziyang\workfile\neg.csv', sep=',')
    neg_datas = neg_data[['P5', 'P10', 'P20', 'OverTimes']]
    # datas = datas.append(neg_datas)
    for i in np.arange(0, dataset_num, 1):
        datas[i] = pos_datas.append(neg_datas.sample(n=pos_datas.shape[0], replace=False, axis=0)).values
    # 此处label负标签设置与正标签相同数量，用于均衡预测
    labels += [1 for i in range(0, pos_datas.shape[0])]
    # 直接使用lgbm，不用k折交叉验证
    clfs = []
    for i in np.arange(0, dataset_num, 1):
        clfs.append(lgbm(np.array(datas[i]), np.array(labels), 2))

    # 计算score
    score = 0.0
    for clf in clfs:
        print(clf.predict(pos_datas.iloc[0]))

    # label_ls = [pos_datas.shape[0], pos_datas[0].shape[0]]
    # print(ret)
    # # 决策树
    # DecisionTree(datas, labels, label_ls)