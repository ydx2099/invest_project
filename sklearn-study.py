from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import time
import lightgbm as lgb
from sklearn import datasets
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.metrics import f1_score
import sklearn

# sklearn决策树
def dt():
    wine = load_wine()
    print(wine.data.shape)
    print(wine.target)
    print(wine.feature_names)
    print(wine.target_names)

    xtrain, xtest, ytrain, ytest = train_test_split(wine.data, wine.target, test_size=0.3)
    
    c_weight = {0:1, 1:5, 2:10}
    clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=3, class_weight=c_weight)
    clf.fit(xtrain, ytrain)
    score = clf.score(xtest, ytest)
    print(score)
    score = clf.score(xtrain, ytrain)
    print(score)

    feature_name = ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']
    
    import graphviz
    dot_data = tree.export_graphviz(clf, feature_names=feature_name, class_names=['1','2','3'], filled=True, rounded=True)
    graph = graphviz.Source(dot_data)
    
    # graph.view()
    print(list(zip(feature_name, clf.feature_importances_)))

dt()