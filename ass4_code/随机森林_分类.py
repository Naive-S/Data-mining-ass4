# 随机森林算法
# 数据集：wdbc，iris
"""
分类任务：分别对两个数据集按照自己设定的比例进行训练集、测试集的划分，
使用训练集分别训练随机森林模型跟AdaBoost(基分类器采用决策树模型)分类器，
并分别用测试集测试其性能
"""
# SONG
# 2023/5/9
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('../dataset/iris/iris.data', header=None)

data = df.iloc[:, :-1]
target = df.iloc[:, -1]

# 数据预处理
target = target.replace({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data.values, target.values, test_size=0.25)
# 创建随机森林模型，使用10棵决策树，每棵树的最大深度为2
rf = RandomForestClassifier(n_estimators=10, max_depth=2)

# 训练模型
rf.fit(X_train, y_train)

# 预测测试集结果
y_pred = rf.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
