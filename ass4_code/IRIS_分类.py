# 随机森林算法、adaboost算法
# 数据集：iris
"""
分类任务：对IRIS数据集按照自己设定的比例进行训练集、测试集的划分，
使用训练集分别训练随机森林模型跟AdaBoost(基分类器采用决策树模型)分类器，
并分别用测试集测试其性能，比较二者性能。
"""
# SONG
# 2023/5/9
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

################### 随机森林 ###############################
df = pd.read_csv('../dataset/iris/iris.data', header=None)

data = df.iloc[:, :-1]
target = df.iloc[:, -1]

# 数据预处理
target = target.replace({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data.values, target.values, test_size=0.25,random_state=42)
# 创建随机森林模型，使用10棵决策树，每棵树的最大深度为2
rf = RandomForestClassifier(n_estimators=10, max_depth=4)

# 训练模型
rf.fit(X_train, y_train)

# 预测测试集结果
y_pred = rf.predict(X_test)

# 计算评价指标
accuracy = accuracy_score(y_test, y_pred)
pre_re_f1 = precision_score(y_test, y_pred, average='micro')
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
# 打印评价指标
print("随机森林分类效果：")
print("accuracy:", accuracy)
print("precision_macro:", precision)
print("recall_macro:", recall)
print("f1_macro:", f1)
print("pre_re_d1_micro:",pre_re_f1)


