# 朴素贝叶斯算法
# 数据集:crx.data,iris.data
# 将D1、D2按照一定的比例划分为训练集 Dtrain 和测试集 Dtest（比例自行设定），用 Dtrain 分别训练模型，用 Dtest 测试其性能
# 评价指标： accuracy、precision、recall、F1-measure（micro,macro）
# SONG
# 2023/4/13

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns


class NaiveBayes:
    def __init__(self, alpha=1):
        self.alpha = alpha  # 拉普拉斯平滑参数
        self.classes = None  # 类别列表
        self.class_prior = None  # 先验概率
        self.feature_log_prob = None  # 条件概率的对数值

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)  # 获取所有类别
        n_classes = len(self.classes)

        self.class_prior = np.zeros(n_classes)  # 初始化先验概率
        self.feature_log_prob = np.zeros((n_classes, n_features))  # 初始化条件概率的对数值

        for i, c in enumerate(self.classes):  # 针对每个类别进行处理
            X_c = X[c == y]  # 获取该类别下的所有样本
            self.class_prior[i] = (len(X_c) + self.alpha) / (n_samples + n_classes * self.alpha)  # 计算该类别的先验概率
            self.feature_log_prob[i] = np.log(
                (np.sum(X_c, axis=0) + self.alpha) / (np.sum(X_c) + n_features * self.alpha))  # 计算该类别下每个特征的条件概率的对数值

    def predict(self, X):
        log_posterior = self._log_posterior(X)  # 获取对数后验概率
        return self.classes[np.argmax(log_posterior, axis=1)]  # 返回预测结果

    def _log_posterior(self, X):
        return np.dot(X, self.feature_log_prob.T) + np.log(self.class_prior)  # 计算对数后验概率


##############################  数据集crx.data  #################################################
# 读入数据集
df = pd.read_csv('./dataset/crx/crx.data', header=None)
df.columns = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9',
              'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16']

# 划分特征和标签
X = df.iloc[:, :-1]
y = df.iloc[:, -1].apply(lambda x: 1 if x == '+' else 0)  # +为1，-为0

# 查看数据集
X = X.replace("?", np.NaN)  # 将数据集中的？值替换为Nan
print(X.info())  # 查看数据集信息

# 将类别型特征转换为数值型特征
X['A1'] = X['A1'].map({'b': 0, 'a': 1})
X['A4'] = X['A4'].map({'u': 0, 'y': 1, 'l': 2, 't': 3})
X['A5'] = X['A5'].map({'g': 0, 'p': 1, 'gg': 2})
X['A6'] = X['A6'].map({'c': 0, 'd': 1, 'cc': 2, 'i': 3,
                       'j': 4, 'k': 5, 'm': 6, 'r': 7, 'q': 8,
                       'w': 9, 'v': 10, 'x': 11, 'e': 12})
X['A7'] = X['A7'].map({'v': 0, 'h': 1, 'bb': 2, 'j': 3, 'n': 4, 'z': 5,
                       'dd': 6, 'ff': 7, 'o': 8})
X['A9'] = X['A9'].apply(lambda x: 1 if x == 't' else 0)
X['A10'] = X['A10'].apply(lambda x: 1 if x == 't' else 0)
X['A12'] = X['A12'].apply(lambda x: 1 if x == 't' else 0)
X['A13'] = X['A13'].map({'g': 0, 'p': 1, 's': 2})

# 更改数据类型
for i in ['A1', 'A2', 'A4', 'A5', 'A6', 'A7', 'A9',
          'A10', 'A12', 'A13', 'A14']:
    X[i] = X[i].astype(float)

# 填充缺失值
X.fillna(X.median(), inplace=True)  # 以中位数填充缺失值

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.25)

# 训练模型
model = NaiveBayes()
model.fit(X_train, y_train)

# 预测测试集的类别
y_pred = model.predict(X_test)

# 计算模型的评价指标
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1_1_1 = f1_score(y_test, y_pred, average='micro')
f1_1_2 = f1_score(y_test, y_pred, average='macro')
print("模型准确率：", accuracy)
print("模型精确率：", precision)
print("模型召回率：", recall)
print("模型 F1 值(micro)：", f1_1_1)
print("模型 F1 值(macro)：", f1_1_2)

# 绘制混淆矩阵
cm = confusion_matrix(y_test, y_pred)
# 生成混淆矩阵热图
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix(crx.data)')
plt.show()

###############################iris.data#########################################

df = pd.read_csv('./dataset/iris/iris.data', header=None)

data = df.iloc[:, :-1]
target = df.iloc[:, -1]

# 数据预处理
target = target.replace({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})

# 划分训练集和测试集
X_train2, X_test2, y_train2, y_test2 = train_test_split(data.values, target.values, test_size=0.25)

# 训练决策树分类器
model = NaiveBayes()
model.fit(X_train2, y_train2)

# 预测测试集的类别
y_pred2 = model.predict(X_test2)

# 计算模型的评价指标
accuracy2 = accuracy_score(y_test2, y_pred2)
precision2 = precision_score(y_test2, y_pred2, average='macro')
recall2 = recall_score(y_test2, y_pred2, average='macro')
f1_2_1 = f1_score(y_test2, y_pred2, average='micro')
f1_2_2 = f1_score(y_test2, y_pred2, average='macro')
print("模型准确率：", accuracy2)
print("模型精确率：", precision2)
print("模型召回率：", recall2)
print("模型 F1 值(micro)：", f1_2_1)
print("模型 F1 值(macro)：", f1_2_2)

# 绘制混淆矩阵
cm = confusion_matrix(y_test2, y_pred2)
# 生成混淆矩阵热图
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix(iris.data)')
plt.show()

# 绘制柱状图
# 绘制多组MAE和RMSE的条形图
fig, ax = plt.subplots()
index = np.arange(5)
bar_width = 0.35
opacity = 0.8

list1 = [accuracy, precision, recall, f1_1_1, f1_1_2]
list2 = [accuracy2, precision2, recall2, f1_2_1, f1_2_2]
rects1 = ax.bar(index, list1, bar_width, alpha=opacity, color='b', label='crx')
rects2 = ax.bar(index + bar_width, list2, bar_width, alpha=opacity, color='g', label='iris')

ax.set_xlabel('Iteration')
ax.set_ylabel('Error')
ax.set_title('Evaluation Indicators')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(('1', '2', '3', '4', '5'))
ax.legend()
plt.show()
