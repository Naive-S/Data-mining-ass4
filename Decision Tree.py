# 决策树算法
# 数据集:crx.data,iris.data
# 将D1、D2按照一定的比例划分为训练集 Dtrain 和测试集 Dtest（比例自行设定），用 Dtrain 分别训练一个决策树分类器，用 Dtest 测试其性能
# 评价指标： accuracy、precision、recall、F1-measure
# SONG
# 2023/4/13

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# 定义节点类
class Node:
    def __init__(self, feature_idx=None, feature_val=None, label=None, left=None, right=None):
        self.feature_idx = feature_idx  # 节点所分裂的特征索引
        self.feature_val = feature_val  # 节点所分裂的特征值
        self.label = label  # 叶子节点的类别标签
        self.left = left  # 左子节点
        self.right = right  # 右子节点


# 定义决策树分类器类
class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth  # 决策树最大深度
        self.root = None  # 决策树根节点

    # 计算基尼不纯度
    def _calc_gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)

    # 计算划分后的基尼指数
    def _calc_split_gini(self, X, y, feature_idx, feature_val):
        left_mask = X[:, feature_idx] < feature_val
        right_mask = ~left_mask
        left_gini = self._calc_gini(y[left_mask])
        right_gini = self._calc_gini(y[right_mask])
        n_left, n_right = len(y[left_mask]), len(y[right_mask])
        total_gini = (n_left * left_gini + n_right * right_gini) / len(y)
        return total_gini

    # 选择最优的划分特征和特征值
    def _find_best_split(self, X, y):
        best_feature_idx, best_feature_val, min_gini = None, None, np.inf
        n_samples, n_features = X.shape
        for feature_idx in range(n_features):
            for feature_val in np.unique(X[:, feature_idx]):
                split_gini = self._calc_split_gini(X, y, feature_idx, feature_val)
                if split_gini < min_gini:
                    best_feature_idx, best_feature_val, min_gini = feature_idx, feature_val, split_gini
        return best_feature_idx, best_feature_val

    # 递归构建决策树
    def _build_tree(self, X, y, depth):
        if depth == self.max_depth or len(np.unique(y)) == 1:
            # 如果达到最大深度或者样本属于同一类别，则返回叶子节点
            label = np.bincount(y).argmax()
            return Node(label=label)
        else:
            # 选择最优的划分特征和特征值
            best_feature_idx, best_feature_val = self._find_best_split(X, y)

            # 根据最优划分特征和特征值分裂数据集
            left_mask = X[:, best_feature_idx] < best_feature_val
            right_mask = ~left_mask

            # 如果分裂后的子集过小，则返回叶子节点
            if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
                label = np.bincount(y).argmax()
                return Node(label=label)
            else:
                # 递归构建左右子树
                left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
                right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
                return Node(feature_idx=best_feature_idx, feature_val=best_feature_val, left=left, right=right)

    # 拟合训练数据集
    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)

    # 预测样本的类别
    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            node = self.root
            while node.label is None:
                if X[i, node.feature_idx] < node.feature_val:
                    node = node.left
                else:
                    node = node.right
            y_pred[i] = node.label
        return y_pred


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

# 训练决策树分类器
tree = DecisionTreeClassifier(max_depth=5)
tree.fit(X_train, y_train)

# 预测测试集的类别
y_pred = tree.predict(X_test)

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
tree = DecisionTreeClassifier(max_depth=4)
tree.fit(X_train2, y_train2)

# 预测测试集的类别
y_pred2 = tree.predict(X_test2)

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

list1=[accuracy,precision,recall,f1_1_1,f1_1_2]
list2=[accuracy2,precision2,recall2,f1_2_1,f1_2_2]
rects1 = ax.bar(index,list1 , bar_width, alpha=opacity, color='b', label='crx')
rects2 = ax.bar(index + bar_width,list2 , bar_width, alpha=opacity, color='g', label='iris')

ax.set_xlabel('Iteration')
ax.set_ylabel('Error')
ax.set_title('Evaluation Indicators')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(('1', '2', '3', '4', '5'))
ax.legend()
plt.show()