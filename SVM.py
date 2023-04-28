# SVM算法
# 数据集:wdbc.data,winequality-red.csv
# 将D2、D3按照一定的比例划分为训练集 Dtrain 和测试集 Dtest（比例自行设定），用 Dtrain 分别训练模型，用 Dtest 测试其性能
# 评价指标： accuracy、precision、recall、F1-measure
# SONG
# 2023/4/13

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 加载数据集
df = pd.read_csv('./dataset/wdbc/wdbc.data', header=None)

# 去除第一列ID以及第二列标签列
X = df.iloc[:, 2:]
y = df.iloc[:, 1].apply(lambda x: 1 if x == 'B' else 0)
print(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)

# 训练SVM模型
model = SVC(kernel='linear', C=1)
model.fit(X_train, y_train)

# 预测测试集类别
y_pred = model.predict(X_test)

# 计算评价指标
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
plt.title('Confusion Matrix(wdbc.data)')
plt.show()

################### 数据集winequality.data##############################
# 加载数据集
df2 = pd.read_csv('./dataset/winequality/winequality-red.csv', delimiter=';')

X = df2.iloc[:, :-1].values
y = df2.iloc[:, -1].values

# 划分训练集和测试集
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.2)
# 训练SVM模型
model = SVC(kernel='linear')
model.fit(X_train2, y_train2)

# 预测测试集类别
y_pred2 = model.predict(X_test2)

# 计算模型的评价指标
accuracy2 = accuracy_score(y_test2, y_pred2)
precision2 = precision_score(y_test2, y_pred2, average='macro', zero_division=1)
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
plt.title('Confusion Matrix(winequality-red.csv)')
plt.show()


# 绘制柱状图
fig, ax = plt.subplots()
index = np.arange(5)
bar_width = 0.35
opacity = 0.8

list1 = [accuracy, precision, recall, f1_1_1, f1_1_2]
list2 = [accuracy2, precision2, recall2, f1_2_1, f1_2_2]
rects1 = ax.bar(index, list1, bar_width, alpha=opacity, color='b', label='wdbc')
rects2 = ax.bar(index + bar_width, list2, bar_width, alpha=opacity, color='g', label='winequality-red')

ax.set_xlabel('Iteration')
ax.set_ylabel('Error')
ax.set_title('Evaluation Indicators')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(('1', '2', '3', '4', '5'))
ax.legend()
plt.show()

