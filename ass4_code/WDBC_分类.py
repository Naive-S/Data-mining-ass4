# 随机森林算法、AdaBoost算法、stacking
# 数据集：wdbc
"""
分类任务：对wdbc数据集按照自己设定的比例进行训练集、测试集（和验证集）的划分，
使用训练集分别训练随机森林模型、AdaBoost(基分类器采用决策树模型)分类器以及satcking模型，
并分别用测试集测试其性能，比较三者在统一数据集的性能
"""
# SONG
# 2023/5/9

import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

# 加载数据集
names = ['id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
         'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
         'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se',
         'concave_points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
         'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst',
         'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst']
data = pd.read_csv('../dataset/wdbc/wdbc.data', names=names)

# 将标签列转换为数字
data['diagnosis'] = data['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)

# 分离特征和标签
X = data.drop(['id', 'diagnosis'], axis=1)
y = data['diagnosis']

# 将数据集划分为训练集和测试集
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, random_state=42)

####################################### 随机森林 #########################################################

# 创建随机森林模型，使用100棵决策树，每棵树的最大深度为4
rf = RandomForestClassifier(n_estimators=100, max_depth=4)

# 训练模型
rf.fit(X_trainval, y_trainval)

# 预测测试集结果
y_pred = rf.predict(X_test)

# 计算评价指标
accuracy = accuracy_score(y_test, y_pred)
pre_re_f1 = precision_score(y_test, y_pred, average='micro')
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

##################################### Adaboost（决策树基分类器） ##############################################
# 构建AdaBoost分类器
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=200,
    learning_rate=0.5
)

# 训练分类器
ada.fit(X_trainval, y_trainval)

# 预测测试集标签
y_pred2 = ada.predict(X_test)

# 计算评价指标
accuracy2 = accuracy_score(y_test, y_pred2)
pre_re_f12 = precision_score(y_test, y_pred2, average='micro')
precision2 = precision_score(y_test, y_pred2, average='macro')
recall2 = recall_score(y_test, y_pred2, average='macro')
f12 = f1_score(y_test, y_pred2, average='macro')

####################################### Stacking方法 #############################################
# 特征归一化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 训练第一阶段基础分类器
knn = KNeighborsClassifier(n_neighbors=3)
svm = SVC(kernel='linear', C=1)
dtc = DecisionTreeClassifier(max_depth=5)

knn.fit(X_train_scaled, y_train)
svm.fit(X_train_scaled, y_train)
dtc.fit(X_train_scaled, y_train)

# 在验证集上进行预测并生成元特征
meta_features_train = []
for clf in [knn, svm, dtc]:
    meta_features_train.append(clf.predict(X_val_scaled))
meta_features_train = np.array(meta_features_train).T

# 训练第二阶段元分类器
mlp = MLPClassifier(hidden_layer_sizes=(100,), alpha=0.1, max_iter=2000)

mlp.fit(meta_features_train, y_val)

# 在测试集上进行预测并计算性能指标
meta_features_test = []
for clf in [knn, svm, dtc]:
    meta_features_test.append(clf.predict(X_test_scaled))
meta_features_test = np.array(meta_features_test).T

y_pred3 = mlp.predict(meta_features_test)

# 计算评价指标
accuracy3 = accuracy_score(y_test, y_pred3)
pre_re_f13 = precision_score(y_test, y_pred3, average='micro')
precision3 = precision_score(y_test, y_pred3, average='macro')
recall3 = recall_score(y_test, y_pred3, average='macro')
f13 = f1_score(y_test, y_pred3, average='macro')

########################################## 绘图 #############################################

# 设置每个算法的性能指标
RF_scores = [accuracy, precision, recall, f1, pre_re_f1]
AdaBoost_scores = [accuracy2, precision2, recall2, f12, pre_re_f12]
Stacking_scores = [accuracy3, precision3, recall3, f13, pre_re_f13]

# 设置每个簇的位置和宽度
x = np.arange(5)
width = 0.25

# 绘制柱状图
fig, ax = plt.subplots(figsize=(10, 8))
rects1 = ax.bar(x - width, RF_scores, width, label='RF')
rects2 = ax.bar(x, AdaBoost_scores, width, label='AdaBoost')
rects3 = ax.bar(x + width, Stacking_scores, width, label='Stacking')


# 添加数据标签
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

# 将图例放在右上角
ax.legend(loc='upper right')

# 添加轴标签和标题
ax.set_xticks(x)
ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1-score', 'Pre_Rec_F1'])
ax.set_ylabel('Score')
ax.set_title('Comparison of Algorithms on wdbc Dataset')

# 调整柱状图间距
ax.set_xlim([-1.5 * width, len(x) - width])
fig.tight_layout()
plt.ylim([0.5, 1.1])

# 显示图形
plt.show()
