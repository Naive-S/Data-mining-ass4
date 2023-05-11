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
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

df = pd.read_csv('../dataset/iris/iris.data', header=None)

data = df.iloc[:, :-1]
target = df.iloc[:, -1]

# 数据预处理
target = target.replace({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data.values, target.values, test_size=0.25, random_state=42)

####################################### 随机森林 #########################################################

# 创建随机森林模型，使用100棵决策树，每棵树的最大深度为4
rf = RandomForestClassifier(n_estimators=100, max_depth=4)

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

##################################### Adaboost（决策树基分类器） ##############################################
# 构建AdaBoost分类器
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=200,
    learning_rate=0.5
)

# 训练分类器
ada.fit(X_train, y_train)

# 预测测试集标签
y_pred2 = ada.predict(X_test)

# 计算评价指标
accuracy2 = accuracy_score(y_test, y_pred2)
pre_re_f12 = precision_score(y_test, y_pred2, average='micro')
precision2 = precision_score(y_test, y_pred2, average='macro')
recall2 = recall_score(y_test, y_pred2, average='macro')
f12 = f1_score(y_test, y_pred2, average='macro')

# 定义评价指标名称
metrics = ['Accuracy', 'Precision (macro)', 'Recall (macro)', 'F1-score (macro)', 'Pre_Rec_F1(micro)']

# 提取随机森林和Adaboost分类器的评价指标
rf_scores = [accuracy, precision, recall, f1, pre_re_f1]
ada_scores = [accuracy2, precision2, recall2, f12, pre_re_f12]

# 创建柱状图
fig, ax = plt.subplots(figsize=(8, 5))
bar_width = 0.35
opacity = 0.8
index = list(range(len(metrics)))
rects1 = ax.bar(index, rf_scores, bar_width, alpha=opacity, color='b', label='Random Forest')
rects2 = ax.bar([i + bar_width for i in index], ada_scores, bar_width, alpha=opacity, color='r', label='AdaBoost')

# 添加轴标签和标题
ax.set_xlabel('Metrics')
ax.set_ylabel('Score')
ax.set_title('Comparison of Random Forest and AdaBoost on IRIS dataset')


# 添加数据标签
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

add_labels(rects1)
add_labels(rects2)
# 添加轴标签和标题
ax.set_xticks([i + bar_width / 2 for i in index])
ax.set_xticklabels(metrics)
plt.ylim([0.5, 1.2])

# 添加图例
ax.legend()

# 显示图形
plt.tight_layout()
plt.show()
