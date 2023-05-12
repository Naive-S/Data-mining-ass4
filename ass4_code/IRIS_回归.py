# 随机森林算法，AdaBoost算法
# 数据集：iris
"""
回归任务：忽略数据集中的类别属性，从其余属性中任选一个作为回归任务的目标属性；
采取不同算法分别对数据集按照自行设定的比例进行训练集、测试集的划分，
使用训练集分别训练随机森林跟AdaBoost回归器，并分别用测试集测试其性能。
"""
# SONG
# 2023/5/9

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, explained_variance_score, median_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../dataset/iris/iris.data', header=None)

# 数据预处理
data = df.iloc[:, 1:-1]
target = df.iloc[:, 0]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data.values, target.values, test_size=0.25, random_state=42)

###################################### 随机森林回归器 ###############################################################

# 创建随机森林回归器
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# 拟合模型
rf.fit(X_train, y_train)

# 在测试集上评估模型性能
y_pred = rf.predict(X_test)

print("随机森林算法")
# 计算均方误差（MSE）
mse = np.mean((y_test - y_pred) ** 2)
print('均方误差（MSE）:', mse)

# 计算平均绝对误差（MAE）
mae = mean_absolute_error(y_test, y_pred)
print('平均绝对误差（MAE）:', mae)

# 计算决定系数（R²）
r2 = r2_score(y_test, y_pred)
print('决定系数（R²）:', r2)

# 计算解释方差分数
evs = explained_variance_score(y_test, y_pred)
print('解释方差分数:', evs)

# 计算中位绝对误差
medae = median_absolute_error(y_test, y_pred)
print('中位绝对误差（MedAE）:', medae)

############################################## Adaboost算法 ######################################################

from sklearn.ensemble import AdaBoostRegressor

# 创建AdaBoost回归器，并以决策树为基本估计器
ada_regressor = AdaBoostRegressor(n_estimators=50, random_state=42)

# 使用训练数据拟合模型
ada_regressor.fit(X_train, y_train)

# 在测试数据上进行预测
y_pred2 = ada_regressor.predict(X_test)

print("\nAdaboost算法")
# 计算均方误差（MSE）
mse2 = np.mean((y_test - y_pred2) ** 2)
print('均方误差（MSE）:', mse2)

# 计算平均绝对误差（MAE）
mae2 = mean_absolute_error(y_test, y_pred2)
print('平均绝对误差（MAE）:', mae2)

# 计算决定系数（R²）
r22 = r2_score(y_test, y_pred2)
print('决定系数（R²）:', r22)

# 计算解释方差分数
evs2 = explained_variance_score(y_test, y_pred2)
print('解释方差分数:', evs2)

# 计算中位绝对误差
medae2 = median_absolute_error(y_test, y_pred2)
print('中位绝对误差（MedAE）:', medae2)



################################################ 绘图 ##############################################
# 将评价指标绘制在一张图上进行对比
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制随机森林算法的评价指标
sns.lineplot(x=['MSE', 'MAE', 'R^2', 'Explained Variance', 'MedAE'], y=[mse, mae, r2, evs, medae], label='Random Forest', ax=ax)

# 绘制Adaboost算法的评价指标
sns.lineplot(x=['MSE', 'MAE', 'R^2', 'Explained Variance', 'MedAE'], y=[mse2, mae2, r22, evs2, medae2], label='Adaboost', ax=ax)

# 设置图形标题、轴标签和图例
ax.set_title('Comparison of Evaluation Metrics for Random Forest and Adaboost Algorithms')
ax.set_xlabel('Evaluation Metrics')
ax.set_ylabel('Value')
ax.legend()

plt.show()
