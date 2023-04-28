# 逻辑回归算法
# 数据集：wdbc.data
# 将 D4 按照 Dtrain: Dtest = 80% : 20% 的比例进行划分，用 Dtrain 训练一个逻辑回归分类器，用 Dtest 测试其性能
# 评价指标: accuracy、precision、recall、F1-measure
# SONG
# 2023/4/13
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据集
names = ['id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
         'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
         'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se',
         'concave_points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
         'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst',
         'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst']
data = pd.read_csv('./dataset/wdbc/wdbc.data', names=names)

# 将标签列转换为数字
data['diagnosis'] = data['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)

# 分离特征和标签
X = data.drop(['id', 'diagnosis'], axis=1)
y = data['diagnosis']

# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


# 对训练集进行标准化
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 训练逻辑回归分类器
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# 打印预测结果
# print(y_pred)

# 计算评价指标
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# 打印评价指标
print("accuracy:",accuracy)
print("precision:",precision)
print("recall:",recall)
print("f1:",f1)

# 生成混淆矩阵
cm = confusion_matrix(y_test, y_pred)

# 生成评价指标柱状图
plt.figure(figsize=(8, 6))
sns.barplot(x=['Accuracy', 'Precision', 'Recall', 'F1-measure'], y=[accuracy, precision, recall, f1])
plt.ylim(0, 1)
plt.title('Evaluation Metrics')
plt.show()

# 生成混淆矩阵热图
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
