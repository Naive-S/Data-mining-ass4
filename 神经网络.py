import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class NeuralNetwork:
    # 初始化函数，定义神经网络中的输入层、隐藏层和输出层节点数，并初始化权重和偏置
    def __init__(self, n_inputs, n_hidden, n_outputs):
        self.weights_1 = np.random.randn(n_inputs, n_hidden)
        self.bias_1 = np.zeros((1, n_hidden))
        self.weights_2 = np.random.randn(n_hidden, n_outputs)
        self.bias_2 = np.zeros((1, n_outputs))

    # 定义sigmoid激活函数
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # 定义前向传播函数
    def forward(self, X):
        self.hidden_layer = self.sigmoid(np.dot(X, self.weights_1) + self.bias_1)
        self.output_layer = self.sigmoid(np.dot(self.hidden_layer, self.weights_2) + self.bias_2)
        return self.output_layer

    # 定义sigmoid函数的导数
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    # 定义反向传播函数，更新权重和偏置
    def backward(self, X, y, output):
        d_output = (output - y) * self.sigmoid_derivative(output)
        d_hidden = d_output.dot(self.weights_2.T) * self.sigmoid_derivative(self.hidden_layer)
        delta_weights_2 = self.hidden_layer.T.dot(d_output)
        delta_bias_2 = np.sum(d_output, axis=0)
        delta_weights_1 = X.T.dot(d_hidden)
        delta_bias_1 = np.sum(d_hidden, axis=0)

        self.weights_2 -= learning_rate * delta_weights_2
        self.bias_2 -= learning_rate * delta_bias_2
        self.weights_1 -= learning_rate * delta_weights_1
        self.bias_1 -= learning_rate * delta_bias_1

    # 定义训练函数，迭代n_iterations次，依次进行前向传播、反向传播和权重更新操作，并输出损失函数值
    def train(self, X, y, n_iterations):
        for i in range(n_iterations):
            output = self.forward(X)
            loss = -np.mean(y * np.log(output) + (1 - y) * np.log(1 - output))
            self.backward(X, y, output)

            if i % 100 == 0:
                print(f"Loss after iteration {i}: {loss}")

    # 定义预测函数，在输入数据集X上进行前向传播，并返回二元标签输出
    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)



# 加载wdbc数据集
# 加载数据集
df = pd.read_csv('./dataset/wdbc/wdbc.data', header=None)

# 去除第一列ID以及第二列标签列
X = df.iloc[:, 2:].values
y = df.iloc[:, 1].apply(lambda x: 1 if x == 'B' else 0)
y = y.values

# 数据预处理
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)  # 标准化特征
y = y.reshape(-1, 1)  # 转换标签为列向量
n_samples, n_features = X.shape

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义超参数
learning_rate = 0.1
n_iterations = 1000

# 创建神经网络并训练模型
nn = NeuralNetwork(n_inputs=n_features, n_hidden=10, n_outputs=1)
nn.train(X_train, y_train, n_iterations)

# # 计算评价指标
y_pred = nn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1_1_1 = f1_score(y_test, y_pred, average='micro')
f1_1_2 = f1_score(y_test, y_pred, average='macro')
print(f"Accuracy: {accuracy}")
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

# 数据预处理
X = df2.iloc[:, :-1].values
y = df2.iloc[:, -1].values
y = (y > 5).astype(int).reshape(-1, 1)
n_samples, n_features = X.shape
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# 定义超参数
learning_rate = 0.1
n_iterations = 1000

# 划分训练集和测试集
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.2)

# 创建神经网络并训练模型
nn = NeuralNetwork(n_inputs=n_features, n_hidden=10, n_outputs=1)
nn.train(X_train2, y_train2, n_iterations)

# 预测测试集类别
y_pred2 = nn.predict(X_test2)

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
