import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class NeuralNetwork:
    def __init__(self, n_inputs, n_hidden, n_outputs):
        # 初始化权重和偏差
        self.weights_1 = np.random.randn(n_inputs, n_hidden)
        self.bias_1 = np.zeros((1, n_hidden))
        self.weights_2 = np.random.randn(n_hidden, n_outputs)
        self.bias_2 = np.zeros((1, n_outputs))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, X):
        # 前向传播
        self.hidden_layer = self.sigmoid(np.dot(X, self.weights_1) + self.bias_1)
        self.output_layer = self.sigmoid(np.dot(self.hidden_layer, self.weights_2) + self.bias_2)
        return self.output_layer

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def backward(self, X, y, output):
        # 反向传播
        d_output = (output - y) * self.sigmoid_derivative(output)
        d_hidden = d_output.dot(self.weights_2.T) * self.sigmoid_derivative(self.hidden_layer)
        delta_weights_2 = self.hidden_layer.T.dot(d_output)
        delta_bias_2 = np.sum(d_output, axis=0)
        delta_weights_1 = X.T.dot(d_hidden)
        delta_bias_1 = np.sum(d_hidden, axis=0)

        # 更新权重和偏置
        self.weights_2 -= learning_rate * delta_weights_2
        self.bias_2 -= learning_rate * delta_bias_2
        self.weights_1 -= learning_rate * delta_weights_1
        self.bias_1 -= learning_rate * delta_bias_1

    def train(self, X, y, n_iterations, learning_rate):
        for i in range(n_iterations):
            # 前向传播
            output = self.forward(X)

            # 计算损失函数
            loss = -np.mean(y * np.log(output) + (1 - y) * np.log(1 - output))

            # 反向传播并更新权重和偏置
            self.backward(X, y, output)

            # 打印损失函数值
            if i % 100 == 0:
                print(f"Loss after iteration {i}: {loss}")

    def predict(self, X):
        # 预测
        return (self.forward(X) > 0.5).astype(int)


# 加载wdbc数据集
# 加载数据集
df = pd.read_csv('./dataset/wdbc/wdbc.data', header=None)

# 去除第一列ID以及第二列标签列
X = df.iloc[:, 2:].values
y = df.iloc[:, 1].apply(lambda x: 1 if x == 'B' else 0)
y=y.values


# 数据预处理
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)  # 标准化特征
y = y.reshape(-1, 1)  # 转换标签为列向量
n_samples, n_features = X.shape

# 定义超参数
learning_rate = 0.1
n_iterations = 1000

# 创建神经网络并训练模型
nn = NeuralNetwork(n_inputs=n_features, n_hidden=10, n_outputs=1)
nn.train(X, y, n_iterations, learning_rate)

# # 计算评价指标
y_pred = nn.predict(X)

accuracy = accuracy_score(y,y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1_1_1 = f1_score(y, y_pred, average='micro')
f1_1_2 = f1_score(y, y_pred, average='macro')
print(f"Accuracy: {accuracy}")
print("模型准确率：", accuracy)
print("模型精确率：", precision)
print("模型召回率：", recall)
print("模型 F1 值(micro)：", f1_1_1)
print("模型 F1 值(macro)：", f1_1_2)