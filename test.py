# SVM算法
# 数据集:wdbc.data,winequality-red.csv
# 将D2、D3按照一定的比例划分为训练集 Dtrain 和测试集 Dtest（比例自行设定），用 Dtrain 分别训练模型，用 Dtest 测试其性能
# 评价指标： accuracy、precision、recall、F1-measure
# SONG
# 2023/4/13

# ValueError: could not convert string to float: 'fixed acidity'

import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


