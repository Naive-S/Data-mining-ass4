import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 加载数据集
df2=pd.read_csv('./dataset/winequality/winequality-red.csv',delimiter=';')
print(df2.head())

