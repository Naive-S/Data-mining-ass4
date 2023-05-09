from sklearn.model_selection import train_test_split

df = pd.read_csv('../dataset/iris/iris.data', header=None)

data = df.iloc[:, :-1]
target = df.iloc[:, -1]

# 数据预处理
target = target.replace({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data.values, target.values, test_size=0.25)