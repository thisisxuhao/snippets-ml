import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
import seaborn as sns
import random
from random import randint
import math

random_state = 42


def visualize_results(X, y, model, title):
    """可视化决策边界"""
    plt.figure(figsize=(10, 8))

    # 创建网格点
    h = 0.01
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # 预测网格点的类别
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 绘制决策边界
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.contour(xx, yy, Z, levels=[0.5], linewidths=2, colors='black')

    # 绘制数据点
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', s=25, label='0')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red', s=25, label='1')

    plt.title(title, fontsize=16)
    plt.xlim((x_min, x_max))
    plt.ylim((y_min, y_max))
    plt.legend()

    return plt


class AdditiveLR():
    def __init__(self, n_models):
        self.n_models = n_models
        self.n_classes = None

    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        # 训练多个基础模型
        models = []
        for i in range(self.n_models):
            # 添加一些随机扰动，使每个模型略有不同
            model = LogisticRegression()
            w = np.zeros_like(y)
            random.seed(i * 1003 % 51)
            # s = randint(len(y) // self.n_models * i, len(y) // self.n_models * (i + 1))
            # w[s:s + len(y) // self.n_models * 2] = 1.0
            w[len(y) // 5 * i:int(len(y) // self.n_models * (i + 1.2))] = 1.0
            n_models = 4
            i = i % 4
            X_ = X[len(y) // n_models * i:int(len(y) // n_models * (i + 1.2)), :]
            y_ = y[len(y) // n_models * i:int(len(y) // n_models * (i + 1.2))]
            model.fit(X_, y_)
            models.append(model)
            # visualize_results(X_, y_, model, 'additive logistic')
        self.models = models
        # 使用验证集优化权重
        from scipy.optimize import minimize

        def objective(weights):
            self.weights = weights
            predictions = self.predict(X)
            return -accuracy_score(y, predictions)  # 最小化负准确率 = 最大化准确率

        # 初始权重均匀分布
        np.random.seed(42)
        initial_weights = np.random.rand(self.n_models)
        # initial_weights = np.ones(self.n_models) / self.n_models

        # 约束条件：所有权重之和为1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

        # 边界条件：所有权重都在0和1之间
        bounds = [(0, 1) for _ in range(self.n_models)]

        # 优化权重
        result = minimize(objective, initial_weights, method='SLSQP',)
        weights = result.x / np.sum(result.x)  # 确保权重和为1
        self.weights = weights

    def predict_prob(self, X):
        weights = self.weights / np.sum(self.weights)  # 归一化
        predictions = np.zeros((X.shape[0], self.n_classes))
        n_classes = np.arange(0, self.n_classes)[:, None]
        for i, model in enumerate(self.models):
            proba = np.array(model.predict(X))
            predictions += (proba == n_classes).T * weights[i]
        return predictions

    def predict(self, X):
        return np.argmax(self.predict_prob(X), axis=1)


X, y = make_circles(n_samples=300, noise=0.1, factor=0.5, random_state=random_state)
import pandas as pd

data = pd.DataFrame(np.c_[X, y], columns=['x1', 'x2', 'y'])
data['theta'] = data.apply(lambda s: math.atan(s['x2'] / s['x1']) + np.pi * float(s['x1'] > 0), axis=1)
data.sort_values(by=['theta'], inplace=True)
X = np.array(data[['x1', 'x2']].values)
y = np.array(data['y'].values)
alr = AdditiveLR(5)
alr.fit(X, y)
# np.random.seed(1123)
# alr.weights = np.random.random(20)
print(X.shape)
print(alr.predict_prob(X).shape)
print(alr.predict(X))
print(accuracy_score(y, alr.predict(X)))

# visualize_results(X, y, alr, 'additive logistic')
# for clf in alr.models:
#     x = np.linspace(-1.5, 1.5)
#     x = np.c_(np.ones_like(x), x)
#     print(clf.intercept_, clf.coef_)

# 创建基础逻辑回归分类器
base_estimator = LogisticRegression(
    max_iter=10000,
    # random_state=42,
    solver='liblinear',  # 使用liblinear求解器确保正确处理样本权重
    C=1.0,  # 正则化强度
    class_weight='balanced'  # 平衡类别权重
)

# 创建并训练AdaBoost模型
ada = AdaBoostClassifier(
    base_estimator=base_estimator,
    n_estimators=10,  # 增加分类器数量
    learning_rate=0.5,  # 降低学习率，使权重调整更平滑
    algorithm='SAMME',  # 使用SAMME算法，适合二分类问题
    # random_state=42
)
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
lgb = LGBMClassifier()
# lgb.fit(X, y)
dc = DecisionTreeClassifier()
# dc.fit(X, y)
from sklearn.svm import SVC
svc = SVC()
# svc.fit(X, y)

from sklearn.linear_model import LogisticRegression

# ada.fit(X, y)
# # 打印每个基础分类器的系数和权重
# print("基础分类器系数和权重:")
# for i, (clf, alpha) in enumerate(zip(ada.estimators_, ada.estimator_weights_)):
#     print(f"分类器 {i + 1}:")
#     print(f"  权重: {alpha:.4f}")
#     print(f"  系数: {clf.coef_}")
#     print(f"  截距: {clf.intercept_}")
#     visualize_results(X, y, clf, 'additive logistic')
#     print()

class IntersectLR:
    def fit(self, X, y):
        self.model = LogisticRegression()
        print(X.shape)
        print(X[0,:].shape)
        X = np.c_[X, X[:, 0] * X[:, 0], X[:, 1] * X[:, 1], X[:, 0] * X[:, 1]]
        self.model.fit(X, y)

    def predict(self, X):
        X = np.c_[X, X[:, 0] * X[:, 0], X[:, 1] * X[:, 1], X[:, 0] * X[:, 1]]
        return self.model.predict(X)

ilr = IntersectLR()
ilr.fit(X, y)
visualize_results(X, y, ilr, 'IntersectLR')
plt.show()
