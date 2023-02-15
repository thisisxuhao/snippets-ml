from typing import Optional

from sklearn.base import ClassifierMixin
import numpy as np


class FM2WayModel(ClassifierMixin):
    def __init__(self, n: int, p: int, k: int, learning_rate: float, init_std: float, init_mean: float,
                 seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)

        self.p = p
        self.k = k
        self.learning_rate = learning_rate

        # w, 一维向量, 初始化
        self.w = np.zeros(p)
        # V, n * k维矩阵, 通过正态分布进行初始化, {v1, v2, ... vn}
        self.V = init_mean + np.random.randn(n, k) * init_std

        # M用于计算过程的缓存, 后续求梯度的迭代总会使用多次, Vt(k*n) * X(n*p) = M(k*p), M[f, :] = \sum_{j=1}^n v_{j, f} x_j
        self.M = np.zeros(k, p)

    def compute(self, X, V):
        self.M = np.transpose(V) * X
        pass

    def fit(self, X, y):
        for f in range(0, self.k):
            for i in range(0, X.shape[0]):
                grad = np.dot(self.M[f], X[i, :]) - self.V[i, f] * np.dot(X[i, :], X[i, :])
                self.V[i, f] -= self.learning_rate * (grad + regv * v)
        pass

# X =
# 
# 
# 1.5
# 0.0
# 0.0 −7.9
# 0.0
# 0.0
# 0.0
# 0.0
# 10−5
# 0.0
# 2.0
# 0.0
# 0.0
# 0.0
# 0.0
# 0.0
# 0.0
# 0.0
# 0.0
# 0.0
# 1.0
# 
# , y =
# 
# 
# 4
# 2
# −1
# 
# 
