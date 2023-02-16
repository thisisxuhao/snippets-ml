from typing import Optional

import numpy as np
from sklearn.base import ClassifierMixin


class FM2WayModel(ClassifierMixin):
    """
    2-way FM模型
    :param k: factor数
    :param learning_rate: 学习率
    :param init_std: V初始化时正态分布的标准差
    :param init_mean: V初始化时正态分布的均值
    :param iter_num: 迭代次数
    :param y_min: y分布的最小值, 如果为None则表示不限制
    :param y_max: y分布的最大值, 如果为None则表示不限制
    :param seed: 随机数, 为None则不设置
    """

    def __init__(self, k: int, learning_rate: float, init_mean: float, init_std: float, iter_num: int,
                 epsilon: float = 1e-4, y_min: Optional[int] = None, y_max: Optional[int] = None,
                 seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
        self.n = None
        self.p = None
        self.k = k
        self.learning_rate = learning_rate
        self.iter_num = iter_num
        self.epsilon = epsilon
        self.init_mean = init_mean
        self.init_std = init_std
        self.y_min = y_min
        self.y_max = y_max

        # w0, 截距项
        self.w0 = 0

        # w, 一维向量, 初始化
        self.w = None
        # V, n * k维矩阵, 训练时初始化, 通过正态分布进行初始化, {v1, v2, ... vn}
        self.V = None

        # M用于计算过程的缓存, 后续求梯度的迭代总会使用多次, Vt(k*n) * X(n*p) = M(k*p), M[f, :] = \sum_{j=1}^n v_{j, f} x_j
        self.M = None

        # X沿第二个维度求范式
        self.X_norm = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        训练
        :param X: n*p维
        :param y: n*1维
        :return: 模型本身
        """
        self.n = X.shape[0]
        self.p = X.shape[1]
        # V, n * k维矩阵, 通过正态分布进行初始化, {v1, v2, ... vn}
        self.V = self.init_mean + np.random.randn(self.n, self.k) * self.init_std
        # w, 一维向量, 初始化
        self.w = np.zeros(self.p)
        # V, n * k维矩阵, 训练时初始化, 通过正态分布进行初始化, {v1, v2, ... vn}
        self.V = np.random.randn(self.n, self.k) * self.init_std + self.init_mean
        # M用于计算过程的缓存, 后续求梯度的迭代总会使用多次, Vt(k*n) * X(n*p) = M(k*p), M[f, :] = \sum_{j=1}^n v_{j, f} x_j
        self.M = np.zeros((self.k, self.p))
        # X沿第二个维度求范式
        self.X_norm = np.linalg.norm(X, axis=1)

        epoch = 0
        e = np.inf

        # 迭代
        while epoch < self.iter_num and e > self.epsilon:
            Y_pred = [self.predict(x) for x in X]
            w0_pre = self.w0
            w_pre = np.copy(self.w)
            v_pre = np.copy(self.V)

            for i in range(0, X.shape[0]):
                mult = 2 * (Y_pred[i] - y[i])
                # 截距项
                self.w0 -= self.learning_rate * mult
                # 线性项
                self.w -= self.learning_rate * mult * X[i]
                # 交叉项
                for f in range(0, self.k):
                    grad = np.dot(self.M[f], X[i, :]) - self.V[i, f] * np.dot(X[i, :], X[i, :])
                    self.V[i, f] -= self.learning_rate * grad * mult
            self.V = self.V / (np.linalg.norm(self.V) + 10)

            e = abs(w0_pre - self.w0)
            e += np.linalg.norm(self.w - w_pre, ord=1)
            e += np.linalg.norm(self.V - v_pre, ord=1)
            epoch += 1
        return self

    @staticmethod
    def scale(y, min_value, max_value):
        """
        标准化
        :param y: 原值
        :param min_value: y的最小值
        :param max_value: y的最大值
        :return: 结果
        """
        if min_value is not None:
            y = max(min_value, y)
        if max_value is not None:
            y = min(max_value, y)
        return y

    def predict(self, x: np.ndarray) -> float:
        """
        预测(一条记录)
        :param x: 一条记录, 需要是一维向量
        :return: 预测得到的y
        """
        assert x.ndim == 1

        y_head = self.w0
        y_head += np.dot(self.w, x)

        M_norm2 = np.linalg.norm(self.M, axis=1)
        for _k in range(0, self.k):
            y_head += 0.5 * M_norm2[_k]
            for i in range(0, self.n):
                try:
                    y_head -= 0.5 * self.X_norm[i] * self.V[i, _k] * self.V[i, _k]
                except Exception as e:
                    raise e
        return self.scale(y_head, self.y_min, self.y_max)


if __name__ == '__main__':
    _X = np.random.randn(10, 4)
    _y = np.random.randint(1, 6, 10)

    fm = FM2WayModel(k=2, learning_rate=0.05, init_std=0.5, init_mean=0.0, iter_num=10, y_min=1, y_max=5)
    fm.fit(_X, _y)
    _y_pred = np.array([fm.predict(_x) for _x in _X])
    print(_y_pred, _y)
