import numpy as np
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


class AdaBoostLogisticRegression:
    def __init__(self, n_estimators=10):
        """
        初始化 AdaBoost 分类器
        :param n_estimators: 基础分类器的数量
        """
        self.n_estimators = n_estimators
        self.models = []  # 存储每个基础分类器
        self.alphas = []  # 存储每个基础分类器的权重

    def fit(self, X, y):
        """
        训练 AdaBoost 分类器
        :param X: 训练数据特征
        :param y: 训练数据标签
        """
        n_samples = len(y)
        # 初始化样本权重
        sample_weights = np.ones(n_samples) / n_samples

        for _ in range(self.n_estimators):
            # 训练基础分类器
            model = DecisionTreeClassifier()
            model.fit(X, y, sample_weight=sample_weights)

            # 计算错误率
            predictions = model.predict(X)
            error = np.sum(sample_weights * (predictions != y))

            # 计算分类器权重
            alpha = 0.5 * np.log((1 - error) / max(error, 1e-10))

            # 更新样本权重
            sample_weights *= np.exp(-alpha * y * predictions)
            sample_weights /= np.sum(sample_weights)

            # 保存模型和权重
            self.models.append(model)
            self.alphas.append(alpha)

    def predict(self, X):
        """
        预测新数据的标签
        :param X: 新数据特征
        :return: 预测的标签
        """
        # 对每个基础分类器进行预测并加权求和
        ensemble_predictions = np.zeros(len(X))
        for model, alpha in zip(self.models, self.alphas):
            predictions = model.predict(X)
            ensemble_predictions += alpha * predictions

        # 返回最终预测结果
        return np.sign(ensemble_predictions)



def visualize_results(X, y, model, title):
    """可视化决策边界"""
    plt.figure(figsize=(10, 8))

    # 创建网格点
    h = 0.01
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    x_max = 2.5
    x_min = -2.5
    y_min = -2.5
    y_max = 2.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # 预测网格点的类别
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 绘制决策边界
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.contour(xx, yy, Z, levels=[0.5], linewidths=2, colors='black')

    # 绘制数据点
    plt.scatter(X[y == -1, 0], X[y == -1, 1], c='blue', s=25, label='-1')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red', s=25, label='1')

    plt.title(title, fontsize=16)
    plt.xlim((x_min, x_max))
    plt.ylim((y_min, y_max))
    plt.legend()

    return plt


# 示例用法
if __name__ == "__main__":
    from sklearn.datasets import make_classification, make_circles, make_moons
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # 生成示例数据
    # X, y = make_circles(n_samples=300, noise=0.1, factor=0.5, random_state=42)
    X, y = make_moons(n_samples=3000, noise=0.05, random_state=42)
    y[y == 0] = -1  # 将标签转换为 {-1, 1}

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练 AdaBoost 分类器
    adaboost = AdaBoostLogisticRegression(n_estimators=50)
    adaboost.fit(X_train, y_train)

    # 预测测试集
    y_pred = adaboost.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    visualize_results(X, y, adaboost, 'adaboost')
    plt.show()
