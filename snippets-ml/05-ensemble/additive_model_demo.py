import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd

# 设置随机种子以确保结果可重现
np.random.seed(42)

# # 创建自定义颜色映射
# cmap_light = LinearSegmentedColormap.from_list('light_cmap',
#                                                [(0.9, 0.9, 1.0), (1.0, 0.9, 0.9)],
#                                                N=100)
# cmap_bold = LinearSegmentedColormap.from_list('bold_cmap',
#                                               [(0.2, 0.2, 0.8), (0.8, 0.2, 0.2)],
#                                               N=100)


def visualize_results(X, y, model, title):
    """可视化决策边界"""
    plt.figure(figsize=(10, 8))

    # 创建网格点
    h = 0.02
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
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', s=25, label='类别0')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red', s=25, label='类别1')

    plt.title(title, fontsize=16)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.legend()

    return plt


def main():
    """主函数"""
    # 生成非线性可分数据
    X, y = make_moons(n_samples=300, noise=0.3, random_state=42)

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42, stratify=y)

    print("=" * 50)
    print("使用sklearn实现AdaBoost与逻辑回归基础分类器")
    print("=" * 50)

    # 创建并训练AdaBoost模型
    base_estimator = LogisticRegression(max_iter=1000, random_state=42)
    model = AdaBoostClassifier(
        base_estimator=base_estimator,
        n_estimators=5,
        learning_rate=1.0,
        random_state=42
    )

    model.fit(X_train, y_train)

    # 打印每个基础分类器的系数和权重
    print("基础分类器系数和权重:")
    for i, (clf, alpha) in enumerate(zip(model.estimators_, model.estimator_weights_)):
        print(f"分类器 {i + 1}:")
        print(f"  权重: {alpha:.4f}")
        print(f"  系数: {clf.coef_[0]}")
        print(f"  截距: {clf.intercept_[0]}")
        print()

    # 计算训练集和测试集上的准确率
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    print("=" * 50)
    print(f"训练集准确率: {train_acc:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")

    # 可视化决策边界
    plt = visualize_results(X_scaled, y, model,
                            f"sklearn AdaBoost (逻辑回归基础分类器)\n准确率: {test_acc:.4f}")
    plt.savefig("sklearn_adaboost_logistic_decision_boundary.png", dpi=300, bbox_inches='tight')
    plt.show()

    # 验证加性模型公式
    print("=" * 50)
    print("验证加性模型公式:")

    # 选择几个样本进行验证
    sample_indices = np.random.choice(len(X_test), 3, replace=False)
    for i, idx in enumerate(sample_indices):
        x_sample = X_test[idx:idx + 1]
        y_true = y_test[idx]

        # 使用模型预测
        y_pred_model = model.predict(x_sample)[0]

        # 手动计算预测结果
        weighted_sum = 0
        for clf, alpha in zip(model.estimators_, model.estimator_weights_):
            clf_pred = clf.predict(x_sample)[0]
            # 将0/1转换为-1/1
            clf_pred = 2 * clf_pred - 1  # 0 → -1, 1 → 1
            weighted_sum += alpha * clf_pred

        manual_pred = 1 if weighted_sum > 0 else 0

        print(f"样本 {i + 1}:")
        print(f"  真实标签: {y_true}")
        print(f"  模型预测: {y_pred_model}")
        print(f"  手动计算 (加权和 = {weighted_sum:.4f}): {manual_pred}")
        print(f"  预测一致: {y_pred_model == manual_pred}")
        print()


if __name__ == "__main__":
    main()
