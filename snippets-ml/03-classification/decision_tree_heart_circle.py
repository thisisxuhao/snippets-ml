"""
心形曲线 + 噪声

"""
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt

# 生成随机点
n_samples = 1000
theta_samples = np.linspace(0, 2*np.pi, n_samples)
r_samples = np.random.uniform(0, 2, n_samples)  # 半径范围覆盖心形线

# 计算心形边界对应的r值
r_heart = 1 - np.cos(theta_samples)


# heart_theta = np.linspace(0, 2*np.pi, n_samples * 10)
# heart_rho = 1 - np.cos(heart_theta)
# heart_x = np.cos(heart_rho)
# heart_y = np.sin(heart_rho)
# plt.plot(heart_x, heart_y, color='red')

# 标签：心形线内为1，外为0
y = np.where(r_samples <= r_heart, 1, 0)

# 极坐标转笛卡尔坐标
X = np.column_stack([
    r_samples * np.cos(theta_samples),
    r_samples * np.sin(theta_samples)
])

# 添加噪声
noise_mask = np.random.rand(n_samples) < 0.05
y[noise_mask] = 1 - y[noise_mask]

# 添加极坐标特征
X_augmented = np.column_stack([
    X,  # 原始笛卡尔坐标
    np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2),  # 极径 r
    np.arctan2(X[:, 1], X[:, 0])  # 极角 θ
])

# 训练决策树
tree = DecisionTreeClassifier(random_state=42)
# tree = SVC()
tree.fit(X_augmented, y)


# 可视化边界（需在笛卡尔空间投影）
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    # 为网格点添加极坐标特征
    grid_r = np.sqrt(xx ** 2 + yy ** 2)
    grid_theta = np.arctan2(yy, xx)
    grid_augmented = np.column_stack([
        xx.ravel(), yy.ravel(), grid_r.ravel(), grid_theta.ravel()
    ])

    # 预测并绘图
    Z = model.predict(grid_augmented).reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="RdBu")
    plt.axis('equal')


plot_decision_boundary(tree, X, y)
plt.scatter(X[:, 0], X[:, 1], c=y, s=15, edgecolor="k", cmap="RdBu")
plt.title("Decision Tree with Polar Features")
plt.show()
