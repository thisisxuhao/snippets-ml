import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import make_classification
from sklearn.inspection import DecisionBoundaryDisplay
fig, ax = plt.subplots(figsize=(10, 6))
# 生成倾斜线性可分数据
np.random.seed(42)
X, y = make_classification(
    n_samples=500, n_features=2, n_redundant=0, n_clusters_per_class=1,
    flip_y=0, class_sep=1.5, random_state=42
)

x1 = np.linspace(0, 1, 1000)
# 两条抛物线 (x1-0.5) ^ 2 + 0.5, 外加一个偏离中心的异方差
x2 = 8 * (x1 - 0.5) * (x1-0.5) + np.random.normal(scale=0.1, size=len(x1)) * np.power(x1 - 0.5, 2) * 4
x3 = 8 * (x1 - 0.5) * (x1-0.5) - 1 + np.random.normal(scale=0.2, size=len(x1)) * np.power(x1 - 0.5, 2) * 4
# x2 = np.sin(x1 * np.pi * 1.1) + np.random.normal(scale=0.05, size=len(x1))
# x3 = np.sin(x1 * np.pi) + 0.5 + np.random.normal(scale=0.1, size=len(x1))

X = np.concatenate([np.c_[x1, x2], np.c_[x1, x3]])
y = np.concatenate([np.zeros(len(x1)), np.ones(len(x1))])
print(X.shape)
print(y.shape)
noise_mask = np.random.rand(len(y)) < 0.05

# 决策树对噪声较为敏感, 加入噪声后鞠策书的边界更加"不美观"了
y[noise_mask] = 1 - y[noise_mask]
rotation_matrix = np.array([[0.8, -0.6], [0.6, 0.8]])  # 旋转45度
X = np.dot(X, rotation_matrix)

# 训练决策树
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
# clf = NearestCentroid()
# clf = DecisionTreeClassifier()
# clf = RandomForestClassifier(n_estimators=30)

clf.fit(X, y)

# 绘制决策边界

DecisionBoundaryDisplay.from_estimator(
    clf, X, response_method="predict", cmap="RdBu", alpha=0.5, ax=ax
)
ax.scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolor="k", cmap="RdBu")
ax.set_title("Decision Tree Boundary (Tilted Linear Data)\nAxis-aligned splits inefficient", fontsize=12)
plt.xlim((-0.5, 2))
plt.ylim((-1.5, 2))
plt.show()
