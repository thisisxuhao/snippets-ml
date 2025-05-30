from sklearn.datasets import make_circles

# 生成同心圆数据
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
import numpy as np

X_circles, y_circles = make_circles(n_samples=500, noise=0.1, factor=0.5, random_state=42)
noise_mask = np.random.rand(len(y_circles)) < 0.05
y_circles[noise_mask] = 1 - y_circles[noise_mask]
# 训练决策树
tree_circles = DecisionTreeClassifier(max_depth=5, random_state=42)
# tree_circles = RandomForestClassifier(n_estimators=30)
tree_circles.fit(X_circles, y_circles)

# 绘制决策边界
fig, ax = plt.subplots(figsize=(10, 6))
DecisionBoundaryDisplay.from_estimator(
    tree_circles, X_circles, response_method="predict", cmap="RdBu", alpha=0.5, ax=ax
)
ax.scatter(X_circles[:, 0], X_circles[:, 1], c=y_circles, s=30, edgecolor="k", cmap="RdBu")
ax.set_title("Decision Tree Boundary (Concentric Circles)\nRequires deep tree to approximate curves", fontsize=12)
plt.show()

# 可视化树结构（验证复杂度）
plt.figure(figsize=(12, 8))
plot_tree(tree_circles, filled=True, feature_names=["x1", "x2"], class_names=["0", "1"])
plt.show()
