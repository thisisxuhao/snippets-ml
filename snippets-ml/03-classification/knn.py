# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    RandomForestClassifier,
)
from sklearn import datasets
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.neighbors import NearestCentroid

# import some data to play with
iris = datasets.load_iris()


# we only take the first two features. We could avoid this ugly
# slicing by using a two-dim dataset
X_all = iris.data[:, [1, 2]]
y_all = iris.target

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)
print(X_train.shape, X_test.shape)
# Create color maps
cmap_light = ListedColormap(["orange", "cyan", "cornflowerblue"])
cmap_bold = ListedColormap(["darkorange", "c", "darkblue"])

# for shrinkage in [None, 0.2]:
# we create an instance of Nearest Centroid Classifier and fit the data.
# clf = NearestCentroid()
# clf = DecisionTreeClassifier()
clf = RandomForestClassifier(n_estimators=30)
clf.fit(X_train, y_train)
y_test_pred = clf.predict(X_test)
print('f1:', f1_score(y_test, y_test_pred, average='macro'))
print('accuracy:', accuracy_score(y_test, y_test_pred))


_, ax = plt.subplots()
DecisionBoundaryDisplay.from_estimator(
    clf, X_all, cmap=cmap_light, ax=ax, response_method="predict"
)

# Plot also the training points
plt.scatter(X_all[:, 0], X_all[:, 1], c=y_all, cmap=cmap_bold, edgecolor="k", s=20)
plt.axis("tight")

plt.show()