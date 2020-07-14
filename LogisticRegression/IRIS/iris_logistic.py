"""Train a logistic regression classifier to predict whether a flower is iris viriginica or not"""

from types import LambdaType
from sklearn import datasets
import numpy as np
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt

iris = datasets.load_iris()

# Iris dataset keys
iris.keys()

# print data of iris
iris['data']

#print target of iris
iris['target']

# description of iris dataset
print(iris['DESCR'])


# Storing dependent and independent data in variables

X = iris['data'][:,3:]
y = (iris['target'] == 2).astype(np.int) # 2 represents class Virginica here

clf = LogisticRegression()
clf.fit(X, y)
prediction = clf.predict(([[1.6]]))
prediction

prediction = clf.predict(([[2.6]]))
prediction

# Using matplotlib to plot the visualization

X_new = np.linspace(0, 3, 1000).reshape(-1,1)
print(X_new)

y_prob = clf.predict_proba(X_new)

plt.plot(X_new, y_prob[:,1], 'g-', label='virginica')


print(y_prob)