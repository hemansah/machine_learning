from ridge_lasso import lasso_regressor, ridge_regressor
from matplotlib.pyplot import scatter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('Advertising.csv')
data.head()

data.drop(['Unnamed: 0'], axis=1, inplace=True)
data.head()

def scatter_plot(feature, target):
    plt.figure(figsize=(16,8))
    plt.scatter(
        data[feature],
        data[target],
        c='black'
    )

    plt.xlabel("Money spent on {} ads ($)".format(feature))
    plt.ylabel("Sales ($k)")
    plt.show()


scatter_plot('TV','sales')
scatter_plot('radio','sales')
scatter_plot('newspaper','sales')


## Performing Linear Regression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

X = data.drop(['sales'], axis=1)
y = data['sales'].values.reshape(-1,1)

lin_reg = LinearRegression()
MSE = cross_val_score(lin_reg, X, y, scoring='neg_mean_squared_error', cv=5)
print(np.mean(MSE))

## Performing Ridge Regression

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

ridge = Ridge()

parameters = {'alpha':[1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1,5,10,20]}
ridge_regressor = GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error', cv=5)
ridge_regressor.fit(X, y)

print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)


from sklearn.linear_model import Lasso
lasso = Lasso()
parameters = {'alpha':[1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1,5,10,20]}
lasso_regressor = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv=5)
lasso_regressor.fit(X, y)

print(lasso_regressor.best_score_)
print(lasso_regressor.best_params_)